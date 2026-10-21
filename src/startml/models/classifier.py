import inspect
from collections.abc import Callable, Mapping
from typing import Any

import torch
from lightning import LightningModule
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MaxMetric, MeanMetric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

type Batch = tuple[Tensor, Tensor] | Mapping[str, Tensor]


class ClassifierModule(LightningModule):
    """Generic multiclass classifier around any `nn.Module` that maps inputs to logits.

    Batches are either `(inputs, targets)` tuples or dicts with a `labels` key (Hugging Face
    style), in which case the remaining keys are passed to the network as keyword arguments.

    `optimizer` and `scheduler` are partials (Hydra `_partial_: true`) completed with the
    parameters / optimizer in `configure_optimizers`.
    """

    def __init__(
        self,
        net: nn.Module,
        optimizer: Callable[..., Optimizer],
        num_classes: int,
        scheduler: Callable[..., LRScheduler] | None = None,
        scheduler_interval: str = "epoch",
        label_smoothing: float = 0.0,
        compile: bool = False,
    ) -> None:
        super().__init__()
        # Keep checkpoints loadable with `weights_only=True`: the net lives in the state dict and
        # the optimizer/scheduler partials (not plain data) are kept as attributes.
        self.save_hyperparameters(logger=False, ignore=["net", "optimizer", "scheduler"])
        self.net = net
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.net(*args, **kwargs)

    def model_step(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(batch, Mapping):
            inputs = dict(batch)
            targets = inputs.pop("labels")
            logits = self(**inputs)
        else:
            x, targets = batch
            logits = self(x)
        loss = self.criterion(logits, targets)
        return loss, logits.argmax(dim=-1), targets

    def on_train_start(self) -> None:
        # Sanity-check validation runs before training; don't let it pollute the best metric.
        self.val_loss.reset()
        self.val_metrics.reset()
        self.val_acc_best.reset()

    def training_step(self, batch: Batch, batch_idx: int) -> Tensor:
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_metrics(preds, targets)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_metrics(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        # torchmetrics' `compute` wrapper confuses type checkers (reports a missing `self`).
        acc = self.val_metrics["acc"].compute()  # ty: ignore[missing-argument]
        self.val_acc_best(acc)
        best = self.val_acc_best.compute()  # ty: ignore[missing-argument]
        self.log("val/acc_best", best, sync_dist=True, prog_bar=True)

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_metrics(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, batch: Batch, batch_idx: int) -> Tensor:
        if isinstance(batch, Mapping):
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            logits = self(**inputs)
        else:
            x = batch[0] if isinstance(batch, list | tuple) else batch
            logits = self(x)
        return torch.softmax(logits, dim=-1)

    def setup(self, stage: str) -> None:
        # In-place compile keeps state dict keys unchanged, so checkpoints stay portable.
        if self.hparams["compile"] and stage == "fit":
            self.net.compile()

    def configure_optimizers(self) -> Any:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.optimizer_factory(params=params)
        if self.scheduler_factory is None:
            return {"optimizer": optimizer}
        kwargs: dict[str, Any] = {}
        # Schedules that need the run length (e.g. warmup + decay from `transformers`).
        if "num_training_steps" in inspect.signature(self.scheduler_factory).parameters:
            kwargs["num_training_steps"] = int(self.trainer.estimated_stepping_batches)
        scheduler = self.scheduler_factory(optimizer=optimizer, **kwargs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": self.hparams["scheduler_interval"],
                "frequency": 1,
            },
        }
