from typing import Any

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig

from startml import CONFIG_DIR
from startml.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__)


@task_wrapper
def train(
    cfg: DictConfig, extra_callbacks: list[Callback] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the model, then optionally test the best checkpoint. Returns `(metrics, objects)`."""
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.data, _convert_="all")

    log.info(f"instantiating model <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model, _convert_="all")

    callbacks = instantiate_callbacks(cfg.get("callbacks")) + (extra_callbacks or [])
    loggers = instantiate_loggers(cfg.get("logger"))
    if not loggers:
        # LearningRateMonitor raises when there is no logger.
        callbacks = [c for c in callbacks if not isinstance(c, LearningRateMonitor)]

    log.info(f"instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers or False)

    objects = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }
    log_hyperparameters(cfg, model, trainer)

    if cfg.get("train"):
        log.info("starting training")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = dict(trainer.callback_metrics)

    if cfg.get("test"):
        log.info("starting testing")
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if not ckpt_path:
            log.warning("best checkpoint not found, testing current weights")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=True)
        log.info(f"best checkpoint: {ckpt_path}")

    metrics = {**train_metrics, **trainer.callback_metrics}
    return metrics, objects


@hydra.main(config_path=str(CONFIG_DIR), config_name="train")
def main(cfg: DictConfig) -> float | None:
    extras(cfg)
    metrics, _ = train(cfg)
    # Returned value is the objective for Hydra multirun sweepers.
    return get_metric_value(metrics, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
