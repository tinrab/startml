import functools
import sys
import warnings
from collections.abc import Callable
from typing import Any, cast

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from startml.utils.logging import RankedLogger
from startml.utils.rich_utils import print_config_tree

log = RankedLogger(__name__)

type TaskResult = tuple[dict[str, Any], dict[str, Any]]


def extras(cfg: DictConfig) -> None:
    """Apply optional process-wide settings from `cfg.extras`."""
    if not (extra := cfg.get("extras")):
        return

    if extra.get("ignore_warnings"):
        log.info("disabling python warnings (extras.ignore_warnings=True)")
        warnings.filterwarnings("ignore")

    if precision := extra.get("float32_matmul_precision"):
        torch.set_float32_matmul_precision(precision)

    if extra.get("print_config"):
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper[**P](task: Callable[P, TaskResult]) -> Callable[P, TaskResult]:
    """Wrap a task (first argument: the config) so failures are logged and loggers closed.

    Closing W&B explicitly matters for multiruns/sweeps: otherwise the next run in the same
    process would log into the previous run.
    """

    @functools.wraps(task)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> TaskResult:
        cfg = args[0] if args else kwargs["cfg"]
        assert isinstance(cfg, DictConfig)
        try:
            return task(*args, **kwargs)
        except Exception:
            log.exception("task failed")
            raise
        finally:
            log.info(f"output dir: {cfg.paths.output_dir}")
            if "wandb" in sys.modules:
                import wandb

                if wandb.run is not None:
                    log.info("closing wandb")
                    wandb.finish()

    return wrapper


@rank_zero_only
def log_hyperparameters(cfg: DictConfig, model: LightningModule, trainer: Trainer) -> None:
    """Send the full resolved config plus parameter counts to all loggers."""
    if not trainer.loggers:
        return

    hparams = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    hparams.pop("hydra", None)
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(metrics: dict[str, Any], name: str | None) -> float | None:
    """Fetch a scalar metric (used as the objective for hyperparameter search)."""
    if not name:
        return None
    if name not in metrics:
        raise KeyError(
            f"metric {name!r} not found, make sure it is logged in the LightningModule "
            f"(available: {sorted(metrics)})"
        )
    value = metrics[name]
    return float(value.item() if isinstance(value, torch.Tensor) else value)
