from typing import Any

import hydra
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from startml import CONFIG_DIR
from startml.utils import (
    RankedLogger,
    extras,
    instantiate,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from startml.utils.checkpoint import resolve_checkpoint

log = RankedLogger(__name__)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test a checkpoint (or a run directory's best checkpoint) on the datamodule's test set."""
    ckpt_path = resolve_checkpoint(cfg.ckpt_path, cfg.get("ckpt_which", "best"))

    log.info(f"instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = instantiate(cfg.data, _convert_="all")

    log.info(f"instantiating model <{cfg.model._target_}>")
    model: LightningModule = instantiate(cfg.model, _convert_="all")

    loggers = instantiate_loggers(cfg.get("logger"))

    log.info(f"instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = instantiate(cfg.trainer, logger=loggers or False)

    objects = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": loggers,
        "trainer": trainer,
    }
    log_hyperparameters(cfg, model, trainer)

    log.info(f"testing checkpoint: {ckpt_path}")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=True)

    return dict(trainer.callback_metrics), objects


@hydra.main(config_path=str(CONFIG_DIR), config_name="eval")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
