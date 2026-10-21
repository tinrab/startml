from typing import Any

import hydra
from lightning.pytorch import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from startml.utils.logging import RankedLogger

log = RankedLogger(__name__)

# Packages whose callables configs may reference in `_target_` (Hydra >= 1.4 requires an
# explicit allowlist). Add your own dependencies here.
EXECUTION_WHITELIST = [
    "startml.*",
    "torch.*",
    "torchvision.*",
    "torchmetrics.*",
    "lightning.*",
    "transformers.*",
    "optuna.*",
]


def instantiate(cfg: Any, *args: Any, **kwargs: Any) -> Any:
    """`hydra.utils.instantiate` restricted to `EXECUTION_WHITELIST`."""
    return hydra.utils.instantiate(cfg, *args, _execution_whitelist_=EXECUTION_WHITELIST, **kwargs)


def _instantiate_group[T](cfg: DictConfig | None, kind: str) -> list[T]:
    """Instantiate every entry of a config group that has a `_target_` (null entries are skipped)."""
    objects: list[T] = []
    if not cfg:
        log.warning(f"no {kind} configs found, skipping")
        return objects
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"{kind} config must be a DictConfig")
    for name, conf in cfg.items():
        if isinstance(conf, DictConfig) and "_target_" in conf:
            log.info(f"instantiating {kind} <{conf._target_}> ({name})")
            objects.append(instantiate(conf))
    return objects


def instantiate_callbacks(cfg: DictConfig | None) -> list[Callback]:
    return _instantiate_group(cfg, "callback")


def instantiate_loggers(cfg: DictConfig | None) -> list[Logger]:
    return _instantiate_group(cfg, "logger")
