from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from startml import CONFIG_DIR

# Overrides that make any train config fast, offline and CPU-only.
FAST_OVERRIDES = [
    "data=random",
    "logger=csv",
    "trainer=cpu",
    "trainer.max_epochs=1",
    "extras.print_config=false",
]

type ComposeFn = Callable[..., DictConfig]


@pytest.fixture
def compose_cfg(tmp_path: Path) -> Iterator[ComposeFn]:
    """Compose a config by name with overrides, writing all outputs under `tmp_path`."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):

        def _compose(config_name: str = "train", overrides: list[str] | None = None) -> DictConfig:
            cfg = compose(config_name=config_name, overrides=overrides or [])
            if "paths" in cfg:
                with open_dict(cfg):
                    cfg.paths.output_dir = str(tmp_path)
                    cfg.paths.log_dir = str(tmp_path)
            return cfg

        yield _compose
    GlobalHydra.instance().clear()


@pytest.fixture
def cfg_train(compose_cfg: ComposeFn) -> DictConfig:
    return compose_cfg("train", FAST_OVERRIDES)
