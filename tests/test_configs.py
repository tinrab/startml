import pytest
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from startml import CONFIG_DIR
from startml.utils import instantiate, instantiate_callbacks, instantiate_loggers
from tests.conftest import FAST_OVERRIDES, ComposeFn


def group_options(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIG_DIR / group).glob("*.yaml"))


# Configs that need network access (Hub models/datasets) to instantiate.
NEEDS_NETWORK = {"sst2", "distilbert", "sst2_distilbert"}


def test_train_config(cfg_train: DictConfig) -> None:
    assert cfg_train.data and cfg_train.model and cfg_train.trainer
    datamodule = instantiate(cfg_train.data, _convert_="all")
    model = instantiate(cfg_train.model, _convert_="all")
    trainer = instantiate(cfg_train.trainer)
    assert isinstance(datamodule, LightningDataModule)
    assert isinstance(model, LightningModule)
    assert isinstance(trainer, Trainer)
    assert instantiate_callbacks(cfg_train.callbacks)
    assert instantiate_loggers(cfg_train.logger)


def test_eval_config(compose_cfg: ComposeFn) -> None:
    cfg = compose_cfg("eval", ["ckpt_path=some.ckpt"])
    assert cfg.ckpt_path == "some.ckpt"
    instantiate(cfg.model, _convert_="all")


def test_sweep_config(compose_cfg: ComposeFn) -> None:
    for search in group_options("search"):
        cfg = compose_cfg("sweep", [f"search={search}"])
        assert cfg.search.params
        instantiate(cfg.sampler)
        instantiate(cfg.pruner)


@pytest.mark.parametrize("experiment", group_options("experiment"))
def test_experiment_configs(compose_cfg: ComposeFn, experiment: str) -> None:
    cfg = compose_cfg("train", [f"experiment={experiment}", "trainer=cpu"])
    if experiment in NEEDS_NETWORK:
        pytest.skip("needs network")
    instantiate(cfg.model, _convert_="all")
    instantiate(cfg.data, _convert_="all")


@pytest.mark.parametrize("debug", [d for d in group_options("debug") if d != "default"])
def test_debug_configs(compose_cfg: ComposeFn, debug: str) -> None:
    cfg = compose_cfg("train", [*FAST_OVERRIDES, f"debug={debug}"])
    assert cfg.task_name == "debug"
    instantiate(cfg.trainer)


@pytest.mark.parametrize("trainer", group_options("trainer"))
def test_trainer_configs_compose(compose_cfg: ComposeFn, trainer: str) -> None:
    cfg = compose_cfg("train", [f"trainer={trainer}"])
    assert cfg.trainer._target_


@pytest.mark.parametrize("model", sorted(set(group_options("model")) - NEEDS_NETWORK))
def test_model_configs(compose_cfg: ComposeFn, model: str) -> None:
    cfg = compose_cfg("train", [f"model={model}"])
    assert isinstance(instantiate(cfg.model, _convert_="all"), LightningModule)
