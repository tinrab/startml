from pathlib import Path

import pytest
from omegaconf import OmegaConf, open_dict

from startml.sweep import format_override, run
from tests.conftest import FAST_OVERRIDES, ComposeFn


def test_format_override() -> None:
    assert format_override("a.b", 0.5) == "a.b=0.5"
    assert format_override("net.sizes", [64, 32]) == "net.sizes=[64,32]"


@pytest.mark.slow
def test_sweep(compose_cfg: ComposeFn, tmp_path: Path) -> None:
    cfg = compose_cfg("sweep", ["n_trials=3", f"output_dir={tmp_path}"])
    with open_dict(cfg):
        cfg.storage = f"sqlite:///{tmp_path}/optuna.db"
        cfg.study_name = "test"
        cfg.search.overrides = [
            *FAST_OVERRIDES,
            "trainer.max_epochs=2",
            "test=false",
            "callbacks.model_checkpoint=null",
        ]

    study = run(cfg)

    assert len(study.trials) == 3
    (summary_path,) = tmp_path.glob("*/best.yaml")
    summary = OmegaConf.load(summary_path)
    assert set(summary.best_params) == set(cfg.search.params)
    assert len(list(tmp_path.glob("*/trial_*"))) == 3
