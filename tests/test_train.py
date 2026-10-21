from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from startml.eval import evaluate
from startml.train import train
from startml.utils.checkpoint import load_model_from_run, resolve_checkpoint
from tests.conftest import FAST_OVERRIDES, ComposeFn


def test_fast_dev_run(compose_cfg: ComposeFn) -> None:
    cfg = compose_cfg("train", [*FAST_OVERRIDES, "debug=fdr"])
    metrics, _ = train(cfg)
    assert "test/acc" in metrics


@pytest.mark.slow
def test_train_eval_load(cfg_train: DictConfig, compose_cfg: ComposeFn, tmp_path: Path) -> None:
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 3
    metrics, _ = train(cfg_train)

    assert metrics["val/acc_best"] > 0.5  # random guess is 0.1
    assert (tmp_path / "checkpoints" / "best.ckpt").is_file()
    assert (tmp_path / "checkpoints" / "last.ckpt").is_file()
    assert (tmp_path / "csv" / "metrics.csv").is_file()

    cfg_eval = compose_cfg(
        "eval", ["data=random", "trainer=cpu", "extras.print_config=false", f"ckpt_path={tmp_path}"]
    )
    eval_metrics, _ = evaluate(cfg_eval)
    assert eval_metrics["test/acc"] == pytest.approx(metrics["test/acc"].item(), abs=1e-6)

    # Hydra writes `.hydra/config.yaml` for CLI runs; mimic it.
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(cfg_train, tmp_path / ".hydra" / "config.yaml")
    model, _ = load_model_from_run(tmp_path)
    assert model(torch.randn(2, 1, 28, 28)).shape == (2, 10)


@pytest.mark.slow
def test_resume(cfg_train: DictConfig, tmp_path: Path) -> None:
    train(cfg_train)
    last = resolve_checkpoint(tmp_path, "last")
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.ckpt_path = str(last)
    _, objects = train(cfg_train)
    assert objects["trainer"].current_epoch == 2


def test_resolve_checkpoint_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_train_gpu_bf16(compose_cfg: ComposeFn) -> None:
    overrides = [o for o in FAST_OVERRIDES if not o.startswith("trainer=")]
    cfg = compose_cfg("train", [*overrides, "trainer=gpu", "test=false"])
    metrics, _ = train(cfg)
    assert "val/acc" in metrics
