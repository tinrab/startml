"""Hyperparameter search with Optuna over the Hydra training config.

Usage: `uv run sweep search=mnist_mlp n_trials=50` (arguments are Hydra overrides for
`configs/sweep.yaml`). Each trial composes `configs/train.yaml` with the search's fixed
`overrides` plus the sampled `params`, and trains in-process with an epoch-level pruning callback.
The study lives in `storage` (SQLite by default), so re-running the same command resumes it.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from rich.logging import RichHandler

from startml import CONFIG_DIR
from startml.callbacks import OptunaPruning
from startml.train import train
from startml.utils import extras, get_metric_value, instantiate

log = logging.getLogger(__name__)


def suggest(trial: optuna.Trial, name: str, spec: DictConfig) -> Any:
    """Sample one parameter. `spec.type` is `float`, `int` or `categorical`."""
    match spec.type:
        case "float":
            return trial.suggest_float(
                name, spec.low, spec.high, log=spec.get("log", False), step=spec.get("step")
            )
        case "int":
            return trial.suggest_int(
                name, spec.low, spec.high, log=spec.get("log", False), step=spec.get("step", 1)
            )
        case "categorical":
            return trial.suggest_categorical(name, list(spec.choices))
        case other:
            raise ValueError(f"unknown parameter type {other!r} for {name!r}")


def format_override(name: str, value: Any) -> str:
    # Lists (e.g. hidden sizes) must be written as Hydra list literals.
    if isinstance(value, list | tuple):
        value = "[" + ",".join(str(v) for v in value) + "]"
    return f"{name}={value}"


class Objective:
    def __init__(self, cfg: DictConfig, sweep_dir: Path) -> None:
        self.cfg = cfg
        self.sweep_dir = sweep_dir

    def __call__(self, trial: optuna.Trial) -> float:
        search = self.cfg.search
        sampled = {name: suggest(trial, name, spec) for name, spec in search.params.items()}
        output_dir = self.sweep_dir / f"trial_{trial.number:03d}"
        overrides = [
            *search.get("overrides", []),
            *(format_override(k, v) for k, v in sampled.items()),
            f"paths.output_dir={output_dir}",
            f"optimized_metric={self.cfg.metric}",
            # Logger configs use `task_name` for project/group names.
            f"task_name={self.cfg.study_name}",
        ]
        log.info(f"trial {trial.number}: {sampled}")

        train_cfg = compose(config_name="train", overrides=overrides)
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(train_cfg, output_dir / "config.yaml")
        extras(train_cfg)

        pruning = OptunaPruning(trial, monitor=self.cfg.metric)
        metrics, _ = train(train_cfg, extra_callbacks=[pruning])
        if pruning.pruned:
            raise optuna.TrialPruned()

        value = get_metric_value(metrics, self.cfg.metric)
        assert value is not None
        return value


def run(cfg: DictConfig) -> optuna.Study:
    sweep_dir = Path(cfg.output_dir) / f"{datetime.now():%Y-%m-%d_%H-%M-%S}"
    if cfg.storage and cfg.storage.startswith("sqlite:///"):
        Path(cfg.storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=cfg.storage,
        direction=cfg.direction,
        sampler=instantiate(cfg.sampler),
        pruner=instantiate(cfg.pruner),
        load_if_exists=True,
    )
    study.optimize(
        Objective(cfg, sweep_dir),
        n_trials=cfg.n_trials,
        timeout=cfg.get("timeout"),
        catch=(RuntimeError,) if cfg.get("continue_on_error") else (),
        gc_after_trial=True,
    )

    best = study.best_trial
    log.info(f"best trial: #{best.number} {cfg.metric}={best.value}")
    for name, value in best.params.items():
        log.info(f"  {name}: {value}")

    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "study_name": cfg.study_name,
        "metric": cfg.metric,
        "best_value": best.value,
        "best_trial": best.number,
        "best_params": best.params,
        "best_overrides": [format_override(k, v) for k, v in best.params.items()],
    }
    OmegaConf.save(OmegaConf.create(summary), sweep_dir / "best.yaml")
    log.info(f"summary saved to {sweep_dir / 'best.yaml'}")
    return study


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)]
    )
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()

    overrides = sys.argv[1:] if argv is None else argv
    with initialize_config_dir(config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="sweep", overrides=overrides)
        run(cfg)


if __name__ == "__main__":
    main()
