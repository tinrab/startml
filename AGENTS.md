# Project notes for coding agents

ML training project: PyTorch Lightning + Hydra configs + W&B/TensorBoard logging + Optuna sweeps,
managed with uv. Python 3.14.

## Layout

- `src/startml/train.py`, `eval.py`, `sweep.py`: entry points (`uv run train|evaluate|sweep`).
- `src/startml/models/`: `LightningModule`s (`classifier.py`) and plain `nn.Module` networks (`nets/`).
- `src/startml/data/`: `LightningDataModule`s. `random.py` is synthetic data for tests.
- `src/startml/utils/`: instantiation (Hydra execution whitelist), logging, checkpoint loading.
- `configs/`: Hydra configs. `train.yaml` composes groups (`data/`, `model/`, `trainer/`,
  `logger/`, `callbacks/`, `experiment/`, `debug/`); `sweep.yaml` + `search/` for Optuna.
- `tests/`: pytest; `conftest.py` composes configs with fast CPU overrides into `tmp_path`.

## Conventions

- Everything a run needs is selected through configs (`_target_`), never hardcoded in `train.py`.
  New model/data: add a class + a YAML in the matching config group + an `experiment/` config.
- Instantiate with `startml.utils.instantiate` (not `hydra.utils.instantiate`): it enforces
  `EXECUTION_WHITELIST`; add new third-party packages there.
- Data/model configs are instantiated with `_convert_="all"` so modules get plain Python types and
  checkpoints load with `weights_only=True`. Keep `save_hyperparameters` to plain data.
- Metrics are logged as `train/…`, `val/…`, `test/…`; checkpoints/early stopping monitor `val/acc`.
- Optimizers/schedulers are Hydra partials (`_partial_: true`).

## Commands

- `just check`: format check, lint (ruff), type-check (ty), tests. Run before finishing a change.
- `just smoke data=random`: one-batch end-to-end run.
- `just config <overrides>`: print the composed config.
- `uv add <pkg>` for dependencies (never pip).
