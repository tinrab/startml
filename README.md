# startml

A starting point for training models: PyTorch + Lightning + Hydra, with experiment tracking, checkpointing, early stopping, hyperparameter search and a path from laptop to cloud GPU. Copy it, rename it, replace the example model and data, and train.

| Concern | Tool |
| --- | --- |
| Environment, Python, lockfile | [uv](https://docs.astral.sh/uv/) (+ Nix flake/direnv for system bits) |
| Training loop, devices, precision, DDP/FSDP | [Lightning](https://lightning.ai/docs/pytorch/stable/) |
| Configs and CLI overrides | [Hydra](https://hydra.cc) + OmegaConf |
| Experiment tracking | TensorBoard (default), [W&B](https://wandb.ai), CSV |
| Metrics | torchmetrics |
| Checkpoints, early stopping | Lightning callbacks |
| Hyperparameter search | [Optuna](https://optuna.org) (with pruning) |
| Models and data from the Hub | transformers, datasets |
| Lint, format, types, tests | ruff, ty, pytest, pre-commit, GitHub Actions |
| Task runner | [just](https://just.systems) |
| Reproducible runtime | Docker |

## Quickstart

```bash
# 1. Make a new project from this template
cp -r startml my-project && cd my-project && rm -rf .git .venv outputs && git init
just rename my_project          # renames the package, imports and configs; re-locks

# 2. Environment (Nix + direnv; otherwise install uv and just yourself)
direnv allow                    # or: nix develop
uv sync
cp .env.example .env            # add WANDB_API_KEY, HF_TOKEN

# 3. Check that everything works
just gpu                        # PyTorch sees the GPU?
just test                       # tests (CPU, synthetic data, ~30s)
just smoke                      # one batch through train/val/test on MNIST

# 4. Train
just train experiment=mnist_cnn
```

Each run writes to `outputs/<task_name>/runs/<timestamp>/`: the saved config (`.hydra/`), a log file, `checkpoints/{best,last}.ckpt`, and the logger's files.

## Layout

```
configs/
  train.yaml  eval.yaml  sweep.yaml   entry configs
  data/  model/  trainer/             what to train, and how
  callbacks/  logger/                 checkpointing, early stopping, progress, tracking
  experiment/                         pinned, versioned setups (start here)
  debug/                              fdr, limit, overfit, profiler
  search/                             Optuna search spaces
  local/                              per-machine overrides (gitignored)
src/startml/
  train.py  eval.py  sweep.py         entry points: `uv run train|evaluate|sweep`
  models/classifier.py                generic LightningModule (any net -> logits)
  models/nets/                        plain nn.Modules: MLP, SimpleCNN, HF wrapper
  data/                               LightningDataModules: MNIST, HF text, synthetic
  callbacks/  utils/
notebooks/                            explore.ipynb (inspect a run), colab.ipynb
scripts/                              rename_project.py, setup_remote.sh
tests/
data/  outputs/                       gitignored
```

## Training

All settings come from Hydra configs, and you can override any of them from the command line:

```bash
uv run train                                         # defaults from configs/train.yaml
uv run train experiment=mnist_cnn                    # a pinned experiment
uv run train model=mnist_cnn trainer=gpu data.batch_size=256 model.optimizer.lr=3e-4
uv run train trainer.max_epochs=50 callbacks.early_stopping.patience=10
uv run train logger=wandb tags='[baseline,v2]'
uv run train model.compile=true                      # torch.compile
uv run train ckpt_path=outputs/train/runs/<run>/checkpoints/last.ckpt   # resume

just config experiment=mnist_cnn                     # print the composed config, no run
uv run train --help                                  # list config groups and options
```

Trainer presets: `trainer=default` (auto device, fp32), `gpu` (bf16-mixed), `cpu`, `mps`, `ddp` (all GPUs), `fsdp`.

Debugging presets, which use their own `outputs/debug/` dir and disable loggers:

```bash
just smoke                  # debug=fdr: 1 batch of train/val/test
just overfit                # debug=overfit: loss on 3 batches should go to ~0
uv run train debug=limit    # 3 epochs on 5% of the data
uv run train debug=profiler # where does time go?
```

Hydra multirun gives quick grids. Each run gets its own dir under `outputs/train/multiruns/`:

```bash
uv run train -m model.optimizer.lr=1e-3,3e-4 data.batch_size=64,128
```

### Adding your own model and data

1. **Data:** add `src/startml/data/<name>.py` with a `LightningDataModule` (use `mnist.py` as the reference: download in `prepare_data`, build datasets in `setup`). Add `configs/data/<name>.yaml` with `_target_` and its arguments.
2. **Network:** add an `nn.Module` in `src/startml/models/nets/` and a `configs/model/<name>.yaml`. `ClassifierModule` already handles any network that returns logits, with batches as `(x, y)` tuples or dicts with `labels`.
3. **Other tasks** (regression, segmentation, LMs): copy `models/classifier.py`, change the loss and metrics, and keep the `train/`, `val/` and `test/` metric prefixes. If you change the monitored metric, update `callbacks/default.yaml` (`monitor`, `mode`) and `optimized_metric` in `train.yaml`.
4. **Experiment:** add `configs/experiment/<name>.yaml` that pins data, model, trainer and hyperparameters, then run `uv run train experiment=<name>`. Commit experiment configs so every result can be reproduced.
5. Targets outside the listed packages must be added to `EXECUTION_WHITELIST` in `utils/instantiate.py`. Hydra 1.4 refuses to instantiate anything else.

## Tracking, checkpoints, early stopping

- **Metrics:** `self.log("val/acc", ...)` in the module. Loggers are picked with `logger=`: `tensorboard` (default, view with `just tensorboard`), `wandb`, `csv`, or `many` (all three). The full resolved config and parameter counts are logged as hyperparameters.
- **W&B:** `uv run wandb login`, or set `WANDB_API_KEY` in `.env`. Without network, use `WANDB_MODE=offline uv run train logger=wandb`, then `just wandb-sync <run>/wandb/offline-run-*`. To upload checkpoints as artifacts, set `logger.wandb.log_model=true`.
- **Checkpoints:** `checkpoints/best.ckpt` (top-1 by `val/acc`) and `last.ckpt`. After training, `test: true` evaluates the best checkpoint.
- **Early stopping:** stops after `callbacks.early_stopping.patience` epochs (default 5) without improvement in `val/acc`.
- **Evaluate later:** `uv run evaluate experiment=mnist_cnn ckpt_path=outputs/train/runs/<run>`. A run dir resolves to its `best.ckpt`.
- **In Python or notebooks:**

  ```python
  from startml.utils.checkpoint import load_model_from_run

  model, cfg = load_model_from_run("outputs/train/runs/<run>")  # eval mode, best weights
  ```

  Run `just kernel` once to register the Jupyter kernel (it installs the `notebook` dependency group).

## Hyperparameter search

`uv run sweep` runs an Optuna study over the training config. Each search space lives in `configs/search/` and lists fixed overrides plus parameters to sample:

```bash
just sweep search=mnist_mlp n_trials=50
just optuna-dashboard                     # browse studies
```

- Trials train in-process. Weak trials are pruned after each epoch (median pruner).
- The study is stored in `outputs/optuna.db`, so re-running the same command resumes it.
- The best parameters are written to `outputs/sweeps/<study>/<time>/best.yaml` as ready-to-paste overrides.
- Failed trials (such as CUDA OOM) are recorded as failed and the sweep continues.

## Multi-GPU

```bash
uv run train trainer=ddp                    # all GPUs on this node
uv run train trainer=ddp trainer.devices=2
uv run train trainer=fsdp                   # shard large models
```

`data.batch_size` is global for `MNISTDataModule` and is split across devices.

## Data and model versioning

Git only tracks code and configs. `data/`, `outputs/`, checkpoints and `.env` are gitignored.

- **DVC** for datasets:
  ```bash
  uv tool install "dvc[s3]"       # or [gs], [azure], [ssh]
  dvc init
  dvc remote add -d storage s3://my-bucket/dvc
  dvc add data/my_dataset && git add data/my_dataset.dvc data/.gitignore
  dvc push                        # on another machine: git pull && dvc pull
  ```
- **W&B artifacts** for checkpoints (`logger.wandb.log_model=true`), or `rclone` to any bucket.

## Running elsewhere

**Cloud GPU box** (RunPod, Lambda, Vast.ai, and similar). Use VS Code / Cursor Remote-SSH and the same repo:

```bash
git clone <repo> project && cd project && bash scripts/setup_remote.sh
uv run train experiment=mnist_cnn logger=wandb
```

`setup_remote.sh` installs uv and just, runs `uv sync --locked`, creates `.env`, and checks the GPU. Use `tmux` so runs survive disconnects.

**Google Colab:** open `notebooks/colab.ipynb`. It clones the repo, runs the same setup script, reads secrets from Colab's Secrets panel, and links `outputs/` to Google Drive. Colab works for quick experiments; sessions are ephemeral.

**Docker:**

```bash
just docker-build
just docker-train experiment=mnist_cnn trainer=gpu
```

The image is built from `uv.lock`. PyTorch wheels ship their own CUDA, so the host only needs the NVIDIA driver and container toolkit. On NixOS with CDI, use `--device nvidia.com/gpu=all` instead of `--gpus all`.

## Environment notes

- **PyTorch/CUDA variant:** on Linux, `torch` and `torchvision` come from the CUDA 13.0 index (`[tool.uv.sources]` in `pyproject.toml`); elsewhere they come from PyPI (CPU/MPS). For another CUDA version or CPU-only, point the sources at another index (for example `https://download.pytorch.org/whl/cu128` or the predefined `pytorch-cpu`), then run `uv lock`.
- **NixOS:** the flake's dev shell sets up what PyPI wheels need: `LD_LIBRARY_PATH` with `/run/opengl-driver/lib` (libcuda), `SSL_CERT_FILE` (CA bundle for uv's Python), and `TRITON_LIBCUDA_PATH` (torch.compile). uv-managed Python needs `programs.nix-ld.enable = true` in your system config.
- **Python 3.14** needs Hydra ≥ 1.4 (1.3 breaks on 3.14's argparse). It is pinned to `>=1.4.0rc1` until 1.4.0 is final.
- **Machine-specific settings** (paths, workers, precision) go in `configs/local/default.yaml`. See `default.yaml.example`.

## Development

```bash
just fmt          # ruff format + fix
just check        # format check, ruff, ty, pytest
just test -m "not slow"
uv run pre-commit install
```

Tests run offline on CPU with synthetic data. They cover config composition for every config group, fast dev runs, train → evaluate → reload, resuming, the sweep, and (if a GPU is present) a bf16 run. CI (`.github/workflows/ci.yml`) runs the same checks.
