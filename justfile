# List recipes
default:
    @just --list

# Install/update the environment from uv.lock
sync:
    uv sync

# Train; pass Hydra overrides, e.g. `just train experiment=mnist_cnn trainer=gpu`
train *args:
    uv run train {{args}}

# Evaluate a checkpoint or run dir: `just eval ckpt_path=outputs/train/runs/<run>`
eval *args:
    uv run evaluate {{args}}

# Optuna hyperparameter search: `just sweep search=mnist_mlp n_trials=50`
sweep *args:
    uv run sweep {{args}}

# One batch through train/val/test to catch bugs quickly
smoke *args:
    uv run train debug=fdr {{args}}

# Overfit a few batches (loss should approach 0)
overfit *args:
    uv run train debug=overfit {{args}}

# Print the composed config without running
config *args:
    uv run train --cfg job --resolve {{args}}

# Run tests (`just test -m "not slow"` for the quick ones)
test *args:
    uv run pytest {{args}}

# Format and lint-fix
fmt:
    uv run ruff format .
    uv run ruff check --fix .

# Lint and type-check without modifying files
lint:
    uv run ruff format --check .
    uv run ruff check .
    uv run ty check src tests scripts

# Format, lint, type-check and test
check: lint test

# TensorBoard over all runs
tensorboard dir="outputs":
    uv run tensorboard --logdir {{dir}}

# Optuna dashboard for sweeps
optuna-dashboard:
    uvx optuna-dashboard sqlite:///outputs/optuna.db

# Upload offline W&B runs: `just wandb-sync outputs/train/runs/<run>/wandb/offline-run-*`
wandb-sync *paths:
    uv run wandb sync {{paths}}

# Check that PyTorch sees the GPU
gpu:
    uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('available:', torch.cuda.is_available()); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]"

# Jupyter kernel for this project (VS Code / JupyterLab)
kernel:
    uv sync --group notebook
    uv run python -m ipykernel install --user --name startml --display-name "Python (startml)"

# Rename the project/package: `just rename my_project`
rename name:
    uv run python scripts/rename_project.py {{name}}

# Delete all run outputs
clean-outputs:
    rm -rf outputs

# Build the Docker image
docker-build tag="startml":
    docker build -t {{tag}} .

# Run training in Docker with GPUs: `just docker-train trainer=gpu`
docker-train *args:
    docker run --rm -it --gpus all --ipc=host --env-file .env -v ./data:/app/data -v ./outputs:/app/outputs startml train {{args}}
