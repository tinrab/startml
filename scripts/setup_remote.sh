#!/usr/bin/env bash
# Bootstrap a fresh Linux machine (cloud GPU box, Colab, CI) from a clone of this repo:
#   git clone <repo-url> project && cd project && bash scripts/setup_remote.sh
# Afterwards: `uv run train trainer=gpu logger=wandb` (see README).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
	echo "installing uv"
	curl -LsSf https://astral.sh/uv/install.sh | sh
	export PATH="$HOME/.local/bin:$PATH"
fi

# `just` as a uv tool, so the same recipes work everywhere.
command -v just >/dev/null 2>&1 || uv tool install rust-just

uv sync --locked

if [[ ! -f .env ]]; then
	cp .env.example .env
	echo "created .env from .env.example; add WANDB_API_KEY / HF_TOKEN there"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

uv run python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

echo "ready: run \`uv run train\` (or \`just train\`)"
