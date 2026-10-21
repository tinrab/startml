import os
from pathlib import Path

from dotenv import load_dotenv

# Project root (the directory containing `pyproject.toml`). Configs refer to it via
# `${oc.env:PROJECT_ROOT}`, so set it before Hydra composes anything.
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"

os.environ.setdefault("PROJECT_ROOT", str(ROOT_DIR))

# Secrets and machine-specific settings (WANDB_API_KEY, HF_TOKEN, ...); see `.env.example`.
load_dotenv(ROOT_DIR / ".env", override=False)
