from pathlib import Path

import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from startml.utils.instantiate import instantiate


def resolve_checkpoint(path: str | Path, which: str = "best") -> Path:
    """Accept either a checkpoint file or a run directory (then use `checkpoints/{which}.ckpt`)."""
    path = Path(path)
    if path.is_dir():
        path = path / "checkpoints" / f"{which}.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def load_run_config(run_dir: str | Path) -> DictConfig:
    """Load the config Hydra saved for a run (`<run_dir>/.hydra/config.yaml`)."""
    cfg = OmegaConf.load(Path(run_dir) / ".hydra" / "config.yaml")
    assert isinstance(cfg, DictConfig)
    return cfg


def load_model_from_run(
    run_dir: str | Path,
    which: str = "best",
    map_location: str | torch.device = "cpu",
) -> tuple[LightningModule, DictConfig]:
    """Rebuild a run's model from its saved config and load its checkpoint weights.

    Handy in notebooks: `model, cfg = load_model_from_run("outputs/train/runs/<timestamp>")`.
    """
    cfg = load_run_config(run_dir)
    model: LightningModule = instantiate(cfg.model, _convert_="all")
    checkpoint = torch.load(
        resolve_checkpoint(run_dir, which), map_location=map_location, weights_only=True
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, cfg
