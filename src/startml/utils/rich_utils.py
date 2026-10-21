from collections.abc import Sequence
from pathlib import Path

import rich
import rich.syntax
import rich.tree
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

DEFAULT_PRINT_ORDER = (
    "data",
    "model",
    "callbacks",
    "logger",
    "trainer",
    "paths",
    "extras",
)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = DEFAULT_PRINT_ORDER,
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print the composed config as a rich tree, optionally saving it to the run's output dir."""
    tree = rich.tree.Tree("CONFIG", style="dim", guide_style="dim")

    fields = [f for f in print_order if f in cfg]
    fields += [str(f) for f in cfg if f not in fields]

    for field in fields:
        branch = tree.add(str(field), style="dim", guide_style="dim")
        value = cfg[field]
        if isinstance(value, DictConfig):
            content = OmegaConf.to_yaml(value, resolve=resolve)
        else:
            content = str(value)
        branch.add(rich.syntax.Syntax(content, "yaml"))

    rich.print(tree)

    if save_to_file:
        path = Path(cfg.paths.output_dir) / "config_tree.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            rich.print(tree, file=file)
