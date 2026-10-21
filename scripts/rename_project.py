"""Rename the template's project/package (`startml`) to a new name.

Usage: `just rename my_project` (or `uv run python scripts/rename_project.py my_project`).

Replaces the whole word `startml` in project text files, moves `src/startml` to `src/<name>`,
and re-locks. Run it once, right after creating a project from the template.
"""

import argparse
import re
import shutil
import subprocess
from pathlib import Path

OLD = "startml"
ROOT = Path(__file__).resolve().parents[1]

# Skipped anywhere in the tree / only at the project root (`configs/data` must be renamed).
SKIP_DIRS = {".git", ".venv", ".direnv", "wandb", "__pycache__", ".ruff_cache", ".pytest_cache"}
SKIP_ROOT_DIRS = {"outputs", "data"}
SKIP_FILES = {"uv.lock", "rename_project.py"}
TEXT_SUFFIXES = {".py", ".toml", ".yaml", ".yml", ".md", ".nix", ".ipynb", ".example", ".cfg"}
TEXT_NAMES = {"justfile", "Dockerfile", ".envrc", ".env", ".gitignore", ".dockerignore"}


def text_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if parts[0] in SKIP_ROOT_DIRS or any(part in SKIP_DIRS for part in parts):
            continue
        if not path.is_file() or path.name in SKIP_FILES:
            continue
        if path.suffix in TEXT_SUFFIXES or path.name in TEXT_NAMES:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="new package name (lowercase, underscores allowed)")
    parser.add_argument("--root", type=Path, default=ROOT, help="project root")
    parser.add_argument("--no-lock", action="store_true", help="skip `uv lock`")
    args = parser.parse_args()

    name: str = args.name
    root: Path = args.root.resolve()
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
        parser.error(f"invalid package name {name!r}: use lowercase letters, digits and '_'")

    src = root / "src" / OLD
    if not src.is_dir():
        parser.error(f"{src} not found, was the project already renamed?")

    pattern = re.compile(rf"\b{OLD}\b")
    for path in text_files(root):
        text = path.read_text()
        new_text = pattern.sub(name, text)
        if new_text != text:
            path.write_text(new_text)
            print(f"updated {path.relative_to(root)}")

    shutil.move(src, root / "src" / name)
    print(f"moved src/{OLD} -> src/{name}")

    if not args.no_lock:
        subprocess.run(["uv", "lock"], cwd=root, check=True)
        # A different name length can change import order and line wrapping.
        subprocess.run(["uv", "run", "ruff", "check", "--fix", "-q", "."], cwd=root, check=False)
        subprocess.run(["uv", "run", "ruff", "format", "-q", "."], cwd=root, check=False)
    print("done; run `uv sync` and `just test`")


if __name__ == "__main__":
    main()
