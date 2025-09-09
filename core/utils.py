import os
from pathlib import Path


def ensure_dir(path: str) -> None:
    """Creates a directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

