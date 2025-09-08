import os
from pathlib import Path
from typing import Dict, Tuple


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    env: Dict[str, str] = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip()
    return env


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def pick_video_path(env: Dict[str, str]) -> Tuple[str, float]:
    candidates = []
    if env.get("VIDEO_PATH"):
        candidates.append(env["VIDEO_PATH"]) 
    candidates.extend(["data/input.mov", "data/input.mp4"]) 
    for p in candidates:
        if p and os.path.exists(p):
            return p, 0.0
    raise FileNotFoundError("No video found. Expected data/input.mov or data/input.mp4, or set VIDEO_PATH in .env")

