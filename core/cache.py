"""Helpers for organizing cache directory layout."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path


SAFE_SEGMENT_RE = re.compile(r"[^0-9A-Za-z]+")


def _safe_segment(value: str, fallback: str) -> str:
    segment = SAFE_SEGMENT_RE.sub("_", value).strip("_")
    return segment or fallback


def video_cache_root(video_path: Path, cache_root: Path) -> Path:
    """Return the root cache directory for a specific video input."""
    try:
        resolved = video_path.resolve()
    except Exception:
        resolved = video_path

    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
    stem = _safe_segment(resolved.stem or "video", "video")
    return cache_root / "videos" / f"{stem}_{digest}"


def detection_cache_dir(video_path: Path, cache_root: Path, target: str) -> Path:
    """Return the cache directory for detection results of a target."""
    video_root = video_cache_root(video_path, cache_root)
    target_segment = _safe_segment(target, "target")
    return video_root / "detection" / target_segment


__all__ = ["video_cache_root", "detection_cache_dir"]
