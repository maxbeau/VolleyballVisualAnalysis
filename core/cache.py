"""Helpers for organizing cache directory layout."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any
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


def detection_cache_signature(*, backend: str, model_id: str, confidence: float, fps_sample: float, max_frames: int | None) -> str:
    """Return a stable cache signature for one detection configuration."""
    payload: dict[str, Any] = {
        "backend": backend,
        "model_id": model_id,
        "confidence": round(float(confidence), 6),
        "fps_sample": round(float(fps_sample), 6),
        "max_frames": max_frames,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    backend_segment = _safe_segment(backend, "backend")
    return f"{backend_segment}_{hashlib.sha1(raw).hexdigest()[:10]}"


def detection_cache_dir(video_path: Path, cache_root: Path, target: str, signature: str | None = None) -> Path:
    """Return the cache directory for detection results of a target."""
    video_root = video_cache_root(video_path, cache_root)
    target_segment = _safe_segment(target, "target")
    root = video_root / "detection" / target_segment
    return root / _safe_segment(signature, "default") if signature else root


__all__ = ["video_cache_root", "detection_cache_dir", "detection_cache_signature"]
