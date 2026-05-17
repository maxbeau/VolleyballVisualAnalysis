"""Detection backend factory helpers."""
from __future__ import annotations

from typing import Any, Dict, Type

from .backends.base import DetectionBackend
from .backends.roboflow import RoboflowBackend
from .backends.ultralytics import UltralyticsBackend
from .backends.yolo import LocalYOLOBackend


_BACKENDS: Dict[str, Type[DetectionBackend]] = {
    RoboflowBackend.name: RoboflowBackend,
    UltralyticsBackend.name: UltralyticsBackend,
    LocalYOLOBackend.name: LocalYOLOBackend,
}


def create_detection_backend(common_settings: Any, target_settings: Dict) -> DetectionBackend:
    """Create a detection backend from explicit settings."""
    backend_key = str(target_settings.get("backend") or common_settings.backend).strip().lower()
    if backend_key not in _BACKENDS:
        raise ValueError(
            f"Unsupported detection backend '{backend_key}' in config/detection.yaml. "
            f"Available: {', '.join(sorted(_BACKENDS.keys()))}."
        )
    backend_cls = _BACKENDS[backend_key]
    backend = backend_cls(common_settings, target_settings)
    backend.ensure_ready()
    backend.warmup()
    return backend


__all__ = ["create_detection_backend"]
