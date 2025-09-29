"""Detection backend factory helpers."""
from __future__ import annotations

from typing import Dict, Type

from .backends.base import DetectionBackend
from .backends.roboflow import RoboflowBackend
from .backends.yolo import LocalYOLOBackend


_BACKENDS: Dict[str, Type[DetectionBackend]] = {
    RoboflowBackend.name: RoboflowBackend,
    LocalYOLOBackend.name: LocalYOLOBackend,
}


from pipeline.config import settings

def create_detection_backend(target_settings: Dict) -> DetectionBackend:
    """Creates a detection backend based on the global pipeline settings."""
    backend_key = settings.detection.backend.strip().lower()
    if backend_key not in _BACKENDS:
        raise ValueError(
            f"Unsupported detection backend '{backend_key}' in pipeline.yaml. "
            f"Available: {', '.join(sorted(_BACKENDS.keys()))}."
        )
    backend_cls = _BACKENDS[backend_key]
    backend = backend_cls(settings.detection, target_settings)
    backend.ensure_ready()
    backend.warmup()
    return backend


__all__ = ["create_detection_backend"]
