"""Backend implementations for detection."""

from .roboflow import RoboflowBackend
from .yolo import LocalYOLOBackend

__all__ = ["RoboflowBackend", "LocalYOLOBackend"]
