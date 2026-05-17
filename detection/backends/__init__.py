"""Backend implementations for detection."""

from .roboflow import RoboflowBackend
from .ultralytics import UltralyticsBackend
from .yolo import LocalYOLOBackend

__all__ = ["RoboflowBackend", "UltralyticsBackend", "LocalYOLOBackend"]
