"""Detection namespace exports."""

from .pipeline import DetectionPipeline
from .factory import create_detection_backend

__all__ = ["DetectionPipeline", "create_detection_backend"]
