"""Abstract detection backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class DetectionBackend(ABC):
    """Base interface shared by all detection backends."""

    name: str = "abstract"

    def __init__(self, common_settings, target_settings) -> None:
        self.common = common_settings
        self.target = target_settings

    def warmup(self) -> None:
        """Optional warmup hook."""

    def shutdown(self) -> None:
        """Optional teardown hook."""

    @abstractmethod
    def infer(self, frame, *, frame_idx: int, model_id: str, confidence: float) -> Dict[str, Any]:
        """Run inference on a single frame and return a prediction payload."""

    def ensure_ready(self) -> None:
        """Backend specific readiness checks."""


def ensure_confidence(confidence: Optional[float]) -> float:
    if confidence is None:
        return 0.0
    try:
        return float(confidence)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid confidence value: {confidence!r}") from exc


__all__ = ["DetectionBackend", "ensure_confidence"]
