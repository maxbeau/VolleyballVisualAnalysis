"""
Legacy compatibility wrapper for the detection pipeline.

The real implementation now lives in `volleyball.detection.pipeline`.
This module remains to surface a clear runtime error if older entry-points
are executed without updating their imports.
"""
from __future__ import annotations

import warnings

from volleyball.detection.pipeline import DetectionPipeline as _DetectionPipeline

__all__ = ["DetectionPipeline"]


class DetectionPipeline(_DetectionPipeline):
    def __init__(self, *args, **kwargs):  # pragma: no cover - compatibility guard
        warnings.warn(
            "core.pipeline.DetectionPipeline is deprecated. Import "
            "volleyball.detection.pipeline.DetectionPipeline instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "common_settings" not in kwargs or "target_settings" not in kwargs or "target_name" not in kwargs:
            raise TypeError(
                "The refactored DetectionPipeline requires 'target_name', 'common_settings' and 'target_settings'. "
                "Update your call-site to the new API."
            )
        super().__init__(*args, **kwargs)
