from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

from orchestration.config import PipelineConfig, get_settings


def _safe_segment(value: str, fallback: str) -> str:
    """Create a safe directory segment from a string."""
    segment = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value).strip("_")
    return segment or fallback


class PipelineContext:
    """
    Manages the execution context for a pipeline run, including configuration,
    paths, and shared state between steps.
    """

    def __init__(self, config: PipelineConfig, create_output_dir: bool = True):
        self.config = config
        self.video_path = self.config.global_settings.video_path
        self.create_output_dir = create_output_dir
        self.output_dir = self._get_video_output_dir()
        self.artifacts: Dict[str, Any] = {}

    def _get_video_output_dir(self) -> Path:
        """
        Return the root output directory for a specific video input,
        e.g., 'outputs/myvideo_a1b2c3d4/'.
        """
        output_dir = self.config.global_settings.output_dir
        try:
            resolved = self.video_path.resolve()
        except Exception:
            resolved = self.video_path

        digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:8]
        stem = _safe_segment(resolved.stem or "video", "video")
        video_output_dir = output_dir / f"{stem}_{digest}"
        if self.create_output_dir:
            video_output_dir.mkdir(parents=True, exist_ok=True)
        return video_output_dir

    def get_artifact_path(self, name: str) -> Path:
        """Get the full path for a registered artifact."""
        if name not in self.artifacts:
            raise KeyError(f"Artifact '{name}' not found in context.")
        
        # Assuming artifacts store relative paths
        return self.output_dir / self.artifacts[name]

    def register_artifact(self, name: str, relative_path: str | Path):
        """Register an artifact with a relative path."""
        self.artifacts[name] = Path(relative_path)

    @classmethod
    def from_settings(cls) -> "PipelineContext":
        """Create a context instance from the global settings."""
        return cls(config=get_settings())
