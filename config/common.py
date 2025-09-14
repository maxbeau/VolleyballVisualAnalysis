import os
from typing import Optional, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class CommonSettings(BaseSettings):
    """
    Centralized, common configuration for the application.
    """
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Roboflow
    ROBOFLOW_API_KEY: Optional[str] = None

    # Input and Cache
    VIDEO_PATH: str = ""  # Will be dynamically set
    INFER_FPS: int = 12
    CACHE_DIR: str = "outputs/preds"

    # Overlay Output
    OVERLAY_MIN_CONF: float = 0.1
    SHOW_BOX_LABELS: bool = False
    OVERLAY_CODEC: str = "avc1"
    BALL_OVERLAY_FULL: str = "outputs/ball_overlay_full.mp4"

    @field_validator("VIDEO_PATH", mode="before")
    @classmethod
    def _pick_video_path(cls, v: Any) -> str:
        """Finds the video file to process."""
        # Use the value from .env if it's set and valid
        if v and os.path.exists(v):
            return v
        
        # Otherwise, check candidate paths
        candidates = ["data/input.mov", "data/input.mp4"]
        for p in candidates:
            if os.path.exists(p):
                return p
        
        raise FileNotFoundError(
            "No video found. Set VIDEO_PATH in .env or place video at data/input.mov or data/input.mp4"
        )
