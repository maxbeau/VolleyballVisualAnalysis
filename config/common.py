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
    VIDEO_PATH: str
    INFER_FPS: int = 12
    
    # Dynamic output paths based on video name
    OUTPUT_DIR: str = ""
    CACHE_DIR: str = ""
    BALL_OVERLAY_FULL: str = ""

    # Overlay Output
    OVERLAY_MIN_CONF: float = 0.1
    SHOW_BOX_LABELS: bool = False
    OVERLAY_CODEC: str = "avc1"

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

    def __init__(self, **data: Any):
        super().__init__(**data)
        if self.VIDEO_PATH:
            video_name = os.path.splitext(os.path.basename(self.VIDEO_PATH))[0]
            self.OUTPUT_DIR = f"outputs/{video_name}"
            self.CACHE_DIR = f"{self.OUTPUT_DIR}/preds"
            self.BALL_OVERLAY_FULL = f"{self.OUTPUT_DIR}/ball_overlay_full.mp4"
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)
            os.makedirs(self.CACHE_DIR, exist_ok=True)
