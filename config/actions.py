import os
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from .common import CommonSettings

class ActionsSettings(BaseSettings):
    """
    Configuration specific to actions detection.
    """
    model_config = SettingsConfigDict(env_prefix='ACTIONS_', env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Detection
    DETECTIONS_JSONL: str = "actions_detections.jsonl"
    CACHE_DIR: str = "actions_preds"
    SAVE_FRAME_JSON: bool = True
    INFER_FPS: int = 8
    MODEL_ID: str = "actions-zzid2/6"

    # Overlay
    OVERLAY_FULL: str = "actions_overlay.mp4"
    SHOW_BOX: bool = True

    # Processed clips output
    CLIPS_JSONL: str = "actions_clips.jsonl"

    def __init__(self, **data: Any):
        super().__init__(**data)
        base_out = CommonSettings().OUTPUT_DIR
        self.DETECTIONS_JSONL = os.path.join(base_out, self.DETECTIONS_JSONL)
        self.CACHE_DIR = os.path.join(base_out, self.CACHE_DIR)
        self.OVERLAY_FULL = os.path.join(base_out, self.OVERLAY_FULL)
        self.CLIPS_JSONL = os.path.join(base_out, self.CLIPS_JSONL)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
