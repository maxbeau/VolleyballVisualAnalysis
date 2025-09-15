import os
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from .common import CommonSettings

class PlayersSettings(BaseSettings):
    """
    Configuration specific to players detection and tracking.
    """
    model_config = SettingsConfigDict(env_prefix='PLAYERS_', env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Detection
    DETECTIONS_JSONL: str = "players_detections.jsonl"
    SAVE_FRAME_JSON: bool = True
    INFER_FPS: int = 6
    MODEL_ID: str = "players-dataset-fusrb/1"

    # Tracking
    TRACKS_JSONL: str = "players_tracks.jsonl"
    TRACK_THRESH: float = 0.35
    MATCH_IOU: float = 0.3
    LOW_TRACK_THRESH: float = 0.1
    REID_WEIGHT: float = 0.55
    MAX_AGE: int = 30
    MIN_HITS: int = 3

    # ReID backend
    REID_BACKEND: str = "onnx"  # onnx | hist | torch
    REID_ONNX: str = "weights/osnet_x0_25_msmt17.onnx"
    REID_AUTO_DOWNLOAD: bool = True

    # Interpolation and Hold
    INTERP_ENABLE: bool = True
    INTERP_MAX_GAP: int = 6
    HOLD_TTL_FRAMES: int = 8

    # Overlay
    SHOW_BOX: bool = False
    OVERLAY_FULL: str = "players_overlay.mp4"

    def __init__(self, **data: Any):
        super().__init__(**data)
        base_out = CommonSettings().OUTPUT_DIR
        self.DETECTIONS_JSONL = os.path.join(base_out, self.DETECTIONS_JSONL)
        self.TRACKS_JSONL = os.path.join(base_out, self.TRACKS_JSONL)
        self.OVERLAY_FULL = os.path.join(base_out, self.OVERLAY_FULL)
