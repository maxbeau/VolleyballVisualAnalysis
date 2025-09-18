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
    TRACK_THRESH: float = 0.4
    MATCH_IOU: float = 0.35
    LOW_TRACK_THRESH: float = 0.15
    REID_WEIGHT: float = 0.6
    MAX_AGE: int = 24
    MIN_HITS: int = 3
    REID_MIN_SIM: float = 0.3
    SIZE_CHANGE_MAX_RATIO: float = 1.8
    ID_LOCK_AGE: int = 2
    SWITCH_MIN_SIM: float = 0.45
    REID_EXPAND_RATIO: float = 0.15
    REID_FOCUS_TOP: float = 0.78
    REID_MIN_CROP_PX: int = 24
    REID_PROFILE_NEW_THRESH: float = 0.28
    REID_PROFILE_MERGE_THRESH: float = 0.55
    REID_PROFILE_BETA: float = 0.35
    REID_PROFILE_MAX: int = 4
    REID_PROFILE_TTL: int = 120
    COURT_MARGIN: float = 0.18
    COURT_SOFT_MARGIN: float = 0.35
    COURT_MIN_KEEP: int = 6
    COURT_MIN_RATIO: float = 0.55
    COURT_CONF_PENALTY: float = 0.12
    COURT_ALLOW_OUTSIDE: bool = True
    COURT_FALLBACK_RATIO: float = 0.75
    COURT_OUTSIDE_BAND_Y: float = 1.1
    COURT_OUTSIDE_BAND_X: float = 0.2
    MIN_BOX_RATIO: float = 0.022
    MAX_ASPECT_RATIO: float = 3.5
    CONF_BONUS_INSIDE_COURT: float = 0.05
    ASPECT_BYPASS_CONF: float = 0.9
    ASPECT_BYPASS_HEIGHT_RATIO: float = 0.22

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
