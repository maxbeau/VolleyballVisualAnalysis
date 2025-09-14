from typing import Tuple
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class CourtSettings(BaseSettings):
    """
    Configuration specific to court detection, tracking, and overlay.
    """
    model_config = SettingsConfigDict(env_prefix='COURT_', env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # IO
    DETECTIONS_JSONL: str = "outputs/court_detections.jsonl"
    TRACKING_JSONL: str = "outputs/court_tracking.jsonl"
    TRACKING_META: str = "outputs/court_tracking_meta.json"
    SAVE_JPEGS: bool = False

    # Detection quality (template precision gating)
    DET_USE_TEMPLATE_SCORE: bool = True
    DET_MIN_TEMPLATE_PREC: float = 0.28
    DET_TEMPLATE_LINE_PX: int = 8

    # Tracker advanced params
    LK_ROI_EXPAND_RATIO: float = 0.12
    MAX_SCALE_CHANGE_PER_FRAME: float = 0.08
    KF_ADAPTIVE_FROM_TEMPLATE: bool = True
    KF_R_API_MIN: float = 0.8
    KF_R_API_MAX: float = 2.5

    # Overlay diagnostics
    SHOW_DIAG: bool = False

    # Court Overlay
    OVERLAY: bool = False
    OVERLAY_METHOD: str = "timeseries"
    COLOR: Tuple[int, int, int] = Field(default=(0, 255, 0))
    THICKNESS: int = 2
    
    # Mini bird's-eye overlay
    MINI_ENABLE: bool = True
    MINI_ORIENT_MODE: str = "template"  # template | geometry | force_horizontal | force_vertical
    MINI_SHOW_LABEL: bool = True
    MINI_PLACEMENT: str = "top-right"
    MINI_SCALE: float = 0.24
    MINI_DRAW_POLY: bool = True
    
    # Extra court lines
    CENTER_COLOR: Tuple[int, int, int] = Field(default=(0, 255, 255))
    ATTACK_COLOR: Tuple[int, int, int] = Field(default=(255, 0, 255))

    # Optional ROI filter
    ROI_FILTER: bool = False