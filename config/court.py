import os
from typing import Tuple, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from .common import CommonSettings

class CourtSettings(BaseSettings):
    """
    Configuration specific to court detection, tracking, and overlay.
    """
    model_config = SettingsConfigDict(env_prefix='COURT_', env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # IO
    DETECTIONS_JSONL: str = "court_detections.jsonl"
    TRACKING_JSONL: str = "court_tracking.jsonl"
    TRACKING_META: str = "court_tracking_meta.json"
    # Detection
    MODEL_ID: str = "volleyball-court-lurkn/1"
    INFER_FPS: int = 1
    SAVE_FRAME_JSON: bool = True
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

    # Court Overlay (enable by default for integrated overlay)
    OVERLAY: bool = True
    # Auto-generate court detections/tracking when missing (overlay-time)
    AUTO_GENERATE: bool = True
    OVERLAY_METHOD: str = "timeseries"
    COLOR: Tuple[int, int, int] = Field(default=(0, 255, 0))
    THICKNESS: int = 2
    
    # Mini bird's-eye overlay
    MINI_ENABLE: bool = True
    MINI_ORIENT_MODE: str = "template"  # template | geometry | force_horizontal | force_vertical
    MINI_SHOW_LABEL: bool = True
    # Show team names (TeamA/TeamB) on mini bird's-eye
    MINI_SHOW_TEAMS: bool = True
    MINI_PLACEMENT: str = "top-right"
    MINI_SCALE: float = 0.24
    MINI_DRAW_POLY: bool = True
    
    # Extra court lines
    CENTER_COLOR: Tuple[int, int, int] = Field(default=(0, 255, 255))
    ATTACK_COLOR: Tuple[int, int, int] = Field(default=(255, 0, 255))

    # Optional ROI filter
    ROI_FILTER: bool = False

    def __init__(self, **data: Any):
        super().__init__(**data)
        base_out = CommonSettings().OUTPUT_DIR

        def _resolve(p: str) -> str:
            # Absolute path: keep as-is
            if os.path.isabs(p):
                return p
            # If user mistakenly provided a path with a leading "outputs/",
            # normalize to the current video output directory with basename only.
            head = os.path.dirname(p)
            if head and head.split(os.sep)[0].lower() == "outputs":
                return os.path.join(base_out, os.path.basename(p))
            # If already under current base_out, keep
            try:
                if os.path.commonpath([os.path.abspath(os.path.join(os.getcwd(), p)), os.path.abspath(base_out)]) == os.path.abspath(base_out):
                    return p
            except Exception:
                pass
            # Default: join under current video output dir
            return os.path.join(base_out, p)

        self.DETECTIONS_JSONL = _resolve(self.DETECTIONS_JSONL)
        self.TRACKING_JSONL = _resolve(self.TRACKING_JSONL)
        self.TRACKING_META = _resolve(self.TRACKING_META)
