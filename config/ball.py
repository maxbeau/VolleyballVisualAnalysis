import os
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from .common import CommonSettings

class BallSettings(BaseSettings):
    """
    Configuration specific to ball detection and analysis.
    """
    model_config = SettingsConfigDict(env_prefix='BALL_', env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Detection
    DETECTIONS_JSONL: str = "ball_detections.jsonl"
    SAVE_FRAME_JSON: bool = False
    MODEL_ID: str = "volleyball_v2/3"

    # Manual per-video exclude list for frames (e.g., "20-33,244,252,312,334,346,390-414,545,546")
    EXCLUDE_FRAMES: str = ""

    # Smoothing and Interpolation
    MAX_INTERP_GAP_FRAMES: int = 15
    # Default off per request: disable smoothing/hold by default
    HOLD_MODE: bool = False
    SMOOTHING_ENABLE: bool = False
    HOLD_TTL_FRAMES: int = 5

    # Gating and Gravity
    OBS_GATE_CHISQ_THRESH: float = 18.4
    OBS_GATE_USE_CONF: bool = True
    GRAVITY_PPS2: float = 0.0

    # Soft Weight Filtering
    FILTER_MIN_ASPECT_RATIO: float = 0.7
    FILTER_MAX_ASPECT_RATIO: float = 1.5
    FILTER_AR_SOFT_ALPHA: float = 0.5

    # Kinematic Filtering (image-space)
    KINEMATIC_FILTER_ENABLE: bool = False
    KIN_MAX_SPEED_PX_PER_S: float = 2200
    KIN_MAX_ACCEL_PX_PER_S2: float = 18000
    KIN_MAX_DIR_CHANGE_DEG: float = 135
    KIN_MAX_SIZE_FRAC_PER_S: float = 4.0
    KIN_STATIC_FILTER_ENABLE: bool = True
    KIN_STATIC_MIN_SPEED_PX_PER_S: float = 30
    KIN_STATIC_MIN_FRAMES: int = 8
    KIN_ENABLE_SPEED_GATE: bool = True
    KIN_ENABLE_ACCEL_GATE: bool = True
    KIN_ENABLE_DIR_GATE: bool = True
    KIN_ENABLE_SIZE_GATE: bool = True
    KIN_DYN_ENABLE: bool = True
    KIN_DYN_MIN_MULT: float = 0.7
    KIN_DYN_MAX_MULT: float = 1.5

    # Candidate selection (confidence vs continuity)
    USE_CONTINUITY_SELECTION: bool = True
    CONT_MAX_JUMP_PX: float = 120
    CONT_SEARCH_TOPK: int = 5
    CONT_RESEED_MISSES: int = 3
    RETRO_MIN_SEG_LEN: int = 2
    RETRO_MIN_SEG_MOVE_PX: float = 15
    CONFIRM_RESEED_LOOKAHEAD: int = 2
    CONFIRM_RESEED_MIN_MOVE_PX: float = 8
    CONFIRM_MIN_CONF: float = 0.6
    CONFIRM_MAX_AR_DEV: float = 0.5  # |w/h - 1|

    # Viterbi/DP global path selection
    USE_VITERBI_SELECTION: bool = False
    VIT_TOPK: int = 5
    VIT_MAX_GAP_FRAMES: int = 3
    VIT_GAP_PENALTY: float = 1.0
    VIT_W_CONF: float = 1.0
    VIT_W_AR: float = 0.2
    VIT_W_CIRCLE: float = 0.0
    VIT_W_BORDER: float = 0.0
    IMAGE_BORDER_MARGIN_PX: float = 24
    VIT_W_DIST: float = 0.1
    VIT_W_SIZE: float = 0.1
    VIT_W_DIR: float = 0.0
    VIT_W_ACCEL: float = 0.0
    VIT_DIR_MAX_DEG: float = 180
    VIT_START_PENALTY: float = 0.5

    # Evaluation-only
    EVAL_NONBALL_FRAMES: str = ""

    # Overlay label toggle for near-box tags
    SHOW_NEAR_BOX_TAGS: bool = True  # Show near-box tags like KEPT/FILT

    # Trajectory tail rendering
    TAIL_ENABLE: bool = True
    TAIL_MAX_AGE_FRAMES: int = 12  # shorter default trail length
    TAIL_THICKNESS: int = 2        # line thickness and dot radius
    TAIL_BASE_ALPHA: float = 0.7   # 0..1, newest segment opacity
    TAIL_COLOR: tuple = Field(default=(255, 255, 255))  # BGR white

    def __init__(self, **data: Any):
        super().__init__(**data)
        base_out = CommonSettings().OUTPUT_DIR
        self.DETECTIONS_JSONL = os.path.join(base_out, self.DETECTIONS_JSONL)
        # Backward-compatible env fallbacks for trail settings
        try:
            import os as _os
            def _parse_bool(v: Any) -> bool:
                return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
            def _parse_color(v: str):
                try:
                    parts = [int(p.strip()) for p in str(v).split(',')]
                    if len(parts) == 3:
                        return (parts[0], parts[1], parts[2])
                except Exception:
                    pass
                return self.TAIL_COLOR
            v_show = _os.getenv('BALL_SHOW_TRAIL')
            if v_show is not None:
                self.TAIL_ENABLE = _parse_bool(v_show)
            v_len = _os.getenv('BALL_TRAIL_LENGTH')
            if v_len is not None:
                try:
                    self.TAIL_MAX_AGE_FRAMES = int(float(v_len))
                except Exception:
                    pass
            v_fade = _os.getenv('BALL_TRAIL_FADE_FRAMES')
            if v_fade is not None:
                try:
                    # Interpret fade frames as max age if provided
                    self.TAIL_MAX_AGE_FRAMES = int(float(v_fade))
                except Exception:
                    pass
            v_color = _os.getenv('BALL_TAIL_COLOR')
            if v_color is not None:
                self.TAIL_COLOR = _parse_color(v_color)
        except Exception:
            pass
