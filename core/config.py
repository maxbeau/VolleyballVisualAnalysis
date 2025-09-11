import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Centralized configuration for the application.
    """

    def __init__(self):
        # Roboflow
        self.ROBOFLOW_API_KEY: Optional[str] = os.getenv("ROBOFLOW_API_KEY")

        # Input and Cache
        self.VIDEO_PATH: str = self._pick_video_path()
        self.INFER_FPS: int = int(os.getenv("INFER_FPS", "12"))
        self.CACHE_DIR: str = os.getenv("CACHE_DIR", "outputs/preds")
        
        # Ball Detection
        self.BALL_DETECTIONS_JSONL: str = os.getenv("BALL_DETECTIONS_JSONL", "outputs/ball_detections.jsonl")
        self.BALL_SAVE_FRAME_JSON: bool = os.getenv("BALL_SAVE_FRAME_JSON", "false").lower() == "true"

        # Court Detection
        self.COURT_DETECTIONS_JSONL: str = os.getenv("COURT_DETECTIONS_JSONL", "outputs/court_detections.jsonl")
        self.COURT_TRACKING_JSONL: str = os.getenv("COURT_TRACKING_JSONL", "outputs/court_tracking.jsonl")
        self.COURT_SAVE_JPEGS: bool = os.getenv("COURT_SAVE_JPEGS", "false").lower() == "true"
        # Court tracker advanced params
        self.LK_ROI_EXPAND_RATIO: float = float(os.getenv("LK_ROI_EXPAND_RATIO", "0.12"))
        self.MAX_SCALE_CHANGE_PER_FRAME: float = float(os.getenv("MAX_SCALE_CHANGE_PER_FRAME", "0.08"))
        self.KF_ADAPTIVE_FROM_TEMPLATE: bool = os.getenv("KF_ADAPTIVE_FROM_TEMPLATE", "true").lower() == "true"
        self.KF_R_API_MIN: float = float(os.getenv("KF_R_API_MIN", "0.8"))
        self.KF_R_API_MAX: float = float(os.getenv("KF_R_API_MAX", "2.5"))
        # Overlay diagnostics
        self.COURT_SHOW_DIAG: bool = os.getenv("COURT_SHOW_DIAG", "false").lower() == "true"

        # Overlay Output
        self.BALL_OVERLAY_FULL: str = os.getenv("BALL_OVERLAY_FULL", "outputs/ball_overlay_full.mp4")
        self.OVERLAY_MIN_CONF: float = float(os.getenv("OVERLAY_MIN_CONF", "0.1"))
        self.SHOW_BOX_LABELS: bool = os.getenv("SHOW_BOX_LABELS", "false").lower() == "true"
        self.OVERLAY_CODEC: str = os.getenv("OVERLAY_CODEC", "avc1")

        # Smoothing and Interpolation
        self.MAX_INTERP_GAP_FRAMES: int = int(os.getenv("MAX_INTERP_GAP_FRAMES", "15"))
        self.HOLD_MODE: bool = os.getenv("HOLD_MODE", "true").lower() == "true"
        self.SMOOTHING_ENABLE: bool = os.getenv("SMOOTHING_ENABLE", "true").lower() == "true"
        self.HOLD_TTL_FRAMES: int = int(os.getenv("HOLD_TTL_FRAMES", "5"))

        # Gating and Gravity
        self.OBS_GATE_CHISQ_THRESH: float = float(os.getenv("OBS_GATE_CHISQ_THRESH", "18.4"))
        self.OBS_GATE_USE_CONF: bool = os.getenv("OBS_GATE_USE_CONF", "true").lower() == "true"
        self.GRAVITY_PPS2: float = float(os.getenv("GRAVITY_PPS2", "0.0"))

        # Soft Weight Filtering
        self.FILTER_MIN_ASPECT_RATIO: float = float(os.getenv("FILTER_MIN_ASPECT_RATIO", "0.7"))
        self.FILTER_MAX_ASPECT_RATIO: float = float(os.getenv("FILTER_MAX_ASPECT_RATIO", "1.5"))
        self.FILTER_AR_SOFT_ALPHA: float = float(os.getenv("FILTER_AR_SOFT_ALPHA", "0.5"))

        # Kinematic Filtering (image-space)
        self.KINEMATIC_FILTER_ENABLE: bool = os.getenv("KINEMATIC_FILTER_ENABLE", "false").lower() == "true"
        # Speed/accel thresholds in pixels per second
        self.KIN_MAX_SPEED_PX_PER_S: float = float(os.getenv("KIN_MAX_SPEED_PX_PER_S", "2200"))
        self.KIN_MAX_ACCEL_PX_PER_S2: float = float(os.getenv("KIN_MAX_ACCEL_PX_PER_S2", "18000"))
        # Max turn angle between successive velocity vectors
        self.KIN_MAX_DIR_CHANGE_DEG: float = float(os.getenv("KIN_MAX_DIR_CHANGE_DEG", "135"))
        # Max fractional size change per second (e.g., 4.0 => 400%/s)
        self.KIN_MAX_SIZE_FRAC_PER_S: float = float(os.getenv("KIN_MAX_SIZE_FRAC_PER_S", "4.0"))
        # Static content filter
        self.KIN_STATIC_FILTER_ENABLE: bool = os.getenv("KIN_STATIC_FILTER_ENABLE", "true").lower() == "true"
        self.KIN_STATIC_MIN_SPEED_PX_PER_S: float = float(os.getenv("KIN_STATIC_MIN_SPEED_PX_PER_S", "30"))
        self.KIN_STATIC_MIN_FRAMES: int = int(os.getenv("KIN_STATIC_MIN_FRAMES", "8"))
        # Gate toggles
        self.KIN_ENABLE_SPEED_GATE: bool = os.getenv("KIN_ENABLE_SPEED_GATE", "true").lower() == "true"
        self.KIN_ENABLE_ACCEL_GATE: bool = os.getenv("KIN_ENABLE_ACCEL_GATE", "true").lower() == "true"
        self.KIN_ENABLE_DIR_GATE: bool = os.getenv("KIN_ENABLE_DIR_GATE", "true").lower() == "true"
        self.KIN_ENABLE_SIZE_GATE: bool = os.getenv("KIN_ENABLE_SIZE_GATE", "true").lower() == "true"
        # Dynamic thresholding by confidence
        self.KIN_DYN_ENABLE: bool = os.getenv("KIN_DYN_ENABLE", "true").lower() == "true"
        self.KIN_DYN_MIN_MULT: float = float(os.getenv("KIN_DYN_MIN_MULT", "0.7"))
        self.KIN_DYN_MAX_MULT: float = float(os.getenv("KIN_DYN_MAX_MULT", "1.5"))

        # Manual per-video exclude list for frames (e.g., "20-33,244,252,312,334,346,390-414,545,546")
        self.BALL_EXCLUDE_FRAMES: str = os.getenv("BALL_EXCLUDE_FRAMES", "")

        # Court Overlay
        self.COURT_OVERLAY: bool = os.getenv("COURT_OVERLAY", "false").lower() == "true"
        self.COURT_OVERLAY_METHOD: str = os.getenv("COURT_OVERLAY_METHOD", "timeseries")
        self.COURT_COLOR: tuple = tuple(map(int, os.getenv("COURT_COLOR", "0,255,0").split(',')))
        self.COURT_THICKNESS: int = int(os.getenv("COURT_THICKNESS", "2"))
        # Extra court lines (center and 3m attack lines)
        self.COURT_CENTER_COLOR: tuple = tuple(map(int, os.getenv("COURT_CENTER_COLOR", "0,255,255").split(',')))
        self.COURT_ATTACK_COLOR: tuple = tuple(map(int, os.getenv("COURT_ATTACK_COLOR", "255,0,255").split(',')))
        # Optional ROI filter: drop ball detections outside court polygon (timeseries)
        self.COURT_ROI_FILTER: bool = os.getenv("COURT_ROI_FILTER", "false").lower() == "true"

        # Candidate selection (confidence vs continuity)
        self.USE_CONTINUITY_SELECTION: bool = os.getenv("USE_CONTINUITY_SELECTION", "true").lower() == "true"
        self.CONT_MAX_JUMP_PX: float = float(os.getenv("CONT_MAX_JUMP_PX", "120"))
        self.CONT_SEARCH_TOPK: int = int(os.getenv("CONT_SEARCH_TOPK", "5"))
        self.CONT_RESEED_MISSES: int = int(os.getenv("CONT_RESEED_MISSES", "3"))
        # Retro pruning of short/static segments (post-pass)
        self.RETRO_MIN_SEG_LEN: int = int(os.getenv("RETRO_MIN_SEG_LEN", "2"))
        self.RETRO_MIN_SEG_MOVE_PX: float = float(os.getenv("RETRO_MIN_SEG_MOVE_PX", "15"))
        # Reseed confirmation (post-pass)
        self.CONFIRM_RESEED_LOOKAHEAD: int = int(os.getenv("CONFIRM_RESEED_LOOKAHEAD", "2"))
        self.CONFIRM_RESEED_MIN_MOVE_PX: float = float(os.getenv("CONFIRM_RESEED_MIN_MOVE_PX", "8"))
        self.CONFIRM_MIN_CONF: float = float(os.getenv("CONFIRM_MIN_CONF", "0.6"))
        self.CONFIRM_MAX_AR_DEV: float = float(os.getenv("CONFIRM_MAX_AR_DEV", "0.5"))  # |w/h - 1|

        # Viterbi/DP global path selection
        self.USE_VITERBI_SELECTION: bool = os.getenv("USE_VITERBI_SELECTION", "false").lower() == "true"
        self.VIT_TOPK: int = int(os.getenv("VIT_TOPK", "5"))
        self.VIT_MAX_GAP_FRAMES: int = int(os.getenv("VIT_MAX_GAP_FRAMES", "3"))
        self.VIT_GAP_PENALTY: float = float(os.getenv("VIT_GAP_PENALTY", "1.0"))
        # Node costs
        self.VIT_W_CONF: float = float(os.getenv("VIT_W_CONF", "1.0"))
        self.VIT_W_AR: float = float(os.getenv("VIT_W_AR", "0.2"))
        self.VIT_W_CIRCLE: float = float(os.getenv("VIT_W_CIRCLE", "0.0"))  # use (1 - q)
        self.VIT_W_BORDER: float = float(os.getenv("VIT_W_BORDER", "0.0"))  # near-frame-edge penalty
        self.IMAGE_BORDER_MARGIN_PX: float = float(os.getenv("IMAGE_BORDER_MARGIN_PX", "24"))
        # Edge costs
        self.VIT_W_DIST: float = float(os.getenv("VIT_W_DIST", "0.1"))
        self.VIT_W_SIZE: float = float(os.getenv("VIT_W_SIZE", "0.1"))
        self.VIT_W_DIR: float = float(os.getenv("VIT_W_DIR", "0.0"))
        self.VIT_W_ACCEL: float = float(os.getenv("VIT_W_ACCEL", "0.0"))
        self.VIT_DIR_MAX_DEG: float = float(os.getenv("VIT_DIR_MAX_DEG", "180"))
        # Start penalty (starting later should not dominate)
        self.VIT_START_PENALTY: float = float(os.getenv("VIT_START_PENALTY", "0.5"))

        # Evaluation-only known non-ball frames (comma/range string). Not used for filtering.
        self.EVAL_NONBALL_FRAMES: str = os.getenv("EVAL_NONBALL_FRAMES", "")

    def _pick_video_path(self) -> str:
        """Finds the video file to process."""
        env_path = os.getenv("VIDEO_PATH")
        if env_path and os.path.exists(env_path):
            return env_path
        
        candidates = ["data/input.mov", "data/input.mp4"]
        for p in candidates:
            if os.path.exists(p):
                return p
        
        raise FileNotFoundError(
            "No video found. Set VIDEO_PATH in .env or place video at data/input.mov or data/input.mp4"
        )

# Create a single, project-wide instance of the settings
settings = Settings()
