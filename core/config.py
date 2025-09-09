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

        # Overlay Output
        self.BALL_OVERLAY_FULL: str = os.getenv("BALL_OVERLAY_FULL", "outputs/ball_overlay_full.mp4")
        self.OVERLAY_MIN_CONF: float = float(os.getenv("OVERLAY_MIN_CONF", "0.1"))
        self.SHOW_BOX_LABELS: bool = os.getenv("SHOW_BOX_LABELS", "false").lower() == "true"
        self.OVERLAY_CODEC: str = os.getenv("OVERLAY_CODEC", "avc1")

        # Smoothing and Interpolation
        self.MAX_INTERP_GAP_FRAMES: int = int(os.getenv("MAX_INTERP_GAP_FRAMES", "15"))
        self.HOLD_MODE: bool = os.getenv("HOLD_MODE", "true").lower() == "true"
        self.HOLD_TTL_FRAMES: int = int(os.getenv("HOLD_TTL_FRAMES", "5"))

        # Gating and Gravity
        self.OBS_GATE_CHISQ_THRESH: float = float(os.getenv("OBS_GATE_CHISQ_THRESH", "18.4"))
        self.OBS_GATE_USE_CONF: bool = os.getenv("OBS_GATE_USE_CONF", "true").lower() == "true"
        self.GRAVITY_PPS2: float = float(os.getenv("GRAVITY_PPS2", "0.0"))

        # Soft Weight Filtering
        self.FILTER_MIN_ASPECT_RATIO: float = float(os.getenv("FILTER_MIN_ASPECT_RATIO", "0.7"))
        self.FILTER_MAX_ASPECT_RATIO: float = float(os.getenv("FILTER_MAX_ASPECT_RATIO", "1.5"))
        self.FILTER_AR_SOFT_ALPHA: float = float(os.getenv("FILTER_AR_SOFT_ALPHA", "0.5"))

        # Court Overlay
        self.COURT_OVERLAY: bool = os.getenv("COURT_OVERLAY", "false").lower() == "true"
        self.COURT_OVERLAY_METHOD: str = os.getenv("COURT_OVERLAY_METHOD", "timeseries")
        self.COURT_COLOR: tuple = tuple(map(int, os.getenv("COURT_COLOR", "0,255,0").split(',')))
        self.COURT_THICKNESS: int = int(os.getenv("COURT_THICKNESS", "2"))

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