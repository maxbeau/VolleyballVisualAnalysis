"""
Configuration loader for the Volleyball Visual Analysis pipeline.

This module uses Pydantic V2 to define a typed configuration structure that
is loaded from and validated against the `pipeline.yaml` file. It provides
a single, reliable source of truth for all pipeline settings.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Any

import yaml
from pydantic import BaseModel, Field, model_validator, SecretStr
from dotenv import load_dotenv

load_dotenv()

# --- Configuration Models ------------------------------------------

class GlobalConfig(BaseModel):
    """Global settings for the pipeline."""
    video_path: Path
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")
    min_confidence: float = Field(0.2, ge=0.0, le=1.0)
    show_box_labels: bool = True

class StepsConfig(BaseModel):
    """Switches to enable or disable pipeline stages."""
    detection: bool = True
    court_processing: bool = True
    court_homography: bool = True
    players_tracking: bool = True
    trajectory_analysis: bool = True
    overlay: bool = True

class DetectionYoloConfig(BaseModel):
    """Settings for the local YOLO detection backend."""
    device: str = "auto"
    default_model: Path = Path("weights/yolov8n.pt")
    court: Path | None = None
    players: Path | None = None
    ball: Path | None = None
    actions: Path | None = None

class DetectionConfig(BaseModel):
    """Settings for the object detection stage."""
    backend: Literal["roboflow", "local-yolo"] = "roboflow"
    cache_policy: Literal["cache_first", "cache_only", "always_infer"] = "cache_first"
    roboflow_api_key: Optional[SecretStr] = Field(
        default=None,
        env="ROBOFLOW_API_KEY",
        exclude=True,
        repr=False,
        description="Loaded exclusively from the ROBOFLOW_API_KEY environment variable",
    )
    infer_fps: Dict[str, int] = {
        "court": 3, "players": 6, "ball": 12, "actions": 2
    }
    models_roboflow: Dict[str, str]
    models_yolo: DetectionYoloConfig

    @model_validator(mode='after')
    def check_roboflow_key(self) -> 'DetectionConfig':
        """Ensure a valid API key is present when using the Roboflow backend."""
        if self.backend == 'roboflow':
            key_val: str | None = None

            if self.roboflow_api_key is not None:
                try:
                    key_val = self.roboflow_api_key.get_secret_value()
                except Exception:
                    key_val = str(self.roboflow_api_key)

            if not key_val or key_val.strip() in ("", "YOUR_API_KEY_HERE"):
                env_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
                if env_key:
                    self.roboflow_api_key = SecretStr(env_key)
                    key_val = env_key

            if not key_val:
                raise ValueError(
                    "ROBOFLOW_API_KEY must be provided via environment when using the 'roboflow' backend. "
                    "Add it to your .env file or export it before running the pipeline."
                )
        return self

class CourtCoreConfig(BaseModel):
    """Core court tracking behavior settings."""
    use_homography: bool = True
    ransac_reproj_thresh: float = 3.0
    hold_ttl_frames: int = 8
    min_inlier_ratio: float = 0.15
    min_inliers: int = 12
    fb_reproj_thresh: float = 2.5
    ema_alpha: float = 0.02

class CourtFeatureConfig(BaseModel):
    """Settings for feature sampling and ROI management."""
    feature_max: int = 220
    roi_expand_ratio: float = 0.16
    lk_roi_expand_ratio: float = 0.12
    reseed_min_tracks: int = 60
    subpix_win: int = 9
    subpix_stride: int = 6

class CourtGateConfig(BaseModel):
    """Settings for geometric and template-based gating."""
    max_jump_px: float = 10.0
    ratio_tolerance: float = 0.30
    area_tolerance: float = 0.50
    max_scale_change_per_frame: float = 0.10
    use_template_score: bool = True
    template_line_px: int = 8
    template_min_precision: float = 0.34
    template_stride: int = 10
    early_motion_gate: bool = True
    early_motion_mad_gray_thr: float = 3.0

class CourtKalmanConfig(BaseModel):
    """Settings for the Kalman filter."""
    use_kalman: bool = True
    kalman_q_pos: float = 1e-2
    kalman_q_vel: float = 5e-2
    kalman_r_meas: float = 2.0
    kf_adaptive_from_template: bool = True
    kf_r_api_min: float = 0.8
    kf_r_api_max: float = 2.5
    kalman_q_scale_from_motion: bool = True
    kalman_q_scale_lo: float = 0.5
    kalman_q_scale_hi: float = 3.0
    motion_md_ref_px: float = 2.0

class CourtPerformanceConfig(BaseModel):
    """Settings for performance and robustness tuning."""
    use_roi_downsample: bool = True
    roi_downsample_scale: float = 0.5
    model_fallback_affine_on_fail: bool = True

class CourtHomographyConfig(BaseModel):
    """Settings for homography and bird's-eye view generation."""
    scale_px_per_meter: float = 20.0
    birdseye_frame_index: int = 100
    model_width: Optional[int] = None
    model_height: Optional[int] = None

class CourtOutputConfig(BaseModel):
    """File paths for court processing outputs."""
    output_tracking_jsonl: str = "court_tracking.jsonl"
    output_meta_json: str = "court_meta.json"
    output_homography_npy: str = "homography.npy"
    output_birdseye_jpg: str = "birdseye.jpg"

class CourtConfig(BaseModel):
    """Settings for court processing and tracking."""
    core: CourtCoreConfig
    features: CourtFeatureConfig
    gates: CourtGateConfig
    kalman: CourtKalmanConfig
    performance: CourtPerformanceConfig
    homography: CourtHomographyConfig
    outputs: CourtOutputConfig

class PlayersTrackingConfig(BaseModel):
    """Configuration for the player tracking step."""

    track_thresh: float = Field(0.35, description="Detection confidence threshold for tracks")
    match_iou: float = Field(0.8, description="IoU threshold for associating detections with existing tracks")
    max_age: int = Field(50, description="Maximum frames to keep a track alive without a detection")
    min_hits: int = Field(3, description="Minimum detections before confirming a track")
    tracker_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="Advanced overrides passed directly to TrackerConfig",
    )
    max_frames: int | None = Field(None, description="Optional cap on frames processed for tracking")
    hold_ttl_frames: int = Field(8, description="Visualization TTL for player boxes after disappearance")
    show_box: bool = Field(True, description="Whether to draw player boxes on the overlay")
    output_tracks_jsonl: str = "players_tracks.jsonl"

    def tracker_kwargs(self) -> Dict[str, Any]:
        """Map high-level settings to TrackerConfig kwargs."""
        kwargs: Dict[str, Any] = {
            "detection_thresh": self.track_thresh,
            "association_iou": self.match_iou,
            "max_age": self.max_age,
            "min_hits": self.min_hits,
        }
        kwargs.update(self.tracker_overrides)
        return kwargs


class TrajectoryConfig(BaseModel):
    """Settings for trajectory analysis."""

    output_jsonl: str = "ball_trajectory.jsonl"
    output_csv: str = "ball_trajectory.csv"
    output_path_img: str = "ball_path.png"
    ar_filter_min: float = 0.5
    ar_filter_max: float = 2.2
    ar_filter_alpha: float = 3.0
    max_interp_gap_frames: int = 10
    hold_ttl_frames: int = 4
    obs_gate_chisq: float = 9.0
    obs_gate_use_conf: bool = True
    gravity_pps2: float = 350.0

    @property
    def max_interp_gap(self) -> int:
        return self.max_interp_gap_frames

    @property
    def hold_ttl(self) -> int:
        return self.hold_ttl_frames


class BallContinuityConfig(BaseModel):
    """Settings for the continuity-based ball selection algorithm."""
    max_jump_px: float = 100.0
    search_topk: int = 3
    reseed_misses: int = 5
    reseed_lookahead: int = 5
    reseed_min_move_px: float = 10.0
    reseed_min_conf: float = 0.1
    reseed_max_ar_dev: float = 1.5
    retro_min_seg_len: int = 3
    retro_min_seg_move_px: float = 20.0

class BallViterbiConfig(BaseModel):
    """Settings for the Viterbi-based ball selection algorithm."""
    topk: int = 5
    gap_penalty: float = 10.0
    start_penalty: float = 5.0
    w_conf: float = 1.0
    w_ar: float = 2.0
    w_circle: float = 0.0
    w_border: float = 0.0
    w_dist: float = 1.0
    w_size: float = 0.5
    w_dir: float = 2.0
    w_accel: float = 1.0
    dir_max_deg: float = 60.0
    image_border_margin_px: float = 10.0

class BallSelectionConfig(BaseModel):
    """Configuration for the ball selection algorithm."""
    method: Literal["continuity", "viterbi", "best_confidence"] = "continuity"
    continuity: BallContinuityConfig
    viterbi: BallViterbiConfig

class BallVizConfig(BaseModel):
    """Visual settings for the ball overlay."""
    hold_mode: bool = True
    hold_ttl_frames: int = 4
    tail_enable: bool = True
    tail_max_age_frames: int = 16
    tail_thickness: int = 2
    tail_base_alpha: float = 0.7
    tail_color: tuple[int, int, int] = (255, 250, 250)
    show_near_box_tags: bool = True

class BallFilterConfig(BaseModel):
    """Settings for filtering ball detections."""
    exclude_frames: str = ""
    min_aspect_ratio: float = 0.5
    max_aspect_ratio: float = 2.0
    ar_soft_alpha: float = 3.0
    kinematic_filter_enable: bool = False

class KinematicFilterConfig(BaseModel):
    """Settings for the kinematic (physics-based) filter."""
    max_speed_px_per_s: float = 3000.0
    max_accel_px_per_s2: float = 6000.0
    max_dir_change_deg: float = 45.0
    max_size_change_frac_per_s: float = 10.0
    static_filter_enable: bool = True
    static_min_speed_px_per_s: float = 20.0
    static_min_frames: int = 4
    enable_speed_gate: bool = True
    enable_accel_gate: bool = True
    enable_dir_gate: bool = True
    enable_size_gate: bool = True
    dyn_enable: bool = True
    dyn_min_mult: float = 0.5
    dyn_max_mult: float = 2.0

class BallSmoothingConfig(BaseModel):
    """Settings for ball trajectory smoothing."""
    enable: bool = False
    obs_gate_chisq_thresh: float = 9.0
    obs_gate_use_conf: bool = True
    gravity_pps2: float = 350.0

class BallEvalConfig(BaseModel):
    """Settings for ball detection evaluation."""
    nonball_frames: str = ""

class OverlayBallConfig(BaseModel):
    """Root configuration for all ball processing in the overlay."""
    selection: BallSelectionConfig
    visualization: BallVizConfig
    filter: BallFilterConfig
    kinematic_filter: KinematicFilterConfig
    smoothing: BallSmoothingConfig
    evaluation: BallEvalConfig

class OverlayCourtConfig(BaseModel):
    enable: bool = True
    method: Literal["timeseries"] = "timeseries"
    color: tuple[int, int, int] = (255, 255, 255)
    thickness: int = 2
    center_color: tuple[int, int, int] = (200, 200, 200)
    attack_color: tuple[int, int, int] = (200, 200, 200)
    roi_filter: bool = True
    show_diag: bool = False
    mini_enable: bool = True
    mini_show_teams: bool = True
    mini_orient_mode: str = "template"
    mini_placement: str = "top-right"
    mini_scale: float = 0.24
    mini_show_label: bool = True
    mini_draw_poly: bool = True

class OverlayPlayersConfig(BaseModel):
    hold_ttl_frames: int = 8
    show_box: bool = True

class OverlayActionsConfig(BaseModel):
    show_box: bool = True
    clips_jsonl: str = "action_clips.jsonl"

class TeamBindingConfig(BaseModel):
    team_a_name: str = "TeamA"
    team_b_name: str = "TeamB"
    action_min_conf: float = 0.25
    serve_cooldown_frames: int = 20
    bind_strategy: str = "earliest"
    team_a_side: str = "auto"
    bind_window_frames: int = 240
    bind_block_to_serve: bool = True
    bind_set_to_receive: bool = True
    bind_block_oppose_spike: bool = True
    bind_block_oppose_spike_window_frames: int = 24
    bind_block_oppose_set: bool = True
    bind_block_oppose_set_window_frames: int = 48
    bind_rally_max_gap_frames: int = 600

class OverlayConfig(BaseModel):
    """Settings for the final video overlay."""
    output_video_path: str = "full_overlay.mp4"
    codec: str = "mp4v"
    ball: OverlayBallConfig
    court: OverlayCourtConfig
    players: OverlayPlayersConfig
    actions: OverlayActionsConfig
    teams: TeamBindingConfig


class PipelineConfig(BaseModel):
    """Root model for the entire pipeline configuration."""
    global_settings: GlobalConfig = Field(..., alias='global')
    steps: StepsConfig
    detection: DetectionConfig
    court: CourtConfig
    players: PlayersTrackingConfig
    trajectory_analysis: TrajectoryConfig
    overlay: OverlayConfig

# --- Loader Function -----------------------------------------------

def load_config(config_path: str | Path = "pipeline.yaml") -> PipelineConfig:
    """
    Loads, validates, and returns the pipeline configuration from a YAML file.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A validated PipelineConfig object.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    return PipelineConfig.parse_obj(config_data)

# --- Singleton Instance --------------------------------------------

# Load the configuration once and make it available for import across the project.
# This avoids reloading the file and ensures a single source of truth.
try:
    settings = load_config()
except FileNotFoundError:
    logging.warning("pipeline.yaml not found. Using default settings.")
    settings = PipelineConfig.parse_obj({})
except Exception as e:
    logging.error(f"Error loading pipeline.yaml: {e}", exc_info=True)
    # Fallback to default settings to allow basic imports to succeed.
    settings = PipelineConfig.parse_obj({})
