"""
Configuration loader for the Volleyball Visual Analysis orchestration.

This module uses Pydantic V2 to define a typed configuration structure that
is loaded from and validated against YAML files in the `config/` directory.
It provides a single, reliable source of truth for all orchestration settings.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Any, List

import yaml
from pydantic import BaseModel, Field, model_validator, SecretStr
from dotenv import load_dotenv

load_dotenv()

# --- Configuration Models ------------------------------------------

class GlobalConfig(BaseModel):
    """Global settings for the orchestration."""
    video_path: Path
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")
    min_confidence: float = Field(0.2, ge=0.0, le=1.0)
    show_box_labels: bool = True

class StepsConfig(BaseModel):
    """Switches to enable or disable orchestration stages."""
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
                    "Add it to your .env file or export it before running the orchestration."
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


class BallViterbiConfig(BaseModel):
    """Settings for the Viterbi-based ball selection algorithm."""
    topk: int = 5
    gap_penalty: float = 10.0
    start_penalty: float = 5.0
    max_jump_px: float = 120.0
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


class TrajectorySegmentationSettings(BaseModel):
    enable: bool = True
    gap_frame_threshold: int = 8
    smoothing_window_frames: int = 3
    min_segment_frames: int = 6
    min_speed_mps: float = 1.0
    min_heading_speed_mps: float = 3.0
    speed_jump_abs_mps: float = 2.8
    speed_jump_mad_multiplier: float = 3.0
    speed_jump_ratio_min: float = 2.8
    heading_change_abs_deg: float = 35.0
    heading_change_mad_multiplier: float = 3.0
    height_change_abs_m: float = 0.35
    height_change_mad_multiplier: float = 2.5
    min_height_for_event_m: float = 0.35
    require_speed_or_height_for_flip: bool = True
    gap_guard_frames: int = 4
    vertical_zero_cross_enable: bool = True
    min_vertical_speed_mps: float = 2.0
    speed_jump_noheight_multiplier: float = 1.6
    max_event_height_m: float = 0.45
    combined_score_threshold: float = 1.2
    merge_event_window_frames: int = 3
    speed_drop_ratio: float = 0.55
    annotate_segments: bool = True


class TrajectoryConfig(BaseModel):
    """Settings for trajectory analysis."""

    output_jsonl: str = "ball_trajectory.jsonl"
    output_csv: str = "ball_trajectory.csv"
    output_path_img: str = "ball_path.png"
    output_segments_img: str = "ball_segments_timeline.png"
    height_max_m: float = 3.3
    ar_filter_min: float = 0.5
    ar_filter_max: float = 2.2
    ar_filter_alpha: float = 3.0
    max_interp_gap_frames: int = 10
    hold_ttl_frames: int = 4
    obs_gate_chisq: float = 9.0
    obs_gate_use_conf: bool = True
    max_speed_px_per_frame: float = Field(280.0, description="Max allowable image-plane speed between detections (px/frame)")
    max_accel_px_per_frame2: float = Field(800.0, description="Max allowable change in speed per frame (px/frame^2)")
    speed_reset_frame_gap: int = Field(10, description="Frames after which kinematic gates reset to allow new tracks")
    static_filter_enable: bool = Field(True, description="Enable rejection of near-static detection clusters")
    static_window_frames: int = Field(5, description="Frames required before evaluating static filter")
    static_min_motion_px: float = Field(20.0, description="Minimum displacement across the window to keep detections")
    continuity_filter_enable: bool = Field(True, description="Reject detections that deviate from recent trajectory continuity")
    continuity_window_frames: int = Field(6, description="History window (frames) used to estimate trajectory")
    continuity_max_error_px: float = Field(140.0, description="Maximum allowed deviation from predicted position (pixels)")
    continuity_error_growth_px: float = Field(6.0, description="Additional error budget per frame of gap beyond the first")
    ball_diameter_m: float = Field(0.215, description="Approximate real-world diameter of the volleyball (meters)")
    size_model_min_samples: int = Field(12, description="Minimum samples needed to fit the ground-size regression")
    measurement_confidence_floor: float = Field(0.05, description="Minimum detection confidence considered for smoothing")
    viterbi: BallViterbiConfig = Field(default_factory=BallViterbiConfig)
    segmentation: TrajectorySegmentationSettings = Field(default_factory=TrajectorySegmentationSettings)

    @property
    def max_interp_gap(self) -> int:
        return self.max_interp_gap_frames

    @property
    def hold_ttl(self) -> int:
        return self.hold_ttl_frames

    @property
    def speed_reset_frames(self) -> int:
        return max(1, self.speed_reset_frame_gap)

    @property
    def static_window(self) -> int:
        return max(1, int(self.static_window_frames or 1))

    @property
    def continuity_window(self) -> int:
        return max(2, int(self.continuity_window_frames or 2))


class BallVizConfig(BaseModel):
    """Visual settings for the ball overlay."""
    hold_mode: bool = True
    hold_ttl_frames: int = 4
    tail_enable: bool = True
    tail_max_age_frames: int = 16
    tail_thickness: int = 2
    tail_base_alpha: float = 0.7
    tail_color: tuple[int, int, int] = (255, 250, 250)


class BallSegmentationVizConfig(BaseModel):
    enable: bool = True
    palette: List[tuple[int, int, int]] = Field(
        default_factory=lambda: [
            (244, 67, 54),
            (0, 188, 212),
            (76, 175, 80),
            (255, 235, 59),
            (156, 39, 176),
            (33, 150, 243),
            (255, 152, 0),
            (121, 85, 72),
            (205, 220, 57),
            (63, 81, 181),
        ]
    )
    tail_length_frames: int = 32
    tail_thickness: int = 3
    event_marker_radius: int = 10
    event_marker_color: tuple[int, int, int] = (255, 255, 255)
    label_color: tuple[int, int, int] = (240, 240, 240)
    label_bg: tuple[int, int, int] = (0, 0, 0)
    label_scale: float = 0.55


class BallFilterConfig(BaseModel):
    """Filtering options for ball detections before rendering."""
    kinematic_filter_enable: bool = False
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


class OverlayBallConfig(BaseModel):
    """Root configuration for ball visualization in the overlay."""
    min_confidence: float = 0.2
    filter: BallFilterConfig = Field(default_factory=BallFilterConfig)
    visualization: BallVizConfig
    segmentation_viz: BallSegmentationVizConfig = Field(default_factory=BallSegmentationVizConfig)

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
    enable: bool = True
    hold_ttl_frames: int = 8
    show_box: bool = True

class OverlayActionsConfig(BaseModel):
    enable: bool = True
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
    """Root model for the entire orchestration configuration."""
    global_settings: GlobalConfig = Field(..., alias='global')
    steps: StepsConfig
    detection: Optional[DetectionConfig] = None
    court: Optional[CourtConfig] = None
    players: Optional[PlayersTrackingConfig] = None
    trajectory_analysis: Optional[TrajectoryConfig] = None
    overlay: Optional[OverlayConfig] = None

# --- Loader Function -----------------------------------------------

def load_config(config_dir: str | Path = "config") -> PipelineConfig:
    """
    Loads, validates, and returns the orchestration configuration by merging
    all YAML files in the specified directory.

    Args:
        config_dir: The path to the configuration directory.

    Returns:
        A validated PipelineConfig object.
    """
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    merged_config = {}
    for yaml_file in sorted(config_dir.glob("*.yaml")):
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            if data:
                merged_config.update(data)

    return PipelineConfig.parse_obj(merged_config)

# --- Singleton Instance --------------------------------------------

# Load the configuration once and make it available for import across the project.
# This avoids reloading the file and ensures a single source of truth.
try:
    settings = load_config()
except FileNotFoundError:
    logging.warning("Configuration directory 'config' not found or empty. Using default settings.")
    settings = PipelineConfig.parse_obj({})
except Exception as e:
    logging.error(f"Error loading configuration from 'config' directory: {e}", exc_info=True)
    # Fallback to default settings to allow basic imports to succeed.
    settings = PipelineConfig.parse_obj({})
