from dataclasses import dataclass


@dataclass
class CourtTrackerConfig:
    # Method
    use_homography: bool = True

    # Core RANSAC / hold
    ransac_reproj_thresh: float = 3.0
    hold_ttl_frames: int = 8
    min_inlier_ratio: float = 0.15
    min_inliers: int = 12
    fb_reproj_thresh: float = 2.5

    # ROI / features / subpixel
    feature_max: int = 220
    roi_expand_ratio: float = 0.16
    lk_roi_expand_ratio: float = 0.12
    reseed_min_tracks: int = 60
    subpix_win: int = 9
    subpix_stride: int = 6

    # Geometry gates
    max_jump_px: float = 10.0
    ratio_tolerance: float = 0.30
    area_tolerance: float = 0.50
    max_scale_change_per_frame: float = 0.10

    # Template precision gating
    use_template_score: bool = True
    template_line_px: int = 8
    template_min_precision: float = 0.34
    template_stride: int = 10

    # Kalman and adaptive measurement
    use_kalman: bool = True
    kalman_q_pos: float = 1e-2
    kalman_q_vel: float = 5e-2
    kalman_r_meas: float = 2.0
    kf_adaptive_from_template: bool = True
    kf_r_api_min: float = 0.8
    kf_r_api_max: float = 2.5

    # Performance / robustness
    use_roi_downsample: bool = True
    roi_downsample_scale: float = 0.5
    early_motion_gate: bool = True
    early_motion_mad_gray_thr: float = 3.0
    model_fallback_affine_on_fail: bool = True

    # Adaptive Q from motion magnitude
    kalman_q_scale_from_motion: bool = True
    kalman_q_scale_lo: float = 0.5
    kalman_q_scale_hi: float = 3.0
    motion_md_ref_px: float = 2.0

    # EMA smoothing for corners inside tracker
    ema_alpha: float = 0.02

__all__ = ["CourtTrackerConfig"]

