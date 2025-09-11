from dataclasses import dataclass


@dataclass
class CourtTrackerConfig:
    feature_max: int = 240
    roi_expand_ratio: float = 0.08
    # Optical flow ROI比播种更大一些，减少大位移丢失
    lk_roi_expand_ratio: float = 0.14
    ransac_reproj_thresh: float = 3.0
    min_inlier_ratio: float = 0.40
    min_inliers: int = 28
    reseed_min_tracks: int = 48
    hold_ttl_frames: int = 10
    use_homography: bool = True
    # Smoothing (Kalman)
    use_kalman: bool = True
    kalman_q_pos: float = 5e-3
    kalman_q_vel: float = 3e-2
    kalman_r_meas: float = 1.5  # pixels std (default)
    # Adaptive measurement noise (when updating with API detections)
    kf_adaptive_from_template: bool = True
    kf_r_api_min: float = 0.7   # lower std bound for high-quality detections
    kf_r_api_max: float = 3.0   # upper std bound for low-quality detections
    ema_alpha: float = 0.9  # fallback if Kalman disabled
    # Geometry gates
    max_jump_px: float = 10.0
    ratio_tolerance: float = 0.28
    area_tolerance: float = 0.45
    # Template score
    use_template_score: bool = True
    template_line_px: int = 8
    template_min_precision: float = 0.28
    # Optical flow robustness
    fb_reproj_thresh: float = 1.1  # forward-backward error threshold (px)
    subpix_win: int = 7  # cornerSubPix half window size
    # Per-frame motion sanity
    max_scale_change_per_frame: float = 0.10  # allow ±10% scale change per frame


__all__ = ["CourtTrackerConfig"]
