from dataclasses import dataclass


@dataclass
class CourtTrackerConfig:
    feature_max: int = 200
    roi_expand_ratio: float = 0.06
    # Optical flow ROI比播种更大一些，减少大位移丢失
    lk_roi_expand_ratio: float = 0.12
    # ROI downsample for LK (speed-up)
    use_roi_downsample: bool = True
    roi_downsample_scale: float = 0.5  # 0.5 => quarter pixels processed
    ransac_reproj_thresh: float = 3.0
    min_inlier_ratio: float = 0.35
    min_inliers: int = 20
    reseed_min_tracks: int = 40
    hold_ttl_frames: int = 8
    use_homography: bool = True
    # Smoothing (Kalman)
    use_kalman: bool = True
    kalman_q_pos: float = 1e-2
    kalman_q_vel: float = 5e-2
    kalman_r_meas: float = 2.0  # pixels std (default)
    # Adapt Q by motion magnitude (median LK displacement)
    kalman_q_scale_from_motion: bool = True
    kalman_q_scale_lo: float = 1.0
    kalman_q_scale_hi: float = 6.0
    motion_md_ref_px: float = 12.0  # md=ref => use hi scale
    # Adaptive measurement noise (when updating with API detections)
    kf_adaptive_from_template: bool = True
    kf_r_api_min: float = 0.8   # lower std bound for high-quality detections
    kf_r_api_max: float = 2.5   # upper std bound for low-quality detections
    ema_alpha: float = 0.85  # fallback if Kalman disabled
    # Geometry gates
    max_jump_px: float = 8.0
    ratio_tolerance: float = 0.3
    area_tolerance: float = 0.5
    # Template score
    use_template_score: bool = True
    template_line_px: int = 8
    template_min_precision: float = 0.28
    template_stride: int = 3  # compute template score every N frames
    # Optical flow robustness
    fb_reproj_thresh: float = 1.2  # forward-backward error threshold (px)
    subpix_win: int = 5  # cornerSubPix half window size
    subpix_stride: int = 3  # refine every N frames (reduce cost)
    # Per-frame motion sanity
    max_scale_change_per_frame: float = 0.08  # allow ±8% scale change per frame
    # Early motion gate (skip LK when scene nearly static in ROI)
    early_motion_gate: bool = True
    early_motion_mad_gray_thr: float = 2.5  # mean abs diff in gray (0-255)
    # Fallback model when homography path fails
    model_fallback_affine_on_fail: bool = True


__all__ = ["CourtTrackerConfig"]
