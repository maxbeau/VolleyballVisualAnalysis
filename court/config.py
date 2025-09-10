from dataclasses import dataclass


@dataclass
class CourtTrackerConfig:
    feature_max: int = 200
    roi_expand_ratio: float = 0.06
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
    kalman_r_meas: float = 2.0  # pixels std
    ema_alpha: float = 0.85  # fallback if Kalman disabled
    # Geometry gates
    max_jump_px: float = 8.0
    ratio_tolerance: float = 0.3
    area_tolerance: float = 0.5
    # Template score
    use_template_score: bool = True
    template_line_px: int = 8
    template_min_precision: float = 0.28
    # Optical flow robustness
    fb_reproj_thresh: float = 1.2  # forward-backward error threshold (px)
    subpix_win: int = 5  # cornerSubPix half window size


__all__ = ["CourtTrackerConfig"]

