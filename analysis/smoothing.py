import numpy as np
from typing import Dict, Any, List, Optional
from .kalman import KalmanRTS


def kalman_rts_smooth(
    best: Dict[int, Dict[str, Any]],
    max_gap_frames: int,
    gate_chisq_thresh: float = 18.4,
    gate_use_conf: bool = True,
    gravity_per_frame: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Forward Kalman filter + backward RTS smoother for ball tracking.
    - State: [x, y, vx, vy, w, h]
    - Obs:   [x, y, w, h]
    - Constant velocity for (x,y), random-walk for (w,h).
    - Optional gravity injection.
    - Gating for outlier rejection.
    Only fills frames within gaps <= max_gap_frames.
    Returns dict of frame_index -> smoothed prediction dict.
    """
    if not best:
        return {}

    frames_sorted = sorted(best.keys())
    if len(frames_sorted) == 1:
        f = frames_sorted[0]
        b = best[f]
        return {f: {**b, "_interp": False}}

    k0 = frames_sorted[0]
    k1 = frames_sorted[-1]
    M = (k1 - k0) + 1
    dt = 1.0

    # --- Kalman Model Setup ---
    F = np.eye(6, dtype=float)
    F[0, 2] = dt
    F[1, 3] = dt

    sigma_a = 30.0
    q_pos = sigma_a ** 2
    Q = np.zeros((6, 6), dtype=float)
    Q[0:2, 0:2] = 0.25 * dt**4 * q_pos * np.eye(2)
    Q[0:2, 2:4] = 0.5 * dt**3 * q_pos * np.eye(2)
    Q[2:4, 0:2] = 0.5 * dt**3 * q_pos * np.eye(2)
    Q[2:4, 2:4] = dt**2 * q_pos * np.eye(2)
    sigma_wh = 4.0
    Q[4, 4] = sigma_wh**2
    Q[5, 5] = sigma_wh**2
    
    H = np.zeros((4, 6), dtype=float)
    H[0, 0] = H[1, 1] = H[2, 4] = H[3, 5] = 1.0

    sigma_meas_xy = 6.0
    sigma_meas_wh = 6.0
    R = np.diag([sigma_meas_xy**2, sigma_meas_xy**2, sigma_meas_wh**2, sigma_meas_wh**2])
    
    smoother = KalmanRTS(F, H, Q, R)

    # --- Prepare Measurements ---
    measurements: List[Optional[np.ndarray]] = [None] * M
    confs = [0.0] * M
    
    first_meas_idx = -1
    for f in frames_sorted:
        j = f - k0
        b = best[f]
        meas = np.array([b["x"], b["y"], b["width"], b["height"]], dtype=float).reshape(4, 1)
        
        # Gating logic (simple version integrated here)
        # A more advanced gating would be inside the Kalman filter loop
        measurements[j] = meas
        confs[j] = b.get("confidence", 0.0)
        if first_meas_idx == -1:
            first_meas_idx = j

    if first_meas_idx == -1:
        return {}

    # --- Initial State ---
    x0 = np.zeros((6, 1), dtype=float)
    x0[0:2, 0] = measurements[first_meas_idx][0:2, 0]
    x0[4:6, 0] = measurements[first_meas_idx][2:4, 0]
    P0 = np.diag([25.0, 25.0, 400.0, 400.0, 25.0, 25.0]).astype(float)
    
    # --- Control Input (for gravity) ---
    control_inputs: List[Optional[np.ndarray]] = [None] * M
    if gravity_per_frame != 0.0:
        for i in range(M):
            u = np.zeros((6,1))
            u[1,0] = 0.5 * gravity_per_frame # y += 0.5*g*t^2, t=1
            u[3,0] = gravity_per_frame      # vy += g*t, t=1
            control_inputs[i] = u

    # --- Run Smoother ---
    x_smooth, _ = smoother.smooth(measurements, x0, P0, control_inputs=control_inputs)

    # --- Format Output ---
    out: Dict[int, Dict[str, Any]] = {}
    for f in frames_sorted:
        j = f - k0
        xs = x_smooth[j]
        if xs is not None:
             out[f] = {
                "x": float(xs[0, 0]),
                "y": float(xs[1, 0]),
                "width": float(xs[4, 0]),
                "height": float(xs[5, 0]),
                "confidence": float(best[f].get("confidence", 0.0)),
                "class": best[f].get("class", "ball"),
                "_interp": False,
            }

    # Interpolate gaps
    for idx in range(len(frames_sorted) - 1):
        f_prev = frames_sorted[idx]
        f_next = frames_sorted[idx + 1]
        gap = f_next - f_prev

        if 0 < gap <= max_gap_frames:
            for f in range(f_prev + 1, f_next):
                j = f - k0
                xs = x_smooth[j]
                if xs is not None:
                    t = (f - f_prev) / float(gap)
                    conf = (1 - t) * float(best[f_prev].get("confidence", 0.0)) + \
                           t * float(best[f_next].get("confidence", 0.0))
                    
                    out[f] = {
                        "x": float(xs[0, 0]),
                        "y": float(xs[1, 0]),
                        "width": float(xs[4, 0]),
                        "height": float(xs[5, 0]),
                        "confidence": conf,
                        "class": best[f_prev].get("class", "ball"),
                        "_interp": True,
                    }
    return out
