import numpy as np
from typing import Dict, List, Tuple, Optional
from analysis.kalman import KalmanRTS


def _kalman_rts_2d(
    meas: Dict[int, Tuple[float, float]],
    first: int,
    last: int,
    q_var: float,
    r_var: float,
) -> Dict[int, Tuple[float, float]]:
    if last < first:
        return {}
    n = (last - first) + 1
    
    # --- Kalman Model Setup ---
    F = np.eye(4, dtype=float)
    F[0, 2] = 1.0
    F[1, 3] = 1.0
    
    H = np.zeros((2, 4), dtype=float)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    
    Q = np.zeros((4, 4), dtype=float)
    Q[0, 0] = 0.25 * q_var
    Q[0, 2] = 0.5 * q_var
    Q[1, 1] = 0.25 * q_var
    Q[1, 3] = 0.5 * q_var
    Q[2, 2] = q_var
    Q[3, 3] = q_var
    
    R = np.eye(2, dtype=float) * r_var
    smoother = KalmanRTS(F, H, Q, R)

    # --- Prepare Measurements ---
    measurements: List[Optional[np.ndarray]] = [None] * n
    first_meas_idx_local = -1
    first_meas_frame = -1

    for f in range(first, last + 1):
        if f in meas:
            j = f - first
            measurements[j] = np.array(meas[f], dtype=float).reshape(2, 1)
            if first_meas_idx_local == -1:
                first_meas_idx_local = j
                first_meas_frame = f
    
    if first_meas_idx_local == -1:
        return {}

    # --- Initial State ---
    x0 = np.zeros((4, 1), dtype=float)
    x0[0:2, 0] = measurements[first_meas_idx_local].flatten()
    P0 = np.diag([100.0, 100.0, 400.0, 400.0]).astype(float)

    # --- Run Smoother ---
    x_smooth, _ = smoother.smooth(measurements, x0, P0)

    # --- Format Output ---
    out: Dict[int, Tuple[float, float]] = {}
    for f in range(first, last + 1):
        j = f - first
        xs = x_smooth[j]
        if xs is not None:
            out[f] = (float(xs[0, 0]), float(xs[1, 0]))
            
    return out


def smooth_corners_timeseries(
    samples: Dict[int, List[Tuple[float, float]]],
    total_frames: int,
    q_var: float = 400.0,
    r_var: float = 36.0,
    hold_ttl: int = 0,
) -> Dict[int, List[Tuple[float, float]]]:
    if not samples:
        return {}
    frames = sorted(samples.keys())
    first = 0
    last = max(total_frames - 1, frames[-1])

    by_corner: List[Dict[int, Tuple[float, float]]] = [dict() for _ in range(4)]
    for f, pts in samples.items():
        if not pts or len(pts) < 4:
            continue
        for i in range(4):
            by_corner[i][f] = (float(pts[i][0]), float(pts[i][1]))

    smoothed_xy: List[Dict[int, Tuple[float, float]]] = []
    for i in range(4):
        smoothed_xy.append(_kalman_rts_2d(by_corner[i], first, last, q_var, r_var))

    out: Dict[int, List[Tuple[float, float]]] = {}
    last_ok: Optional[List[Tuple[float, float]]] = None
    last_ok_frame: Optional[int] = None
    for f in range(first, last + 1):
        pts_f: List[Tuple[float, float]] = []
        ok = True
        for i in range(4):
            xy = smoothed_xy[i].get(f)
            if xy is None:
                ok = False
                break
            pts_f.append((float(xy[0]), float(xy[1])))
        if ok:
            out[f] = pts_f
            last_ok = pts_f
            last_ok_frame = f
        else:
            if hold_ttl > 0 and last_ok is not None and last_ok_frame is not None and (f - last_ok_frame) <= hold_ttl:
                out[f] = last_ok
    return out


def smooth_xy_timeseries(
    samples: Dict[int, Tuple[float, float]],
    total_frames: int,
    q_var: float = 400.0,
    r_var: float = 36.0,
    hold_ttl: int = 0,
) -> Dict[int, Tuple[float, float]]:
    """
    Smooth a single 2D point timeseries over frames [0, total_frames-1]
    using a constant-velocity Kalman + RTS model.
    Fills frames with hold-back if configured and nearby valid frames exist.
    """
    if not samples:
        return {}
    first = 0
    last = max(0, total_frames - 1)
    sm = _kalman_rts_2d(samples, first, last, q_var, r_var)
    if hold_ttl <= 0:
        return sm
    out: Dict[int, Tuple[float, float]] = dict(sm)
    last_ok: Optional[Tuple[float, float]] = None
    last_ok_frame: Optional[int] = None
    for f in range(first, last + 1):
        if f in sm:
            last_ok = sm[f]
            last_ok_frame = f
            continue
        if last_ok is not None and last_ok_frame is not None and (f - last_ok_frame) <= hold_ttl:
            out[f] = last_ok
    return out
