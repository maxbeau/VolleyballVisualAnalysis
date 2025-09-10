from typing import Dict, Tuple
import numpy as np


def smooth_xy_timeseries(
    meas: Dict[int, Tuple[float, float]],
    total_frames: int,
    q_var: float = 1.0,
    r_var: float = 4.0,
    hold_ttl: int = 0,
) -> Dict[int, Tuple[float, float]]:
    """
    Light 2D constant-velocity Kalman smoothing for sparse (frame->(x,y)) measurements.
    Returns smoothed positions for frames with measurements; optionally fills short gaps
    via prediction up to hold_ttl frames.
    """
    if not meas:
        return {}

    frames = sorted(meas.keys())
    # State x=[x,y,vx,vy]
    x = None
    P = None
    R = np.eye(2) * float(r_var)
    out: Dict[int, Tuple[float, float]] = {}
    last_frame = None
    gap_pred_left = 0

    for f in range(frames[0], frames[-1] + 1):
        has_z = f in meas
        # Compute dt
        if last_frame is None:
            dt = 1.0
        else:
            dt = float(max(1, f - last_frame))
        # Build model
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]], dtype=float)
        Q = (float(q_var) * (G @ G.T))

        if x is None:
            if not has_z:
                continue
            zx, zy = meas[f]
            x = np.array([zx, zy, 0.0, 0.0], dtype=float)
            P = np.eye(4) * 100.0
            z = np.array([zx, zy], dtype=float)
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(4) - K @ H) @ P
            out[f] = (float(x[0]), float(x[1]))
            last_frame = f
            gap_pred_left = hold_ttl
            continue

        # Predict
        x = F @ x
        P = F @ P @ F.T + Q

        if has_z:
            zx, zy = meas[f]
            z = np.array([zx, zy], dtype=float)
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ (z - H @ x)
            P = (np.eye(4) - K @ H) @ P
            out[f] = (float(x[0]), float(x[1]))
            gap_pred_left = hold_ttl
        else:
            # optionally fill short gaps with prediction
            if gap_pred_left > 0:
                out[f] = (float(x[0]), float(x[1]))
                gap_pred_left -= 1
        last_frame = f

    return out


__all__ = ["smooth_xy_timeseries"]

