import numpy as np
from typing import Dict, Any, List


def kalman_rts_smooth(
    best: Dict[int, Dict[str, Any]],
    max_gap_frames: int,
    gate_chisq_thresh: float = 18.4,
    gate_use_conf: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """
    Forward Kalman filter + backward RTS smoother over [first,last] observed frames.
    - State: [x, y, vx, vy, w, h]
    - Obs:   [x, y, w, h]
    - Constant velocity model for (x,y). Random-walk for (w,h).
    Only fills frames within gaps <= max_gap_frames. Others are left to hold logic.
    Returns dict of frame_index -> smoothed prediction dict.
    """
    if not best:
        return {}

    frames_sorted = sorted(best.keys())
    if len(frames_sorted) == 1:
        f = frames_sorted[0]
        b = best[f]
        return {
            f: {
                "x": float(b.get("x", 0.0)),
                "y": float(b.get("y", 0.0)),
                "width": float(b.get("width", 0.0)),
                "height": float(b.get("height", 0.0)),
                "confidence": float(b.get("confidence", 0.0)),
                "class": b.get("class", "ball"),
                "_interp": False,
            }
        }

    k0 = frames_sorted[0]
    k1 = frames_sorted[-1]
    M = (k1 - k0) + 1
    dt = 1.0

    F = np.eye(6, dtype=float)
    F[0, 2] = dt
    F[1, 3] = dt

    sigma_a = 30.0
    q_pos = sigma_a ** 2
    Q = np.zeros((6, 6), dtype=float)
    Q[0, 0] = 0.25 * dt ** 4 * q_pos
    Q[0, 2] = 0.5 * dt ** 3 * q_pos
    Q[1, 1] = 0.25 * dt ** 4 * q_pos
    Q[1, 3] = 0.5 * dt ** 3 * q_pos
    Q[2, 0] = 0.5 * dt ** 3 * q_pos
    Q[2, 2] = dt ** 2 * q_pos
    Q[3, 1] = 0.5 * dt ** 3 * q_pos
    Q[3, 3] = dt ** 2 * q_pos
    sigma_wh = 4.0
    Q[4, 4] = sigma_wh ** 2
    Q[5, 5] = sigma_wh ** 2

    H = np.zeros((4, 6), dtype=float)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    H[2, 4] = 1.0
    H[3, 5] = 1.0

    sigma_meas_xy = 6.0
    sigma_meas_wh = 6.0
    R = np.diag([sigma_meas_xy ** 2, sigma_meas_xy ** 2, sigma_meas_wh ** 2, sigma_meas_wh ** 2])

    z: List[np.ndarray] = [None] * M  # type: ignore
    has_meas = [False] * M
    confs = [0.0] * M
    classes = ["ball"] * M
    for f in frames_sorted:
        j = f - k0
        b = best[f]
        z[j] = np.array([
            float(b.get("x", 0.0)),
            float(b.get("y", 0.0)),
            float(b.get("width", 0.0)),
            float(b.get("height", 0.0)),
        ], dtype=float)
        has_meas[j] = True
        confs[j] = float(b.get("confidence", 0.0))
        classes[j] = b.get("class", "ball")

    first_idx = next((j for j in range(M) if has_meas[j]), None)
    if first_idx is None:
        return {}
    x0 = np.zeros((6, 1), dtype=float)
    x0[0, 0] = z[first_idx][0]
    x0[1, 0] = z[first_idx][1]
    x0[2, 0] = 0.0
    x0[3, 0] = 0.0
    x0[4, 0] = z[first_idx][2]
    x0[5, 0] = z[first_idx][3]
    P0 = np.diag([25.0, 25.0, 400.0, 400.0, 25.0, 25.0]).astype(float)

    x_filt: List[np.ndarray] = [None] * M  # type: ignore
    P_filt: List[np.ndarray] = [None] * M  # type: ignore
    x_pred: List[np.ndarray] = [None] * M  # type: ignore
    P_pred: List[np.ndarray] = [None] * M  # type: ignore
    I = np.eye(6, dtype=float)

    x_prev = x0
    P_prev = P0
    for j in range(M):
        if j == first_idx:
            x_pr = F @ x_prev
            P_pr = F @ P_prev @ F.T + Q
            if has_meas[j]:
                y = (z[j].reshape(4, 1) - (H @ x_pr))
                S = H @ P_pr @ H.T + R
                S = S + 1e-6 * np.eye(4)
                K = P_pr @ H.T @ np.linalg.inv(S)
                x_upd = x_pr + K @ y
                P_upd = (I - K @ H) @ P_pr
            else:
                x_upd, P_upd = x_pr, P_pr
            x_pred[j], P_pred[j] = x_pr, P_pr
            x_filt[j], P_filt[j] = x_upd, P_upd
            x_prev, P_prev = x_upd, P_upd
        elif j > first_idx:
            x_pr = F @ x_prev
            P_pr = F @ P_prev @ F.T + Q
            if has_meas[j]:
                y = (z[j].reshape(4, 1) - (H @ x_pr))
                S = H @ P_pr @ H.T + R
                S = S + 1e-6 * np.eye(4)
                try:
                    S_inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    S_inv = np.linalg.pinv(S)
                d2 = float((y.T @ S_inv @ y).ravel()[0])
                thr = gate_chisq_thresh
                if gate_use_conf:
                    conf = float(confs[j]) if j < len(confs) else 1.0
                    thr = gate_chisq_thresh * (0.5 + 0.5 * max(0.0, min(1.0, conf)))
                if d2 <= thr:
                    K = P_pr @ H.T @ S_inv
                    x_upd = x_pr + K @ y
                    P_upd = (I - K @ H) @ P_pr
                else:
                    x_upd, P_upd = x_pr, P_pr
            else:
                x_upd, P_upd = x_pr, P_pr
            x_pred[j], P_pred[j] = x_pr, P_pr
            x_filt[j], P_filt[j] = x_upd, P_upd
            x_prev, P_prev = x_upd, P_upd
        else:
            x_pred[j], P_pred[j] = None, None
            x_filt[j], P_filt[j] = None, None

    last_idx = max(i for i in range(M) if has_meas[i])
    for j in range(first_idx + 1, last_idx + 1):
        if x_pred[j] is None:
            x_pr = F @ x_filt[j - 1]
            P_pr = F @ P_filt[j - 1] @ F.T + Q
            x_pred[j], P_pred[j] = x_pr, P_pr
            x_filt[j], P_filt[j] = x_pr, P_pr

    x_smooth: List[np.ndarray] = [None] * M  # type: ignore
    P_smooth: List[np.ndarray] = [None] * M  # type: ignore
    x_smooth[last_idx] = x_filt[last_idx]
    P_smooth[last_idx] = P_filt[last_idx]
    for j in range(last_idx - 1, first_idx - 1, -1):
        if x_filt[j] is None:
            continue
        Pj = P_filt[j]
        Pj1_pr = P_pred[j + 1]
        if Pj1_pr is None:
            x_smooth[j] = x_filt[j]
            P_smooth[j] = P_filt[j]
            continue
        J = Pj @ F.T @ np.linalg.inv(Pj1_pr)
        x_smooth[j] = x_filt[j] + J @ (x_smooth[j + 1] - x_pred[j + 1])
        P_smooth[j] = Pj + J @ (P_smooth[j + 1] - Pj1_pr) @ J.T

    out: Dict[int, Dict[str, Any]] = {}
    for idx in range(len(frames_sorted) - 1):
        f_prev = frames_sorted[idx]
        f_next = frames_sorted[idx + 1]
        gap = f_next - f_prev
        if gap <= 0:
            continue
        for f in (f_prev, f_next):
            j = f - k0
            xs = x_smooth[j] if x_smooth[j] is not None else x_filt[j]
            if xs is None:
                b = best[f]
                out[f] = {
                    "x": float(b.get("x", 0.0)),
                    "y": float(b.get("y", 0.0)),
                    "width": float(b.get("width", 0.0)),
                    "height": float(b.get("height", 0.0)),
                    "confidence": float(b.get("confidence", 0.0)),
                    "class": b.get("class", "ball"),
                    "_interp": False,
                }
            else:
                out[f] = {
                    "x": float(xs[0, 0]),
                    "y": float(xs[1, 0]),
                    "width": float(xs[4, 0]),
                    "height": float(xs[5, 0]),
                    "confidence": float(best[f].get("confidence", 0.0)),
                    "class": best[f].get("class", "ball"),
                    "_interp": False,
                }

        if gap <= max_gap_frames:
            for f in range(f_prev + 1, f_next):
                j = f - k0
                xs = x_smooth[j] if x_smooth[j] is not None else x_filt[j]
                if xs is None:
                    continue
                t = (f - f_prev) / float(gap)
                # blend confidence linearly between endpoints
                conf = (1 - t) * float(best[f_prev].get("confidence", 0.0)) + t * float(best[f_next].get("confidence", 0.0))
                cls = best[f_prev].get("class", "ball") or best[f_next].get("class", "ball")
                out[f] = {
                    "x": float(xs[0, 0]),
                    "y": float(xs[1, 0]),
                    "width": float(xs[4, 0]),
                    "height": float(xs[5, 0]),
                    "confidence": float(conf),
                    "class": cls,
                    "_interp": True,
                }

    return out

