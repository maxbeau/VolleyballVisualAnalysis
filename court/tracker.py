from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2

from orchestration.config import CourtConfig
from court.utils import (
    order_corners,
    apply_homography_points,
    compute_homography,
    build_court_model_template,
    template_precision_score,
    shape_metrics as _shape_metrics_util,
    within_tol as _within_tol_util,
)


Point = Tuple[float, float]


class MultiCornerKalman:
    """Kalman filter for 4 corners with constant-velocity model.
    State: positions(8) + velocities(8) = 16-dim; Measurement: positions(8).
    """
    def __init__(self, q_pos: float = 1e-2, q_vel: float = 5e-2, r_meas: float = 2.0) -> None:
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_meas = float(r_meas)
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.F: Optional[np.ndarray] = None
        self.H: Optional[np.ndarray] = None
        self.Q: Optional[np.ndarray] = None
        self.R: Optional[np.ndarray] = None

    def reset(self, corners: np.ndarray) -> None:
        c = corners.reshape(4, 2).astype(np.float64)
        pos = c.flatten()  # 8
        vel = np.zeros_like(pos)
        self.x = np.concatenate([pos, vel])  # (16,)
        self.P = np.eye(16) * 10.0
        I8 = np.eye(8)
        Z8 = np.zeros((8, 8))
        # F for dt=1: [I, I; 0, I]
        self.F = np.block([[I8, I8], [Z8, I8]])
        # H selects position components
        self.H = np.block([np.eye(8), np.zeros((8, 8))])
        # Process and measurement noise
        Qpos = np.eye(8) * self.q_pos
        Qvel = np.eye(8) * self.q_vel
        self.Q = np.block([[Qpos, np.zeros((8, 8))], [np.zeros((8, 8)), Qvel]])
        self.R = np.eye(8) * (self.r_meas ** 2)

    def predict(self) -> None:
        if self.x is None:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, corners: Optional[np.ndarray]) -> np.ndarray:
        # Predict
        self.predict()
        # Initialize on first call if needed
        if self.x is None:
            if corners is None:
                raise RuntimeError("Kalman: no state and no measurement")
            self.reset(corners)
            return corners.astype(np.float64)
        # Update if measurement available
        if corners is not None:
            z = corners.reshape(-1).astype(np.float64)
            y = z - (self.H @ self.x)
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            I = np.eye(self.P.shape[0])
            self.P = (I - K @ self.H) @ self.P
        # Return smoothed positions
        pos = (self.H @ self.x).reshape(4, 2)
        return pos

    def set_process_scale(self, scale: float) -> None:
        """Rebuild process noise Q using base q_pos/q_vel scaled by factor."""
        try:
            s = float(scale)
            if s <= 0:
                s = 1.0
            I8 = np.eye(8)
            Qpos = I8 * (self.q_pos * s)
            Qvel = I8 * (self.q_vel * s)
            self.Q = np.block([[Qpos, np.zeros((8, 8))], [np.zeros((8, 8)), Qvel]])
        except Exception:
            pass


class CourtLKTracker:
    """
    Court tracker that fills frames between sparse keyframes using LK optical
    flow + robust motion estimation.

    Responsibilities are separated as follows:
    - LK/robust model: estimate per-frame motion increment H_prev_curr from
      prev_gray to gray with FB check + RANSAC, then accumulate into curr_H.
    - Kalman: only updated on keyframes with high-quality API detections
      (optionally with adaptive measurement noise R); between keyframes it
      only predicts. We do NOT feed optical-flow "absolute corners" as
      measurements to avoid reinforcing drift.
    - Gating: geometry (ratio/area/jump), template precision, conditioning,
      FB error, inlier stats, scale-change per frame.
    - Refinement: cornerSubPix for the 4 corners projected by curr_H.
    """

    def __init__(self, cfg: CourtConfig) -> None:
        self.cfg = cfg

        # State
        self.keyframe_idx: Optional[int] = None
        self.keyframe_gray: Optional[np.ndarray] = None
        self.keyframe_corners: Optional[np.ndarray] = None  # (4,2)
        self.curr_H: Optional[np.ndarray] = None  # 3x3 mapping keyframe->current (accumulated)
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None  # (N,1,2) current-frame coords
        # In sliding-window mode we no longer need to carry keyframe-space points
        self.orig_pts: Optional[np.ndarray] = None  # kept for compat; unused
        self.hold_left: int = 0
        self.ema_corners: Optional[np.ndarray] = None  # (4,2) stores smoothed corners
        self.ref_ratio: Optional[float] = None
        self.ref_area: Optional[float] = None
        # keyframe <-> model
        self.H_key_img_to_model: Optional[np.ndarray] = None
        self.model_size: Optional[Tuple[int, int]] = None
        self.model_template: Optional[np.ndarray] = None  # uint8 mask (W,H)
        # Kalman filter
        self.KF: Optional[MultiCornerKalman] = None

        # Cached OpenCV params to avoid recreating tuples every frame
        self._lk_win = (21, 21)
        self._lk_levels = 3
        self._lk_term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        self._subpix_term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01)
        # Throttle expensive template scoring: compute every N frames (configurable)
        self._tpl_stride = int(self.cfg.gates.template_stride)
        self._tpl_tick = 0
        # ROI dynamic scale factor (>=1.0)
        self._roi_scale = 1.0
        # Last template precision (for keyframes)
        self.last_tpl_prec: Optional[float] = None
        # Frame tick for throttling certain ops (e.g., subpix)
        self._frame_tick: int = 0
        # Background executor to parallelize heavy OpenCV routines
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=3)
        # Track recent motion magnitudes for adaptive filtering
        self._motion_history: deque[float] = deque(maxlen=20)
        # Slowly varying global reference geometry
        self.reference_corners: Optional[np.ndarray] = None

    # ---------- adaptive KF helpers ----------
    def _adaptive_kf_sigma(self, tscore: Optional[float]) -> float:
        # Base sigma range
        s_min = float(self.cfg.kalman.kf_r_api_min)
        s_max = float(self.cfg.kalman.kf_r_api_max)
        if tscore is None or not self.cfg.kalman.kf_adaptive_from_template:
            return float(self.cfg.kalman.kalman_r_meas)
        # Map template precision to [0,1]
        # Use soft window: 0.2 -> 0, 0.7 -> 1
        q = (float(tscore) - 0.2) / 0.5
        q = max(0.0, min(1.0, q))
        # Higher quality -> smaller sigma
        sigma = s_max - (s_max - s_min) * q
        return sigma

    # ---------- helpers ----------
    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    @staticmethod
    def _poly_roi_bounds(pts: np.ndarray, expand_ratio: float, W: int, H: int) -> Tuple[int, int, int, int]:
        xs = pts[:, 0]
        ys = pts[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        dx = (x2 - x1) * expand_ratio
        dy = (y2 - y1) * expand_ratio
        x1 = max(0, int(np.floor(x1 - dx)))
        y1 = max(0, int(np.floor(y1 - dy)))
        x2 = min(W - 1, int(np.ceil(x2 + dx)))
        y2 = min(H - 1, int(np.ceil(y2 + dy)))
        return x1, y1, x2, y2

    def _seed_features(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        Hh, Ww = gray.shape[:2]
        x1, y1, x2, y2 = self._poly_roi_bounds(corners, self.cfg.features.roi_expand_ratio, Ww, Hh)
        mask = np.zeros_like(gray, dtype=np.uint8)
        mask[y1:y2 + 1, x1:x2 + 1] = 255

        pts: Optional[np.ndarray] = None
        try:
            n_feat = max(32, int(self.cfg.features.feature_max * 2))
            orb = cv2.ORB_create(
                nfeatures=n_feat,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                fastThreshold=12,
            )
            keypoints = orb.detect(gray, mask)
            if keypoints:
                keypoints.sort(key=lambda kp: kp.response, reverse=True)
                selected = keypoints[: int(self.cfg.features.feature_max)]
                coords = np.array([kp.pt for kp in selected], dtype=np.float32)
                if coords.size > 0:
                    pts = coords.reshape(-1, 1, 2)
        except Exception:
            pts = None

        if pts is None or pts.size == 0:
            pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=self.cfg.features.feature_max,
                qualityLevel=0.01,
                minDistance=7,
                blockSize=7,
                mask=mask,
            )

        if pts is None or pts.size == 0:
            return np.empty((0, 1, 2), dtype=np.float32)

        # Subpixel refinement of initial seeds
        try:
            term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
            win = (int(self.cfg.features.subpix_win), int(self.cfg.features.subpix_win))
            pts_ref = cv2.cornerSubPix(gray, pts.astype(np.float32), win, (-1, -1), term)
            return pts_ref
        except Exception:
            return pts.astype(np.float32)

    @staticmethod
    def _shape_metrics(corners: np.ndarray) -> Tuple[float, float]:
        # Delegate to shared utility for consistency
        return _shape_metrics_util(corners)

    @staticmethod
    def _within_tol(val: float, ref: float, tol: float) -> bool:
        # Delegate to shared utility for consistency
        return _within_tol_util(val, ref, tol)

    def _build_model_template(self, model_size: Tuple[int, int], line_px: int) -> np.ndarray:
        W, H = model_size
        return build_court_model_template(W, H, line_px=line_px, orientation="horizontal")

    def _template_precision_score(self, gray: np.ndarray, H_key_to_curr: np.ndarray) -> float:
        if self.H_key_img_to_model is None or self.model_template is None:
            return 1.0
        H_model_to_key = np.linalg.inv(self.H_key_img_to_model)
        H_model_to_curr = H_model_to_key @ H_key_to_curr
        return template_precision_score(gray, H_model_to_curr, self.model_template)

    def _lk_flow(
        self,
        prev_img: np.ndarray,
        curr_img: np.ndarray,
        pts: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            return cv2.calcOpticalFlowPyrLK(
                prev_img,
                curr_img,
                pts,
                None,
                winSize=self._lk_win,
                maxLevel=self._lk_levels,
                criteria=self._lk_term,
            )
        except Exception:
            return None, None, None

    def _evaluate_model(
        self,
        model_type: str,
        matrix: np.ndarray,
        inliers: Optional[np.ndarray],
        p0: np.ndarray,
        curr_surv: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        try:
            if model_type == "homography":
                H_prev_curr = matrix.astype(np.float64)
            else:
                H_prev_curr = np.eye(3, dtype=np.float64)
                H_prev_curr[:2, :] = matrix
        except Exception:
            return None

        mask = inliers.reshape(-1).astype(bool) if inliers is not None else np.ones(len(p0), dtype=bool)
        inlier_count = int(mask.sum())
        inlier_ratio = float(inlier_count / max(1, len(p0)))

        try:
            cond_val = float(np.linalg.cond(H_prev_curr))
        except Exception:
            cond_val = None

        if inlier_count > 0:
            prev_in = p0[mask]
            curr_in = curr_surv[mask]
            proj = np.array(
                apply_homography_points([(float(x), float(y)) for x, y in prev_in], H_prev_curr),
                dtype=np.float32,
            )
            errs = np.linalg.norm(curr_in - proj, axis=1)
            med_err = float(np.median(errs)) if errs.size > 0 else 1e9
        else:
            med_err = 1e9

        scale_x = None
        scale_y = None
        scale_ok = True
        try:
            basis0 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            basis1 = cv2.perspectiveTransform(basis0.reshape(-1, 1, 2), H_prev_curr).reshape(-1, 2)
            scale_x = float(np.linalg.norm(basis1[1] - basis1[0]))
            scale_y = float(np.linalg.norm(basis1[2] - basis1[0]))
            lo = 1.0 - float(self.cfg.gates.max_scale_change_per_frame)
            hi = 1.0 + float(self.cfg.gates.max_scale_change_per_frame)
            scale_ok = (lo <= scale_x <= hi) and (lo <= scale_y <= hi)
        except Exception:
            pass

        cond_cap = 120.0 if model_type == "homography" else 400.0
        ratio_floor = max(self.cfg.core.min_inlier_ratio, 0.5 if model_type == "homography" else 0.4)
        err_cap = self.cfg.core.ransac_reproj_thresh * (1.5 if model_type == "affine" else 1.2)

        passes_primary = (
            inlier_ratio >= ratio_floor
            and inlier_count >= self.cfg.core.min_inliers
            and med_err <= err_cap
            and (cond_val is None or cond_val <= cond_cap)
            and scale_ok
        )

        return {
            "type": model_type,
            "H": H_prev_curr,
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "median_error": med_err,
            "cond": cond_val,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "scale_ok": scale_ok,
            "passes_primary": passes_primary,
        }

    def _estimate_models(
        self,
        p0: np.ndarray,
        curr_surv: np.ndarray,
    ) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []

        try:
            H, inliers = cv2.findHomography(
                p0,
                curr_surv,
                cv2.RANSAC,
                ransacReprojThreshold=self.cfg.core.ransac_reproj_thresh,
            )
            if H is not None:
                evaluated = self._evaluate_model("homography", H, inliers, p0, curr_surv)
                if evaluated:
                    models.append(evaluated)
        except Exception:
            pass

        try:
            M, inliers = cv2.estimateAffinePartial2D(
                p0,
                curr_surv,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.core.ransac_reproj_thresh,
            )
            if M is not None:
                evaluated = self._evaluate_model("affine", M, inliers, p0, curr_surv)
                if evaluated:
                    models.append(evaluated)
        except Exception:
            pass

        return models

    # ---------- API ----------
    def set_keyframe(self, frame_index: int, frame_bgr: np.ndarray, key_corners: List[Point]) -> None:
        gray = self._to_gray(frame_bgr)
        corners_ord = np.array(order_corners(key_corners), dtype=np.float32)
        if self.ema_corners is not None:
            prev = self.ema_corners.astype(np.float32)
            diff = np.linalg.norm(corners_ord - prev, axis=1)
            med_diff = float(np.median(diff)) if diff.size > 0 else 0.0
            if med_diff > 1.0:
                blend = float(np.clip(1.0 / med_diff, 0.05, 1.0))
                corners_ord = prev * (1.0 - blend) + corners_ord * blend
        if self.reference_corners is None:
            self.reference_corners = corners_ord.astype(np.float32)
        else:
            self.reference_corners = (
                0.98 * self.reference_corners + 0.02 * corners_ord.astype(np.float32)
            )
        self.keyframe_idx = frame_index
        self.keyframe_gray = gray
        self.keyframe_corners = corners_ord
        # Reset accumulated transform to identity at keyframe
        self.curr_H = np.eye(3, dtype=np.float64)
        self.prev_gray = gray
        self.prev_pts = self._executor.submit(self._seed_features, gray, corners_ord).result()
        # Sliding-window: no need to store keyframe-space correspondences
        self.orig_pts = np.empty((0, 1, 2), dtype=np.float32)
        self.hold_left = self.cfg.core.hold_ttl_frames
        # Build model mapping and template once per keyframe
        try:
            H_img2model, model_size = compute_homography(order_corners(corners_ord.tolist()))
            self.H_key_img_to_model = H_img2model
            self.model_size = model_size
            self.model_template = self._build_model_template(model_size, self.cfg.gates.template_line_px)
        except Exception:
            self.H_key_img_to_model = None
            self.model_size = None
            self.model_template = None
        # Compute template precision score at keyframe (H=I)
        tscore = None
        try:
            tscore = self._template_precision_score(gray, np.eye(3, dtype=np.float64))
        except Exception:
            tscore = None
        self.last_tpl_prec = float(tscore) if tscore is not None else None
        # Fuse keyframe detection into Kalman softly (do not reset velocity/state)
        if self.cfg.kalman.use_kalman:
            sigma = self._adaptive_kf_sigma(tscore)
            if self.KF is None:
                self.KF = MultiCornerKalman(q_pos=self.cfg.kalman.kalman_q_pos, q_vel=self.cfg.kalman.kalman_q_vel, r_meas=sigma)
                self.KF.reset(corners_ord)
                self.ema_corners = corners_ord.astype(np.float64)
            else:
                # adapt R per-keyframe before update
                self.KF.R = np.eye(8, dtype=np.float64) * (sigma ** 2)
                smoothed = self.KF.update(corners_ord)
                self.ema_corners = smoothed.astype(np.float64)
        else:
            # EMA fallback: blend toward new detection以避免大跳变
            if self.ema_corners is None:
                self.ema_corners = corners_ord.astype(np.float64)
            else:
                a = 0.6  # keep history more
                self.ema_corners = self.ema_corners * a + corners_ord.astype(np.float64) * (1.0 - a)
        # Update reference shape metrics gently to avoid abrupt resets
        r_new, a_new = self._shape_metrics(corners_ord)
        self.ref_ratio = r_new if self.ref_ratio is None else (0.9 * self.ref_ratio + 0.1 * r_new)
        self.ref_area = a_new if self.ref_area is None else (0.9 * self.ref_area + 0.1 * a_new)
        # Note: Kalman state was softly fused above; no hard reset here

    def update(self, frame_bgr: np.ndarray) -> Tuple[Optional[List[Point]], Dict[str, Any]]:
        info: Dict[str, Any] = {
            "inliers": 0,
            "matches": 0,
            "inlier_ratio": 0.0,
            "method": None,
            "condH": None,
            "reseed": False,
            "hold": False,
            "hold_left": self.hold_left,
        }
        if self.keyframe_gray is None or self.keyframe_corners is None or self.curr_H is None:
            return None, info

        gray = self._to_gray(frame_bgr)
        # Track features from prev to curr (prefer ROI LK with fallback to full-frame)
        if self.prev_pts is None or len(self.prev_pts) < 4:
            seed_future = self._executor.submit(self._seed_features, self.prev_gray, self.keyframe_corners)
            self.prev_pts = seed_future.result()
        if self.prev_pts is None or len(self.prev_pts) < 4:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # ROI around previous smoothed corners to speed up LK and reduce drift
        base_corners = (self.ema_corners if self.ema_corners is not None else self.keyframe_corners).astype(np.float32)
        Hh, Ww = gray.shape[:2]
        eff_expand = max(self.cfg.features.roi_expand_ratio, float(self.cfg.features.lk_roi_expand_ratio) * float(self._roi_scale))
        rx1, ry1, rx2, ry2 = self._poly_roi_bounds(base_corners, eff_expand, Ww, Hh)
        try:
            prev_roi = self.prev_gray[ry1:ry2 + 1, rx1:rx2 + 1]
            curr_roi = gray[ry1:ry2 + 1, rx1:rx2 + 1]
            # Early motion gate: skip LK if ROI nearly static
            if self.cfg.gates.early_motion_gate:
                try:
                    mad = float(np.mean(np.abs(curr_roi.astype(np.int16) - prev_roi.astype(np.int16))))
                    info["roi_mad"] = mad
                    if mad <= float(self.cfg.gates.early_motion_mad_gray_thr):
                        # Predict only and output EMA/current projection
                        if self.cfg.kalman.use_kalman and self.KF is not None:
                            try:
                                self.KF.predict()
                                pos = (self.KF.H @ self.KF.x).reshape(4, 2)
                                self.ema_corners = pos.astype(np.float64)
                            except Exception:
                                pass
                        if self.ema_corners is None:
                            base = np.array(apply_homography_points(self.keyframe_corners.tolist(), self.curr_H), dtype=np.float32)
                        else:
                            base = self.ema_corners.astype(np.float32)
                        out_pts = [(float(x), float(y)) for x, y in order_corners(base.tolist())]
                        info["early_stop"] = True
                        self.prev_gray = gray
                        self.hold_left = self.cfg.core.hold_ttl_frames
                        self._frame_tick = (self._frame_tick + 1) % 1000000
                        return out_pts, info
                except Exception:
                    pass
            pts0 = self.prev_pts.reshape(-1, 2).astype(np.float32)
            pts0_roi = (pts0 - np.array([rx1, ry1], dtype=np.float32)).reshape(-1, 1, 2)
            # Optional ROI downsample for speed
            use_ds = bool(self.cfg.performance.use_roi_downsample) and (0.0 < float(self.cfg.performance.roi_downsample_scale) < 1.0)
            if use_ds:
                s = float(self.cfg.performance.roi_downsample_scale)
                new_w = max(2, int(round(prev_roi.shape[1] * s)))
                new_h = max(2, int(round(prev_roi.shape[0] * s)))
                prev_roi_ds = cv2.resize(prev_roi, (new_w, new_h))
                curr_roi_ds = cv2.resize(curr_roi, (new_w, new_h))
                pts0_roi_ds = pts0_roi * s
                flow_future = self._executor.submit(self._lk_flow, prev_roi_ds, curr_roi_ds, pts0_roi_ds)
                next_pts_roi_ds, st, err = flow_future.result()
                next_pts_roi = (next_pts_roi_ds / s) if next_pts_roi_ds is not None else None
            else:
                flow_future = self._executor.submit(self._lk_flow, prev_roi, curr_roi, pts0_roi)
                next_pts_roi, st, err = flow_future.result()
            st = st.reshape(-1) if st is not None else None
            next_pts = (next_pts_roi.reshape(-1, 2) + np.array([rx1, ry1], dtype=np.float32)).reshape(-1, 1, 2) if next_pts_roi is not None else None
            used_roi = True
        except Exception:
            next_pts, st, err = None, None, None
            used_roi = False
        # Fallback to full-frame LK if ROI fails
        if next_pts is None or st is None:
            flow_future = self._executor.submit(self._lk_flow, self.prev_gray, gray, self.prev_pts)
            next_pts, st, err = flow_future.result()
            st = st.reshape(-1) if st is not None else None
            used_roi = False
        if next_pts is None or st is None:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Forward-backward check to prune unstable tracks
        try:
            if used_roi:
                # Backward on ROI as well
                next_pts_roi = (next_pts.reshape(-1, 2) - np.array([rx1, ry1], dtype=np.float32)).reshape(-1, 1, 2)
                back_future = self._executor.submit(self._lk_flow, curr_roi, prev_roi, next_pts_roi)
                back_pts_roi, st_back, _ = back_future.result()
                back_pts = (back_pts_roi.reshape(-1, 2) + np.array([rx1, ry1], dtype=np.float32)).reshape(-1, 1, 2) if back_pts_roi is not None else None
            else:
                back_future = self._executor.submit(self._lk_flow, gray, self.prev_gray, next_pts)
                back_pts, st_back, _ = back_future.result()
            st_back = st_back.reshape(-1) if st_back is not None else None
        except Exception:
            back_pts, st_back = None, None

        p0_all = self.prev_pts.reshape(-1, 2)
        p1_all = next_pts.reshape(-1, 2)
        good = (st == 1)
        if back_pts is not None and st_back is not None:
            p0_back = back_pts.reshape(-1, 2)
            good = good & (st_back == 1)
            fb_err = np.linalg.norm(p0_all - p0_back, axis=1)
            good = good & (fb_err <= float(self.cfg.core.fb_reproj_thresh))

        p0 = p0_all[good]
        p1 = p1_all[good]
        info["matches"] = int(len(p0))
        if len(p0) < 4:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info
        # Update ROI scale and Kalman gains based on motion statistics
        try:
            disp = np.linalg.norm(p1 - p0, axis=1)
            md = float(np.median(disp)) if disp.size > 0 else 0.0
            self._motion_history.append(md)

            motion_ref = float(max(1e-6, self.cfg.kalman.motion_md_ref_px))
            motion_norm = float(np.clip(md / motion_ref, 0.0, 1.5))
            desired_tracks = max(1, int(self.cfg.features.reseed_min_tracks))
            feature_frac = float(len(p1)) / float(desired_tracks)
            feature_frac = float(np.clip(feature_frac, 0.0, 1.5))
            feature_pressure = float(np.clip(1.0 - feature_frac, 0.0, 1.0))
            target = 1.0 + 0.35 * motion_norm + 0.45 * feature_pressure
            self._roi_scale = float(np.clip(0.6 * self._roi_scale + 0.4 * target, 1.0, 2.0))

            if self.cfg.kalman.use_kalman and self.KF is not None and self.cfg.kalman.kalman_q_scale_from_motion:
                hist = np.array(self._motion_history, dtype=np.float32)
                avg_motion = float(np.mean(hist)) if hist.size else md
                trend = 0.0
                if hist.size >= 2:
                    trend = float((hist[-1] - hist[0]) / max(1, hist.size - 1))
                burst = float((float(hist.max()) - float(hist.min())) / max(1, hist.size)) if hist.size else 0.0
                norm_avg = float(np.clip(avg_motion / motion_ref, 0.0, 2.0))
                norm_trend = float(np.clip(abs(trend) / motion_ref, 0.0, 1.5))
                norm_burst = float(np.clip(burst / motion_ref, 0.0, 1.5))
                blend = min(1.25, 0.55 * norm_avg + 0.3 * norm_trend + 0.15 * norm_burst)
                qlo = float(self.cfg.kalman.kalman_q_scale_lo)
                qhi = float(self.cfg.kalman.kalman_q_scale_hi)
                scale = qlo + (qhi - qlo) * min(1.0, blend)
                self.KF.set_process_scale(scale)
        except Exception:
            pass

        # Estimate prev->curr transform (sliding window)
        curr_surv = p1
        models_future = self._executor.submit(self._estimate_models, p0, curr_surv)
        models = models_future.result()
        if not models:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        preferred = "homography" if self.cfg.core.use_homography else "affine"

        def model_score(model: Dict[str, Any]) -> Tuple[int, int, float, float]:
            return (
                0 if model["type"] == preferred else 1,
                0 if model["passes_primary"] else 1,
                -model["inliers"],
                model["median_error"],
            )

        passing = [m for m in models if m["passes_primary"]]
        if not passing:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        candidate_pool = passing
        candidate_pool.sort(key=model_score)
        selected = candidate_pool[0]

        H_prev_curr = selected["H"]
        info["method"] = f"{selected['type']}(prev->curr)"
        info["inliers"] = int(selected["inliers"])
        info["inlier_ratio"] = float(selected["inlier_ratio"])
        info["condH"] = selected["cond"]
        sx = selected.get("scale_x")
        sy = selected.get("scale_y")
        med_err = float(selected["median_error"])
        s_ok = bool(selected["scale_ok"])
        cond_cap = 120.0 if selected["type"] == "homography" else 400.0
        err_cap_final = self.cfg.core.ransac_reproj_thresh * (1.2 if selected["type"] == "homography" else 1.5)

        gating_fail = (
            info["inlier_ratio"] < max(self.cfg.core.min_inlier_ratio, 0.5 if selected["type"] == "homography" else 0.4)
            or info["inliers"] < self.cfg.core.min_inliers
            or med_err > err_cap_final
            or (info["condH"] is not None and info["condH"] > cond_cap)
            or (not s_ok)
        )

        if gating_fail:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Accept: accumulate transform keyframe->current
        self.curr_H = H_prev_curr @ self.curr_H
        # Normalize to keep H numerically stable
        if abs(self.curr_H[2, 2]) > 1e-12:
            self.curr_H = self.curr_H / self.curr_H[2, 2]

        # Apply to keyframe corners
        curr_corners = np.array(apply_homography_points(self.keyframe_corners.tolist(), self.curr_H), dtype=np.float32)
        if self.ema_corners is not None:
            prev = self.ema_corners.astype(np.float32)
            delta = np.linalg.norm(curr_corners - prev, axis=1)
            med_delta = float(np.median(delta)) if delta.size > 0 else 0.0
            if med_delta > 1e-6:
                blend = float(np.clip(1.0 / max(1.0, med_delta), 0.05, 0.6))
                curr_corners = prev * (1.0 - blend) + curr_corners * blend
        if self.reference_corners is not None:
            baseline = self.reference_corners.astype(np.float32)
            curr_corners = baseline * 0.4 + curr_corners * 0.6
        # Subpixel refine the 4 corners for better spatial stability (throttled)
        do_refine = (self._frame_tick % max(1, int(self.cfg.features.subpix_stride))) == 0
        if do_refine:
            try:
                win = (int(self.cfg.features.subpix_win), int(self.cfg.features.subpix_win))
                corners_nx1x2 = curr_corners.reshape(-1, 1, 2).astype(np.float32)
                corners_ref = cv2.cornerSubPix(gray, corners_nx1x2, win, (-1, -1), self._subpix_term)
                if corners_ref is not None and len(corners_ref) == 4:
                    curr_corners = corners_ref.reshape(4, 2)
            except Exception:
                pass

        # Geometric sanity checks (ratio/area and jump vs smoothed)
        ratio, area = self._shape_metrics(curr_corners)
        ok_geo = True
        if self.ref_ratio is not None:
            if not self._within_tol(ratio, self.ref_ratio, self.cfg.gates.ratio_tolerance):
                ok_geo = False
        if self.ref_area is not None and self.ref_area > 1e-6:
            if not (1.0 - self.cfg.gates.area_tolerance) * self.ref_area <= area <= (1.0 + self.cfg.gates.area_tolerance) * self.ref_area:
                ok_geo = False
        if self.ema_corners is not None:
            d = np.linalg.norm(curr_corners.astype(np.float64) - self.ema_corners, axis=1)
            if float(np.median(d)) > self.cfg.gates.max_jump_px:
                ok_geo = False
        # Optional template precision score (edge alignment)
        if ok_geo and self.cfg.gates.use_template_score:
            do_score = (self._tpl_tick % max(1, int(self._tpl_stride))) == 0
            if do_score:
                try:
                    tscore = self._template_precision_score(gray, self.curr_H)
                    info["tpl_prec"] = tscore
                    self.last_tpl_prec = float(tscore)
                    if tscore < self.cfg.gates.template_min_precision:
                        ok_geo = False
                except Exception:
                    pass
            self._tpl_tick = (self._tpl_tick + 1) % 1000000
        if not ok_geo:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            # reseed features around last good corners projected (ema or keyframe)
            base_corners = (self.ema_corners if self.ema_corners is not None else self.keyframe_corners).astype(np.float32)
            seed_future = self._executor.submit(self._seed_features, gray, base_corners)
            self.prev_pts = seed_future.result()
            self.prev_gray = gray
            return None, info

        # Re-seed features if tracks are low
        if len(curr_surv) < self.cfg.features.reseed_min_tracks:
            info["reseed"] = True
            seed_future = self._executor.submit(self._seed_features, gray, curr_corners)
            self.prev_pts = seed_future.result()
        else:
            self.prev_pts = curr_surv.reshape(-1, 1, 2).astype(np.float32)

        # Step frame
        self.prev_gray = gray
        self.hold_left = self.cfg.core.hold_ttl_frames

        if self.cfg.kalman.use_kalman and self.KF is not None:
            try:
                smoothed = self.KF.update(curr_corners)
                self.ema_corners = smoothed.astype(np.float64)
            except Exception:
                if self.ema_corners is None:
                    self.ema_corners = curr_corners.astype(np.float64)
                else:
                    a = self.cfg.core.ema_alpha
                    self.ema_corners = self.ema_corners * a + curr_corners.astype(np.float64) * (1.0 - a)
        else:
            if self.ema_corners is None:
                self.ema_corners = curr_corners.astype(np.float64)
            else:
                a = self.cfg.core.ema_alpha
                self.ema_corners = self.ema_corners * a + curr_corners.astype(np.float64) * (1.0 - a)
        if self.reference_corners is None:
            self.reference_corners = curr_corners.astype(np.float32)
        else:
            self.reference_corners = (
                0.995 * self.reference_corners + 0.005 * curr_corners.astype(np.float32)
            )
        # Update reference metrics slowly
        r, a = self._shape_metrics(self.ema_corners.astype(np.float32))
        self.ref_ratio = r if self.ref_ratio is None else (self.ref_ratio * 0.98 + r * 0.02)
        self.ref_area = a if self.ref_area is None else (self.ref_area * 0.98 + a * 0.02)

        # enrich info for diagnostics
        info["roi_used"] = bool(used_roi)
        info["roi_scale"] = float(self._roi_scale)
        info["med_reproj_err"] = float(med_err)
        try:
            info["scale_x"] = float(sx)
            info["scale_y"] = float(sy)
        except Exception:
            pass
        out_pts = [(float(x), float(y)) for x, y in order_corners(self.ema_corners.tolist())]
        self._frame_tick = (self._frame_tick + 1) % 1000000
        return out_pts, info


    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


__all__ = ["CourtLKTracker", "MultiCornerKalman"]
