from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict, Any
import math

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
from court.sentinel import DriftSentinel
from court.net_tracker import NetTracker
from court.motion_estimator import MotionEstimator


Point = Tuple[float, float]


class HomographyKalman:
    """Lightweight Kalman filter that operates directly on the 8 DoF homography."""

    def __init__(
        self,
        q: float = 5e-4,
        r: float = 2.0,
        *,
        static_scale: float = 0.1,
    ) -> None:
        self.base_q = float(max(q, 1e-12))
        self.base_r = float(max(r, 1e-12))
        self.static_scale = float(max(min(static_scale, 1.0), 1e-3))
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.q_scale: float = 1.0
        self._static = False

    @staticmethod
    def _normalize_h(H: np.ndarray) -> np.ndarray:
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError("Homography must be 3x3")
        if abs(H[2, 2]) < 1e-12:
            return H
        return H / H[2, 2]

    @staticmethod
    def _to_vec(H: np.ndarray) -> np.ndarray:
        Hn = HomographyKalman._normalize_h(H)
        return np.array(
            [
                Hn[0, 0],
                Hn[0, 1],
                Hn[0, 2],
                Hn[1, 0],
                Hn[1, 1],
                Hn[1, 2],
                Hn[2, 0],
                Hn[2, 1],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _to_matrix(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64).reshape(8)
        H = np.array(
            [
                [v[0], v[1], v[2]],
                [v[3], v[4], v[5]],
                [v[6], v[7], 1.0],
            ],
            dtype=np.float64,
        )
        return HomographyKalman._normalize_h(H)

    def reset(self, H: np.ndarray) -> None:
        vec = self._to_vec(H)
        self.x = vec.copy()
        self.P = np.eye(8, dtype=np.float64) * self.base_r

    def set_process_scale(self, scale: float) -> None:
        try:
            self.q_scale = float(scale)
            if self.q_scale <= 0:
                self.q_scale = 1.0
        except Exception:
            self.q_scale = 1.0

    def set_static(self, is_static: bool) -> None:
        self._static = bool(is_static)

    def predict(self) -> Optional[np.ndarray]:
        if self.x is None or self.P is None:
            return None
        q = self.base_q * self.q_scale
        if self._static:
            q *= self.static_scale
        Q = np.eye(8, dtype=np.float64) * q
        self.P = self.P + Q
        return self.x.copy()

    def update(self, H_meas: np.ndarray, *, meas_scale: float = 1.0) -> np.ndarray:
        if H_meas is None:
            if self.x is None:
                raise RuntimeError("HomographyKalman: measurement required for initialisation")
            return self.x.copy()

        z = self._to_vec(H_meas)
        if not np.all(np.isfinite(z)):
            return self.x.copy() if self.x is not None else z

        if self.x is None or self.P is None:
            self.reset(H_meas)
            return self.x.copy()

        R_scale = float(max(meas_scale, 1e-6))
        R = np.eye(8, dtype=np.float64) * (self.base_r * R_scale)

        S = self.P + R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.x)
        I = np.eye(8, dtype=np.float64)
        self.P = (I - K) @ self.P
        return self.x.copy()

    def current_matrix(self) -> Optional[np.ndarray]:
        if self.x is None:
            return None
        try:
            return self._to_matrix(self.x)
        except Exception:
            return None


class ScalarKalman1D:
    """Lightweight scalar Kalman filter for net height smoothing."""

    def __init__(self, q: float, r: float) -> None:
        self.q = float(max(q, 1e-9))
        self.r = float(max(r, 1e-9))
        self.x: Optional[float] = None
        self.P: float = float(self.r)

    def reset(self, value: float, variance: Optional[float] = None) -> float:
        self.x = float(value)
        self.P = float(variance if (variance is not None and variance > 0) else self.r)
        return self.x

    def predict(self) -> Optional[float]:
        if self.x is None:
            return None
        self.P += self.q
        return self.x

    def update(self, measurement: float, r: Optional[float] = None) -> Optional[float]:
        if measurement is None:
            return self.x
        meas = float(measurement)
        if self.x is None:
            return self.reset(meas, r)
        R = float(r) if r is not None and r > 0 else self.r
        if R <= 0:
            R = self.r
        K = self.P / (self.P + R)
        self.x = self.x + K * (meas - self.x)
        self.P = (1.0 - K) * self.P
        return self.x


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
        self.fallback_cfg = getattr(cfg, "fallback", None)

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
        self.H_model_to_key_img: Optional[np.ndarray] = None
        self.model_size: Optional[Tuple[int, int]] = None
        self.model_template: Optional[np.ndarray] = None  # uint8 mask (W,H)
        # Homography filter
        self.H_filter: Optional[HomographyKalman] = None
        
        # Initialize Drift Sentinel
        self._sentinel = DriftSentinel(cfg)
        
        # Initialize Net Tracker
        self._net_tracker = NetTracker(getattr(cfg, "net", None))
        
        # Cached OpenCV params to avoid recreating tuples every frame
        self._subpix_term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01)
        # Throttle expensive template scoring: compute every N frames (configurable)
        self._tpl_stride = int(self.cfg.gates.template_stride)
        self._tpl_tick = 0
        # ROI dynamic scale factor (>=1.0)
        self._roi_scale = 1.0
        # Last template precision (for keyframes)
        # Frame tick for throttling certain ops (e.g., subpix)
        self._frame_tick: int = 0
        # Background executor to parallelize heavy OpenCV routines
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=3)
        
        # Initialize Motion Estimator (after executor is created)
        self._motion_estimator = MotionEstimator(cfg, self._executor)
        # Track recent motion magnitudes for adaptive filtering
        self._motion_history: deque[float] = deque(maxlen=20)
        self._early_stop_streak: int = 0
        # Slowly varying global reference geometry
        self.reference_corners: Optional[np.ndarray] = None
        # Bootstrap meta cache for diagnostics
        self._bootstrap_meta: Optional[Dict[str, Any]] = None
        # Track absolute frame index for sentinel bookkeeping
        self._frame_index: int = -1
        
        # Store last template precision directly, as it's used by external code
        self._last_tpl_prec: Optional[float] = None
        
    @property
    def net_state(self) -> Optional[Dict[str, Any]]:
        """Access net state from the NetTracker."""
        return self._net_tracker.net_state
    
    @property
    def last_tpl_prec(self) -> Optional[float]:
        """Access last template precision."""
        return self._last_tpl_prec

    # ---------- bootstrap helpers ----------
    def build_bootstrap_reference(
        self,
        detections: List[Dict[str, Any]],
        *,
        threshold_px: Optional[float] = None,
        min_inliers: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Aggregate multiple detections into a single robust reference corners set."""
        if not detections:
            return None

        samples: List[np.ndarray] = []
        frames: List[int] = []
        for det in detections:
            corners = det.get("corners") if isinstance(det, dict) else det
            if corners is None:
                continue
            try:
                arr = np.array(order_corners(corners), dtype=np.float32).reshape(4, 2)
            except Exception:
                continue
            samples.append(arr)
            try:
                frames.append(int(det.get("frame", -1)) if isinstance(det, dict) else -1)
            except Exception:
                frames.append(-1)

        if not samples:
            return None

        stack = np.stack(samples, axis=0)
        thresh = float(threshold_px if threshold_px is not None else self.cfg.bootstrap.ransac_threshold_px)
        min_req = max(1, int(min_inliers if min_inliers is not None else self.cfg.bootstrap.min_inliers))

        best_idx: Optional[int] = None
        best_mask: Optional[np.ndarray] = None
        best_score = -1e12

        for idx in range(stack.shape[0]):
            anchor = stack[idx]
            residuals = np.linalg.norm(stack - anchor[None, ...], axis=2)
            med_res = np.median(residuals, axis=1)
            mask = med_res <= thresh
            count = int(mask.sum())
            if count < min_req:
                continue
            spread = float(np.median(med_res[mask])) if count else 0.0
            score = float(count) - 0.01 * spread
            if score > best_score:
                best_score = score
                best_idx = idx
                best_mask = mask

        if best_idx is None or best_mask is None:
            consensus = np.median(stack, axis=0)
            ordered = order_corners(consensus.tolist())
            residual = float(np.median(np.linalg.norm(stack - consensus[None, ...], axis=2)))
            meta = {
                "method": "median",
                "inliers": int(stack.shape[0]),
                "support_frames": frames,
                "residual_median": residual,
                "threshold_px": thresh,
            }
            self._bootstrap_meta = meta
            return {
                "corners": [(float(x), float(y)) for x, y in ordered],
                "meta": meta,
            }

        inlier_stack = stack[best_mask]
        consensus = np.median(inlier_stack, axis=0)
        ordered = order_corners(consensus.tolist())
        residual = float(np.median(np.linalg.norm(inlier_stack - consensus[None, ...], axis=2)))
        support_frames = [frames[i] for i, keep in enumerate(best_mask) if keep and i < len(frames)]
        anchor_frame = frames[best_idx] if 0 <= best_idx < len(frames) else -1
        meta = {
            "method": "ransac-median",
            "inliers": int(len(inlier_stack)),
            "support_frames": support_frames,
            "anchor_frame": anchor_frame,
            "residual_median": residual,
            "threshold_px": thresh,
        }
        self._bootstrap_meta = meta
        return {
            "corners": [(float(x), float(y)) for x, y in ordered],
            "meta": meta,
        }

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
    def _corners_degenerate(corners: np.ndarray, *, tol: float = 1.0) -> bool:
        try:
            pts = corners.reshape(-1, 2).astype(np.float64)
        except Exception:
            return True
        unique: List[np.ndarray] = []
        for p in pts:
            if not any(np.linalg.norm(p - q) <= tol for q in unique):
                unique.append(p)
        if len(unique) < 4:
            return True
        # no need to check area separately; uniqueness is main guard
        return False

    def _poly_roi_bounds(self, pts: np.ndarray, expand_ratio: float, W: int, H: int) -> Tuple[int, int, int, int]:
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

    # ---------- API ----------
    def set_keyframe(
        self,
        frame_index: int,
        frame_bgr: np.ndarray,
        key_corners: List[Point],
        net_measurement: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._frame_index = int(frame_index)
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
        self._early_stop_streak = 0
        self.hold_left = self.cfg.core.hold_ttl_frames
        # Build model mapping and template once per keyframe
        try:
            H_img2model, model_size = compute_homography(order_corners(corners_ord.tolist()))
            self.H_key_img_to_model = H_img2model
            try:
                H_model2img = np.linalg.inv(H_img2model)
                if abs(H_model2img[2, 2]) > 1e-12:
                    H_model2img = H_model2img / H_model2img[2, 2]
                self.H_model_to_key_img = H_model2img
            except Exception:
                self.H_model_to_key_img = None
            self.model_size = model_size
            self.model_template = self._build_model_template(model_size, self.cfg.gates.template_line_px)
        except Exception:
            self.H_key_img_to_model = None
            self.H_model_to_key_img = None
            self.model_size = None
            self.model_template = None
        # Compute template precision score at keyframe (H=I)
        tscore = None
        try:
            tscore = self._template_precision_score(gray, np.eye(3, dtype=np.float64))
        except Exception:
            tscore = None
        if tscore is not None:
            self._last_tpl_prec = float(tscore)
        # Initialise homography filter around keyframe
        if self.cfg.kalman.use_kalman:
            if self.H_filter is None:
                self.H_filter = HomographyKalman(
                    q=getattr(self.cfg.kalman, "homography_q", self.cfg.kalman.kalman_q_pos),
                    r=getattr(self.cfg.kalman, "homography_r", self.cfg.kalman.kalman_r_meas),
                    static_scale=getattr(self.cfg.kalman, "homography_static_scale", 0.1),
                )
            self.H_filter.reset(self.curr_H)
            meas_scale = 1.0
            if self.cfg.kalman.kf_adaptive_from_template and tscore is not None:
                base_r = max(getattr(self.cfg.kalman, "homography_r", self.cfg.kalman.kalman_r_meas), 1e-6)
                sigma = self._adaptive_kf_sigma(tscore)
                meas_scale = max(sigma / base_r, 1e-3)
            self.H_filter.update(self.curr_H, meas_scale=meas_scale)
            filtered = self.H_filter.current_matrix()
            if filtered is not None:
                self.curr_H = filtered
            curr_pts = np.array(
                apply_homography_points(self.keyframe_corners.tolist(), self.curr_H),
                dtype=np.float32,
            )
            self.ema_corners = curr_pts.astype(np.float64)
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
        # Reset sentinel state
        self._sentinel.reset(frame_index=frame_index)
        if self._last_tpl_prec is not None:
            self._sentinel.set_template_ref(float(self._last_tpl_prec))
        # Note: Kalman state was softly fused above; no hard reset here
        self._net_tracker._missing_frames = 0
        self._net_tracker.update(
            frame_index,
            net_measurement,
            self.H_model_to_key_img,
            self.curr_H,
            self.model_size,
            court_corners=self.keyframe_corners.astype(np.float64) if self.keyframe_corners is not None else None,
        )

    def update(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: Optional[int] = None,
        net_measurement: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[List[Point]], Dict[str, Any]]:
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
        if frame_index is not None:
            self._frame_index = int(frame_index)
        else:
            self._frame_index += 1
        current_frame_idx = self._frame_index
        info["frame_idx"] = current_frame_idx
        def _finalize(corners_out: Optional[List[Point]]) -> Tuple[Optional[List[Point]], Dict[str, Any]]:
            if corners_out is not None:
                corner_hint = np.array(order_corners(corners_out), dtype=np.float64)
            elif self.ema_corners is not None:
                corner_hint = self.ema_corners
            elif self.keyframe_corners is not None:
                corner_hint = self.keyframe_corners.astype(np.float64)
            else:
                corner_hint = None
            info["net"] = self._net_tracker.update(
                current_frame_idx,
                net_measurement,
                self.H_model_to_key_img,
                self.curr_H,
                self.model_size,
                court_corners=corner_hint,
            )
            return corners_out, info

        if self.keyframe_gray is None or self.keyframe_corners is None or self.curr_H is None:
            return _finalize(None)

        gray = self._to_gray(frame_bgr)
        # Track features from prev to curr (prefer ROI LK with fallback to full-frame)
        if self.prev_pts is None or len(self.prev_pts) < 4:
            seed_future = self._executor.submit(self._seed_features, self.prev_gray, self.keyframe_corners)
            self.prev_pts = seed_future.result()
        if self.prev_pts is None or len(self.prev_pts) < 4:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            self._sentinel.on_hold(info, frame_idx=current_frame_idx, reason="seed_features")
            return _finalize(None)

        # ROI around previous smoothed corners to speed up LK and reduce drift
        base_corners = (self.ema_corners if self.ema_corners is not None else self.keyframe_corners).astype(np.float32)
        Hh, Ww = gray.shape[:2]
        eff_expand = max(self.cfg.features.roi_expand_ratio, float(self.cfg.features.lk_roi_expand_ratio) * float(self._roi_scale))
        rx1, ry1, rx2, ry2 = self._poly_roi_bounds(base_corners, eff_expand, Ww, Hh)
        prev_roi = self.prev_gray[ry1:ry2 + 1, rx1:rx2 + 1]
        curr_roi = gray[ry1:ry2 + 1, rx1:rx2 + 1]
        # Early motion gate: skip LK if ROI nearly static
        if self.cfg.gates.early_motion_gate:
            try:
                mad = float(np.mean(np.abs(curr_roi.astype(np.int16) - prev_roi.astype(np.int16))))
                info["roi_mad"] = mad
                if mad <= float(self.cfg.gates.early_motion_mad_gray_thr):
                    # Predict only and output EMA/current projection
                    base: np.ndarray
                    if self.cfg.kalman.use_kalman and self.H_filter is not None:
                        try:
                            self.H_filter.set_static(True)
                            self.H_filter.predict()
                        except Exception:
                            pass
                        predicted = self.H_filter.current_matrix()
                        if predicted is not None:
                            self.curr_H = predicted
                            base = np.array(
                                apply_homography_points(self.keyframe_corners.tolist(), self.curr_H),
                                dtype=np.float32,
                            )
                            self.ema_corners = base.astype(np.float64)
                        else:
                            base = (
                                self.ema_corners.astype(np.float32)
                                if self.ema_corners is not None
                                else np.array(
                                    apply_homography_points(self.keyframe_corners.tolist(), self.curr_H),
                                    dtype=np.float32,
                                )
                            )
                    else:
                        if self.ema_corners is None:
                            base = np.array(
                                apply_homography_points(self.keyframe_corners.tolist(), self.curr_H),
                                dtype=np.float32,
                            )
                        else:
                            base = self.ema_corners.astype(np.float32)
                    if self._corners_degenerate(base):
                        info["degenerate"] = True
                        info["hold"] = True
                        self.hold_left = max(0, self.hold_left - 1)
                        self._sentinel.on_hold(info, frame_idx=current_frame_idx, reason="degenerate")
                        self.prev_gray = gray
                        self._early_stop_streak = 0
                        # If we are here, it means motion was detected, so ensure KF is in dynamic mode
                        if self.cfg.kalman.use_kalman and self.H_filter is not None:
                            try:
                                self.H_filter.set_static(False)
                            except Exception:
                                pass
                        return _finalize(None)
                    out_pts = [(float(x), float(y)) for x, y in order_corners(base.tolist())]
                    info["early_stop"] = True
                    self.prev_gray = gray
                    self.hold_left = self.cfg.core.hold_ttl_frames
                    self._frame_tick = (self._frame_tick + 1) % 1000000
                    self._early_stop_streak += 1
                    max_stall = max(int(self.cfg.core.hold_ttl_frames), 12)
                    if self._early_stop_streak >= max_stall:
                        info["needs_redetect"] = True
                        reasons = info.setdefault("sentinel_reasons", [])
                        reasons.append("stalled")
                    else:
                        info.pop("needs_redetect", None)
                    return _finalize(out_pts)
            except Exception:
                pass
        self._early_stop_streak = 0
        used_roi = rx2 > rx1 and ry2 > ry1
        info["roi_used"] = used_roi

        # Estimate motion using MotionEstimator
        p0_result, curr_surv, selected_model = self._motion_estimator.estimate_motion(
            gray=gray,
            prev_gray=self.prev_gray,
            prev_pts=self.prev_pts,
            base_corners=base_corners,
            info=info,
            _roi_scale=self._roi_scale,
            last_tpl_prec=self.last_tpl_prec,
            used_roi=used_roi,
            rx1=rx1,
            ry1=ry1,
            rx2=rx2,
            ry2=ry2,
        )
        
        if p0_result is not None and curr_surv is not None:
            try:
                disp = np.linalg.norm(curr_surv - p0_result, axis=1)
                md = float(np.median(disp)) if disp.size > 0 else 0.0
                self._motion_history.append(md)

                motion_ref = float(max(1e-6, self.cfg.kalman.motion_md_ref_px))
                motion_norm = float(np.clip(md / motion_ref, 0.0, 1.5))
                desired_tracks = max(1, int(self.cfg.features.reseed_min_tracks))
                feature_frac = float(len(curr_surv)) / float(desired_tracks)
                feature_frac = float(np.clip(feature_frac, 0.0, 1.5))
                feature_pressure = float(np.clip(1.0 - feature_frac, 0.0, 1.0))
                target = 1.0 + 0.35 * motion_norm + 0.45 * feature_pressure
                self._roi_scale = float(np.clip(0.6 * self._roi_scale + 0.4 * target, 1.0, 2.0))

                if self.cfg.kalman.use_kalman and self.H_filter is not None and self.cfg.kalman.kalman_q_scale_from_motion:
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
                    self.H_filter.set_process_scale(scale)
            except Exception:
                pass
        
        if p0_result is None or curr_surv is None or selected_model is None:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            self._sentinel.on_hold(info, frame_idx=current_frame_idx, reason="model_estimation")
            return _finalize(None)
            
        # Update info with model results
        info.update({
            "method": f"{selected_model['type']}(prev->curr)",
            "inliers": int(selected_model["inliers"]),
            "inlier_ratio": float(selected_model["inlier_ratio"]),
            "condH": selected_model.get("cond"),
            "med_reproj_err": float(selected_model["median_error"]),
        })
        
        if selected_model.get("ecc_score") is not None:
            info["ecc_score"] = float(selected_model["ecc_score"])
            
        H_prev_curr = selected_model["H"]

        # Accept: accumulate transform keyframe->current
        self.curr_H = H_prev_curr @ self.curr_H
        # Normalize to keep H numerically stable
        if abs(self.curr_H[2, 2]) > 1e-12:
            self.curr_H = self.curr_H / self.curr_H[2, 2]

        if self.cfg.kalman.use_kalman:
            if self.H_filter is None:
                self.H_filter = HomographyKalman(
                    q=getattr(self.cfg.kalman, "homography_q", self.cfg.kalman.kalman_q_pos),
                    r=getattr(self.cfg.kalman, "homography_r", self.cfg.kalman.kalman_r_meas),
                    static_scale=getattr(self.cfg.kalman, "homography_static_scale", 0.1),
                )
                self.H_filter.reset(self.curr_H)
            else:
                try:
                    self.H_filter.set_static(False)
                    self.H_filter.predict()
                except Exception:
                    pass
                try:
                    meas_scale = 1.0 + info.get("med_reproj_err", 0.0) / max(self.cfg.core.ransac_reproj_thresh, 1e-6)
                    if self.cfg.kalman.kf_adaptive_from_template and self.last_tpl_prec is not None:
                        base_r = max(getattr(self.cfg.kalman, "homography_r", self.cfg.kalman.kalman_r_meas), 1e-6)
                        sigma = self._adaptive_kf_sigma(self.last_tpl_prec)
                        meas_scale = max(meas_scale * max(sigma / base_r, 1e-3), 1e-3)
                    self.H_filter.update(self.curr_H, meas_scale=meas_scale)
                    filtered = self.H_filter.current_matrix()
                    if filtered is not None:
                        self.curr_H = filtered
                except Exception:
                    pass

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

        if self._corners_degenerate(curr_corners):
            info["hold"] = True
            info["degenerate"] = True
            self.hold_left = max(0, self.hold_left - 1)
            seed_future = self._executor.submit(self._seed_features, gray, curr_corners.astype(np.float32))
            self.prev_pts = seed_future.result()
            self.prev_gray = gray
            self._sentinel.on_hold(info, frame_idx=current_frame_idx, reason="degenerate")
            return _finalize(None)

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
                    if tscore is not None:
                        self._last_tpl_prec = float(tscore)
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
            self._sentinel.on_hold(info, frame_idx=current_frame_idx, reason="geometry")
            return _finalize(None)

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
        try:
            info["scale_x"] = float(selected_model.get("scale_x", 1.0))
            info["scale_y"] = float(selected_model.get("scale_y", 1.0))
        except Exception:
            pass
        # Prepare info for sentinel
        info["ref_ratio"] = self.ref_ratio
        info["ref_area"] = self.ref_area
        
        self._sentinel.on_success(
            info,
            frame_idx=current_frame_idx,
            ratio=float(ratio),
            area=float(area),
            med_err=float(info.get("med_reproj_err", 0.0)),
            matches=int(len(curr_surv)),
            inlier_ratio=float(info.get("inlier_ratio", 0.0)),
            template_score=info.get("tpl_prec", self._last_tpl_prec),
        )
        out_pts = [(float(x), float(y)) for x, y in order_corners(self.ema_corners.tolist())]
        self._frame_tick = (self._frame_tick + 1) % 1000000
        return _finalize(out_pts)


    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


__all__ = ["CourtLKTracker", "HomographyKalman", "ScalarKalman1D"]
