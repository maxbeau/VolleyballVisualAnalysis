from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2

from court.config import CourtTrackerConfig
from court.utils import order_corners, apply_homography_points, compute_homography


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


class CourtLKTracker:
    """
    Lightweight court tracker between sparse keyframes using LK optical flow
    and robust motion estimation (affine/homography). Designed to minimize API
    usage by filling frames between low-frequency detections.
    """

    def __init__(self, cfg: Optional[CourtTrackerConfig] = None, **kwargs) -> None:
        self.cfg = cfg or CourtTrackerConfig()
        # allow kwargs overrides for backwards compat
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

        # State
        self.keyframe_idx: Optional[int] = None
        self.keyframe_gray: Optional[np.ndarray] = None
        self.keyframe_corners: Optional[np.ndarray] = None  # (4,2)
        self.curr_H: Optional[np.ndarray] = None  # 3x3 mapping keyframe->current
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None  # (N,1,2) current-frame coords
        self.orig_pts: Optional[np.ndarray] = None  # (N,1,2) keyframe coords
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
        x1, y1, x2, y2 = self._poly_roi_bounds(corners, self.cfg.roi_expand_ratio, Ww, Hh)
        mask = np.zeros_like(gray)
        mask[y1:y2 + 1, x1:x2 + 1] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.feature_max,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            mask=mask,
        )
        if pts is None:
            return np.empty((0, 1, 2), dtype=np.float32)
        # Subpixel refinement of initial seeds
        try:
            term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
            win = (int(self.cfg.subpix_win), int(self.cfg.subpix_win))
            pts_ref = cv2.cornerSubPix(gray, pts.astype(np.float32), win, (-1, -1), term)
            return pts_ref
        except Exception:
            return pts

    @staticmethod
    def _shape_metrics(corners: np.ndarray) -> Tuple[float, float]:
        c = corners.reshape(4, 2).astype(np.float64)
        tl, tr, br, bl = c
        top = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left = np.linalg.norm(bl - tl)
        right = np.linalg.norm(br - tr)
        w = max(1e-6, 0.5 * (top + bottom))
        h = max(1e-6, 0.5 * (left + right))
        ratio = float(w / h)
        x = c[:, 0]; y = c[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        return ratio, area

    @staticmethod
    def _within_tol(val: float, ref: float, tol: float) -> bool:
        if ref == 0:
            return False
        return (ref * (1 - tol)) <= val <= (ref * (1 + tol))

    def _build_model_template(self, model_size: Tuple[int, int], line_px: int) -> np.ndarray:
        W, H = model_size
        canvas = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), 255, thickness=line_px)
        cy = int(round(H / 2))
        cv2.line(canvas, (0, cy), (W - 1, cy), 255, thickness=line_px)
        a1 = int(round(H * (1.5 / 9.0)))
        a2 = int(round(H * (7.5 / 9.0)))
        cv2.line(canvas, (0, a1), (W - 1, a1), 255, thickness=line_px)
        cv2.line(canvas, (0, a2), (W - 1, a2), 255, thickness=line_px)
        return canvas

    def _template_precision_score(self, gray: np.ndarray, H_key_to_curr: np.ndarray) -> float:
        if self.H_key_img_to_model is None or self.model_template is None:
            return 1.0
        H_model_to_key = np.linalg.inv(self.H_key_img_to_model)
        H_model_to_curr = H_model_to_key @ H_key_to_curr
        Hh, Ww = gray.shape[:2]
        warped = cv2.warpPerspective(self.model_template, H_model_to_curr, (Ww, Hh), flags=cv2.INTER_NEAREST)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        tmask = warped > 0
        if not np.any(tmask):
            return 0.0
        overlap = (edges > 0) & tmask
        prec = float(overlap.sum()) / float(tmask.sum())
        return prec

    # ---------- API ----------
    def set_keyframe(self, frame_index: int, frame_bgr: np.ndarray, key_corners: List[Point]) -> None:
        gray = self._to_gray(frame_bgr)
        corners_ord = np.array(order_corners(key_corners), dtype=np.float32)
        self.keyframe_idx = frame_index
        self.keyframe_gray = gray
        self.keyframe_corners = corners_ord
        self.curr_H = np.eye(3, dtype=np.float64)
        self.prev_gray = gray
        self.prev_pts = self._seed_features(gray, corners_ord)
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            self.orig_pts = self.prev_pts.copy()
        else:
            self.orig_pts = np.empty((0, 1, 2), dtype=np.float32)
        self.hold_left = self.cfg.hold_ttl_frames
        self.ema_corners = corners_ord.astype(np.float64)
        self.ref_ratio, self.ref_area = self._shape_metrics(corners_ord)
        # Build model mapping and template once per keyframe
        try:
            H_img2model, model_size = compute_homography(order_corners(corners_ord.tolist()))
            self.H_key_img_to_model = H_img2model
            self.model_size = model_size
            self.model_template = self._build_model_template(model_size, self.cfg.template_line_px)
        except Exception:
            self.H_key_img_to_model = None
            self.model_size = None
            self.model_template = None
        # Init/Reset Kalman with keyframe
        if self.cfg.use_kalman:
            if self.KF is None:
                self.KF = MultiCornerKalman(q_pos=self.cfg.kalman_q_pos, q_vel=self.cfg.kalman_q_vel, r_meas=self.cfg.kalman_r_meas)
            self.KF.reset(corners_ord)

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
        # Track features from prev to curr
        if self.prev_pts is None or len(self.prev_pts) < 4:
            self.prev_pts = self._seed_features(self.prev_gray, self.keyframe_corners)
        if self.prev_pts is None or len(self.prev_pts) < 4:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        st = st.reshape(-1) if st is not None else None
        if next_pts is None or st is None:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Forward-backward check to prune unstable tracks
        try:
            back_pts, st_back, _ = cv2.calcOpticalFlowPyrLK(
                gray, self.prev_gray, next_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
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
            good = good & (fb_err <= float(self.cfg.fb_reproj_thresh))

        p0 = p0_all[good]
        p1 = p1_all[good]
        info["matches"] = int(len(p0))
        if len(p0) < 4:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Build keyframe->current correspondences using surviving tracks
        if self.orig_pts is None or len(self.orig_pts) == 0:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info
        orig_all = self.orig_pts.reshape(-1, 2)
        try:
            orig_surv = orig_all[good]
        except Exception:
            orig_surv = orig_all[: len(p1)]
        curr_surv = p1
        if len(orig_surv) < 4:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Estimate direct keyframe->current transform
        H = None
        M = None
        inliers = None
        if self.cfg.use_homography:
            H, inliers = cv2.findHomography(orig_surv, curr_surv, cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_reproj_thresh)
            info["method"] = "homography"
        else:
            M, inliers = cv2.estimateAffinePartial2D(orig_surv, curr_surv, method=cv2.RANSAC, ransacReprojThreshold=self.cfg.ransac_reproj_thresh)
            info["method"] = "affine"
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        info["inliers"] = inlier_count
        info["inlier_ratio"] = float(inlier_count / max(1, len(orig_surv)))

        if self.cfg.use_homography and H is not None:
            curr_H = H.astype(np.float64)
        elif (not self.cfg.use_homography) and M is not None:
            curr_H = np.eye(3, dtype=np.float64)
            curr_H[:2, :] = M
        else:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Quality checks: conditioning and median reprojection error
        try:
            info["condH"] = float(np.linalg.cond(curr_H))
        except Exception:
            info["condH"] = None
        if inliers is not None and inlier_count > 0:
            mask = inliers.reshape(-1).astype(bool)
            orig_in = orig_surv[mask]
            curr_in = curr_surv[mask]
            proj = np.array(apply_homography_points([(float(x), float(y)) for x, y in orig_in], curr_H), dtype=np.float32)
            errs = np.linalg.norm(curr_in - proj, axis=1)
            med_err = float(np.median(errs)) if errs.size > 0 else 1e9
        else:
            med_err = 1e9
        if (
            info["inlier_ratio"] < self.cfg.min_inlier_ratio
            or inlier_count < self.cfg.min_inliers
            or med_err > (self.cfg.ransac_reproj_thresh * 2.0)
            or (info["condH"] is not None and info["condH"] > 1e4)
        ):
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Accept and set current H
        self.curr_H = curr_H

        # Apply to keyframe corners
        curr_corners = np.array(apply_homography_points(self.keyframe_corners.tolist(), self.curr_H), dtype=np.float32)
        # Subpixel refine the 4 corners for better spatial stability
        try:
            term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.01)
            win = (int(self.cfg.subpix_win), int(self.cfg.subpix_win))
            corners_nx1x2 = curr_corners.reshape(-1, 1, 2).astype(np.float32)
            corners_ref = cv2.cornerSubPix(gray, corners_nx1x2, win, (-1, -1), term)
            if corners_ref is not None and len(corners_ref) == 4:
                curr_corners = corners_ref.reshape(4, 2)
        except Exception:
            pass

        # Geometric sanity checks (ratio/area and jump vs smoothed)
        ratio, area = self._shape_metrics(curr_corners)
        ok_geo = True
        if self.ref_ratio is not None:
            if not self._within_tol(ratio, self.ref_ratio, self.cfg.ratio_tolerance):
                ok_geo = False
        if self.ref_area is not None and self.ref_area > 1e-6:
            if not (1.0 - self.cfg.area_tolerance) * self.ref_area <= area <= (1.0 + self.cfg.area_tolerance) * self.ref_area:
                ok_geo = False
        if self.ema_corners is not None:
            d = np.linalg.norm(curr_corners.astype(np.float64) - self.ema_corners, axis=1)
            if float(np.median(d)) > self.cfg.max_jump_px:
                ok_geo = False
        # Optional template precision score (edge alignment)
        if ok_geo and self.cfg.use_template_score:
            try:
                tscore = self._template_precision_score(gray, self.curr_H)
                info["tpl_prec"] = tscore
                if tscore < self.cfg.template_min_precision:
                    ok_geo = False
            except Exception:
                pass
        if not ok_geo:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            # reseed features around last good corners projected (ema or keyframe)
            base_corners = (self.ema_corners if self.ema_corners is not None else self.keyframe_corners).astype(np.float32)
            self.prev_pts = self._seed_features(gray, base_corners)
            try:
                Hinv = np.linalg.inv(self.curr_H)
                if self.prev_pts is not None and len(self.prev_pts) > 0:
                    pts = self.prev_pts.reshape(-1, 2)
                    key_pts = np.array(apply_homography_points([(float(x), float(y)) for x, y in pts], Hinv), dtype=np.float32).reshape(-1, 1, 2)
                    self.orig_pts = key_pts
                else:
                    self.orig_pts = np.empty((0, 1, 2), dtype=np.float32)
            except Exception:
                self.orig_pts = np.empty((0, 1, 2), dtype=np.float32)
            self.prev_gray = gray
            return None, info

        # Re-seed features if tracks are low
        if len(curr_surv) < self.cfg.reseed_min_tracks:
            info["reseed"] = True
            self.prev_pts = self._seed_features(gray, curr_corners)
            if self.prev_pts is not None and len(self.prev_pts) > 0:
                try:
                    Hinv = np.linalg.inv(self.curr_H)
                    pts = self.prev_pts.reshape(-1, 2)
                    key_pts = np.array(apply_homography_points([(float(x), float(y)) for x, y in pts], Hinv), dtype=np.float32).reshape(-1, 1, 2)
                    self.orig_pts = key_pts
                except Exception:
                    self.orig_pts = np.empty((0, 1, 2), dtype=np.float32)
            else:
                self.orig_pts = np.empty((0, 1, 2), dtype=np.float32)
        else:
            self.prev_pts = curr_surv.reshape(-1, 1, 2).astype(np.float32)
            self.orig_pts = orig_surv.reshape(-1, 1, 2).astype(np.float32)

        # Step frame
        self.prev_gray = gray
        self.hold_left = self.cfg.hold_ttl_frames

        # Kalman smoothing (fallback to EMA)
        if self.cfg.use_kalman and self.KF is not None:
            smoothed = self.KF.update(curr_corners)
            self.ema_corners = smoothed.astype(np.float64)
        else:
            if self.ema_corners is None:
                self.ema_corners = curr_corners.astype(np.float64)
            else:
                a = self.cfg.ema_alpha
                self.ema_corners = self.ema_corners * a + curr_corners.astype(np.float64) * (1.0 - a)
        # Update reference metrics slowly
        r, a = self._shape_metrics(self.ema_corners.astype(np.float32))
        self.ref_ratio = r if self.ref_ratio is None else (self.ref_ratio * 0.98 + r * 0.02)
        self.ref_area = a if self.ref_area is None else (self.ref_area * 0.98 + a * 0.02)

        out_pts = [(float(x), float(y)) for x, y in order_corners(self.ema_corners.tolist())]
        return out_pts, info


__all__ = ["CourtLKTracker", "MultiCornerKalman"]

