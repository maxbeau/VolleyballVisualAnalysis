import os
import json
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2

from core.config import settings
from core.utils import ensure_dir
from court.utils import order_corners
from court.config import CourtTrackerConfig
from court.tracker import CourtLKTracker as ExtCourtLKTracker
from court.io import load_detections


Point = Tuple[float, float]


Point = Tuple[float, float]


class CourtLKTracker:
    """
    Lightweight court tracker between sparse keyframes using LK optical flow
    and robust motion estimation (affine/homography). Designed to minimize API
    usage by filling frames between low-frequency detections.
    """

    def __init__(
        self,
        feature_max: int = 200,
        roi_expand_ratio: float = 0.06,
        ransac_reproj_thresh: float = 3.0,
        min_inlier_ratio: float = 0.35,
        min_inliers: int = 20,
        reseed_min_tracks: int = 40,
        hold_ttl_frames: int = 8,
        use_homography: bool = True,
        ema_alpha: float = 0.85,
        max_jump_px: float = 8.0,
        ratio_tolerance: float = 0.3,  # +-30%
        area_tolerance: float = 0.5,   # accept [1-area_tol, 1+area_tol]
        use_template_score: bool = True,
        template_line_px: int = 8,
        template_min_precision: float = 0.28,
        cfg: Optional[CourtTrackerConfig] = None,
    ) -> None:
        # Backward-compatible args; prefer cfg if provided
        if cfg is None:
            self.cfg = CourtTrackerConfig(
                feature_max=feature_max,
                roi_expand_ratio=roi_expand_ratio,
                ransac_reproj_thresh=ransac_reproj_thresh,
                min_inlier_ratio=min_inlier_ratio,
                min_inliers=min_inliers,
                reseed_min_tracks=reseed_min_tracks,
                hold_ttl_frames=hold_ttl_frames,
                use_homography=use_homography,
                ema_alpha=ema_alpha,
                max_jump_px=max_jump_px,
                ratio_tolerance=ratio_tolerance,
                area_tolerance=area_tolerance,
                use_template_score=use_template_score,
                template_line_px=template_line_px,
                template_min_precision=template_min_precision,
            )
        else:
            self.cfg = cfg
        # Mirror important fields for existing code
        self.feature_max = self.cfg.feature_max
        self.roi_expand_ratio = self.cfg.roi_expand_ratio
        self.ransac_reproj_thresh = self.cfg.ransac_reproj_thresh
        self.min_inlier_ratio = self.cfg.min_inlier_ratio
        self.min_inliers = self.cfg.min_inliers
        self.reseed_min_tracks = self.cfg.reseed_min_tracks
        self.hold_ttl_frames = self.cfg.hold_ttl_frames
        self.use_homography = self.cfg.use_homography
        self.ema_alpha = self.cfg.ema_alpha
        self.max_jump_px = self.cfg.max_jump_px
        self.ratio_tolerance = self.cfg.ratio_tolerance
        self.area_tolerance = self.cfg.area_tolerance
        self.use_template_score = self.cfg.use_template_score
        self.template_line_px = self.cfg.template_line_px
        self.template_min_precision = self.cfg.template_min_precision

        # State
        self.keyframe_idx: Optional[int] = None
        self.keyframe_gray: Optional[np.ndarray] = None
        self.keyframe_corners: Optional[np.ndarray] = None  # (4,2)
        self.curr_H: Optional[np.ndarray] = None  # 3x3 mapping keyframe->current
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None  # (N,1,2) current-frame coords
        self.orig_pts: Optional[np.ndarray] = None  # (N,1,2) keyframe coords
        self.hold_left: int = 0
        self.ema_corners: Optional[np.ndarray] = None  # (4,2)
        self.ref_ratio: Optional[float] = None
        self.ref_area: Optional[float] = None
        # keyframe <-> model
        self.H_key_img_to_model: Optional[np.ndarray] = None
        self.model_size: Optional[Tuple[int, int]] = None
        self.model_template: Optional[np.ndarray] = None  # uint8 mask (W,H)
        # Kalman filter
        self.KF: Optional[MultiCornerKalman] = None

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
        H, W = gray.shape[:2]
        x1, y1, x2, y2 = self._poly_roi_bounds(corners, self.roi_expand_ratio, W, H)
        mask = np.zeros_like(gray)
        mask[y1:y2 + 1, x1:x2 + 1] = 255

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.feature_max,
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
            win = (int(self.cfg.subpix_win), int(self.cfg.subpix_win)) if hasattr(self, 'cfg') and self.cfg else (5, 5)
            pts_ref = cv2.cornerSubPix(gray, pts.astype(np.float32), win, (-1, -1), term)
            return pts_ref
        except Exception:
            return pts

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
            # At keyframe, original (keyframe) coords equal current
            self.orig_pts = self.prev_pts.copy()
        else:
            self.orig_pts = np.empty((0, 1, 2), dtype=np.float32)
        self.hold_left = self.hold_ttl_frames
        self.ema_corners = corners_ord.astype(np.float64)
        # initialize refs
        self.ref_ratio, self.ref_area = self._shape_metrics(corners_ord)
        # Build model mapping and template once per keyframe
        try:
            H_img2model, model_size = compute_homography(order_corners(corners_ord.tolist()))
            self.H_key_img_to_model = H_img2model
            self.model_size = model_size
            self.model_template = self._build_model_template(model_size, self.template_line_px)
        except Exception:
            self.H_key_img_to_model = None
            self.model_size = None
            self.model_template = None
        # Init/Reset Kalman with keyframe
        if self.cfg.use_kalman:
            if self.KF is None:
                self.KF = MultiCornerKalman(
                    q_pos=self.cfg.kalman_q_pos, q_vel=self.cfg.kalman_q_vel, r_meas=self.cfg.kalman_r_meas
                )
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
            thr = getattr(self, 'cfg', None).fb_reproj_thresh if getattr(self, 'cfg', None) else 1.2
            good = good & (fb_err <= float(thr))

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
        curr_surv = p1
        # Use the same 'good' mask that filtered p1
        try:
            orig_surv = orig_all[good]
        except Exception:
            # Fallback if mask mismatch
            orig_surv = orig_all[: len(curr_surv)]
        if len(orig_surv) < 4:
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            return None, info

        # Estimate direct keyframe->current transform
        H = None
        M = None
        inliers = None
        if self.use_homography:
            H, inliers = cv2.findHomography(orig_surv, curr_surv, cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_thresh)
            info["method"] = "homography"
        else:
            M, inliers = cv2.estimateAffinePartial2D(orig_surv, curr_surv, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_reproj_thresh)
            info["method"] = "affine"
        inlier_count = int(inliers.sum()) if inliers is not None else 0
        info["inliers"] = inlier_count
        info["inlier_ratio"] = float(inlier_count / max(1, len(orig_surv)))

        # Form 3x3 H
        if self.use_homography and H is not None:
            curr_H = H.astype(np.float64)
        elif not self.use_homography and M is not None:
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
        if info["inlier_ratio"] < self.min_inlier_ratio or inlier_count < self.min_inliers or med_err > (self.ransac_reproj_thresh * 2.0) or (info["condH"] is not None and info["condH"] > 1e4):
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
            win = (int(self.cfg.subpix_win), int(self.cfg.subpix_win)) if getattr(self, 'cfg', None) else (5, 5)
            corners_nx1x2 = curr_corners.reshape(-1, 1, 2).astype(np.float32)
            corners_ref = cv2.cornerSubPix(gray, corners_nx1x2, win, (-1, -1), term)
            if corners_ref is not None and len(corners_ref) == 4:
                curr_corners = corners_ref.reshape(4, 2)
        except Exception:
            pass

        # Geometric sanity checks (ratio/area and jump vs EMA)
        ratio, area = self._shape_metrics(curr_corners)
        ok_geo = True
        if self.ref_ratio is not None:
            if not self._within_tol(ratio, self.ref_ratio, self.ratio_tolerance):
                ok_geo = False
        if self.ref_area is not None and self.ref_area > 1e-6:
            if not (1.0 - self.area_tolerance) * self.ref_area <= area <= (1.0 + self.area_tolerance) * self.ref_area:
                ok_geo = False
        if self.ema_corners is not None:
            d = np.linalg.norm(curr_corners.astype(np.float64) - self.ema_corners, axis=1)
            if float(np.median(d)) > self.max_jump_px:
                ok_geo = False
        # Optional template precision score (edge alignment)
        if ok_geo and self.use_template_score:
            try:
                tscore = self._template_precision_score(gray, self.curr_H)
                info["tpl_prec"] = tscore
                if tscore < self.template_min_precision:
                    ok_geo = False
            except Exception:
                pass

        if not ok_geo:
            # reject update: keep last EMA, reseed around last EMA in current frame to stay in sync
            info["hold"] = True
            self.hold_left = max(0, self.hold_left - 1)
            # reseed features around last good corners projected (ema_corners)
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
        if len(curr_surv) < self.reseed_min_tracks:
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
        self.hold_left = self.hold_ttl_frames

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
        if self.ref_ratio is None:
            self.ref_ratio = r
        else:
            self.ref_ratio = self.ref_ratio * 0.98 + r * 0.02
        if self.ref_area is None:
            self.ref_area = a
        else:
            self.ref_area = self.ref_area * 0.98 + a * 0.02
        ema_out = [(float(x), float(y)) for x, y in order_corners(self.ema_corners.tolist())]
        return ema_out, info

    @staticmethod
    def _shape_metrics(corners: np.ndarray) -> Tuple[float, float]:
        """Return (aspect_ratio approx, polygon area) for 4-point quad.
        aspect ~ average(top,bottom)/average(left,right)."""
        c = corners.reshape(4, 2).astype(np.float64)
        tl, tr, br, bl = c
        top = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left = np.linalg.norm(bl - tl)
        right = np.linalg.norm(br - tr)
        w = max(1e-6, 0.5 * (top + bottom))
        h = max(1e-6, 0.5 * (left + right))
        ratio = float(w / h)
        # polygon area (shoelace)
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
        # Outer rectangle
        cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), 255, thickness=line_px)
        # Center line
        cy = int(round(H / 2))
        cv2.line(canvas, (0, cy), (W - 1, cy), 255, thickness=line_px)
        # Attack lines at 3m from center (model scale assumed via model_size)
        # Standard court is 18m x 9m => H corresponds to 9m, center at 4.5m, attack at 1.5m/7.5m => y=H*(1.5/9), y=H*(7.5/9)
        a1 = int(round(H * (1.5 / 9.0)))
        a2 = int(round(H * (7.5 / 9.0)))
        cv2.line(canvas, (0, a1), (W - 1, a1), 255, thickness=line_px)
        cv2.line(canvas, (0, a2), (W - 1, a2), 255, thickness=line_px)
        return canvas

    def _template_precision_score(self, gray: np.ndarray, H_key_to_curr: np.ndarray) -> float:
        if self.H_key_img_to_model is None or self.model_template is None:
            return 1.0  # can't score; don't block
        # Model -> keyframe is inverse of keyframe image -> model
        H_model_to_key = np.linalg.inv(self.H_key_img_to_model)
        # Model -> current
        H_model_to_curr = H_model_to_key @ H_key_to_curr
        Hh, Ww = gray.shape[:2]
        warped = cv2.warpPerspective(self.model_template, H_model_to_curr, (Ww, Hh), flags=cv2.INTER_NEAREST)
        # Edge map of image
        edges = cv2.Canny(gray, 50, 150)
        # Slightly dilate edges to allow tolerance
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # Compute precision: how many template edge pixels hit image edges
        tmask = warped > 0
        if not np.any(tmask):
            return 0.0
        overlap = (edges > 0) & tmask
        prec = float(overlap.sum()) / float(tmask.sum())
        return prec


def _load_detections(detections_jsonl: str) -> List[Dict[str, Any]]:
    return load_detections(detections_jsonl)


## Kalman and tracker implementation moved to court.tracker

# Deprecated alias: ensure any import of CourtLKTracker from this module
# resolves to the canonical implementation in court.tracker
CourtLKTracker = ExtCourtLKTracker


def run_tracking(
    video_path: str,
    detections_jsonl: str,
    tracking_jsonl: str,
    use_homography: bool = True,
    ransac_thresh: float = 3.0,
    hold_ttl: int = 8,
) -> None:
    ensure_dir(os.path.dirname(tracking_jsonl) or ".")
    dets = _load_detections(detections_jsonl)
    if not dets:
        raise RuntimeError(f"No usable detections found in {detections_jsonl}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Build tracker with explicit overrides via kwargs
    tracker = ExtCourtLKTracker(
        cfg=CourtTrackerConfig(
            lk_roi_expand_ratio=settings.LK_ROI_EXPAND_RATIO,
            max_scale_change_per_frame=settings.MAX_SCALE_CHANGE_PER_FRAME,
            kf_adaptive_from_template=settings.KF_ADAPTIVE_FROM_TEMPLATE,
            kf_r_api_min=settings.KF_R_API_MIN,
            kf_r_api_max=settings.KF_R_API_MAX,
        ),
        use_homography=use_homography,
        ransac_reproj_thresh=ransac_thresh,
        hold_ttl_frames=hold_ttl,
    )

    det_idx = 0
    next_key = dets[det_idx]
    next_key_frame = int(next_key["frame"]) if next_key else None
    prev_det_corners: Optional[List[Point]] = None

    with open(tracking_jsonl, "w", encoding="utf-8") as out_f:
        frame_i = 0
        last_corners: Optional[List[Point]] = None
        while frame_i < total_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # If this frame is a keyframe per detections, reset tracker
            if next_key is not None and frame_i == next_key_frame:
                accept = True
                det_corners = order_corners(next_key["corners"])  # TL,TR,BR,BL
                # If we already have a track, validate keyframe against last corners and shape refs
                if last_corners is not None and tracker.ema_corners is not None:
                    det_arr = np.array(det_corners, dtype=np.float32)
                    last_arr = np.array(last_corners, dtype=np.float32)
                    # displacement
                    d = np.linalg.norm(det_arr - last_arr, axis=1)
                    med_d = float(np.median(d))
                    # shape metrics
                    r_det, a_det = tracker._shape_metrics(det_arr)
                    ok_shape = True
                    if tracker.ref_ratio is not None and not tracker._within_tol(r_det, tracker.ref_ratio, max(0.35, tracker.cfg.ratio_tolerance)):
                        ok_shape = False
                    if tracker.ref_area is not None and tracker.ref_area > 1e-6:
                        lo = (1.0 - max(0.6, tracker.cfg.area_tolerance)) * tracker.ref_area
                        hi = (1.0 + max(0.6, tracker.cfg.area_tolerance)) * tracker.ref_area
                        if not (lo <= a_det <= hi):
                            ok_shape = False
                    if med_d > max(12.0, tracker.cfg.max_jump_px * 1.5) or not ok_shape:
                        accept = False
                # Additional fallback: compare to previous detection corners as well
                if accept and prev_det_corners is not None:
                    det_arr = np.array(det_corners, dtype=np.float32)
                    prev_arr = np.array(prev_det_corners, dtype=np.float32)
                    d2 = np.linalg.norm(det_arr - prev_arr, axis=1)
                    med_d2 = float(np.median(d2))
                    if med_d2 > 20.0:  # hard limit across detections
                        accept = False

                if accept:
                    tracker.set_keyframe(frame_i, frame, det_corners)
                    # Write smoothed corners at keyframe to avoid a visual jump
                    if tracker.ema_corners is not None:
                        sm = [(float(x), float(y)) for x, y in order_corners(tracker.ema_corners.tolist())]
                        info = {"keyframe": True, "tpl_prec": tracker.last_tpl_prec}
                        out_f.write(json.dumps({"frame": frame_i, "corners": sm, "info": info}, ensure_ascii=False) + "\n")
                        last_corners = sm
                    else:
                        info = {"keyframe": True, "tpl_prec": tracker.last_tpl_prec}
                        out_f.write(json.dumps({"frame": frame_i, "corners": det_corners, "info": info}, ensure_ascii=False) + "\n")
                        last_corners = det_corners
                else:
                    # Reject suspicious keyframe; attempt tracking update instead
                    corners, info = tracker.update(frame)
                    if corners is not None:
                        out_f.write(json.dumps({"frame": frame_i, "corners": corners}, ensure_ascii=False) + "\n")
                        last_corners = corners
                    elif last_corners is not None:
                        out_f.write(json.dumps({"frame": frame_i, "corners": last_corners}, ensure_ascii=False) + "\n")

                # advance to next detection
                det_idx += 1
                next_key = dets[det_idx] if det_idx < len(dets) else None
                next_key_frame = int(next_key["frame"]) if next_key is not None else None
                prev_det_corners = det_corners
            else:
                # Regular frame: predict via LK+RANSAC
                corners, info = tracker.update(frame)
                if corners is not None:
                    out_f.write(json.dumps({"frame": frame_i, "corners": corners, "info": info}, ensure_ascii=False) + "\n")
                    last_corners = corners
                else:
                    # If tracker is in hold window and we have last corners, repeat for continuity
                    if info.get("hold") and info.get("hold_left", 0) > 0 and last_corners is not None:
                        out_f.write(json.dumps({"frame": frame_i, "corners": last_corners, "info": info}, ensure_ascii=False) + "\n")

            frame_i += 1

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Track court corners between low-frequency detections using LK+RANSAC")
    parser.add_argument("--detections-jsonl", default=settings.COURT_DETECTIONS_JSONL)
    parser.add_argument("--tracking-jsonl", default=settings.COURT_TRACKING_JSONL)
    parser.add_argument("--use-homography", action="store_true", help="Use homography (default). If not set, uses affine.")
    parser.add_argument("--affine", action="store_true", help="Force affine instead of homography")
    parser.add_argument("--ransac-thresh", type=float, default=3.0)
    parser.add_argument("--hold-ttl", type=int, default=8)
    args = parser.parse_args()

    # Resolve method flags
    use_h = True
    if args.affine:
        use_h = False
    elif args.use_homography:
        use_h = True

    run_tracking(
        video_path=settings.VIDEO_PATH,
        detections_jsonl=args.detections_jsonl,
        tracking_jsonl=args.tracking_jsonl,
        use_homography=use_h,
        ransac_thresh=args.ransac_thresh,
        hold_ttl=args.hold_ttl,
    )

    print(f"Court tracking saved: {args.tracking_jsonl}")


if __name__ == "__main__":
    main()
