from typing import Any, Optional, Tuple
import numpy as np
import cv2


class FeatureManager:
    """Manages seeding, tracking, and filtering of feature points for motion estimation."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self._lk_win = (21, 21)
        self._lk_levels = 3
        self._lk_term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    def seed_features(self, gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Seeds new feature points within a region of interest defined by corners."""
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

        try:
            term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
            win = (int(self.cfg.features.subpix_win), int(self.cfg.features.subpix_win))
            pts_ref = cv2.cornerSubPix(gray, pts.astype(np.float32), win, (-1, -1), term)
            return pts_ref
        except Exception:
            return pts.astype(np.float32)

    def track_features(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
        use_roi: bool = False,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Tracks points from prev_gray to curr_gray and performs a forward-backward check."""
        next_pts, st = self._lk_forward(prev_gray, curr_gray, prev_pts, use_roi, roi_rect)
        if next_pts is None or st is None:
            return None, None

        back_pts, st_back = self._lk_backward(prev_gray, curr_gray, next_pts, use_roi, roi_rect)
        if back_pts is None or st_back is None:
            return None, None

        # Filter good tracks
        p0_all = prev_pts.reshape(-1, 2)
        st = st.reshape(-1)
        st_back = st_back.reshape(-1)
        good = (st == 1) & (st_back == 1)

        p0_back = back_pts.reshape(-1, 2)
        fb_err = np.linalg.norm(p0_all - p0_back, axis=1)
        good = good & (fb_err <= float(self.cfg.core.fb_reproj_thresh))

        p0_good = p0_all[good]
        p1_good = next_pts.reshape(-1, 2)[good]

        if len(p0_good) < 4:
            return None, None

        return p0_good, p1_good

    def _lk_forward(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
        use_roi: bool,
        roi_rect: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        next_pts, st, _ = self._lk_flow_roi(prev_gray, curr_gray, prev_pts, use_roi, roi_rect)
        return next_pts, st

    def _lk_backward(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        next_pts: np.ndarray,
        use_roi: bool,
        roi_rect: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        back_pts, st_back, _ = self._lk_flow_roi(curr_gray, prev_gray, next_pts, use_roi, roi_rect)
        return back_pts, st_back

    def _lk_flow_roi(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        pts: np.ndarray,
        use_roi: bool,
        roi_rect: Optional[Tuple[int, int, int, int]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if use_roi and roi_rect:
            try:
                rx1, ry1, rx2, ry2 = roi_rect
                img1_roi = img1[ry1:ry2 + 1, rx1:rx2 + 1]
                img2_roi = img2[ry1:ry2 + 1, rx1:rx2 + 1]
                pts_roi = (pts.reshape(-1, 2) - np.array([rx1, ry1], dtype=np.float32)).reshape(-1, 1, 2)

                next_pts_roi, st, err = self._lk_flow(img1_roi, img2_roi, pts_roi)

                if next_pts_roi is not None:
                    next_pts = (next_pts_roi.reshape(-1, 2) + np.array([rx1, ry1], dtype=np.float32)).reshape(-1, 1, 2)
                    return next_pts, st, err
            except Exception:
                pass  # Fallback to full frame

        return self._lk_flow(img1, img2, pts)

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