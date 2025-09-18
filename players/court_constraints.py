import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CourtConstraint:
    """Projects detections into court space and applies soft filtering safeguards."""

    def __init__(
        self,
        homography_path: str,
        meta_path: Optional[str] = None,
        margin: float = 0.08,
        soft_margin: Optional[float] = None,
        min_keep: int = 0,
        min_ratio: float = 0.0,
        penalty: float = 0.0,
        allow_outside: bool = True,
        fallback_ratio: float = 0.0,
        band_y: float = 0.4,
        band_x: float = 0.12,
    ):
        self.enabled = False
        self.margin = float(max(0.0, margin))
        self.soft_margin = float(max(self.margin, soft_margin if soft_margin is not None else self.margin))
        self.min_keep = max(0, int(min_keep))
        self.min_ratio = float(max(0.0, min(1.0, min_ratio)))
        self.penalty = float(max(0.0, penalty))
        self.allow_outside = bool(allow_outside)
        self.fallback_ratio = float(max(0.0, min(1.0, fallback_ratio)))
        self.band_y = float(max(0.0, band_y))
        self.band_x = float(max(0.0, band_x))
        self.dst_size: Tuple[int, int] = (0, 0)
        self._H = None
        self._court_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._load(homography_path, meta_path)

    def _load(self, homography_path: str, meta_path: Optional[str]):
        try:
            if not os.path.exists(homography_path):
                return
            H = np.load(homography_path)
            if H.shape != (3, 3):
                return
            W = 1800
            Hh = 900
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    dst = meta.get("dst_size")
                    if isinstance(dst, dict):
                        W = int(dst.get("w", W))
                        Hh = int(dst.get("h", Hh))
                except Exception:
                    pass
            self._H = H.astype(np.float64)
            self.dst_size = (float(W), float(Hh))
            self._court_bounds = (0.0, float(W), 0.0, float(Hh))
            self.enabled = True
        except Exception:
            self._H = None
            self.enabled = False

    def _project_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.concatenate([pts, ones], axis=1)
        proj = pts_h @ self._H.T
        proj /= np.clip(proj[:, 2:3], 1e-6, None)
        return proj[:, :2]

    def evaluate_detection(self, detection: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        if not self.enabled or self._H is None:
            return True, {}
        try:
            cx = float(detection.get("x", detection.get("cx", detection.get("center_x", 0.0))))
            cy = float(detection.get("y", detection.get("cy", detection.get("center_y", 0.0))))
        except Exception:
            return False, {}
        pts = np.array([[cx, cy]], dtype=np.float64)
        court_xy = self._project_points(pts)[0]
        W, Hh = self.dst_size
        mx0 = -W * self.margin
        mx1 = W * (1.0 + self.margin)
        my0 = -Hh * self.margin
        my1 = Hh * (1.0 + self.margin)
        sx0 = -W * self.soft_margin
        sx1 = W * (1.0 + self.soft_margin)
        sy0 = -Hh * self.soft_margin
        sy1 = Hh * (1.0 + self.soft_margin)
        x_court, y_court = float(court_xy[0]), float(court_xy[1])
        ok = (mx0 <= x_court <= mx1) and (my0 <= y_court <= my1)
        soft_ok = (sx0 <= x_court <= sx1) and (sy0 <= y_court <= sy1)
        norm_x = x_court / max(1e-6, W)
        norm_y = y_court / max(1e-6, Hh)
        meta = {
            "x": x_court,
            "y": y_court,
            "norm_x": norm_x,
            "norm_y": norm_y,
            "side": "left" if norm_x < 0.5 else "right",
            "inside": ok,
            "soft_inside": soft_ok,
        }
        return ok, meta

    def filter_predictions(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.enabled:
            return detections
        filtered: List[Dict[str, Any]] = []
        outside_pool: List[Dict[str, Any]] = []
        for det in detections:
            ok, meta = self.evaluate_detection(det)
            det_copy = det.copy()
            det_copy["_court"] = meta
            if ok:
                filtered.append(det_copy)
                continue

            if meta.get("soft_inside", False):
                if self.penalty > 0.0:
                    det_copy["confidence"] = max(0.0, float(det_copy.get("confidence", 0.0)) - 0.5 * self.penalty)
                meta_out = dict(det_copy.get("_court", {}))
                meta_out["outside"] = True
                det_copy["_court"] = meta_out
                filtered.append(det_copy)
                continue

            if self.allow_outside:
                if self._accept_outside(meta, det_copy):
                    if self.penalty > 0.0:
                        det_copy["confidence"] = max(0.0, float(det_copy.get("confidence", 0.0)) - self.penalty)
                    meta_out = dict(det_copy.get("_court", {}))
                    meta_out["outside"] = True
                    det_copy["_court"] = meta_out
                    filtered.append(det_copy)
                else:
                    outside_pool.append(det_copy)
                continue

            outside_pool.append(det_copy)
        if outside_pool:
            keep_target = max(
                self.min_keep,
                int(round(len(detections) * self.min_ratio))
            )
            if keep_target > 0 and len(filtered) < keep_target:
                need = keep_target - len(filtered)
                candidate_pool = sorted(outside_pool, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
                taken = 0
                for det in candidate_pool:
                    if taken >= need:
                        break
                    det_soft = det.copy()
                    court_meta = dict(det_soft.get("_court", {}))
                    court_meta["outside"] = True
                    det_soft["_court"] = court_meta
                    if self.penalty > 0.0:
                        det_soft["confidence"] = max(0.0, float(det_soft.get("confidence", 0.0)) - self.penalty)
                    filtered.append(det_soft)
                    taken += 1

        if not filtered:
            return self._annotate_all(detections)

        ratio = len(filtered) / max(1, len(detections))
        if self.fallback_ratio > 0.0 and ratio < self.fallback_ratio:
            return self._annotate_all(detections)
        return filtered

    def _annotate_all(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        annotated: List[Dict[str, Any]] = []
        for det in detections:
            ok, meta = self.evaluate_detection(det)
            det_copy = det.copy()
            if meta:
                det_copy["_court"] = meta
            if self.penalty > 0.0 and not ok:
                det_copy["confidence"] = max(0.0, float(det_copy.get("confidence", 0.0)) - 0.5 * self.penalty)
            annotated.append(det_copy)
        return annotated

    def _accept_outside(self, meta: Dict[str, Any], det: Dict[str, Any]) -> bool:
        if not meta:
            return False
        nx = meta.get("norm_x")
        ny = meta.get("norm_y")
        if nx is None or ny is None:
            return False
        bx = self.band_x
        by = self.band_y
        if not (-bx <= nx <= 1.0 + bx):
            return False
        if (-by <= ny <= 1.0 + by):
            return True
        return False


def build_court_constraint(settings) -> Optional[CourtConstraint]:
    try:
        base_out = settings.common.OUTPUT_DIR
        if not base_out:
            return None
        homography_path = os.path.join(base_out, "court_homography.npy")
        meta_path = os.path.join(base_out, "court_homography.json")
        players_cfg = getattr(settings, "players", None)
        margin = getattr(players_cfg, "COURT_MARGIN", 0.08)
        soft_margin = getattr(players_cfg, "COURT_SOFT_MARGIN", margin)
        min_keep = getattr(players_cfg, "COURT_MIN_KEEP", 0)
        min_ratio = getattr(players_cfg, "COURT_MIN_RATIO", 0.0)
        penalty = getattr(players_cfg, "COURT_CONF_PENALTY", 0.0)
        allow_outside = getattr(players_cfg, "COURT_ALLOW_OUTSIDE", True)
        fallback_ratio = getattr(players_cfg, "COURT_FALLBACK_RATIO", 0.0)
        band_y = getattr(players_cfg, "COURT_OUTSIDE_BAND_Y", 0.4)
        band_x = getattr(players_cfg, "COURT_OUTSIDE_BAND_X", 0.12)
        constraint = CourtConstraint(
            homography_path=homography_path,
            meta_path=meta_path,
            margin=margin,
            soft_margin=soft_margin,
            min_keep=min_keep,
            min_ratio=min_ratio,
            penalty=penalty,
            allow_outside=allow_outside,
            fallback_ratio=fallback_ratio,
            band_y=band_y,
            band_x=band_x,
        )
        return constraint if constraint.enabled else None
    except Exception:
        return None


__all__ = ["CourtConstraint", "build_court_constraint"]
