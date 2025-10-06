from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2


class MotionEstimator:
    """
    Estimates motion between frames by fitting a robust model to feature point matches.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.fallback_cfg = getattr(cfg, "fallback", None)

    def _evaluate_model(
        self,
        model_type: str,
        matrix: np.ndarray,
        inliers: Optional[np.ndarray],
        p0: np.ndarray,
        p1: np.ndarray,
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
            curr_in = p1[mask]
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
        err_cap = self.cfg.core.ransac_reproj_thresh * (1.5 if model_type == "homography" else 1.2)

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
            "mask": mask,
        }

    def _estimate_models(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []

        try:
            H, inliers = cv2.findHomography(
                p0,
                p1,
                cv2.RANSAC,
                ransacReprojThreshold=self.cfg.core.ransac_reproj_thresh,
            )
            if H is not None:
                evaluated = self._evaluate_model("homography", H, inliers, p0, p1)
                if evaluated:
                    models.append(evaluated)
        except Exception:
            pass

        try:
            M, inliers = cv2.estimateAffinePartial2D(
                p0,
                p1,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.cfg.core.ransac_reproj_thresh,
            )
            if M is not None:
                evaluated = self._evaluate_model("affine", M, inliers, p0, p1)
                if evaluated:
                    models.append(evaluated)
        except Exception:
            pass

        return models

    def _fallback_ecc_enabled(self) -> bool:
        cfg = getattr(self, "fallback_cfg", None)
        return bool(cfg and getattr(cfg, "ecc_enable", False))

    def _make_ecc_roi(
        self,
        gray: np.ndarray,
        prev_gray: np.ndarray,
        base_corners: np.ndarray,
        _roi_scale: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int, int]]:
        if prev_gray is None:
            return None
        try:
            corners = base_corners.reshape(-1, 2).astype(np.float32)
        except Exception:
            return None
        Hh, Ww = gray.shape[:2]
        expand = max(
            float(self.cfg.features.roi_expand_ratio),
            float(self.cfg.features.lk_roi_expand_ratio) * float(_roi_scale),
        )
        
        xs = corners[:, 0]
        ys = corners[:, 1]
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        dx = (x2 - x1) * expand
        dy = (y2 - y1) * expand
        rx1 = max(0, int(np.floor(x1 - dx)))
        ry1 = max(0, int(np.floor(y1 - dy)))
        rx2 = min(Ww - 1, int(np.ceil(x2 + dx)))
        ry2 = min(Hh - 1, int(np.ceil(y2 + dy)))
        
        if rx2 <= rx1 or ry2 <= ry1:
            return None
        prev_roi = prev_gray[ry1:ry2 + 1, rx1:rx2 + 1]
        curr_roi = gray[ry1:ry2 + 1, rx1:rx2 + 1]
        if prev_roi.size == 0 or curr_roi.size == 0:
            return None
        if prev_roi.shape[0] < 8 or prev_roi.shape[1] < 8:
            return None
        return prev_roi, curr_roi, rx1, ry1

    def _try_ecc_motion(
        self,
        gray: np.ndarray,
        prev_gray: np.ndarray,
        base_corners: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        info: Dict[str, Any],
        _roi_scale: float,
        last_tpl_prec: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        if not self._fallback_ecc_enabled():
            return None
        cfg = getattr(self, "fallback_cfg", None)
        if cfg is None or prev_gray is None:
            return None
        if p0 is None or p1 is None or len(p0) < max(self.cfg.core.min_inliers, 4):
            return None
        min_tpl = float(getattr(cfg, "ecc_min_tpl_precision", 0.0) or 0.0)
        if min_tpl > 0.0 and last_tpl_prec is not None and last_tpl_prec < min_tpl:
            return None
        roi = self._make_ecc_roi(gray, prev_gray, base_corners, _roi_scale)
        if roi is None:
            return None
        prev_roi, curr_roi, rx1, ry1 = roi
        info["ecc_attempt"] = True
        prev_ecc = prev_roi
        curr_ecc = curr_roi
        scale_x = 1.0
        scale_y = 1.0
        try:
            down = float(getattr(cfg, "ecc_downscale", 1.0))
        except Exception:
            down = 1.0
        if 0.1 < down < 1.0:
            new_w = max(8, int(round(prev_roi.shape[1] * down)))
            new_h = max(8, int(round(prev_roi.shape[0] * down)))
            prev_ecc = cv2.resize(prev_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            curr_ecc = cv2.resize(curr_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scale_x = float(prev_roi.shape[1]) / float(prev_ecc.shape[1])
            scale_y = float(prev_roi.shape[0]) / float(prev_ecc.shape[0])
        prev_f = prev_ecc.astype(np.float32)
        curr_f = curr_ecc.astype(np.float32)
        if prev_f.max() > 1.5:
            prev_f *= 1.0 / 255.0
            curr_f *= 1.0 / 255.0
        try:
            gauss = int(getattr(cfg, "ecc_gauss_kernel", 5))
        except Exception:
            gauss = 5
        if gauss < 0:
            gauss = 0
        if gauss % 2 == 0:
            gauss += 1
        if gauss >= 3:
            prev_f = cv2.GaussianBlur(prev_f, (gauss, gauss), 0)
            curr_f = cv2.GaussianBlur(curr_f, (gauss, gauss), 0)
        max_it = max(1, int(getattr(cfg, "ecc_max_iterations", 30)))
        eps = float(getattr(cfg, "ecc_epsilon", 1e-4))
        term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_it, max(eps, 1e-7))
        warp = np.eye(3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(prev_f, curr_f, warp, cv2.MOTION_HOMOGRAPHY, term, None, gauss if gauss >= 3 else 1)
        except cv2.error:
            return None
        except Exception:
            return None
        warp64 = warp.astype(np.float64)
        if not np.isfinite(warp64).all():
            return None
        if scale_x != 1.0 or scale_y != 1.0:
            S = np.array([[1.0 / scale_x, 0.0, 0.0], [0.0, 1.0 / scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            S_inv = np.array([[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            warp64 = S_inv @ warp64 @ S
        offset = np.array([[1.0, 0.0, float(rx1)], [0.0, 1.0, float(ry1)], [0.0, 0.0, 1.0]], dtype=np.float64)
        offset_inv = np.array([[1.0, 0.0, -float(rx1)], [0.0, 1.0, -float(ry1)], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_prev_curr = offset @ warp64 @ offset_inv
        if abs(H_prev_curr[2, 2]) > 1e-12:
            H_prev_curr = H_prev_curr / H_prev_curr[2, 2]
        ones = np.ones((len(p0), 1), dtype=np.uint8)
        evaluated = self._evaluate_model("homography", H_prev_curr, ones, p0, p1)
        if evaluated is None:
            return None
        evaluated["type"] = "ecc"
        evaluated["ecc_score"] = float(cc)
        info["ecc_score"] = float(cc)
        return evaluated

    def estimate_motion(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        gray: np.ndarray,
        prev_gray: np.ndarray,
        base_corners: np.ndarray,
        info: Dict[str, Any],
        _roi_scale: float,
        last_tpl_prec: Optional[float],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        
        info["matches"] = int(len(p0))
        
        if len(p0) < 4:
            return None, None, None

        # --- Estimate motion model ---
        models = self._estimate_models(p0, p1)
        
        if not models:
            ecc_model = self._try_ecc_motion(
                gray, prev_gray, base_corners, p0, p1, info, _roi_scale, last_tpl_prec
            )
            if ecc_model is not None:
                models = [ecc_model]
        
        if not models:
            return None, None, None

        # --- Select best model ---
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
            return None, None, None

        candidate_pool = passing
        candidate_pool.sort(key=model_score)
        selected = candidate_pool[0]

        # --- Finalize surviving points based on model inliers ---
        mask = selected["mask"]
        p0_final = p0[mask]
        p1_final = p1[mask]

        return p0_final, p1_final, selected


# Helper function to avoid circular imports
def apply_homography_points(points: List[Tuple[float, float]], H: np.ndarray) -> List[Tuple[float, float]]:
    """Apply homography to a list of points."""
    try:
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, H)
        return [tuple(p) for p in warped.reshape(-1, 2)]
    except Exception:
        return points
