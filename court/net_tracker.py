from typing import Dict, Any, Optional, Tuple
import numpy as np
import cv2


class ScalarKalman1D:
    """Simple scalar Kalman filter used for smoothing 1-D quantities."""

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


class LinearKalman:
    """Lightweight linear Kalman filter for low-dimensional states."""

    def __init__(self, dim: int, q: float, r: float) -> None:
        self.dim = int(max(dim, 1))
        self.base_q = float(max(q, 1e-9))
        self.base_r = float(max(r, 1e-9))
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None

    def reset(self, value: np.ndarray, variance: Optional[float] = None) -> np.ndarray:
        vec = np.asarray(value, dtype=np.float64).reshape(self.dim)
        var = float(variance if variance is not None and variance > 0 else self.base_r)
        cov = np.eye(self.dim, dtype=np.float64) * var
        self.x = vec.copy()
        self.P = cov
        return self.x

    def predict(self, q_scale: float = 1.0) -> Optional[np.ndarray]:
        if self.x is None or self.P is None:
            return None
        q = self.base_q * max(q_scale, 1e-6)
        self.P = self.P + np.eye(self.dim, dtype=np.float64) * q
        return self.x.copy()

    def update(self, measurement: np.ndarray, r_scale: float = 1.0) -> Optional[np.ndarray]:
        if measurement is None:
            return self.x.copy() if self.x is not None else None
        meas = np.asarray(measurement, dtype=np.float64).reshape(self.dim)
        if not np.all(np.isfinite(meas)):
            return self.x.copy() if self.x is not None else None
        if self.x is None or self.P is None:
            return self.reset(meas, self.base_r)
        R = self.base_r * max(r_scale, 1e-6)
        cov_meas = np.eye(self.dim, dtype=np.float64) * R
        S = self.P + cov_meas
        try:
            K = self.P @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self.x.copy()
        innovation = meas - self.x
        self.x = self.x + K @ innovation
        I = np.eye(self.dim, dtype=np.float64)
        self.P = (I - K) @ self.P
        return self.x.copy()


class NetTracker:
    """Tracks the volleyball net by estimating its baseline in model space."""

    _COURT_LENGTH_M: float = 18.0
    _COURT_WIDTH_M: float = 9.0

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.net_state: Optional[Dict[str, Any]] = None

        self._baseline_filter: Optional[LinearKalman] = None
        self._px_per_meter_filter: Optional[ScalarKalman1D] = None
        self._model_size: Optional[Tuple[int, int]] = None
        self._scale_px_per_meter: Optional[Tuple[float, float]] = None

        self._court_length_m = float(getattr(cfg, "court_length_m", self._COURT_LENGTH_M) or self._COURT_LENGTH_M) if cfg else self._COURT_LENGTH_M
        self._court_width_m = float(getattr(cfg, "court_width_m", self._COURT_WIDTH_M) or self._COURT_WIDTH_M) if cfg else self._COURT_WIDTH_M
        self._net_height_m = float(getattr(cfg, "physical_height_m", 2.43) or 2.43) if cfg else 2.43
        self._fallback_height_px = float(getattr(cfg, "fallback_height_px", 0.0) or 0.0) if cfg else 0.0

        self._direction_vec = np.array([0.0, -1.0], dtype=np.float64)
        self._vanish_point: Optional[np.ndarray] = None
        self._missing_frames: int = 0
        self._seen_measurement: bool = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> Optional[np.ndarray]:
        try:
            a = np.array([float(p1[0]), float(p1[1]), 1.0], dtype=np.float64)
            b = np.array([float(p2[0]), float(p2[1]), 1.0], dtype=np.float64)
            c = np.array([float(p3[0]), float(p3[1]), 1.0], dtype=np.float64)
            d = np.array([float(p4[0]), float(p4[1]), 1.0], dtype=np.float64)
        except Exception:
            return None
        line1 = np.cross(a, b)
        line2 = np.cross(c, d)
        pt = np.cross(line1, line2)
        if abs(pt[2]) < 1e-9:
            return None
        return pt[:2] / pt[2]

    @staticmethod
    def _net_measurement_valid(corners: np.ndarray) -> bool:
        try:
            arr = corners.reshape(4, 2).astype(np.float64)
        except Exception:
            return False
        tl, tr, br, bl = arr

        def _col_metrics(top: np.ndarray, bottom: np.ndarray) -> Tuple[float, float]:
            vec = top - bottom
            height = float(np.linalg.norm(vec))
            horiz = float(abs(vec[0]))
            return height, horiz

        hl, dx_l = _col_metrics(tl, bl)
        hr, dx_r = _col_metrics(tr, br)
        if not (np.isfinite(hl) and np.isfinite(hr)):
            return False
        min_height = min(hl, hr)
        if min_height < 20.0:
            return False
        avg_h = 0.5 * (hl + hr)
        if abs(hl - hr) > max(0.15 * avg_h, 6.0):
            return False
        slant = max(dx_l, dx_r)
        if slant > max(0.18 * avg_h, 8.0):
            return False
        top_span = float(np.linalg.norm(tr - tl))
        bottom_span = float(np.linalg.norm(br - bl))
        if not np.isfinite(top_span) or not np.isfinite(bottom_span):
            return False
        if abs(top_span - bottom_span) > max(0.2 * avg_h, 10.0):
            return False
        return True

    @staticmethod
    def _combine_homography(H_model_to_key_img: Optional[np.ndarray], curr_H: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if H_model_to_key_img is None or curr_H is None:
            return None
        try:
            H = curr_H @ H_model_to_key_img
            if abs(H[2, 2]) > 1e-12:
                H = H / H[2, 2]
            return H
        except Exception:
            return None

    def _ensure_filters(self) -> None:
        cfg = self.cfg
        if not cfg or not getattr(cfg, "enable", True):
            self._baseline_filter = None
            self._px_per_meter_filter = None
            self.net_state = None
            self._missing_frames = 0
            self._seen_measurement = False
            return
        if self._baseline_filter is None:
            q_m = float(getattr(cfg, "model_process_q_m", 1e-3) or 1e-3)
            r_m = float(getattr(cfg, "model_measure_r_m", 2e-2) or 2e-2)
            self._baseline_filter = LinearKalman(4, q_m, r_m)
            default_state = self._default_baseline_state()
            self._baseline_filter.reset(default_state, variance=r_m)
        if self._px_per_meter_filter is None:
            q_px = float(getattr(cfg, "px_per_meter_kalman_q", 0.8) or 0.8)
            r_px = float(getattr(cfg, "px_per_meter_kalman_r", 25.0) or 25.0)
            self._px_per_meter_filter = ScalarKalman1D(q_px, r_px)
            if self._fallback_height_px > 0.0 and self._net_height_m > 1e-6:
                guess = self._fallback_height_px / self._net_height_m
                self._px_per_meter_filter.reset(guess, r_px)

    def _default_baseline_state(self) -> np.ndarray:
        mid_y = self._court_width_m * 0.5
        return np.array([0.0, mid_y, self._court_length_m, mid_y], dtype=np.float64)

    def _update_scale(self, model_size: Optional[Tuple[int, int]]) -> None:
        if model_size is None:
            return
        if self._model_size == model_size and self._scale_px_per_meter is not None:
            return
        W, H = model_size
        if W <= 0 or H <= 0:
            return
        sx = W / max(self._court_length_m, 1e-6)
        sy = H / max(self._court_width_m, 1e-6)
        self._model_size = (int(W), int(H))
        self._scale_px_per_meter = (float(sx), float(sy))

    def _model_m_to_px(self, pts_m: np.ndarray) -> np.ndarray:
        if self._scale_px_per_meter is None:
            return pts_m.copy()
        sx, sy = self._scale_px_per_meter
        arr = np.asarray(pts_m, dtype=np.float64).copy()
        arr[:, 0] *= sx
        arr[:, 1] *= sy
        return arr

    def _model_px_to_m(self, pts_px: np.ndarray) -> np.ndarray:
        if self._scale_px_per_meter is None:
            return pts_px.copy()
        sx, sy = self._scale_px_per_meter
        arr = np.asarray(pts_px, dtype=np.float64).copy()
        arr[:, 0] /= max(sx, 1e-6)
        arr[:, 1] /= max(sy, 1e-6)
        return arr

    def _project_baseline(self, state_m: np.ndarray, H_model_to_curr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if H_model_to_curr is None:
            return None
        try:
            pts_model = np.array(
                [
                    [state_m[0], state_m[1]],
                    [state_m[2], state_m[3]],
                ],
                dtype=np.float64,
            )
            pts_model_px = self._model_m_to_px(pts_model)
            warped = cv2.perspectiveTransform(pts_model_px.reshape(-1, 1, 2).astype(np.float32), H_model_to_curr.astype(np.float64))
            return warped.reshape(-1, 2).astype(np.float64)
        except Exception:
            return None

    def _measurement_to_model_state(
        self,
        base_pts_px: Optional[np.ndarray],
        H_model_to_curr: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if base_pts_px is None or H_model_to_curr is None or self._scale_px_per_meter is None:
            return None
        try:
            H_curr_to_model = np.linalg.inv(H_model_to_curr)
        except np.linalg.LinAlgError:
            return None
        try:
            arr = np.asarray(base_pts_px, dtype=np.float64).reshape(-1, 1, 2)
            pts_model_px = cv2.perspectiveTransform(arr.astype(np.float32), H_curr_to_model.astype(np.float64)).reshape(-1, 2)
            pts_model_m = self._model_px_to_m(pts_model_px)
            return np.array([pts_model_m[0, 0], pts_model_m[0, 1], pts_model_m[1, 0], pts_model_m[1, 1]], dtype=np.float64)
        except Exception:
            return None

    def _parse_measurement(self, measurement: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        info = {
            "confidence": 0.0,
            "base": None,
            "top": None,
            "height_px": None,
            "direction": None,
            "vanish": None,
            "bottom_scalar": None,
            "top_scalar": None,
            "raw_corners": None,
            "detection_present": False,
        }
        if not isinstance(measurement, dict):
            return info
        try:
            info["confidence"] = float(measurement.get("confidence", measurement.get("score", 0.0)) or 0.0)
        except Exception:
            info["confidence"] = 0.0
        corners = measurement.get("corners")
        if corners and len(corners) >= 4:
            try:
                arr = np.array(corners[:4], dtype=np.float64)
                if self._net_measurement_valid(arr):
                    info["raw_corners"] = arr
                    tl, tr, br, bl = arr
                    base = np.vstack([bl, br]).astype(np.float64)
                    top = np.vstack([tl, tr]).astype(np.float64)
                    info["base"] = base
                    info["top"] = top
                    col_left = tl - bl
                    col_right = tr - br
                    heights = np.array([np.linalg.norm(col_left), np.linalg.norm(col_right)], dtype=np.float64)
                    info["height_px"] = float(np.mean(heights))
                    avg_vec = (col_left + col_right) * 0.5
                    norm = float(np.linalg.norm(avg_vec))
                    if norm > 1e-6:
                        dir_vec = avg_vec / norm
                        if dir_vec[1] > 0:
                            dir_vec = -dir_vec
                        info["direction"] = dir_vec
                    vanish = self._line_intersection(bl, tl, br, tr)
                    if vanish is not None and np.all(np.isfinite(vanish)):
                        info["vanish"] = vanish.astype(np.float64)
                    info["detection_present"] = True
            except Exception:
                pass
        if info["base"] is None:
            center = measurement.get("center")
            width_val = measurement.get("width", measurement.get("w"))
            bottom_val = measurement.get("bottom")
            top_val = measurement.get("top")
            try:
                cx = float(center[0]) if center and center[0] is not None else None
                cy = float(center[1]) if center and center[1] is not None else None
            except Exception:
                cx = cy = None
            try:
                width_px = float(width_val) if width_val is not None else None
            except Exception:
                width_px = None
            if cx is not None and cy is not None and width_px is not None:
                half_w = width_px * 0.5
                bottom_y = float(bottom_val) if bottom_val is not None else cy
                base = np.array([[cx - half_w, bottom_y], [cx + half_w, bottom_y]], dtype=np.float64)
                info["base"] = base
                info["bottom_scalar"] = bottom_y
                if top_val is not None:
                    try:
                        top_y = float(top_val)
                    except Exception:
                        top_y = bottom_y - 80.0
                else:
                    top_y = bottom_y - 80.0
                top = np.array([[cx - half_w, top_y], [cx + half_w, top_y]], dtype=np.float64)
                info["top"] = top
                height_val = measurement.get("height") or measurement.get("height_px") or measurement.get("h")
                try:
                    info["height_px"] = float(height_val) if height_val is not None else abs(top_y - bottom_y)
                except Exception:
                    info["height_px"] = abs(top_y - bottom_y)
                info["top_scalar"] = top_y
                info["detection_present"] = True
        if info["height_px"] is None:
            height_val = measurement.get("height") or measurement.get("height_px") or measurement.get("h")
            try:
                info["height_px"] = float(height_val) if height_val is not None else None
            except Exception:
                info["height_px"] = None
        if measurement.get("bottom") is not None:
            try:
                info["bottom_scalar"] = float(measurement.get("bottom"))
            except Exception:
                pass
        if measurement.get("top") is not None and info["top_scalar"] is None:
            try:
                info["top_scalar"] = float(measurement.get("top"))
            except Exception:
                pass
        return info

    # ------------------------------------------------------------------
    def update(
        self,
        frame_idx: int,
        measurement: Optional[Dict[str, Any]],
        H_model_to_key_img: Optional[np.ndarray],
        curr_H: Optional[np.ndarray],
        model_size: Optional[Tuple[int, int]],
        court_corners: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        if not cfg or not getattr(cfg, "enable", True):
            self.net_state = None
            return None

        self._ensure_filters()
        if self._baseline_filter is None:
            self.net_state = None
            return None

        self._update_scale(model_size)
        H_model_to_curr = self._combine_homography(H_model_to_key_img, curr_H)

        if self._vanish_point is None and court_corners is not None:
            try:
                cc = np.array(court_corners, dtype=np.float64).reshape(4, 2)
                candidate = self._line_intersection(cc[3], cc[0], cc[2], cc[1])
                if candidate is not None and np.all(np.isfinite(candidate)):
                    self._vanish_point = candidate.astype(np.float64)
            except Exception:
                pass

        meas_info = self._parse_measurement(measurement)
        meas_conf = float(meas_info["confidence"])
        min_conf = float(getattr(cfg, "measurement_min_confidence", 0.0) or 0.0)
        allow_measurement = meas_conf >= min_conf

        baseline_state = self._baseline_filter.predict()
        if baseline_state is None:
            baseline_state = self._baseline_filter.reset(self._default_baseline_state(), variance=self._baseline_filter.base_r)

        predicted_state = baseline_state.copy()
        predicted_base_px: Optional[np.ndarray] = self._project_baseline(predicted_state, H_model_to_curr)

        used_measurement = False
        detection_seen = meas_info["detection_present"]
        measurement_state_raw = self._measurement_to_model_state(meas_info["base"], H_model_to_curr) if allow_measurement else None
        if measurement_state_raw is not None:
            if not self._seen_measurement:
                # First valid measurement, reset the filter state directly to the raw measurement
                baseline_state = self._baseline_filter.reset(measurement_state_raw)
            else:
                # We have a lock, update using a blended measurement for stability
                blend = float(np.clip(getattr(cfg, "model_measure_blend", 0.4), 0.0, 1.0))
                if meas_conf < 1.0:
                    blend *= float(np.clip(meas_conf, 0.0, 1.0))
                measurement_state_m = predicted_state + blend * (measurement_state_raw - predicted_state)

                r_scale = 1.0
                if meas_conf > 1e-3:
                    effective_conf = float(np.clip(meas_conf, 0.3, 0.9))
                    r_scale = 1.0 / effective_conf
                if predicted_base_px is not None and meas_info["base"] is not None:
                    err_vec = np.asarray(meas_info["base"], dtype=np.float64) - predicted_base_px
                    err_norm = float(np.mean(np.linalg.norm(err_vec, axis=1)))
                    gate_px = float(max(getattr(cfg, "model_measure_gate_px", 64.0), 1.0))
                    r_scale *= 1.0 + (err_norm / gate_px) ** 2
                baseline_state = self._baseline_filter.update(measurement_state_m, r_scale=r_scale)

            used_measurement = baseline_state is not None
            self._seen_measurement = True
            self._missing_frames = 0
        else:
            if used_measurement:
                self._missing_frames = 0
            elif detection_seen:
                self._missing_frames = 0
            else:
                self._missing_frames = min(self._missing_frames + 1, 10**6)

        if baseline_state is None:
            baseline_state = self._default_baseline_state()

        # Predict pixel locations of the baseline
        base_img_pts = self._project_baseline(baseline_state, H_model_to_curr)
        if base_img_pts is None:
            if meas_info["base"] is not None:
                base_img_pts = np.asarray(meas_info["base"], dtype=np.float64)
            elif court_corners is not None:
                try:
                    cc = np.array(court_corners, dtype=np.float64).reshape(4, 2)
                    base_img_pts = np.array([(cc[0] + cc[3]) * 0.5, (cc[1] + cc[2]) * 0.5], dtype=np.float64)
                except Exception:
                    base_img_pts = None
            elif self.net_state is not None:
                try:
                    prev_base = np.array(self.net_state.get("base"), dtype=np.float64)
                    if prev_base.shape == (2, 2):
                        base_img_pts = prev_base
                except Exception:
                    base_img_pts = None

        if allow_measurement and meas_info["direction"] is not None:
            new_dir_vec = np.asarray(meas_info["direction"], dtype=np.float64)
            norm = np.linalg.norm(new_dir_vec)
            if norm > 1e-6:
                new_dir_vec = new_dir_vec / norm
                if new_dir_vec[1] > 0:
                    new_dir_vec = -new_dir_vec
                alpha = float(getattr(cfg, "direction_ema_alpha", 0.1))
                self._direction_vec = alpha * new_dir_vec + (1.0 - alpha) * self._direction_vec
                self._direction_vec /= np.linalg.norm(self._direction_vec)
        elif self._vanish_point is not None and base_img_pts is not None:
            try:
                base_center = base_img_pts.mean(axis=0)
                vanish_vec = np.asarray(self._vanish_point, dtype=np.float64) - base_center
                if vanish_vec[1] > 0:
                    vanish_vec = -vanish_vec
                norm = np.linalg.norm(vanish_vec)
                if norm > 1e-6:
                    new_dir_vec = vanish_vec / norm
                    alpha = float(getattr(cfg, "direction_ema_alpha", 0.1))
                    self._direction_vec = alpha * new_dir_vec + (1.0 - alpha) * self._direction_vec
                    self._direction_vec /= np.linalg.norm(self._direction_vec)
            except Exception:
                pass
        dir_vec = self._direction_vec
        if np.linalg.norm(dir_vec) < 1e-6:
            dir_vec = np.array([0.0, -1.0], dtype=np.float64)
            self._direction_vec = dir_vec

        if allow_measurement and meas_info["vanish"] is not None and np.all(np.isfinite(meas_info["vanish"])):
            self._vanish_point = np.asarray(meas_info["vanish"], dtype=np.float64)

        # Update pixel-per-meter estimate using net height
        px_per_meter_est = None
        if self._px_per_meter_filter is not None:
            px_per_meter_est = self._px_per_meter_filter.predict()
        height_meas_px = meas_info["height_px"]
        if allow_measurement and height_meas_px is not None and self._net_height_m > 1e-6:
            ratio = max(height_meas_px / self._net_height_m, 1e-3)
            noise = None
            if self._px_per_meter_filter is not None:
                if meas_conf > 1e-3:
                    noise = self._px_per_meter_filter.r / max(meas_conf, 0.1)
                px_per_meter_est = self._px_per_meter_filter.update(ratio, noise)
            else:
                px_per_meter_est = ratio
        elif self._px_per_meter_filter is not None:
            px_per_meter_est = self._px_per_meter_filter.x

        if px_per_meter_est is None and self._fallback_height_px > 0.0 and self._net_height_m > 1e-6:
            px_per_meter_est = self._fallback_height_px / self._net_height_m

        # Determine height in pixels for output
        if px_per_meter_est is not None:
            height_px = px_per_meter_est * self._net_height_m
        elif height_meas_px is not None:
            height_px = height_meas_px
        elif self._fallback_height_px > 0:
            height_px = self._fallback_height_px
        else:
            height_px = 90.0

        lo, hi = getattr(cfg, "height_bounds_px", (40.0, 420.0))
        lo = float(min(lo, hi))
        hi = float(max(lo, hi))
        height_px = float(np.clip(height_px, lo, hi))

        if base_img_pts is None:
            self.net_state = None
            return None

        base_left = np.array(base_img_pts[0], dtype=np.float64)
        base_right = np.array(base_img_pts[1], dtype=np.float64)

        if self._vanish_point is not None and np.all(np.isfinite(self._vanish_point)):
            try:
                base_center = (base_left + base_right) * 0.5
                vanish_vec = np.asarray(self._vanish_point, dtype=np.float64) - base_center
                if vanish_vec[1] > 0:
                    vanish_vec = -vanish_vec
                norm = np.linalg.norm(vanish_vec)
                if norm > 1e-6:
                    new_dir_vec = vanish_vec / norm
                    alpha = float(getattr(cfg, "direction_ema_alpha", 0.1))
                    self._direction_vec = alpha * new_dir_vec + (1.0 - alpha) * self._direction_vec
                    self._direction_vec /= np.linalg.norm(self._direction_vec)
                dir_vec = self._direction_vec
            except Exception:
                pass
        dir_vec = self._direction_vec

        left_top = base_left + dir_vec * height_px
        right_top = base_right + dir_vec * height_px

        conf_out: float
        if used_measurement:
            conf_out = max(0.0, min(1.0, meas_conf))
        else:
            hold_frames = max(int(getattr(cfg, "hold_frames_without_measure", 0)), 0)
            missing = self._missing_frames
            conf_out = max(0.05, 1.0 - 0.05 * min(missing, hold_frames))
            extra = max(0, missing - hold_frames)
            if extra > 0:
                conf_out = max(0.02, conf_out - 0.08 * extra)

        filter_variance = None
        if self._baseline_filter is not None and self._baseline_filter.P is not None:
            try:
                filter_variance = float(np.mean(np.diag(self._baseline_filter.P)))
            except Exception:
                filter_variance = None

        state = {
            "frame": int(frame_idx),
            "height_px": height_px,
            "base": [tuple(float(v) for v in base_left), tuple(float(v) for v in base_right)],
            "top": [tuple(float(v) for v in left_top), tuple(float(v) for v in right_top)],
            "polygon": [
                (float(base_left[0]), float(base_left[1])),
                (float(base_right[0]), float(base_right[1])),
                (float(right_top[0]), float(right_top[1])),
                (float(left_top[0]), float(left_top[1])),
            ],
            "confidence": conf_out,
            "measurement_conf": float(meas_conf),
            "measurement_height_px": float(height_meas_px) if height_meas_px is not None else None,
            "measurement_bottom": float(meas_info["bottom_scalar"]) if meas_info["bottom_scalar"] is not None else None,
            "measurement_top": float(meas_info["top_scalar"]) if meas_info["top_scalar"] is not None else None,
            "missing_frames": int(self._missing_frames),
            "filter_variance": filter_variance if filter_variance is not None else 0.0,
            "direction": [float(dir_vec[0]), float(dir_vec[1])],
        }
        if self._vanish_point is not None and np.all(np.isfinite(self._vanish_point)):
            state["vanish_point"] = [
                float(self._vanish_point[0]),
                float(self._vanish_point[1]),
            ]

        self.net_state = state
        return state
