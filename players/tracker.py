"""OC-SORT style tracker with optional jersey number OCR."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import math

import cv2
import numpy as np

from players.jersey_ocr import JerseyOCR, JerseyOCROptions
from players.reid_embedder import build_reid_embedder


_LOG = logging.getLogger(__name__)


def iou_tlbr(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def xywh_to_tlbr(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
    half_w = w / 2.0
    half_h = h / 2.0
    return (x - half_w, y - half_h, x + half_w, y + half_h)


def tlbr_to_xywh(tlbr: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = tlbr
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx, cy, w, h


def clip_tlbr(tlbr: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = tlbr
    x1 = int(round(max(0.0, min(float(width - 1), x1))))
    y1 = int(round(max(0.0, min(float(height - 1), y1))))
    x2 = int(round(max(0.0, min(float(width), x2))))
    y2 = int(round(max(0.0, min(float(height), y2))))
    return x1, y1, x2, y2


def tlbr_center(tlbr: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = tlbr
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _extract_reid_crop(
    frame: np.ndarray,
    tlbr: Tuple[int, int, int, int],
    focus: float = 0.75,
    min_px: int = 20,
    expand: float = 0.1,
) -> np.ndarray:
    x1, y1, x2, y2 = tlbr
    if expand > 0.0:
        w = x2 - x1
        h = y2 - y1
        pad_x = int(round(w * expand))
        pad_y = int(round(h * expand))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame.shape[1], x2 + pad_x)
        y2 = min(frame.shape[0], y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return frame[0:1, 0:1]
    crop = frame[y1:y2, x1:x2]
    if focus < 1.0:
        keep = max(min_px, int(round(crop.shape[0] * focus)))
        crop = crop[:keep, :]
    return crop


def _hist_embed(bgr: np.ndarray) -> Optional[np.ndarray]:
    if bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [4], [0, 256])
    feat = np.concatenate([h.ravel(), s.ravel(), v.ravel()]).astype(np.float32)
    n = float(np.linalg.norm(feat))
    if n <= 1e-6:
        return None
    return feat / n


@dataclass
class TrackerConfig:
    detection_thresh: float = 0.4
    association_iou: float = 0.35
    max_age: int = 24
    min_hits: int = 3
    inertia: float = 0.6
    inertia_min: float = 0.2
    inertia_max: float = 0.75
    inertia_accel_thresh: float = 28.0
    inertia_unmatched_decay: float = 0.55
    velocity_beta: float = 0.65
    velocity_beta_min: float = 0.35
    velocity_beta_max: float = 0.8
    velocity_speed_thresh: float = 42.0
    inertia_decay: float = 0.5
    score_decay: float = 0.92
    process_noise: float = 1e-2
    measurement_noise: float = 5e-2
    init_cov: float = 1.0
    keep_history: int = 1
    center_dist_max: float = 90.0
    size_ratio_min: float = 0.35
    size_ratio_max: float = 2.8
    ocr_enable: bool = False
    ocr_allowlist: str = "0123456789"
    ocr_upper_frac: float = 0.6
    ocr_resize_h: int = 320
    ocr_gpu: bool = False
    ocr_min_conf: float = 0.5
    ocr_smooth_beta: float = 0.6
    ocr_history_max: int = 6
    fast_match_iou: float = 0.6
    reid_enable: bool = True
    reid_history: int = 12
    reid_history_decay: float = 0.65
    reid_similarity_gate: float = 0.3
    reid_similarity_strict: float = 0.45
    reid_long_lost_frames: int = 6
    reid_motion_alpha: float = 0.55
    reid_motion_alpha_lost: float = 0.25
    reid_match_max_cost: float = 0.82
    reid_team_side_gate: bool = True
    ocr_reject_diff_conf: float = 0.75
    ocr_match_bonus: float = 0.65


@dataclass
class Detection:
    tlbr: Tuple[float, float, float, float]
    score: float
    meta: Optional[Dict[str, Any]] = None

    @property
    def measurement(self) -> np.ndarray:
        cx, cy, w, h = tlbr_to_xywh(self.tlbr)
        m = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        # Avoid zero-size updates which can destabilise the Kalman filter
        m[2] = max(m[2], 1.0)
        m[3] = max(m[3], 1.0)
        return m


class OCSortTrack:
    def __init__(self, track_id: int, detection: Detection, frame_idx: int, cfg: TrackerConfig):
        self.id = track_id
        self.cfg = cfg
        self.kf = cv2.KalmanFilter(8, 4)
        self._init_kalman(detection.measurement)
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.confidence = float(detection.score)
        self.last_tlbr = detection.tlbr
        self.last_frame = frame_idx
        self.velocity = np.zeros((4, 1), dtype=np.float32)
        self.prev_velocity = np.zeros((4, 1), dtype=np.float32)
        self.dynamic_inertia = float(np.clip(cfg.inertia, cfg.inertia_min, cfg.inertia_max))
        self.dynamic_beta = float(np.clip(cfg.velocity_beta, cfg.velocity_beta_min, cfg.velocity_beta_max))
        x1, y1, x2, y2 = detection.tlbr
        self.size_w = max(1.0, x2 - x1)
        self.size_h = max(1.0, y2 - y1)
        self._last_measurement = detection.measurement.copy()
        self.history: List[Tuple[float, float, float, float]] = []
        self.court: Dict[str, Any] = {}
        self.court_side: Optional[str] = None
        self.number_votes: Dict[str, float] = {}
        self.number: Optional[str] = None
        self.number_conf: float = 0.0
        self.number_history: List[Tuple[int, str, float]] = []
        self.reid_vectors: List[np.ndarray] = []
        self.reid_ema: Optional[np.ndarray] = None
        self.reid_last_frame: int = frame_idx
        self.team_label: Optional[str] = None
        self._update_meta(detection.meta)

    def _init_kalman(self, measurement: np.ndarray):
        dt = 1.0
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = dt
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * float(self.cfg.process_noise)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * float(self.cfg.measurement_noise)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32) * float(self.cfg.init_cov)
        self.kf.statePost = np.zeros((8, 1), dtype=np.float32)
        self.kf.statePost[:4] = measurement
        self.kf.statePre = self.kf.statePost.copy()

    def _effective_inertia(self) -> float:
        inertia = float(getattr(self, "dynamic_inertia", self.cfg.inertia))
        inertia = max(0.0, min(1.0, inertia))
        if self.time_since_update > 0:
            decay = float(getattr(self.cfg, "inertia_unmatched_decay", 0.6))
            decay = max(0.0, min(1.0, decay))
            steps = max(1, int(self.time_since_update))
            inertia *= decay ** steps
        return inertia

    def _desired_beta(self, raw_delta: np.ndarray) -> float:
        vx = float(raw_delta[0])
        vy = float(raw_delta[1])
        speed = math.hypot(vx, vy)
        thresh = max(1e-3, float(getattr(self.cfg, "velocity_speed_thresh", self.cfg.velocity_beta)))
        ratio = min(1.0, speed / thresh)
        beta_min = max(0.0, min(1.0, float(getattr(self.cfg, "velocity_beta_min", self.cfg.velocity_beta))))
        beta_max = max(beta_min, min(1.0, float(getattr(self.cfg, "velocity_beta_max", self.cfg.velocity_beta))))
        return beta_min + (beta_max - beta_min) * ratio

    def _desired_inertia(self, acc_vec: np.ndarray) -> float:
        ax = float(acc_vec[0])
        ay = float(acc_vec[1])
        accel = math.hypot(ax, ay)
        thresh = max(1e-3, float(getattr(self.cfg, "inertia_accel_thresh", self.cfg.inertia)))
        ratio = min(1.0, accel / thresh)
        inertia_min = max(0.0, min(1.0, float(getattr(self.cfg, "inertia_min", self.cfg.inertia))))
        inertia_max = max(inertia_min, min(1.0, float(getattr(self.cfg, "inertia_max", self.cfg.inertia))))
        return inertia_max - (inertia_max - inertia_min) * ratio

    def predict(self) -> Tuple[float, float, float, float]:
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        # Predict next state and inject learned velocity to reduce inertia bias
        state = self.kf.predict()
        if self.hit_streak > 0:
            inertia = self._effective_inertia()
            if inertia > 0:
                state[:4] = state[:4] + inertia * self.velocity
                self.kf.statePre[:4] = state[:4]
        self.last_tlbr = self._state_to_tlbr(state)
        if self.cfg.keep_history > 0:
            self.history.append(self.last_tlbr)
            if len(self.history) > self.cfg.keep_history:
                self.history.pop(0)
        else:
            self.history.clear()
        return self.last_tlbr

    def _state_to_tlbr(self, state: np.ndarray) -> Tuple[float, float, float, float]:
        cx = float(state[0])
        cy = float(state[1])
        w = max(1.0, float(state[2]))
        h = max(1.0, float(state[3]))
        return xywh_to_tlbr(cx, cy, w, h)

    def _update_meta(self, meta: Optional[Dict[str, Any]]):
        if not meta:
            return
        court = meta.get("court") if isinstance(meta.get("court"), dict) else None
        if court is None and all(k in meta for k in ("x", "y", "norm_x", "norm_y")):
            court = meta
        if court:
            self.court = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in court.items()}
            self.court_side = court.get("side", self.court_side)
            if isinstance(court.get("side"), str):
                self.team_label = str(court.get("side"))

    def _update_number(self, frame_idx: int, number: str, conf: float):
        beta = float(self.cfg.ocr_smooth_beta)
        beta = max(0.0, min(1.0, beta))
        # Decay existing votes
        decay = max(0.0, min(1.0, 1.0 - beta * 0.5))
        for key in list(self.number_votes.keys()):
            self.number_votes[key] *= decay
            if self.number_votes[key] < 1e-3:
                del self.number_votes[key]
        self.number_votes[number] = self.number_votes.get(number, 0.0) * (1.0 - beta) + conf * beta
        best = max(self.number_votes.items(), key=lambda kv: kv[1])
        self.number, self.number_conf = best[0], float(best[1])
        self.number_history.append((frame_idx, number, conf))
        if len(self.number_history) > max(1, int(self.cfg.ocr_history_max)):
            self.number_history.pop(0)

    def _update_reid(self, frame_idx: int, vec: np.ndarray, team_hint: Optional[str] = None):
        if not isinstance(vec, np.ndarray):
            return
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        if not np.any(np.isfinite(vec)):
            return
        n = float(np.linalg.norm(vec))
        if n <= 1e-6:
            return
        vec = vec / n
        max_hist = max(1, int(getattr(self.cfg, "reid_history", 12)))
        self.reid_vectors.append(vec)
        if len(self.reid_vectors) > max_hist:
            self.reid_vectors = self.reid_vectors[-max_hist:]
        decay = float(getattr(self.cfg, "reid_history_decay", 0.65))
        decay = max(0.0, min(1.0, decay))
        if self.reid_ema is None or not np.any(np.isfinite(self.reid_ema)):
            self.reid_ema = vec.copy()
        else:
            self.reid_ema = (1.0 - decay) * self.reid_ema + decay * vec
            denom = float(np.linalg.norm(self.reid_ema))
            if denom > 1e-6:
                self.reid_ema = self.reid_ema / denom
        self.reid_last_frame = frame_idx
        if team_hint:
            self.team_label = team_hint

    def get_reid_feature(self) -> Optional[np.ndarray]:
        if self.reid_ema is not None and np.any(np.isfinite(self.reid_ema)):
            return self.reid_ema
        if self.reid_vectors:
            stacked = np.vstack(self.reid_vectors)
            mean_vec = np.mean(stacked, axis=0)
            n = float(np.linalg.norm(mean_vec))
            if n > 1e-6:
                return mean_vec / n
        return None

    def reid_similarity(self, vec: np.ndarray) -> Optional[float]:
        base = self.get_reid_feature()
        if base is None:
            return None
        if vec.ndim != 1:
            vec = vec.reshape(-1)
        n = float(np.linalg.norm(vec))
        if n <= 1e-6:
            return None
        norm_vec = vec / n
        return float(np.clip(np.dot(base, norm_vec), -1.0, 1.0))

    def team_hint(self) -> Optional[str]:
        return self.team_label or self.court_side

    def update(
        self,
        detection: Detection,
        frame_idx: int,
        frame_bgr: np.ndarray,
        ocr: Optional[JerseyOCR],
        reid_vec: Optional[np.ndarray] = None,
        jersey: Optional[Tuple[str, float]] = None,
    ):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.confidence = float(detection.score)
        self.last_frame = frame_idx
        self.kf.correct(detection.measurement)
        meas = detection.measurement
        raw_delta = meas - self._last_measurement
        desired_beta = self._desired_beta(raw_delta)
        self.dynamic_beta = (0.6 * self.dynamic_beta) + (0.4 * desired_beta)
        self.dynamic_beta = max(float(self.cfg.velocity_beta_min), min(float(self.cfg.velocity_beta_max), self.dynamic_beta))
        beta = max(0.0, min(1.0, self.dynamic_beta))
        prev_velocity = self.velocity.copy()
        self.velocity = (1.0 - beta) * self.velocity + beta * raw_delta
        self.prev_velocity = prev_velocity
        acc_vec = self.velocity - prev_velocity
        desired_inertia = self._desired_inertia(acc_vec)
        conf_scale = max(0.3, min(1.0, float(detection.score)))
        desired_inertia *= conf_scale
        self.dynamic_inertia = (0.7 * self.dynamic_inertia) + (0.3 * desired_inertia)
        self.dynamic_inertia = max(float(self.cfg.inertia_min), min(float(self.cfg.inertia_max), self.dynamic_inertia))
        self._last_measurement = meas.copy()
        self.last_tlbr = detection.tlbr
        dx1, dy1, dx2, dy2 = detection.tlbr
        det_w = max(1.0, dx2 - dx1)
        det_h = max(1.0, dy2 - dy1)
        self.size_w = 0.7 * self.size_w + 0.3 * det_w if hasattr(self, "size_w") else det_w
        self.size_h = 0.7 * self.size_h + 0.3 * det_h if hasattr(self, "size_h") else det_h
        meta = detection.meta if isinstance(detection.meta, dict) else None
        team_hint = None
        if meta:
            court = meta.get("court") if isinstance(meta.get("court"), dict) else meta
            if isinstance(court, dict) and isinstance(court.get("side"), str):
                team_hint = str(court.get("side"))
        self._update_meta(meta)
        if reid_vec is not None and bool(getattr(self.cfg, "reid_enable", True)):
            self._update_reid(frame_idx, reid_vec, team_hint)
        ocr_res = jersey
        if ocr_res is None and frame_bgr is not None and ocr is not None and ocr.available():
            bbox = clip_tlbr(self.last_tlbr, frame_bgr.shape[1], frame_bgr.shape[0])
            ocr_res = ocr.infer(frame_bgr, bbox)
        if ocr_res is not None:
            number, conf = ocr_res
            if number and conf >= float(self.cfg.ocr_min_conf):
                self._update_number(frame_idx, number, conf)

    def miss(self):
        self.confidence *= float(self.cfg.score_decay)
        if self.number_votes:
            for key in list(self.number_votes.keys()):
                self.number_votes[key] *= float(self.cfg.inertia_decay)
                if self.number_votes[key] < 1e-3:
                    del self.number_votes[key]
        if not self.number_votes:
            self.number = None
            self.number_conf = 0.0
        self.velocity *= float(self.cfg.inertia_decay)
        decay = float(getattr(self.cfg, "inertia_unmatched_decay", 0.6))
        self.dynamic_inertia = max(float(self.cfg.inertia_min), min(float(self.cfg.inertia_max), self.dynamic_inertia * decay))
        self.dynamic_beta = max(float(self.cfg.velocity_beta_min), min(float(self.cfg.velocity_beta_max), self.dynamic_beta * decay))
        self.size_w = max(1.0, float(getattr(self, "size_w", 1.0)) * 1.02)
        self.size_h = max(1.0, float(getattr(self, "size_h", 1.0)) * 1.02)

    def is_deleted(self) -> bool:
        return self.time_since_update > int(self.cfg.max_age)

    def is_confirmed(self) -> bool:
        return self.hits >= max(1, int(self.cfg.min_hits))

    def to_dict(self, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = self.last_tlbr
        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w), x2))
        y2 = max(0.0, min(float(h), y2))
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        cx = x1 + width / 2.0
        cy = y1 + height / 2.0
        out: Dict[str, Any] = {
            "id": int(self.id),
            "x": float(cx),
            "y": float(cy),
            "width": float(width),
            "height": float(height),
            "confidence": float(max(0.0, min(1.0, self.confidence))),
        }
        if self.court:
            court_meta = dict(self.court)
            if self.court_side and "side" not in court_meta:
                court_meta["side"] = self.court_side
            out["court"] = court_meta
        if self.number:
            out["jersey"] = {
                "text": self.number,
                "confidence": float(self.number_conf),
            }
        return out


class OCSortTracker:
    def __init__(self, cfg: TrackerConfig):
        self.cfg = cfg
        self.tracks: List[OCSortTrack] = []
        self._next_id = 1
        self.ocr: Optional[JerseyOCR] = None
        if cfg.ocr_enable:
            opts = JerseyOCROptions(
                allowlist=cfg.ocr_allowlist,
                upper_frac=cfg.ocr_upper_frac,
                resize_h=cfg.ocr_resize_h,
                gpu=cfg.ocr_gpu,
                min_conf=cfg.ocr_min_conf,
                smooth_beta=cfg.ocr_smooth_beta,
            )
            self.ocr = JerseyOCR(opts)
            if not self.ocr.available():
                _LOG.info("Jersey OCR requested but EasyOCR is unavailable; continuing without OCR.")
        self.reid_embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self._reid_hist_fallback = False
        if bool(getattr(cfg, "reid_enable", True)):
            settings_obj: Any = cfg
            try:
                from config import settings as global_settings  # type: ignore

                settings_obj = global_settings
            except Exception:
                settings_obj = cfg
            backend = None
            players_cfg = getattr(settings_obj, "players", None)
            if players_cfg is not None and hasattr(players_cfg, "REID_BACKEND"):
                backend = str(getattr(players_cfg, "REID_BACKEND", "onnx")).lower()
            else:
                backend = str(getattr(settings_obj, "PLAYERS_REID_BACKEND", "onnx")).lower()
            try:
                self.reid_embedder = build_reid_embedder(settings_obj)
            except Exception as exc:
                _LOG.warning("ReID embedder init failed; continuing without appearance embeddings: %s", exc)
                self.reid_embedder = None
            if self.reid_embedder is None and backend == "hist":
                self._reid_hist_fallback = True
        if not self.reid_embedder and not self._reid_hist_fallback and bool(getattr(cfg, "reid_enable", True)):
            _LOG.info("ReID embeddings unavailable; falling back to motion-only matching.")
        _LOG.debug("OCSortTracker initialised with %s", cfg)

    def _size_compatible(self, track: OCSortTrack, det_tlbr: Tuple[float, float, float, float]) -> bool:
        base_min = max(1e-3, float(self.cfg.size_ratio_min))
        base_max = max(base_min, float(self.cfg.size_ratio_max))
        expected_w = max(1.0, float(getattr(track, "size_w", 0.0)))
        expected_h = max(1.0, float(getattr(track, "size_h", 0.0)))
        if not math.isfinite(expected_w) or expected_w <= 0:
            tx1, ty1, tx2, ty2 = track.last_tlbr
            expected_w = max(1.0, tx2 - tx1)
        if not math.isfinite(expected_h) or expected_h <= 0:
            tx1, ty1, tx2, ty2 = track.last_tlbr
            expected_h = max(1.0, ty2 - ty1)
        dx1, dy1, dx2, dy2 = det_tlbr
        dw = max(1.0, dx2 - dx1)
        dh = max(1.0, dy2 - dy1)
        gap = max(0, int(getattr(track, "time_since_update", 0)))
        expand = 1.0 + min(2.5, 0.55 * gap)
        min_ratio = min(1.0, base_min / expand)
        max_ratio = base_max * expand
        ratio_w = dw / expected_w
        ratio_h = dh / expected_h
        return (min_ratio <= ratio_w <= max_ratio) and (min_ratio <= ratio_h <= max_ratio)

    def _build_detections(self, detections: List[Dict[str, Any]]) -> List[Detection]:
        det_list: List[Detection] = []
        for d in detections:
            try:
                score = float(d.get("confidence", 0.0))
            except Exception:
                score = 0.0
            if score < float(self.cfg.detection_thresh):
                continue
            try:
                cx = float(d.get("x", 0.0))
                cy = float(d.get("y", 0.0))
                w = float(d.get("width", 0.0))
                h = float(d.get("height", 0.0))
            except Exception:
                continue
            w = max(2.0, w)
            h = max(2.0, h)
            tlbr = xywh_to_tlbr(cx, cy, w, h)
            meta = None
            if isinstance(d, dict):
                court = d.get("_court")
                if isinstance(court, dict):
                    meta = {"court": court}
            det_list.append(Detection(tlbr=tlbr, score=score, meta=meta))
        return det_list

    def _motion_cost(
        self,
        pred_tlbr: Tuple[float, float, float, float],
        det_tlbr: Tuple[float, float, float, float],
        age: int,
    ) -> Optional[float]:
        tcx, tcy = tlbr_center(pred_tlbr)
        dcx, dcy = tlbr_center(det_tlbr)
        if not (math.isfinite(tcx) and math.isfinite(tcy) and math.isfinite(dcx) and math.isfinite(dcy)):
            return None
        dist = math.hypot(tcx - dcx, tcy - dcy)
        norm = max(1.0, float(getattr(self.cfg, "center_dist_max", 90.0)))
        long_lost_frames = max(0, int(getattr(self.cfg, "reid_long_lost_frames", 6)))
        if age > long_lost_frames:
            extra = age - long_lost_frames
            norm *= 1.0 + 0.2 * min(10.0, float(extra))
        return min(1.5, dist / norm)

    def _greedy_assignment_from_iou(
        self, iou_matrix: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        rows, cols = iou_matrix.shape
        all_pairs = [
            (t_idx, d_idx)
            for t_idx in range(rows)
            for d_idx in range(cols)
        ]
        all_pairs.sort(key=lambda rc: float(iou_matrix[rc[0], rc[1]]), reverse=True)
        taken_tracks: set[int] = set()
        taken_dets: set[int] = set()
        assignments: List[Tuple[int, int]] = []
        for t_idx, d_idx in all_pairs:
            if t_idx in taken_tracks or d_idx in taken_dets:
                continue
            assignments.append((t_idx, d_idx))
            taken_tracks.add(t_idx)
            taken_dets.add(d_idx)
        if not assignments:
            return [], []
        rows_sel, cols_sel = zip(*assignments)
        return list(rows_sel), list(cols_sel)

    def update(self, frame_bgr: np.ndarray, frame_idx: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dets = self._build_detections(detections)
        if not self.tracks and not dets:
            return []
        frame_shape = frame_bgr.shape[:2]

        preds = [track.predict() for track in self.tracks]

        matched_track_indices: List[int] = []
        matched_det_indices: List[int] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_dets = set(range(len(dets)))
        match_extras: Dict[int, Tuple[Optional[np.ndarray], Optional[Tuple[str, float]]]] = {}

        width = frame_shape[1] if len(frame_shape) > 1 else frame_bgr.shape[1]
        height = frame_shape[0]
        det_bbox_cache: Dict[int, Tuple[int, int, int, int]] = {}
        det_embed_cache: Dict[int, Optional[np.ndarray]] = {}
        det_ocr_cache: Dict[int, Optional[Tuple[str, float]]] = {}

        def _det_bbox(idx: int) -> Tuple[int, int, int, int]:
            if idx in det_bbox_cache:
                return det_bbox_cache[idx]
            bbox = clip_tlbr(dets[idx].tlbr, width, height)
            det_bbox_cache[idx] = bbox
            return bbox

        def _det_team(idx: int) -> Optional[str]:
            meta = dets[idx].meta if isinstance(dets[idx].meta, dict) else None
            if meta:
                court = meta.get("court") if isinstance(meta.get("court"), dict) else meta
                if isinstance(court, dict):
                    side = court.get("side")
                    if isinstance(side, str):
                        return str(side)
            return None

        def _det_embed(idx: int) -> Optional[np.ndarray]:
            if self.reid_embedder is None and not self._reid_hist_fallback:
                return None
            if idx in det_embed_cache:
                return det_embed_cache[idx]
            x1, y1, x2, y2 = _det_bbox(idx)
            if x2 <= x1 or y2 <= y1:
                det_embed_cache[idx] = None
                return None
            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                det_embed_cache[idx] = None
                return None
            try:
                if self.reid_embedder is not None:
                    vec = self.reid_embedder(crop)
                else:
                    vec = _hist_embed(_extract_reid_crop(frame_bgr, (x1, y1, x2, y2)))
            except Exception as exc:
                _LOG.debug("ReID embedding failed for detection %d: %s", idx, exc)
                vec = None
            det_embed_cache[idx] = vec
            return vec

        def _det_ocr(idx: int) -> Optional[Tuple[str, float]]:
            if self.ocr is None or not self.ocr.available():
                return None
            if idx in det_ocr_cache:
                return det_ocr_cache[idx]
            res = self.ocr.infer(frame_bgr, _det_bbox(idx))
            det_ocr_cache[idx] = res
            return res

        ocr_runner: Optional[JerseyOCR] = self.ocr if self.ocr and self.ocr.available() else None

        iou_matrix: Optional[np.ndarray] = None
        if self.tracks and dets:
            iou_matrix = np.zeros((len(self.tracks), len(dets)), dtype=np.float32)
            for t_idx in range(len(self.tracks)):
                for d_idx in range(len(dets)):
                    iou_matrix[t_idx, d_idx] = iou_tlbr(preds[t_idx], dets[d_idx].tlbr)

            fast_thresh = float(getattr(self.cfg, "fast_match_iou", self.cfg.association_iou))
            fast_thresh = max(0.0, fast_thresh)
            if fast_thresh > 0.0:
                cost_matrix = 1.0 - iou_matrix
                try:
                    from scipy.optimize import linear_sum_assignment  # type: ignore

                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                except Exception:
                    row_ind, col_ind = self._greedy_assignment_from_iou(iou_matrix)
                for r, c in zip(row_ind, col_ind):
                    if r not in unmatched_tracks or c not in unmatched_dets:
                        continue
                    iou_val = float(iou_matrix[r, c])
                    if iou_val < fast_thresh:
                        continue
                    matched_track_indices.append(r)
                    matched_det_indices.append(c)
                    unmatched_tracks.discard(r)
                    unmatched_dets.discard(c)

        if unmatched_tracks and unmatched_dets:
            track_order = sorted(
                list(unmatched_tracks),
                key=lambda idx: (self.tracks[idx].time_since_update, -self.tracks[idx].hits),
                reverse=True,
            )
            det_order = list(unmatched_dets)
            if track_order and det_order:
                long_lost_frames = max(0, int(getattr(self.cfg, "reid_long_lost_frames", 6)))
                large_cost = 1e6
                pair_info: Dict[Tuple[int, int], Tuple[float, Optional[np.ndarray], Optional[Tuple[str, float]]]] = {}
                cost_matrix = np.full((len(track_order), len(det_order)), large_cost, dtype=np.float32)
                for t_pos, t_idx in enumerate(track_order):
                    track = self.tracks[t_idx]
                    pred_box = preds[t_idx]
                    track_team = track.team_hint()
                    track_long_lost = track.time_since_update >= long_lost_frames
                    alpha = float(self.cfg.reid_motion_alpha_lost if track_long_lost else self.cfg.reid_motion_alpha)
                    alpha = max(0.0, min(1.0, alpha))
                    sim_gate = float(self.cfg.reid_similarity_strict if track_long_lost else self.cfg.reid_similarity_gate)
                    sim_gate = max(-1.0, min(1.0, sim_gate))
                    max_cost = float(self.cfg.reid_match_max_cost)
                    if track_long_lost:
                        max_cost += 0.08
                    for d_pos, d_idx in enumerate(det_order):
                        det = dets[d_idx]
                        if not self._size_compatible(track, det.tlbr):
                            continue
                        det_team = _det_team(d_idx)
                        if (
                            track_team
                            and det_team
                            and bool(getattr(self.cfg, "reid_team_side_gate", True))
                            and track_team != det_team
                        ):
                            continue
                        iou_val = float(iou_matrix[t_idx, d_idx]) if iou_matrix is not None else iou_tlbr(pred_box, det.tlbr)
                        reid_vec = None
                        reid_sim = None
                        if self.reid_embedder is not None or self._reid_hist_fallback:
                            reid_vec = _det_embed(d_idx)
                            if reid_vec is None:
                                continue
                            reid_sim = track.reid_similarity(reid_vec)
                            if track.reid_vectors and reid_sim is None:
                                continue
                            if reid_sim is not None and reid_sim < sim_gate:
                                continue
                        else:
                            if iou_val < float(self.cfg.association_iou):
                                continue
                        jersey_res = None
                        if track.number and ocr_runner is not None:
                            jersey_res = _det_ocr(d_idx)
                            if jersey_res is not None:
                                number_guess, ocr_conf = jersey_res
                                reject_thresh = float(getattr(self.cfg, "ocr_reject_diff_conf", self.cfg.ocr_min_conf))
                                if number_guess and number_guess != track.number and ocr_conf >= reject_thresh:
                                    continue
                        motion_cost = self._motion_cost(pred_box, det.tlbr, track.time_since_update)
                        if motion_cost is None:
                            continue
                        appearance_base = reid_sim if reid_sim is not None else iou_val
                        appearance_cost = max(0.0, min(1.5, 1.0 - appearance_base))
                        total_cost = alpha * motion_cost + (1.0 - alpha) * appearance_cost
                        if (
                            jersey_res is not None
                            and track.number
                            and jersey_res[0] == track.number
                            and jersey_res[1] >= float(self.cfg.ocr_min_conf)
                        ):
                            total_cost *= float(getattr(self.cfg, "ocr_match_bonus", 0.65))
                        total_cost = max(0.0, total_cost)
                        if total_cost > max_cost:
                            continue
                        cost_matrix[t_pos, d_pos] = total_cost
                        pair_info[(t_pos, d_pos)] = (total_cost, reid_vec, jersey_res)

                if pair_info:
                    try:
                        from scipy.optimize import linear_sum_assignment  # type: ignore

                        row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    except Exception:
                        # greedy fallback if SciPy not available
                        assignments = []
                        taken_d = set()
                        for t_pos in range(cost_matrix.shape[0]):
                            best = None
                            best_cost = large_cost
                            for d_pos in range(cost_matrix.shape[1]):
                                if d_pos in taken_d:
                                    continue
                                c = float(cost_matrix[t_pos, d_pos])
                                if c < best_cost:
                                    best_cost = c
                                    best = d_pos
                            if best is not None and best_cost < large_cost:
                                assignments.append((t_pos, best))
                                taken_d.add(best)
                        row_ind = np.array([a[0] for a in assignments], dtype=int)
                        col_ind = np.array([a[1] for a in assignments], dtype=int)

                    for t_pos, d_pos in zip(row_ind, col_ind):
                        cost_val = float(cost_matrix[t_pos, d_pos])
                        if cost_val >= large_cost:
                            continue
                        info = pair_info.get((t_pos, d_pos))
                        if info is None:
                            continue
                        total_cost, best_vec, best_jersey = info
                        if total_cost >= large_cost:
                            continue
                        t_idx = track_order[t_pos]
                        d_idx = det_order[d_pos]
                        if t_idx not in unmatched_tracks or d_idx not in unmatched_dets:
                            continue
                        unmatched_tracks.discard(t_idx)
                        unmatched_dets.discard(d_idx)
                        matched_track_indices.append(t_idx)
                        matched_det_indices.append(d_idx)
                        if best_vec is not None or best_jersey is not None:
                            match_extras[t_idx] = (best_vec, best_jersey)

        if unmatched_tracks and unmatched_dets:
            dist_thresh = max(0.0, float(self.cfg.center_dist_max))
            if dist_thresh > 0.0:
                tracks_order = sorted(list(unmatched_tracks), key=lambda idx: self.tracks[idx].time_since_update)
                for t_idx in tracks_order:
                    track = self.tracks[t_idx]
                    pred_box = preds[t_idx]
                    tcx, tcy = tlbr_center(pred_box)
                    if not (math.isfinite(tcx) and math.isfinite(tcy)):
                        continue
                    best_idx: Optional[int] = None
                    best_dist: Optional[float] = None
                    for d_idx in list(unmatched_dets):
                        det = dets[d_idx]
                        if not self._size_compatible(track, det.tlbr):
                            continue
                        dcx, dcy = tlbr_center(det.tlbr)
                        if not (math.isfinite(dcx) and math.isfinite(dcy)):
                            continue
                        dist = math.hypot(tcx - dcx, tcy - dcy)
                        if dist > dist_thresh:
                            continue
                        if best_idx is None or (best_dist is not None and dist < best_dist):
                            best_idx = d_idx
                            best_dist = dist
                    if best_idx is None:
                        continue
                    unmatched_tracks.discard(t_idx)
                    unmatched_dets.discard(best_idx)
                    matched_track_indices.append(t_idx)
                    matched_det_indices.append(best_idx)
                    prev_vec, prev_jersey = match_extras.get(t_idx, (None, None))
                    new_vec = _det_embed(best_idx) if (self.reid_embedder is not None or self._reid_hist_fallback) else None
                    if new_vec is None:
                        new_vec = prev_vec
                    jersey_res = prev_jersey
                    if jersey_res is None and ocr_runner is not None and self.tracks[t_idx].number:
                        jersey_res = _det_ocr(best_idx)
                    if new_vec is not None or jersey_res is not None:
                        match_extras[t_idx] = (new_vec, jersey_res)

        for t_idx, d_idx in zip(matched_track_indices, matched_det_indices):
            track = self.tracks[t_idx]
            det = dets[d_idx]
            reid_vec, jersey_res = match_extras.get(t_idx, (None, None))
            if reid_vec is None and (self.reid_embedder is not None or self._reid_hist_fallback):
                reid_vec = _det_embed(d_idx)
            if jersey_res is None and ocr_runner is not None and track.number:
                jersey_res = det_ocr_cache.get(d_idx)
                if jersey_res is None:
                    jersey_res = _det_ocr(d_idx)
            track.update(det, frame_idx, frame_bgr, ocr_runner, reid_vec=reid_vec, jersey=jersey_res)

        for t_idx in unmatched_tracks:
            self.tracks[t_idx].miss()

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        for d_idx in unmatched_dets:
            det = dets[d_idx]
            new_track = OCSortTrack(self._next_id, det, frame_idx, self.cfg)
            self._next_id += 1
            team_hint = _det_team(d_idx)
            if self.reid_embedder is not None or self._reid_hist_fallback:
                init_vec = _det_embed(d_idx)
                if init_vec is not None:
                    new_track._update_reid(frame_idx, init_vec, team_hint)
            self.tracks.append(new_track)

        outputs: List[Dict[str, Any]] = []
        for track in self.tracks:
            if track.is_confirmed() or track.time_since_update == 0:
                outputs.append(track.to_dict(frame_shape))
        return outputs


__all__ = ["TrackerConfig", "OCSortTracker"]
