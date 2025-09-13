import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Callable
from collections import deque

import cv2
import numpy as np


def iou_tlbr(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def xywh_to_tlbr(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + h / 2))
    return x1, y1, x2, y2


def crop_safe(img: np.ndarray, tlbr: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = tlbr
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return img[0:1, 0:1]
    return img[y1:y2, x1:x2]


def hist_embed(img: np.ndarray, bins_h: int = 16, bins_s: int = 8, bins_v: int = 4) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins_h], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [bins_s], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [bins_v], [0, 256])
    vec = np.concatenate([h.flatten(), s.flatten(), v.flatten()]).astype(np.float32)
    n = np.linalg.norm(vec) + 1e-6
    return vec / n


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


@dataclass
class TrackerConfig:
    track_thresh: float = 0.35  # high-score threshold
    low_track_thresh: float = 0.1  # second-association threshold (ByteTrack-style)
    match_iou_thresh: float = 0.3
    reid_weight: float = 0.35  # weight for appearance in [0,1]
    max_age: int = 30          # max frames to keep lost track
    min_hits: int = 3          # confirm after N hits
    hist_bins_h: int = 16
    hist_bins_s: int = 8
    hist_bins_v: int = 4
    # Advanced gating (keep simple gates)
    reid_min_sim: float = 0.25
    size_change_max_ratio: float = 2.0  # allow up to 2x size change between frames
    # Anti-switch guard
    id_lock_age: int = 1              # frames since last hit considered "locked"
    switch_min_sim: float = 0.35      # require at least this ReID sim to switch when locked
    # Jersey OCR
    ocr_enable: bool = True
    ocr_min_conf: float = 0.5
    ocr_upper_frac: float = 0.6
    jersey_min_track_conf: float = 0.6
    # OCR dynamic bonus
    ocr_bonus_max: float = 0.35
    # Kalman removed for players tracking


class Track:
    def __init__(self, tid: int, tlbr: Tuple[int, int, int, int], conf: float, emb: Optional[np.ndarray], frame_idx: int, cfg: Optional[TrackerConfig] = None):
        self.id = tid
        self.tlbr = tlbr
        self.conf = conf
        self.emb = emb
        self.embeds = deque([], maxlen=5)
        if emb is not None:
            self.embeds.append(emb)
        self.age = 0
        self.hits = 1
        self.last_frame = frame_idx
        self.active = True
        self._cfg = cfg or TrackerConfig()
        # Size (EMA)
        x1, y1, x2, y2 = tlbr
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        self.size_w = w
        self.size_h = h
        # Jersey number state
        self.jersey: Optional[str] = None
        self.jersey_conf: float = 0.0
        self._jersey_buf: deque = deque([], maxlen=10)
        # No Kalman state for players

    def update(self, tlbr: Tuple[int, int, int, int], conf: float, emb: Optional[np.ndarray], frame_idx: int, jersey_obs: Optional[Tuple[Optional[str], float]] = None):
        self.tlbr = tlbr
        self.conf = conf
        # Update size EMA
        x1, y1, x2, y2 = tlbr
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        self.size_w = 0.8 * self.size_w + 0.2 * w if hasattr(self, 'size_w') else w
        self.size_h = 0.8 * self.size_h + 0.2 * h if hasattr(self, 'size_h') else h
        # No Kalman measurement update
        if emb is not None:
            # EMA to stabilize appearance
            if self.emb is None:
                self.emb = emb
            else:
                self.emb = 0.7 * self.emb + 0.3 * emb
                n = np.linalg.norm(self.emb) + 1e-6
                self.emb = self.emb / n
            self.embeds.append(emb)
        # Jersey smoothing buffer
        if jersey_obs is not None:
            jtxt, jconf = jersey_obs
            if jtxt and jconf >= self._cfg.ocr_min_conf:
                self._jersey_buf.append((str(jtxt), float(jconf)))
                # vote winner
                counts: Dict[str, List[float]] = {}
                for ttxt, c in self._jersey_buf:
                    counts.setdefault(ttxt, []).append(c)
                best_txt = None
                best_score = -1.0
                for k, vs in counts.items():
                    score = len(vs) + 0.1 * sum(vs)
                    if score > best_score:
                        best_score = score
                        best_txt = k
                if best_txt is not None:
                    conf_avg = float(sum(counts[best_txt]) / len(counts[best_txt]))
                    # lock-in when seen >=3 times with decent conf
                    if len(counts[best_txt]) >= 3 and conf_avg >= max(self._cfg.ocr_min_conf, 0.55):
                        self.jersey = best_txt
                        self.jersey_conf = conf_avg
        self.age = 0
        self.hits += 1
        self.last_frame = frame_idx
        self.active = True


class ByteTrackReID:
    """
    Lightweight ByteTrack-style tracker with simple ReID (HSV color histogram).

    - One-stage greedy assignment using combined IoU and appearance similarity.
    - Keeps IDs stable across short occlusions (max_age frames).
    - Confirms tracks after `min_hits` to reduce flicker.
    """

    def __init__(self, cfg: TrackerConfig, embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None, jersey_ocr: Optional[Callable[[np.ndarray], Tuple[str, float]]] = None):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self._next_id = 1
        self.embedder = embedder if embedder is not None else (lambda img: hist_embed(img, cfg.hist_bins_h, cfg.hist_bins_s, cfg.hist_bins_v))
        self.jersey_ocr = jersey_ocr

    def _score(self, track: Track, det_tlbr: Tuple[int, int, int, int], det_emb: Optional[np.ndarray]) -> float:
        # use predicted state (track.tlbr is kept in sync with prediction)
        iou = iou_tlbr(track.tlbr, det_tlbr)
        sim = 0.0
        if det_emb is not None:
            # use best-of gallery similarity for robustness
            if getattr(track, 'embeds', None):
                sim = max(cos_sim(e, det_emb) for e in track.embeds)
            elif track.emb is not None:
                sim = cos_sim(track.emb, det_emb)
        w = self.cfg.reid_weight
        return (1.0 - w) * iou + w * max(0.0, sim)

    def _ocr_bonus(self, track: Track, jtxt: Optional[str], jconf: float) -> float:
        """Dynamic OCR bonus based on confidence of detection and track jersey.
        - Applies only when both sides have jersey and text matches.
        - Scales bonus by normalized confidences with a cap `ocr_bonus_max`.
        """
        try:
            if not (self.cfg.ocr_enable):
                return 0.0
            if not jtxt or jconf is None:
                return 0.0
            if not track.jersey or str(track.jersey) != str(jtxt):
                return 0.0
            # Require detection OCR above min conf
            if float(jconf) < float(self.cfg.ocr_min_conf):
                return 0.0
            # Normalize confidences to [0,1] from [ocr_min_conf, 1]
            mn = float(self.cfg.ocr_min_conf)
            def _norm(c: float) -> float:
                return max(0.0, min(1.0, (float(c) - mn) / (1.0 - mn + 1e-6)))
            nd = _norm(float(jconf))
            nt = _norm(float(getattr(track, 'jersey_conf', 0.0)))
            strength = math.sqrt(max(0.0, nd * nt))  # conservative combine
            return min(float(self.cfg.ocr_bonus_max), strength * float(self.cfg.ocr_bonus_max))
        except Exception:
            return 0.0

    def _size_gate(self, track: Track, det_tlbr: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = det_tlbr
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        rw = w / max(1e-3, track.size_w)
        rh = h / max(1e-3, track.size_h)
        m = self.cfg.size_change_max_ratio
        return (1.0 / m) <= rw <= m and (1.0 / m) <= rh <= m

    def _cleanup(self):
        kept = []
        for t in self.tracks:
            if t.age <= self.cfg.max_age:
                kept.append(t)
        self.tracks = kept

    def update(self, frame_bgr: np.ndarray, frame_idx: int, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Split detections into high and low confidence sets
        high_dets: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray], Optional[str], float]] = []
        low_dets: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray], Optional[str], float]] = []
        for d in detections:
            try:
                conf = float(d.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            x = float(d.get("x", 0.0))
            y = float(d.get("y", 0.0))
            w = float(d.get("width", 0.0))
            h = float(d.get("height", 0.0))
            tlbr = xywh_to_tlbr(x, y, w, h)
            crop = crop_safe(frame_bgr, tlbr)
            emb = self.embedder(crop)
            # jersey OCR (upper torso)
            jersey_txt: Optional[str] = None
            jersey_conf: float = 0.0
            if self.cfg.ocr_enable and self.jersey_ocr is not None:
                try:
                    x1, y1, x2, y2 = tlbr
                    uh = max(1, int(round((y2 - y1) * self.cfg.ocr_upper_frac)))
                    torso = crop_safe(frame_bgr, (x1, y1, x2, y1 + uh))
                    jtxt, jconf = self.jersey_ocr(torso)
                    if jtxt and jconf >= self.cfg.ocr_min_conf:
                        jersey_txt, jersey_conf = str(jtxt), float(jconf)
                except Exception:
                    pass
            if conf >= self.cfg.track_thresh:
                high_dets.append((tlbr, conf, emb, jersey_txt, jersey_conf))
            elif conf >= self.cfg.low_track_thresh:
                low_dets.append((tlbr, conf, emb, jersey_txt, jersey_conf))

        # Age existing tracks
        for t in self.tracks:
            t.age += 1
            t.active = False

        def _assign(tracks: List[Track], dets: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray], Optional[str], float]]):
            if not tracks or not dets:
                return set(), set(range(len(dets)))
            # Try Hungarian; fallback to greedy
            try:
                from scipy.optimize import linear_sum_assignment  # type: ignore
                # Build cost matrix (1 - score)
                N = len(tracks)
                M = len(dets)
                cost = np.full((N, M), 1.0, dtype=np.float32)
                for i, t in enumerate(tracks):
                    for j, (tlbr, conf, emb, jtxt, jconf) in enumerate(dets):
                        iou = iou_tlbr(t.tlbr, tlbr)
                        sim = 0.0 if emb is None else (
                            max(cos_sim(e, emb) for e in t.embeds) if getattr(t, 'embeds', None) else (cos_sim(t.emb, emb) if t.emb is not None else 0.0)
                        )
                        # Base gates: IoU or ReID
                        if not ( (iou >= self.cfg.match_iou_thresh) or (sim >= self.cfg.reid_min_sim) ):
                            continue
                        if not self._size_gate(t, tlbr):
                            continue
                        # No Kalman gating
                        # Jersey mismatch gate when both available
                        if (t.jersey and t.jersey_conf >= self.cfg.jersey_min_track_conf) and (jtxt and jconf >= self.cfg.ocr_min_conf):
                            if str(t.jersey) != str(jtxt):
                                continue
                        # Anti-switch guard: for recently confirmed tracks, require stronger evidence to switch
                        if t.age <= self.cfg.id_lock_age:
                            if (sim < self.cfg.switch_min_sim) and (iou < max(0.5, self.cfg.match_iou_thresh)):
                                continue
                        s = self._score(t, tlbr, emb)
                        # Dynamic OCR bonus when jersey matches
                        s = min(1.0, s + self._ocr_bonus(t, jtxt, jconf))
                        cost[i, j] = 1.0 - max(0.0, min(1.0, s))
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_t = set()
                used_d = set()
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] <= 0.8:  # allow fairly loose; main gate is IoU
                        tlbr, conf, emb, jtxt, jconf = dets[c]
                        tracks[r].update(tlbr, conf, emb, frame_idx, jersey_obs=(jtxt, jconf))
                        matched_t.add(r)
                        used_d.add(c)
                unmatched_d = set(range(M)) - used_d
                return matched_t, unmatched_d
            except Exception:
                # Greedy fallback
                unmatched = set(range(len(dets)))
                for t in sorted(tracks, key=lambda x: (x.age, -x.hits)):
                    best_idx = -1
                    best_score = -1.0
                    for j in list(unmatched):
                        tlbr, conf, emb, jtxt, jconf = dets[j]
                        iou = iou_tlbr(t.tlbr, tlbr)
                        sim = 0.0 if emb is None else (
                            max(cos_sim(e, emb) for e in t.embeds) if getattr(t, 'embeds', None) else (cos_sim(t.emb, emb) if t.emb is not None else 0.0)
                        )
                        if not ( (iou >= self.cfg.match_iou_thresh) or (sim >= self.cfg.reid_min_sim) ):
                            continue
                        if not self._size_gate(t, tlbr):
                            continue
                        # No Kalman gating
                        if (t.jersey and t.jersey_conf >= self.cfg.jersey_min_track_conf) and (jtxt and jconf >= self.cfg.ocr_min_conf):
                            if str(t.jersey) != str(jtxt):
                                continue
                        if t.age <= self.cfg.id_lock_age:
                            if (sim < self.cfg.switch_min_sim) and (iou < max(0.5, self.cfg.match_iou_thresh)):
                                continue
                        s = self._score(t, tlbr, emb)
                        s = min(1.0, s + self._ocr_bonus(t, jtxt, jconf))
                        if s > best_score:
                            best_score = s
                            best_idx = j
                    if best_idx >= 0:
                        tlbr, conf, emb, jtxt, jconf = dets[best_idx]
                        t.update(tlbr, conf, emb, frame_idx, jersey_obs=(jtxt, jconf))
                        unmatched.discard(best_idx)
                return set(), unmatched

        # First association: high-score detections
        _, high_unmatched = _assign(self.tracks, high_dets)

        # Second association: unassigned tracks with low-score detections
        low_unmatched = set(range(len(low_dets)))
        if low_dets:
            _, low_unmatched = _assign(self.tracks, low_dets)

        # Create new tracks for unmatched high_dets first, then low_dets
        def _spawn(dets_list, idxs):
            for idx in idxs:
                item = dets_list[idx]
                tlbr, conf, emb = item[0], item[1], item[2]
                # Spawn only if not already close to an existing active track (reduce duplicates)
                ok = True
                for t in self.tracks:
                    if iou_tlbr(t.tlbr, tlbr) >= 0.7:
                        ok = False
                        break
                if ok:
                    t = Track(self._next_id, tlbr, conf, emb, frame_idx, cfg=self.cfg)
                    self._next_id += 1
                    self.tracks.append(t)

        _spawn(high_dets, high_unmatched)
        _spawn(low_dets, low_unmatched)

        # Cleanup old tracks
        self._cleanup()

        # Emit confirmed, active tracks
        out: List[Dict[str, Any]] = []
        for t in self.tracks:
            if t.active and t.hits >= max(1, self.cfg.min_hits):
                x1, y1, x2, y2 = t.tlbr
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                cx = x1 + w / 2.0
                cy = y1 + h / 2.0
                row = {
                    "id": int(t.id),
                    "x": float(cx),
                    "y": float(cy),
                    "width": float(w),
                    "height": float(h),
                    "confidence": float(t.conf),
                }
                if t.jersey:
                    row["jersey"] = t.jersey
                # No Kalman debug fields
                out.append(row)
        return out


__all__ = ["TrackerConfig", "ByteTrackReID"]
