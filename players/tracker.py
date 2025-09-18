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


def _clamp_box(tlbr: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = tlbr
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    return x1, y1, x2, y2


def extract_reid_crop(
    img: np.ndarray,
    tlbr: Tuple[int, int, int, int],
    cfg: "TrackerConfig",
) -> np.ndarray:
    """Return a clamped crop that emphasises upper-body appearance cues."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = tlbr
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))

    expand = max(0.0, float(getattr(cfg, "reid_expand_ratio", 0.0)))
    if expand > 0.0:
        pad_x = int(round(bw * expand))
        pad_y = int(round(bh * expand))
        x1 -= pad_x
        x2 += pad_x
        y1 -= pad_y
        y2 += pad_y

    x1, y1, x2, y2 = _clamp_box((x1, y1, x2, y2), w, h)

    # Optionally focus on the torso/upper body by trimming the lower section
    focus = float(getattr(cfg, "reid_focus_top", 1.0))
    focus = max(0.2, min(1.0, focus))
    if focus < 1.0:
        current_h = max(1, y2 - y1)
        keep_h = max(int(round(current_h * focus)), int(getattr(cfg, "reid_min_crop_px", 10)))
        y2 = min(h - 1, y1 + keep_h)

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
    track_thresh: float = 0.4   # high-score threshold
    low_track_thresh: float = 0.15  # second-association threshold (ByteTrack-style)
    match_iou_thresh: float = 0.35
    reid_weight: float = 0.5   # weight for appearance in [0,1]
    max_age: int = 24          # max frames to keep lost track
    min_hits: int = 3          # confirm after N hits
    hist_bins_h: int = 16
    hist_bins_s: int = 8
    hist_bins_v: int = 4
    # Advanced gating (keep simple gates)
    reid_min_sim: float = 0.3
    size_change_max_ratio: float = 1.8  # allow up to 1.8x size change between frames
    # Anti-switch guard
    id_lock_age: int = 2              # frames since last hit considered "locked"
    switch_min_sim: float = 0.45      # require at least this ReID sim to switch when locked
    # Appearance crop tuning
    reid_expand_ratio: float = 0.15   # expand bbox before embedding
    reid_focus_top: float = 0.78      # keep this fraction from top
    reid_min_crop_px: int = 24        # ensure torso crop keeps enough pixels
    # Multi-profile ReID adaptation
    reid_profile_new_thresh: float = 0.28   # spawn new profile below this similarity
    reid_profile_merge_thresh: float = 0.55 # merge into existing profile above this sim
    reid_profile_beta: float = 0.35         # EMA rate when updating profile vectors
    reid_profile_max: int = 4               # cap number of appearance profiles
    reid_profile_ttl: int = 120             # frames to keep a profile without updates
    # OCR removed
    # Kalman removed for players tracking


class Track:
    def __init__(
        self,
        tid: int,
        tlbr: Tuple[int, int, int, int],
        conf: float,
        emb: Optional[np.ndarray],
        frame_idx: int,
        cfg: Optional[TrackerConfig] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.id = tid
        self.tlbr = tlbr
        self.conf = conf
        self.emb = None
        self.embeds = deque([], maxlen=5)
        self.profiles: List[Dict[str, Any]] = []
        self.age = 0
        self.hits = 1
        self.last_frame = frame_idx
        self.active = True
        self._cfg = cfg or TrackerConfig()
        self.meta: Dict[str, Any] = {}
        self.court: Dict[str, Any] = {}
        self.court_side: Optional[str] = None
        # Size (EMA)
        x1, y1, x2, y2 = tlbr
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        self.size_w = w
        self.size_h = h
        # Jersey OCR removed
        # No Kalman state for players
        if meta is not None:
            self._update_meta(meta)
        if emb is not None:
            self._integrate_embedding(emb, frame_idx)

    def similarity(self, emb: Optional[np.ndarray]) -> Tuple[float, float]:
        """Return (best, blended) cosine similarity against stored appearance profiles."""
        if emb is None:
            return 0.0, 0.0
        vec = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(vec) + 1e-6
        vec = vec / n
        sims: List[float] = []
        for profile in getattr(self, "profiles", []):
            sims.append(float(cos_sim(profile.get("vec"), vec)))
        if getattr(self, "embeds", None):
            sims.extend(float(cos_sim(e, vec)) for e in self.embeds if e is not None)
        if getattr(self, "emb", None) is not None:
            sims.append(float(cos_sim(self.emb, vec)))
        sims = [s for s in sims if s == s]
        if not sims:
            return 0.0, 0.0
        best = max(sims)
        mean = sum(sims) / max(1, len(sims))
        blended = 0.6 * best + 0.4 * mean
        return best, blended

    def _update_meta(self, meta: Dict[str, Any]):
        try:
            self.meta.update(meta)
        except Exception:
            pass
        court = meta.get("court") if isinstance(meta.get("court"), dict) else None
        if court is None and all(k in meta for k in ("x", "y", "norm_x", "norm_y")):
            court = meta
        if court:
            self.court = {k: float(v) if isinstance(v, (int, float)) else v for k, v in court.items()}
            self.court_side = court.get("side", self.court_side)

    def update(self, tlbr: Tuple[int, int, int, int], conf: float, emb: Optional[np.ndarray], frame_idx: int, meta: Optional[Dict[str, Any]] = None):
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
            self._integrate_embedding(emb, frame_idx)
        # OCR removed
        self.age = 0
        self.hits += 1
        self.last_frame = frame_idx
        self.active = True
        if meta is not None:
            self._update_meta(meta)

    def _integrate_embedding(self, emb: np.ndarray, frame_idx: int):
        vec = np.asarray(emb, dtype=np.float32)
        n = np.linalg.norm(vec) + 1e-6
        vec = vec / n
        self.embeds.append(vec)
        beta = float(getattr(self._cfg, "reid_profile_beta", 0.35))
        self.emb = self._blend_vectors(self.emb, vec, beta)

        best_idx, best_sim = self._best_profile(vec)
        merge_thresh = float(getattr(self._cfg, "reid_profile_merge_thresh", 0.55))
        spawn_thresh = float(getattr(self._cfg, "reid_profile_new_thresh", 0.28))
        if best_idx is not None and best_sim >= merge_thresh:
            self._update_profile(best_idx, vec, frame_idx, beta)
        elif best_idx is not None and best_sim >= spawn_thresh:
            self._update_profile(best_idx, vec, frame_idx, beta * 0.5)
        else:
            self._add_profile(vec, frame_idx)

    def _best_profile(self, vec: np.ndarray) -> Tuple[Optional[int], float]:
        best_idx: Optional[int] = None
        best_sim = -1.0
        for idx, profile in enumerate(self.profiles):
            sim = float(cos_sim(profile.get("vec"), vec))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        return best_idx, best_sim

    def _update_profile(self, idx: int, vec: np.ndarray, frame_idx: int, alpha: float):
        alpha = float(max(0.0, min(1.0, alpha)))
        profile = self.profiles[idx]
        profile_vec = profile.get("vec")
        profile["vec"] = self._blend_vectors(profile_vec, vec, alpha)
        profile["hits"] = int(profile.get("hits", 0)) + 1
        profile["last_frame"] = frame_idx

    def _add_profile(self, vec: np.ndarray, frame_idx: int):
        profile = {"vec": vec, "hits": 1, "last_frame": frame_idx}
        self.profiles.append(profile)
        self._enforce_profile_cap()

    def _enforce_profile_cap(self):
        max_prof = max(1, int(getattr(self._cfg, "reid_profile_max", 3)))
        if len(self.profiles) <= max_prof:
            return
        self.profiles.sort(key=lambda p: (p.get("hits", 0), p.get("last_frame", -1)))
        while len(self.profiles) > max_prof:
            self.profiles.pop(0)

    def _blend_vectors(self, base: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
        alpha = float(max(0.0, min(1.0, alpha)))
        if base is None:
            return new
        vec = (1.0 - alpha) * base + alpha * new
        n = np.linalg.norm(vec) + 1e-6
        return vec / n

    def decay_profiles(self, frame_idx: int):
        ttl = int(getattr(self._cfg, "reid_profile_ttl", 0))
        if ttl <= 0 or not self.profiles:
            return
        keep: List[Dict[str, Any]] = []
        for p in self.profiles:
            last = int(p.get("last_frame", frame_idx))
            if (frame_idx - last) <= ttl:
                keep.append(p)
        if keep:
            self.profiles = keep
        else:
            recent = max(self.profiles, key=lambda p: p.get("last_frame", -1))
            self.profiles = [recent]


class ByteTrackReID:
    """
    Lightweight ByteTrack-style tracker with simple ReID (HSV color histogram).

    - One-stage greedy assignment using combined IoU and appearance similarity.
    - Keeps IDs stable across short occlusions (max_age frames).
    - Confirms tracks after `min_hits` to reduce flicker.
    """

    def __init__(self, cfg: TrackerConfig, embedder: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.cfg = cfg
        self.tracks: List[Track] = []
        self._next_id = 1
        self.embedder = embedder if embedder is not None else (lambda img: hist_embed(img, cfg.hist_bins_h, cfg.hist_bins_s, cfg.hist_bins_v))
        self.jersey_ocr = None  # OCR removed

    def _score(self, track: Track, det_tlbr: Tuple[int, int, int, int], det_emb: Optional[np.ndarray]) -> float:
        # use predicted state (track.tlbr is kept in sync with prediction)
        iou = iou_tlbr(track.tlbr, det_tlbr)
        _, blended_sim = track.similarity(det_emb)
        w = self.cfg.reid_weight
        sim_use = max(0.0, min(1.0, blended_sim))
        return (1.0 - w) * iou + w * sim_use

    # OCR bonus removed

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
        high_dets: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray], Optional[Dict[str, Any]]]] = []
        low_dets: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray], Optional[Dict[str, Any]]]] = []
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
            crop = extract_reid_crop(frame_bgr, tlbr, self.cfg)
            emb = self.embedder(crop)
            meta = None
            if isinstance(d, dict) and "_court" in d:
                meta = {"court": d.get("_court")}
            if conf >= self.cfg.track_thresh:
                high_dets.append((tlbr, conf, emb, meta))
            elif conf >= self.cfg.low_track_thresh:
                low_dets.append((tlbr, conf, emb, meta))

        # Age existing tracks
        for t in self.tracks:
            t.age += 1
            t.active = False
            t.decay_profiles(frame_idx)

        def _assign(tracks: List[Track], dets: List[Tuple[Tuple[int, int, int, int], float, Optional[np.ndarray]]]):
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
                    for j, (tlbr, conf, emb, meta) in enumerate(dets):
                        iou = iou_tlbr(t.tlbr, tlbr)
                        best_sim, _ = t.similarity(emb)
                        det_side = None
                        if meta and isinstance(meta.get("court"), dict):
                            det_side = meta["court"].get("side")
                        # Base gates: IoU or ReID
                        if not ((iou >= self.cfg.match_iou_thresh) or (best_sim >= self.cfg.reid_min_sim)):
                            continue
                        if not self._size_gate(t, tlbr):
                            continue
                        # No Kalman gating
                        # Anti-switch guard: for recently confirmed tracks, require stronger evidence to switch
                        if t.age <= self.cfg.id_lock_age:
                            if (best_sim < self.cfg.switch_min_sim) and (iou < max(0.5, self.cfg.match_iou_thresh)):
                                continue
                        if det_side and t.court_side and det_side != t.court_side:
                            if (best_sim < max(self.cfg.switch_min_sim, 0.55)) and (iou < max(0.4, self.cfg.match_iou_thresh)):
                                continue
                        s = self._score(t, tlbr, emb)
                        cost[i, j] = 1.0 - max(0.0, min(1.0, s))
                row_ind, col_ind = linear_sum_assignment(cost)
                matched_t = set()
                used_d = set()
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] <= 0.8:  # allow fairly loose; main gate is IoU
                        tlbr, conf, emb, meta = dets[c]
                        tracks[r].update(tlbr, conf, emb, frame_idx, meta=meta)
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
                        tlbr, conf, emb, meta = dets[j]
                        iou = iou_tlbr(t.tlbr, tlbr)
                        best_sim, _ = t.similarity(emb)
                        det_side = None
                        if meta and isinstance(meta.get("court"), dict):
                            det_side = meta["court"].get("side")
                        if not ((iou >= self.cfg.match_iou_thresh) or (best_sim >= self.cfg.reid_min_sim)):
                            continue
                        if not self._size_gate(t, tlbr):
                            continue
                        # No Kalman gating
                        if t.age <= self.cfg.id_lock_age:
                            if (best_sim < self.cfg.switch_min_sim) and (iou < max(0.5, self.cfg.match_iou_thresh)):
                                continue
                        if det_side and t.court_side and det_side != t.court_side:
                            if (best_sim < max(self.cfg.switch_min_sim, 0.55)) and (iou < max(0.4, self.cfg.match_iou_thresh)):
                                continue
                        s = self._score(t, tlbr, emb)
                        if s > best_score:
                            best_score = s
                            best_idx = j
                    if best_idx >= 0:
                        tlbr, conf, emb, meta = dets[best_idx]
                        t.update(tlbr, conf, emb, frame_idx, meta=meta)
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
                tlbr, conf, emb, meta = item[0], item[1], item[2], item[3]
                # Spawn only if not already close to an existing active track (reduce duplicates)
                ok = True
                for t in self.tracks:
                    if iou_tlbr(t.tlbr, tlbr) >= 0.7:
                        ok = False
                        break
                if ok:
                    t = Track(self._next_id, tlbr, conf, emb, frame_idx, cfg=self.cfg, meta=meta)
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
                # OCR removed
                # No Kalman debug fields
                if t.court:
                    court_meta = dict(t.court)
                    if t.court_side and "side" not in court_meta:
                        court_meta["side"] = t.court_side
                    row["court"] = court_meta
                out.append(row)
        return out


__all__ = ["TrackerConfig", "ByteTrackReID"]
