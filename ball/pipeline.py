from typing import Dict, Any, List, Optional, Tuple, Set
import json
import math
import os
import cv2


# ------------------------
# Data loading helpers
# ------------------------

def parse_frame_spec(spec: str) -> List[int]:
    out: List[int] = []
    spec = (spec or "").strip()
    if not spec:
        return out
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                lo = int(a); hi = int(b)
            except Exception:
                continue
            if lo <= hi:
                out.extend(list(range(lo, hi + 1)))
            else:
                out.extend(list(range(hi, lo + 1)))
        else:
            try:
                out.append(int(p))
            except Exception:
                continue
    return out


def load_best_ball_per_frame(jsonl_path: str, allowed_classes: List[str]) -> Dict[int, Dict[str, Any]]:
    best: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frame_idx = int(rec.get("frame", -1))
            if frame_idx < 0:
                continue
            preds = rec.get("predictions", []) or []
            cand = None
            for p in preds:
                cls = p.get("class")
                if cls not in allowed_classes:
                    continue
                if cand is None or float(p.get("confidence", 0.0)) > float(cand.get("confidence", 0.0)):
                    cand = p
            if cand is not None:
                best[frame_idx] = cand
    return best


def load_all_ball_preds_per_frame(jsonl_path: str, allowed_classes: List[str]) -> Dict[int, List[Dict[str, Any]]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frame_idx = int(rec.get("frame", -1))
            if frame_idx < 0:
                continue
            preds = rec.get("predictions", []) or []
            cands: List[Dict[str, Any]] = []
            for p in preds:
                cls = p.get("class")
                if cls in allowed_classes:
                    cands.append(p)
            if cands:
                by_frame[frame_idx] = cands
    return by_frame


# ------------------------
# Track building (forward)
# ------------------------

def select_by_continuity(
    preds_by_frame: Dict[int, List[Dict[str, Any]]],
    max_jump_px: float,
    topk: int,
    reseed_misses: int = 3,
) -> Dict[int, Dict[str, Any]]:
    frames = sorted(preds_by_frame.keys())
    if not frames:
        return {}
    best: Dict[int, Dict[str, Any]] = {}
    # Seed with highest-confidence candidate in first frame (mark reseed)
    f0 = frames[0]
    cands0 = sorted(preds_by_frame[f0], key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
    seed0 = cands0[0].copy()
    seed0.setdefault("_reseed", True)
    best[f0] = seed0

    last_x = float(best[f0].get("x", 0.0))
    last_y = float(best[f0].get("y", 0.0))
    misses = 0
    for f in frames[1:]:
        cands = preds_by_frame[f]
        cands_sorted = sorted(cands, key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
        subset = cands_sorted[: max(1, topk)]
        chosen = None
        best_d = 1e9
        for p in subset:
            dx = float(p.get("x", 0.0)) - last_x
            dy = float(p.get("y", 0.0)) - last_y
            d = (dx * dx + dy * dy) ** 0.5
            if d < best_d:
                best_d = d
                chosen = p
        if chosen is not None and best_d <= max_jump_px:
            best[f] = chosen
            last_x = float(chosen.get("x", 0.0))
            last_y = float(chosen.get("y", 0.0))
            misses = 0
        else:
            misses += 1
            if reseed_misses > 0 and misses >= reseed_misses:
                seed = cands_sorted[0].copy()
                seed.setdefault("_reseed", True)
                best[f] = seed
                last_x = float(seed.get("x", 0.0))
                last_y = float(seed.get("y", 0.0))
                misses = 0
    return best


# ------------------------
# Post-checks (backward / confirm)
# ------------------------

def retro_prune_segments(best: Dict[int, Dict[str, Any]], min_len: int, min_move_px: float, adjacency_gap_max: int = 3) -> Dict[int, Dict[str, Any]]:
    if not best:
        return best
    frames = sorted(best.keys())
    keep = set(frames)
    seg = [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if (b - a) <= max(1, adjacency_gap_max):
            seg.append(b)
        else:
            if len(seg) > 0:
                if len(seg) < max(1, min_len):
                    for f in seg:
                        keep.discard(f)
                else:
                    if len(seg) >= 2:
                        dist = 0.0
                        for u, v in zip(seg, seg[1:]):
                            p0 = best[u]; p1 = best[v]
                            dx = float(p1.get("x", 0.0)) - float(p0.get("x", 0.0))
                            dy = float(p1.get("y", 0.0)) - float(p0.get("y", 0.0))
                            dist += (dx*dx + dy*dy) ** 0.5
                        if dist < max(0.0, min_move_px):
                            for f in seg:
                                keep.discard(f)
            seg = [b]
    if len(seg) > 0:
        if len(seg) < max(1, min_len):
            for f in seg:
                keep.discard(f)
        else:
            if len(seg) >= 2:
                dist = 0.0
                for u, v in zip(seg, seg[1:]):
                    p0 = best[u]; p1 = best[v]
                    dx = float(p1.get("x", 0.0)) - float(p0.get("x", 0.0))
                    dy = float(p1.get("y", 0.0)) - float(p0.get("y", 0.0))
                    dist += (dx*dx + dy*dy) ** 0.5
                if dist < max(0.0, min_move_px):
                    for f in seg:
                        keep.discard(f)
    return {k: v for k, v in best.items() if k in keep}


def _ar_dev(p: Dict[str, Any]) -> float:
    try:
        w = float(p.get("width", 0.0)); h = float(p.get("height", 0.0))
        return abs((w / max(h, 1e-6)) - 1.0)
    except Exception:
        return 999.0


def confirm_reseeds(best: Dict[int, Dict[str, Any]], lookahead: int, min_move_px: float, min_conf: float, max_ar_dev: float) -> Tuple[Dict[int, Dict[str, Any]], List[int], List[Tuple[int,int]]]:
    if not best or lookahead <= 0:
        return best, [], []
    frames = sorted(best.keys())
    keep = set(frames)
    removed: List[int] = []
    replaced: List[Tuple[int,int]] = []
    for f in frames:
        if f not in best:
            continue
        p = best[f]
        if not isinstance(p, dict) or not p.get("_reseed"):
            continue
        chosen_g = None
        for g in range(f + 1, f + lookahead + 1):
            if g in best:
                q = best[g]
                dx = float(q.get("x", 0.0)) - float(p.get("x", 0.0))
                dy = float(q.get("y", 0.0)) - float(p.get("y", 0.0))
                d = (dx * dx + dy * dy) ** 0.5
                if d >= max(0.0, min_move_px):
                    conf_g = float(q.get("confidence", 0.0))
                    if (conf_g >= min_conf) or (_ar_dev(q) <= max_ar_dev):
                        chosen_g = g
                        break
        if chosen_g is None:
            keep.discard(f)
            removed.append(f)
        else:
            keep.discard(f)
            removed.append(f)
            replaced.append((f, chosen_g))
    pruned = {k: v for k, v in best.items() if k in keep}
    return pruned, removed, replaced


# ------------------------
# Orchestrator
# ------------------------

def build_ball_tracks(
    jsonl_path: str,
    allowed_classes: List[str],
    fps: float,
    settings,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Builds ball tracks with forward continuity selection, reseed confirmation,
    retro prune, and kinematic filtering integration points.

    Returns (best_tracks, debug_info).
    """
    # Load detections
    if settings.USE_CONTINUITY_SELECTION:
        preds_by_frame = load_all_ball_preds_per_frame(jsonl_path, allowed_classes)
        best = select_by_continuity(
            preds_by_frame,
            max_jump_px=max(5.0, settings.CONT_MAX_JUMP_PX),
            topk=max(1, settings.CONT_SEARCH_TOPK),
            reseed_misses=max(0, settings.CONT_RESEED_MISSES),
        )
        # Confirm reseeds
        best, confirm_removed, confirm_replaced = confirm_reseeds(
            best,
            settings.CONFIRM_RESEED_LOOKAHEAD,
            settings.CONFIRM_RESEED_MIN_MOVE_PX,
            settings.CONFIRM_MIN_CONF,
            settings.CONFIRM_MAX_AR_DEV,
        )
        confirm_pruned_frames = set(confirm_removed)
        confirm_replaced_targets = set(g for f, g in confirm_replaced)
        # Retro prune
        before_keys = set(best.keys())
        best = retro_prune_segments(best, settings.RETRO_MIN_SEG_LEN, settings.RETRO_MIN_SEG_MOVE_PX)
        after_keys = set(best.keys())
        retro_pruned_frames = before_keys - after_keys
    else:
        best = load_best_ball_per_frame(jsonl_path, allowed_classes)
        retro_pruned_frames = set()
        confirm_pruned_frames = set()
        confirm_replaced_targets = set()

    debug = {
        "frames_raw": set(best.keys()),  # will be overridden by caller after AR-soft etc.
        "retro_pruned": retro_pruned_frames,
        "confirm_pruned": confirm_pruned_frames,
        "confirm_replaced_targets": confirm_replaced_targets,
    }
    return best, debug


__all__ = [
    "parse_frame_spec",
    "load_best_ball_per_frame",
    "load_all_ball_preds_per_frame",
    "select_by_continuity",
    "retro_prune_segments",
    "confirm_reseeds",
    "build_ball_tracks",
]

