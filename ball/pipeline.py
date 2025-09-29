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
# Viterbi/DP global path (optional)
# ------------------------

def _node_cost(p: Dict[str, Any], settings, img_wh: Optional[Tuple[int, int]] = None) -> float:
    conf = max(1e-6, min(1.0, float(p.get("confidence", 0.0))))
    w = float(p.get("width", 0.0)); h = float(p.get("height", 0.0))
    ar_dev = abs((w / max(h, 1e-6)) - 1.0) if (w > 0 and h > 0) else 1.0
    cost = settings.overlay.ball.VIT_W_CONF * (-math.log(conf)) + settings.overlay.ball.VIT_W_AR * ar_dev
    # Circle quality (if available): penalize low q
    try:
        q = float(p.get("circle", {}).get("q", None))
        if q is not None:
            cost += settings.overlay.ball.VIT_W_CIRCLE * (1.0 - max(0.0, min(1.0, q)))
    except Exception:
        pass
    # Border proximity penalty (optional)
    if img_wh is not None and settings.overlay.ball.VIT_W_BORDER > 0.0:
        try:
            iw, ih = int(img_wh[0]), int(img_wh[1])
            # prefer circle center if present
            circ = p.get("circle", {}) if isinstance(p.get("circle"), dict) else {}
            cx = float(circ.get("u", p.get("x", 0.0)))
            cy = float(circ.get("v", p.get("y", 0.0)))
            d_edge = min(cx, iw - cx, cy, ih - cy)
            margin = max(1.0, float(settings.overlay.ball.IMAGE_BORDER_MARGIN_PX))
            if d_edge < margin:
                t = (margin - d_edge) / margin
                cost += settings.overlay.ball.VIT_W_BORDER * t
        except Exception:
            pass
    return cost


def _edge_cost(pa: Dict[str, Any], pb: Dict[str, Any], dt_frames: int, settings) -> float:
    dx = float(pb.get("x", 0.0)) - float(pa.get("x", 0.0))
    dy = float(pb.get("y", 0.0)) - float(pa.get("y", 0.0))
    dist = (dx * dx + dy * dy) ** 0.5
    sigma = max(1.0, settings.overlay.ball.CONT_MAX_JUMP_PX * max(1, dt_frames))
    c_dist = settings.overlay.ball.VIT_W_DIST * (dist / sigma) ** 2
    wa = max(1e-6, float(pa.get("width", 1.0))); ha = max(1e-6, float(pa.get("height", 1.0)))
    wb = max(1e-6, float(pb.get("width", 1.0))); hb = max(1e-6, float(pb.get("height", 1.0)))
    c_size = settings.overlay.ball.VIT_W_SIZE * (abs(math.log(wb/wa)) + abs(math.log(hb/ha)))
    return c_dist + c_size


def _dir_accel_cost(pp: Dict[str, Any], pa: Dict[str, Any], pb: Dict[str, Any], settings) -> float:
    """Second-order kinematic penalty using direction change and acceleration.
    pp -> pa -> pb, dt assumed = 1 frame.
    """
    if settings.overlay.ball.VIT_W_DIR == 0.0 and settings.overlay.ball.VIT_W_ACCEL == 0.0:
        return 0.0
    try:
        xpp, ypp = float(pp.get("x", 0.0)), float(pp.get("y", 0.0))
        xpa, ypa = float(pa.get("x", 0.0)), float(pa.get("y", 0.0))
        xpb, ypb = float(pb.get("x", 0.0)), float(pb.get("y", 0.0))
        v1x, v1y = (xpa - xpp), (ypa - ypp)
        v2x, v2y = (xpb - xpa), (ypb - ypa)
        # Direction change penalty (1 - cos theta)^2
        n1 = max(1e-6, (v1x*v1x + v1y*v1y) ** 0.5)
        n2 = max(1e-6, (v2x*v2x + v2y*v2y) ** 0.5)
        cos_th = (v1x*v2x + v1y*v2y) / (n1 * n2)
        cos_th = max(-1.0, min(1.0, cos_th))
        # Optional hard gate on direction change
        try:
            import math as _m
            ang_deg = _m.degrees(_m.acos(cos_th))
            if ang_deg > max(0.0, float(settings.overlay.ball.VIT_DIR_MAX_DEG)):
                return float('inf')
        except Exception:
            pass
        dir_pen = (1.0 - cos_th)
        dir_cost = settings.overlay.ball.VIT_W_DIR * (dir_pen * dir_pen)
        # Acceleration penalty: |v2 - v1| normalized by CONT_MAX_JUMP_PX
        ax = v2x - v1x; ay = v2y - v1y
        a = (ax*ax + ay*ay) ** 0.5
        sigma_v = max(1.0, float(settings.overlay.ball.CONT_MAX_JUMP_PX))
        acc_cost = settings.overlay.ball.VIT_W_ACCEL * (a / sigma_v) ** 2
        return dir_cost + acc_cost
    except Exception:
        return 0.0


def select_by_viterbi(
    preds_by_frame: Dict[int, List[Dict[str, Any]]],
    fps: float,
    settings,
    img_wh: Optional[Tuple[int, int]] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Viterbi with per-frame Null state to enforce a full-length path.
    - States per frame: K candidates + 1 Null.
    - Node cost: candidate=_node_cost, null=GapPenalty.
    - Edge cost: candidate→candidate uses _edge_cost; null→candidate adds StartPenalty;
                 any→null has no edge cost (pay null node cost already).
    """
    frames = sorted(preds_by_frame.keys())
    if not frames:
        return {}

    # Prepare per-frame top-K candidates
    tops: Dict[int, List[Dict[str, Any]]] = {}
    for f in frames:
        cands = sorted(preds_by_frame[f], key=lambda p: float(p.get("confidence", 0.0)), reverse=True)
        tops[f] = cands[: max(1, settings.overlay.ball.VIT_TOPK)]

    # DP: for each frame f, we store states 0..K (K=index for Null)
    dp_cost: Dict[int, List[float]] = {}
    dp_prev: Dict[int, List[Tuple[Optional[int], Optional[int]]]] = {}

    def state_count(f: int) -> int:
        return len(tops[f]) + 1  # +1 for Null

    # Initialize at first frame
    f0 = frames[0]
    K0 = state_count(f0)
    dp_cost[f0] = [float('inf')] * K0
    dp_prev[f0] = [(None, None)] * K0
    # candidate states
    for i, p in enumerate(tops[f0]):
        dp_cost[f0][i] = _node_cost(p, settings, img_wh) + settings.overlay.ball.VIT_START_PENALTY
    # null state
    dp_cost[f0][K0 - 1] = settings.overlay.ball.VIT_GAP_PENALTY  # null node cost

    # Transition frame by frame
    for t in range(1, len(frames)):
        f = frames[t]
        pf = frames[t - 1]
        K = state_count(f)
        Kp = state_count(pf)
        dp_cost[f] = [float('inf')] * K
        dp_prev[f] = [(None, None)] * K

        # For each current state
        for i in range(K):
            # Current state cost (node)
            if i < len(tops[f]):
                node_c = _node_cost(tops[f][i], settings, img_wh)
            else:
                node_c = settings.overlay.ball.VIT_GAP_PENALTY

            # Try all previous states
            for j in range(Kp):
                prev_c = dp_cost[pf][j]
                if prev_c == float('inf'):
                    continue

                # Edge cost
                edge_c = 0.0
                if j < len(tops[pf]) and i < len(tops[f]):
                    # candidate -> candidate, apply gate
                    pa = tops[pf][j]; pb = tops[f][i]
                    dx = float(pb.get("x", 0.0)) - float(pa.get("x", 0.0))
                    dy = float(pb.get("y", 0.0)) - float(pa.get("y", 0.0))
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist > settings.overlay.ball.CONT_MAX_JUMP_PX * 1.5:
                        continue
                    edge_c = _edge_cost(pa, pb, 1, settings)
                    # Add second-order kinematic term if previous of (pf,j) exists and is candidate
                    ppf, ppj = dp_prev[pf][j]
                    if ppf is not None and ppj is not None and ppj < len(tops[ppf]):
                        pp = tops[ppf][ppj]
                        dk = _dir_accel_cost(pp, pa, pb, settings)
                        if dk == float('inf'):
                            continue
                        edge_c += dk
                elif j == len(tops[pf]) and i < len(tops[f]):
                    # null -> candidate: start penalty
                    edge_c = settings.overlay.ball.VIT_START_PENALTY
                else:
                    # candidate/null -> null : no extra edge cost
                    edge_c = 0.0

                cand = prev_c + edge_c + node_c
                if cand < dp_cost[f][i]:
                    dp_cost[f][i] = cand
                    dp_prev[f][i] = (pf, j)

    # Backtrack best terminal state at last frame
    fend = frames[-1]
    best_i = min(range(state_count(fend)), key=lambda i: dp_cost[fend][i])
    path_states: Dict[int, int] = {}
    f = fend; i = best_i
    while f is not None and i is not None:
        path_states[f] = i
        pf, pj = dp_prev[f][i]
        f, i = pf, pj

    # Map back to candidate dictionary (skip null states)
    path: Dict[int, Dict[str, Any]] = {}
    for f, i in path_states.items():
        if i < len(tops[f]):
            path[f] = tops[f][i]
    return path
# ------------------------
# Post-checks (backward / confirm)
# ------------------------

def _process_segment_for_pruning(seg: List[int], best: Dict[int, Dict[str, Any]], min_len: int, min_move_px: float, keep: Set[int]):
    """Helper to process a single continuous segment and decide whether to prune it."""
    if not seg:
        return

    prune = False
    if len(seg) < max(1, min_len):
        prune = True
    elif len(seg) >= 2:
        dist = 0.0
        for u, v in zip(seg, seg[1:]):
            p0 = best[u]
            p1 = best[v]
            dx = float(p1.get("x", 0.0)) - float(p0.get("x", 0.0))
            dy = float(p1.get("y", 0.0)) - float(p0.get("y", 0.0))
            dist += (dx * dx + dy * dy) ** 0.5
        if dist < max(0.0, min_move_px):
            prune = True

    if prune:
        for f in seg:
            keep.discard(f)


def retro_prune_segments(best: Dict[int, Dict[str, Any]], min_len: int, min_move_px: float, adjacency_gap_max: int = 3) -> Dict[int, Dict[str, Any]]:
    if not best:
        return best
    frames = sorted(best.keys())
    keep = set(frames)

    if not frames:
        return {}

    seg = [frames[0]]
    for a, b in zip(frames, frames[1:]):
        if (b - a) <= max(1, adjacency_gap_max):
            seg.append(b)
        else:
            _process_segment_for_pruning(seg, best, min_len, min_move_px, keep)
            seg = [b]

    # Process the last segment
    _process_segment_for_pruning(seg, best, min_len, min_move_px, keep)

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
            # This reseed is confirmed, so we keep it. The 'replaced' list notes
            # which frame confirmed it, for debugging.
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
    img_wh: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Builds ball tracks with forward continuity selection, reseed confirmation,
    retro prune, and kinematic filtering integration points.

    Returns (best_tracks, debug_info).
    """
    # Load detections
    if getattr(settings.overlay.ball, 'USE_VITERBI_SELECTION', False):
        preds_by_frame = load_all_ball_preds_per_frame(jsonl_path, allowed_classes)
        best = select_by_viterbi(preds_by_frame, fps, settings, img_wh=img_wh)
        retro_pruned_frames = set()
        confirm_pruned_frames = set()
        confirm_replaced_targets = set()
    elif settings.overlay.ball.USE_CONTINUITY_SELECTION:
        preds_by_frame = load_all_ball_preds_per_frame(jsonl_path, allowed_classes)
        best = select_by_continuity(
            preds_by_frame,
            max_jump_px=max(5.0, settings.overlay.ball.CONT_MAX_JUMP_PX),
            topk=max(1, settings.overlay.ball.CONT_SEARCH_TOPK),
            reseed_misses=max(0, settings.overlay.ball.CONT_RESEED_MISSES),
        )
        # Confirm reseeds
        best, confirm_removed, confirm_replaced = confirm_reseeds(
            best,
            settings.overlay.ball.CONFIRM_RESEED_LOOKAHEAD,
            settings.overlay.ball.CONFIRM_RESEED_MIN_MOVE_PX,
            settings.overlay.ball.CONFIRM_MIN_CONF,
            settings.overlay.ball.CONFIRM_MAX_AR_DEV,
        )
        confirm_pruned_frames = set(confirm_removed)
        confirm_replaced_targets = set(g for f, g in confirm_replaced)
        # Retro prune
        before_keys = set(best.keys())
        best = retro_prune_segments(best, settings.overlay.ball.RETRO_MIN_SEG_LEN, settings.overlay.ball.RETRO_MIN_SEG_MOVE_PX)
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
