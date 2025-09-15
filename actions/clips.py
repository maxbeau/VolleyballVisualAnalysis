from typing import Dict, List, Any, Tuple, Optional


def _best_det_for_frame(preds: List[dict], min_conf: float) -> Tuple[dict, float]:
    best = None
    best_c = -1.0
    for p in preds:
        try:
            c = float(p.get("confidence", 0.0))
        except Exception:
            c = 0.0
        if c < min_conf:
            continue
        if c > best_c:
            best = p
            best_c = c
    return (best or {}), float(max(0.0, best_c))


def build_action_clips(
    actions_by_frame: Dict[int, List[dict]],
    classes: List[str],
    *,
    min_conf: float = 0.25,
    max_merge_gap: int = 8,
    pad_start: int = 4,
    pad_end: int = 4,
    min_len: int = 2,
    players_by_frame: Optional[Dict[int, List[dict]]] = None,
    min_player_conf: float = 0.1,
    max_match_dist_px: float = 120.0,
) -> List[Dict[str, Any]]:
    """
    Build action clips by merging sparse detections into continuous clips.
    - For each class, frames with a positive detection (>=min_conf) are merged
      when the gap to the previous is <= max_merge_gap.
    - Each clip is padded by pad_start/pad_end frames to improve continuity.
    - Clips shorter than min_len frames after padding are dropped.
    Returns list of clips with: id, class, start, end, duration, mean_conf, peak_conf.
    """
    clips: List[Dict[str, Any]] = []
    for cls in classes:
        # Collect frames with best det for this class
        frames = []
        confs = []
        for f in sorted(actions_by_frame.keys()):
            preds = [p for p in actions_by_frame[f] if str(p.get("class", "")).lower() == cls]
            if not preds:
                continue
            best, c = _best_det_for_frame(preds, min_conf)
            if c >= min_conf:
                frames.append(f)
                confs.append(c)
        if not frames:
            continue
        # Merge into clips
        st = frames[0]
        cs = [confs[0]]
        prev = frames[0]
        for f, c in zip(frames[1:], confs[1:]):
            if (f - prev) <= max_merge_gap:
                cs.append(c)
            else:
                # finalize prev clip
                start = max(0, st - pad_start)
                end = prev + pad_end
                dur = end - start + 1
                if dur >= min_len:
                    clips.append({
                        "class": cls,
                        "start": int(start),
                        "end": int(end),
                        "duration": int(dur),
                        "mean_conf": float(sum(cs) / max(1, len(cs))),
                        "peak_conf": float(max(cs) if cs else 0.0),
                    })
                # start new
                st = f
                cs = [c]
            prev = f
        # finalize last
        start = max(0, st - pad_start)
        end = prev + pad_end
        dur = end - start + 1
        if dur >= min_len:
            clips.append({
                "class": cls,
                "start": int(start),
                "end": int(end),
                "duration": int(dur),
                "mean_conf": float(sum(cs) / max(1, len(cs))),
                "peak_conf": float(max(cs) if cs else 0.0),
            })
    # Optional: enrich clips with representative player and median bottom-center
    if players_by_frame:
        for clip in clips:
            st = int(clip["start"])
            ed = int(clip["end"])
            btm_pts: List[Tuple[float, float]] = []
            actor_ids: List[Any] = []
            # Side votes from detection bottom centers per-frame
            side_votes = {"left": 0.0, "right": 0.0}
            for f in range(st, ed + 1):
                preds = actions_by_frame.get(f)
                if not preds:
                    continue
                # choose class-matched best pred at this frame
                cls = clip["class"]
                cand = [p for p in preds if str(p.get("class", "")).lower() == cls]
                if not cand:
                    continue
                best = max(cand, key=lambda p: float(p.get("confidence", 0.0)))
                ax = float(best.get("x", 0.0))
                ay = float(best.get("y", 0.0)) + 0.5 * float(best.get("height", 0.0))
                # record detection bottom center
                btm_pts.append((ax, ay))
                # Optional: nearest player guided by same-side heuristic if available (requires court to map side, which isn't here)
                trs = players_by_frame.get(f)
                if trs:
                    best_d2 = None
                    best_id = None
                    best_btm = None
                    for t in trs:
                        try:
                            pc = float(t.get("confidence", 0.0))
                        except Exception:
                            pc = 0.0
                        if pc < min_player_conf:
                            continue
                        cx = float(t.get("x", 0.0))
                        cy = float(t.get("y", 0.0))
                        h = float(t.get("height", 0.0))
                        bx = cx
                        by = cy + 0.5 * h
                        dx = bx - ax
                        dy = by - ay
                        d2 = dx * dx + dy * dy
                        if best_d2 is None or d2 < best_d2:
                            best_d2 = d2
                            best_id = t.get("id")
                            best_btm = (bx, by)
                    if best_btm is not None and (best_d2 is None or best_d2 <= max_match_dist_px * max_match_dist_px):
                        # Keep nearest player's bottom for median calc and actor id votes
                        btm_pts.append(best_btm)
                        if best_id is not None:
                            actor_ids.append(best_id)
            # finalize stats
            if btm_pts:
                xs = sorted([p[0] for p in btm_pts])
                ys = sorted([p[1] for p in btm_pts])
                mxi = len(xs) // 2
                myi = len(ys) // 2
                clip["med_btm_x"] = float(xs[mxi])
                clip["med_btm_y"] = float(ys[myi])
            if actor_ids:
                # modal id
                from collections import Counter
                c = Counter(actor_ids)
                actor_id, _ = c.most_common(1)[0]
                clip["actor_id"] = actor_id

    # Assign IDs per class
    idx: Dict[str, int] = {}
    for clip in clips:
        cls = clip["class"]
        n = idx.get(cls, 0) + 1
        idx[cls] = n
        clip["id"] = f"{cls}_{n:03d}"
    # Sort by start
    clips.sort(key=lambda d: (d["start"], d["class"]))
    return clips


__all__ = ["build_action_clips"]
