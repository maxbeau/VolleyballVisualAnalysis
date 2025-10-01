from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = (
    "load_best_ball_per_frame",
    "load_all_ball_candidates",
    "soft_weight_aspect_ratio",
    "select_ball_track_viterbi",
)


def load_best_ball_per_frame(
    jsonl_path: str, allowed_classes: List[str]
) -> Dict[int, Dict[str, Any]]:
    """Return the highest-confidence ball detection for each frame."""
    best: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            frame_idx = int(record.get("frame", -1))
            if frame_idx < 0:
                continue

            predictions = record.get("predictions", []) or []
            candidate = None
            for pred in predictions:
                if pred.get("class") not in allowed_classes:
                    continue
                if candidate is None or float(pred.get("confidence", 0.0)) > float(
                    candidate.get("confidence", 0.0)
                ):
                    candidate = pred

            if candidate is not None:
                best[frame_idx] = candidate
    return best


def load_all_ball_candidates(
    jsonl_path: str, allowed_classes: List[str]
) -> Dict[int, List[Dict[str, Any]]]:
    """Load every ball candidate detection, grouped by frame."""
    candidates_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            frame_idx = int(record.get("frame", -1))
            if frame_idx < 0:
                continue

            predictions = record.get("predictions", []) or []
            candidates = [pred for pred in predictions if pred.get("class") in allowed_classes]
            if candidates:
                candidates_by_frame[frame_idx] = candidates
    return candidates_by_frame


def soft_weight_aspect_ratio(
    best: Dict[int, Dict[str, Any]],
    min_ar: float,
    max_ar: float,
    alpha: float,
) -> Dict[int, Dict[str, Any]]:
    """Softly penalise detections whose aspect ratio deviates from expected bounds."""
    adjusted: Dict[int, Dict[str, Any]] = {}
    for frame_idx, pred in best.items():
        updated = pred.copy()
        try:
            width = float(updated.get("width", 0.0))
            height = float(updated.get("height", 0.0))
            aspect_ratio = (width / height) if height > 0 else 0.0
            confidence = float(updated.get("confidence", 0.0))
            weight = 1.0
            if aspect_ratio <= 0.0:
                weight = 0.5
            elif aspect_ratio < min_ar:
                delta = (min_ar - aspect_ratio) / max(min_ar, 1e-6)
                weight = float(np.exp(-alpha * delta))
            elif aspect_ratio > max_ar:
                delta = (max_ar - aspect_ratio) / max(max_ar, 1e-6)
                weight = float(np.exp(-alpha * delta))
            if weight < 1.0:
                updated["confidence"] = max(0.0, min(1.0, confidence * weight))
                updated["_ar_weight"] = round(weight, 3)
        except Exception:
            pass
        adjusted[frame_idx] = updated
    return adjusted


def _as_namespace(cfg: Any) -> SimpleNamespace:
    if isinstance(cfg, SimpleNamespace):
        return cfg
    if hasattr(cfg, "model_dump"):
        return SimpleNamespace(**cfg.model_dump())  # type: ignore[arg-type]
    if isinstance(cfg, dict):
        return SimpleNamespace(**cfg)
    attrs = {key: getattr(cfg, key) for key in dir(cfg) if not key.startswith("_")}
    return SimpleNamespace(**attrs)


def _node_cost_viterbi(
    prediction: Dict[str, Any],
    cfg: SimpleNamespace,
    img_wh: Optional[Tuple[int, int]],
) -> float:
    confidence = max(1e-6, min(1.0, float(prediction.get("confidence", 0.0))))
    width = float(prediction.get("width", 0.0))
    height = float(prediction.get("height", 0.0))
    aspect_ratio_deviation = abs((width / max(height, 1e-6)) - 1.0) if (width > 0 and height > 0) else 1.0
    cost = float(cfg.w_conf) * (-math.log(confidence)) + float(cfg.w_ar) * aspect_ratio_deviation

    circle = prediction.get("circle") if isinstance(prediction.get("circle"), dict) else None
    if circle is not None:
        try:
            quality = float(circle.get("q"))
            cost += float(cfg.w_circle) * (1.0 - max(0.0, min(1.0, quality)))
        except Exception:
            pass

    if img_wh is not None and float(cfg.w_border) > 0.0:
        try:
            width_img, height_img = int(img_wh[0]), int(img_wh[1])
            cx = float((circle or {}).get("u", prediction.get("x", 0.0)))
            cy = float((circle or {}).get("v", prediction.get("y", 0.0)))
            margin = max(1.0, float(getattr(cfg, "image_border_margin_px", 0.0)))
            if width_img > 0 and height_img > 0:
                distance_to_edge = min(cx, width_img - cx, cy, height_img - cy)
                if distance_to_edge < margin:
                    ratio = (margin - distance_to_edge) / margin
                    cost += float(cfg.w_border) * ratio
        except Exception:
            pass
    return cost


def _edge_cost_viterbi(
    prev_pred: Dict[str, Any],
    curr_pred: Dict[str, Any],
    cfg: SimpleNamespace,
    dt_frames: int,
) -> float:
    dx = float(curr_pred.get("x", 0.0)) - float(prev_pred.get("x", 0.0))
    dy = float(curr_pred.get("y", 0.0)) - float(prev_pred.get("y", 0.0))
    dist = math.hypot(dx, dy)
    sigma = max(1.0, float(getattr(cfg, "max_jump_px", 120.0)) * max(1, dt_frames))
    cost_distance = float(cfg.w_dist) * (dist / sigma) ** 2

    wa = max(1e-6, float(prev_pred.get("width", 1.0)))
    ha = max(1e-6, float(prev_pred.get("height", 1.0)))
    wb = max(1e-6, float(curr_pred.get("width", 1.0)))
    hb = max(1e-6, float(curr_pred.get("height", 1.0)))
    cost_size = float(cfg.w_size) * (abs(math.log(wb / wa)) + abs(math.log(hb / ha)))
    return cost_distance + cost_size


def _dir_accel_cost(
    pp: Dict[str, Any],
    pa: Dict[str, Any],
    pb: Dict[str, Any],
    cfg: SimpleNamespace,
) -> float:
    if float(cfg.w_dir) == 0.0 and float(cfg.w_accel) == 0.0:
        return 0.0
    try:
        xpp, ypp = float(pp.get("x", 0.0)), float(pp.get("y", 0.0))
        xpa, ypa = float(pa.get("x", 0.0)), float(pa.get("y", 0.0))
        xpb, ypb = float(pb.get("x", 0.0)), float(pb.get("y", 0.0))

        v1x, v1y = xpa - xpp, ypa - ypp
        v2x, v2y = xpb - xpa, ypb - ypa

        n1 = max(1e-6, math.hypot(v1x, v1y))
        n2 = max(1e-6, math.hypot(v2x, v2y))
        cos_theta = (v1x * v2x + v1y * v2y) / (n1 * n2)
        cos_theta = max(-1.0, min(1.0, cos_theta))

        dir_max_deg = float(getattr(cfg, "dir_max_deg", 180.0))
        if dir_max_deg < 180.0:
            angle_deg = math.degrees(math.acos(cos_theta))
            if angle_deg > max(0.0, dir_max_deg):
                return float("inf")

        dir_penalty = 1.0 - cos_theta
        dir_cost = float(cfg.w_dir) * (dir_penalty * dir_penalty)

        ax = v2x - v1x
        ay = v2y - v1y
        accel = math.hypot(ax, ay)
        sigma_v = max(1.0, float(getattr(cfg, "max_jump_px", 120.0)))
        accel_cost = float(cfg.w_accel) * (accel / sigma_v) ** 2
        return dir_cost + accel_cost
    except Exception:
        return 0.0


def select_ball_track_viterbi(
    preds_by_frame: Dict[int, List[Dict[str, Any]]],
    cfg: Any,
    img_wh: Optional[Tuple[int, int]] = None,
) -> Dict[int, Dict[str, Any]]:
    namespace_cfg = _as_namespace(cfg)
    frames = sorted(preds_by_frame.keys())
    if not frames:
        return {}

    topk = max(1, int(getattr(namespace_cfg, "topk", 5)))
    gap_penalty = float(getattr(namespace_cfg, "gap_penalty", 10.0))
    start_penalty = float(getattr(namespace_cfg, "start_penalty", 5.0))
    max_jump_px = float(getattr(namespace_cfg, "max_jump_px", 120.0))

    tops: Dict[int, List[Dict[str, Any]]] = {}
    for frame in frames:
        candidates = sorted(
            preds_by_frame[frame],
            key=lambda pred: float(pred.get("confidence", 0.0)),
            reverse=True,
        )
        tops[frame] = candidates[:topk]

    dp_cost: Dict[int, List[float]] = {}
    dp_prev: Dict[int, List[Tuple[Optional[int], Optional[int]]]] = {}

    def state_count(frame: int) -> int:
        return len(tops[frame]) + 1  # +1 for gap/null state

    first_frame = frames[0]
    k0 = state_count(first_frame)
    dp_cost[first_frame] = [float("inf")] * k0
    dp_prev[first_frame] = [(None, None)] * k0
    for idx, candidate in enumerate(tops[first_frame]):
        dp_cost[first_frame][idx] = _node_cost_viterbi(candidate, namespace_cfg, img_wh) + start_penalty
    dp_cost[first_frame][k0 - 1] = gap_penalty

    for frame_idx in range(1, len(frames)):
        frame = frames[frame_idx]
        prev_frame = frames[frame_idx - 1]
        k_curr = state_count(frame)
        k_prev = state_count(prev_frame)
        dp_cost[frame] = [float("inf")] * k_curr
        dp_prev[frame] = [(None, None)] * k_curr

        for curr_state in range(k_curr):
            node_cost = (
                _node_cost_viterbi(tops[frame][curr_state], namespace_cfg, img_wh)
                if curr_state < len(tops[frame])
                else gap_penalty
            )

            for prev_state in range(k_prev):
                prev_cost = dp_cost[prev_frame][prev_state]
                if prev_cost == float("inf"):
                    continue

                edge_cost = 0.0
                if prev_state < len(tops[prev_frame]) and curr_state < len(tops[frame]):
                    cand_prev = tops[prev_frame][prev_state]
                    cand_curr = tops[frame][curr_state]
                    dx = float(cand_curr.get("x", 0.0)) - float(cand_prev.get("x", 0.0))
                    dy = float(cand_curr.get("y", 0.0)) - float(cand_prev.get("y", 0.0))
                    dist = math.hypot(dx, dy)
                    if dist > max_jump_px * 1.5:
                        continue
                    edge_cost = _edge_cost_viterbi(cand_prev, cand_curr, namespace_cfg, dt_frames=max(1, frame - prev_frame))
                    prev_link = dp_prev[prev_frame][prev_state]
                    if prev_link[0] is not None and prev_link[1] is not None and prev_link[1] < len(tops[prev_link[0]]):
                        cand_pp = tops[prev_link[0]][prev_link[1]]
                        dir_cost = _dir_accel_cost(cand_pp, cand_prev, cand_curr, namespace_cfg)
                        if dir_cost == float("inf"):
                            continue
                        edge_cost += dir_cost
                elif prev_state == len(tops[prev_frame]) and curr_state < len(tops[frame]):
                    edge_cost = start_penalty
                # transitions into gap state just pay node (gap) cost

                candidate_cost = prev_cost + edge_cost + node_cost
                if candidate_cost < dp_cost[frame][curr_state]:
                    dp_cost[frame][curr_state] = candidate_cost
                    dp_prev[frame][curr_state] = (prev_frame, prev_state)

    last_frame = frames[-1]
    last_states = state_count(last_frame)
    best_state = min(range(last_states), key=lambda s: dp_cost[last_frame][s])

    path: Dict[int, int] = {}
    frame_cursor: Optional[int] = last_frame
    state_cursor: Optional[int] = best_state
    while frame_cursor is not None and state_cursor is not None:
        path[frame_cursor] = state_cursor
        frame_cursor, state_cursor = dp_prev[frame_cursor][state_cursor]

    track: Dict[int, Dict[str, Any]] = {}
    for frame, state in path.items():
        if state < len(tops[frame]):
            track[frame] = tops[frame][state]
    return track
