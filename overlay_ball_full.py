import os
import json
import bisect
import cv2
from typing import Dict, Any, List, Optional, Tuple
from utils import load_env_file, ensure_dir, pick_video_path
from smoothing import kalman_rts_smooth


def to_tlbr_from_xywh(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + h / 2))
    return x1, y1, x2, y2


def parse_bool(s: Optional[str], default: bool = False) -> bool:
    if s is None:
        return default
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def parse_color(s: str, default=(0, 255, 0)) -> Tuple[int, int, int]:
    try:
        parts = [int(p.strip()) for p in s.split(",")]
        if len(parts) == 3:
            return (parts[0], parts[1], parts[2])
    except Exception:
        pass
    return default


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
            # choose highest confidence of allowed classes
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


def build_interpolator(best: Dict[int, Dict[str, Any]], max_gap_frames: int) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    # Deprecated; kept for compatibility if other modules import it
    frames = sorted(best.keys())
    return frames, best


def pred_with_kalman_or_hold(
    frames: List[int],
    best: Dict[int, Dict[str, Any]],
    smoothed: Dict[int, Dict[str, Any]],
    i: int,
    hold_mode: str,
    hold_ttl: int,
) -> Optional[Dict[str, Any]]:
    # Prefer Kalman+RTS smoothed output if available
    if i in smoothed:
        return smoothed[i].copy()

    # Otherwise, apply hold fallback based on configuration using raw observations
    if not frames:
        return None
    pos = bisect.bisect_left(frames, i)
    prev_idx = frames[pos - 1] if pos > 0 else None
    next_idx = frames[pos] if pos < len(frames) else None

    hold_mode = (hold_mode or "prev").lower()
    if hold_mode not in ("prev", "next", "both", "none"):
        hold_mode = "prev"

    # Prefer prev hold
    if hold_mode in ("prev", "both") and prev_idx is not None:
        if (i - prev_idx) <= hold_ttl:
            p = best[prev_idx].copy()
            p["_hold"] = True
            return p
    # Optionally allow next hold (forward)
    if hold_mode in ("next", "both") and next_idx is not None:
        if (next_idx - i) <= hold_ttl:
            p = best[next_idx].copy()
            p["_hold"] = True
            return p
    return None


def main():
    env = load_env_file()
    jsonl_path = env.get("COMBINED_JSONL", "outputs/ball_detections.jsonl")
    out_path = env.get("BALL_OVERLAY_FULL", "outputs/ball_overlay_full.mp4")
    max_gap_frames = int(env.get("MAX_INTERP_GAP_FRAMES", 60))
    hold_mode = env.get("HOLD_MODE", "prev")
    hold_ttl = int(env.get("HOLD_TTL_FRAMES", 30))
    allowed = [c.strip() for c in env.get("BALL_CLASSES", "ball,volleyball").split(",")]
    min_conf = float(env.get("OVERLAY_MIN_CONF", 0.0))
    color = (0, 255, 0)
    thickness = 2
    show_labels = parse_bool(env.get("SHOW_BOX_LABELS", "true"), True)
    # Court overlay config
    court_overlay = parse_bool(env.get("COURT_OVERLAY", "false"), False)
    court_json = env.get("COURT_INTEGRATED_JSON", "outputs/court_corners_integrated.json")
    court_method = (env.get("COURT_OVERLAY_METHOD", "median") or "median").lower()
    court_color = parse_color(env.get("COURT_COLOR", "0,200,255"), (0, 200, 255))
    court_thickness = int(env.get("COURT_THICKNESS", 2))

    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}. Run detect_ball.py first.")

    video_path, _ = pick_video_path(env)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Ensure even dims
    out_w = width - (width % 2)
    out_h = height - (height % 2)
    if out_w != width or out_h != height:
        resize_needed = True
    else:
        resize_needed = False

    ensure_dir(os.path.dirname(out_path) or ".")
    codec = env.get("OVERLAY_CODEC", "avc1")
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        alt_path = os.path.splitext(out_path)[0] + ".avi"
        writer = cv2.VideoWriter(alt_path, fourcc, fps, (out_w, out_h))
        out_path = alt_path

    best = load_best_ball_per_frame(jsonl_path, allowed)
    frames_with_pred = sorted(best.keys())
    # Observation gating configuration
    gate_chisq = float(env.get("OBS_GATE_CHISQ_THRESH", 18.4))
    gate_use_conf = parse_bool(env.get("OBS_GATE_USE_CONF", "true"), True)
    smoothed = kalman_rts_smooth(best, max_gap_frames, gate_chisq, gate_use_conf)

    drawn = 0
    interp_count = 0
    hold_count = 0

    # Load court corners once (static court assumption)
    court_corners: Optional[List[Tuple[float, float]]] = None
    if court_overlay and os.path.exists(court_json):
        try:
            with open(court_json, "r", encoding="utf-8") as cf:
                cdata = json.load(cf)
            if court_method == "ema" and cdata.get("ema"):
                court_corners = [(float(x), float(y)) for x, y in cdata["ema"]]
            elif cdata.get("median"):
                court_corners = [(float(x), float(y)) for x, y in cdata["median"]]
        except Exception:
            court_corners = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred = pred_with_kalman_or_hold(frames_with_pred, best, smoothed, i, hold_mode, hold_ttl)
        if pred is not None:
            if float(pred.get("confidence", 0.0)) < min_conf:
                pred = None
        if pred is not None:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            x1, y1, x2, y2 = to_tlbr_from_xywh(x, y, w, h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            if show_labels:
                label = f"{pred.get('class','obj')} {pred.get('confidence', 0):.2f}"
                if pred.get("_interp"):
                    label += " (interp)"
                    interp_count += 1
                if pred.get("_hold"):
                    label += " (hold)"
                    hold_count += 1
                cv2.putText(frame, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            drawn += 1

        # Draw court if available
        if court_corners is not None and len(court_corners) == 4:
            pts = court_corners
            # Order should be TL, TR, BR, BL; connect in order and close
            pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for a, b in pairs:
                ax, ay = int(round(pts[a][0])), int(round(pts[a][1]))
                bx, by = int(round(pts[b][0])), int(round(pts[b][1]))
                cv2.line(frame, (ax, ay), (bx, by), court_color, court_thickness)

        if resize_needed:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        i += 1

    writer.release()
    cap.release()
    print(
        f"Full overlay saved. Frames: {i}/{total_frames}, boxes: {drawn}, interp: {interp_count}, hold: {hold_count}. Output: {out_path}"
    )


if __name__ == "__main__":
    main()
