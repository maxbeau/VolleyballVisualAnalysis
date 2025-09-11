import os
import json
import bisect
import cv2
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from court.utils import standard_court_model_size, build_court_model_template, template_precision_score
from core.config import settings
from core.utils import ensure_dir
from analysis.smoothing import kalman_rts_smooth


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
    jsonl_path = settings.BALL_DETECTIONS_JSONL
    out_path = settings.BALL_OVERLAY_FULL
    max_gap_frames = settings.MAX_INTERP_GAP_FRAMES
    hold_mode = "prev" if settings.HOLD_MODE else "none"
    hold_ttl = settings.HOLD_TTL_FRAMES
    allowed = ["ball", "volleyball"]
    min_conf = settings.OVERLAY_MIN_CONF
    color = (0, 255, 0)
    thickness = 2
    show_labels = settings.SHOW_BOX_LABELS
    # Court overlay config
    court_overlay = settings.COURT_OVERLAY
    court_json = "outputs/court_corners_integrated.json"
    court_tracking_jsonl = settings.COURT_TRACKING_JSONL
    court_method = settings.COURT_OVERLAY_METHOD
    court_color = settings.COURT_COLOR
    court_thickness = settings.COURT_THICKNESS
    court_center_color = settings.COURT_CENTER_COLOR
    court_attack_color = settings.COURT_ATTACK_COLOR

    ball_available = os.path.exists(jsonl_path)
    if not ball_available:
        print(f"Ball JSONL not found: {jsonl_path}. Proceeding without ball boxes.")

    video_path = settings.VIDEO_PATH
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
    codec = settings.OVERLAY_CODEC
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        alt_path = os.path.splitext(out_path)[0] + ".avi"
        writer = cv2.VideoWriter(alt_path, fourcc, fps, (out_w, out_h))
        out_path = alt_path

    if ball_available:
        best = load_best_ball_per_frame(jsonl_path, allowed)
    else:
        best = {}

    # Aspect-ratio soft weighting (no size constraints)
    f_min_ar = settings.FILTER_MIN_ASPECT_RATIO
    f_max_ar = settings.FILTER_MAX_ASPECT_RATIO
    ar_alpha = settings.FILTER_AR_SOFT_ALPHA
    adjusted = {}
    softened = 0
    for k, p in best.items():
        q = p.copy()
        try:
            w = float(q.get("width", 0.0))
            h = float(q.get("height", 0.0))
            ar = (w / h) if h > 0 else 0.0
            conf = float(q.get("confidence", 0.0))
            weight = 1.0
            if ar <= 0.0:
                weight = 0.5  # uncertain, down-weight a bit
            elif ar < f_min_ar:
                t = (f_min_ar - ar) / max(f_min_ar, 1e-6)
                weight = float(math.exp(-ar_alpha * t))
            elif ar > f_max_ar:
                t = (ar - f_max_ar) / max(f_max_ar, 1e-6)
                weight = float(math.exp(-ar_alpha * t))
            # Apply weight softly on confidence
            if weight < 1.0:
                q["confidence"] = max(0.0, min(1.0, conf * weight))
                q["_ar_weight"] = round(weight, 3)
                softened += 1
        except Exception:
            pass
        adjusted[k] = q
    if softened > 0:
        print(f"AR soft-weight: adjusted {softened} frames by aspect-ratio; total {len(adjusted)}")
    best = adjusted
    frames_with_pred = sorted(best.keys())
    # Observation gating configuration
    gate_chisq = settings.OBS_GATE_CHISQ_THRESH
    gate_use_conf = settings.OBS_GATE_USE_CONF
    # Gravity configuration (pixels per second squared)
    gravity_pps2 = settings.GRAVITY_PPS2
    # Convert to per-frame acceleration: g / fps^2
    gravity_per_frame = (gravity_pps2 / (fps * fps)) if fps and fps > 0 else 0.0
    if ball_available and best:
        smoothed = kalman_rts_smooth(
            best,
            max_gap_frames,
            gate_chisq,
            gate_use_conf,
            gravity_per_frame=gravity_per_frame,
        )
    else:
        smoothed = {}

    drawn = 0
    interp_count = 0
    hold_count = 0

    # Load court overlay source
    court_corners: Optional[List[Tuple[float, float]]] = None
    court_timeseries: Optional[Dict[int, List[Tuple[float, float]]]] = None
    court_infos: Optional[Dict[int, Dict[str, Any]]] = None
    if court_overlay:
        if court_method == "timeseries" and os.path.exists(court_tracking_jsonl):
            court_timeseries = {}
            court_infos = {}
            with open(court_tracking_jsonl, "r", encoding="utf-8") as cf:
                for line in cf:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    fi = int(rec.get("frame", -1))
                    cs = rec.get("corners")
                    if fi >= 0 and cs and isinstance(cs, list) and len(cs) >= 4:
                        pts = [(float(cs[0][0]), float(cs[0][1])), (float(cs[1][0]), float(cs[1][1])), (float(cs[2][0]), float(cs[2][1])), (float(cs[3][0]), float(cs[3][1]))]
                        court_timeseries[fi] = pts
                        if isinstance(rec.get("info"), dict):
                            court_infos[fi] = rec["info"]
        elif os.path.exists(court_json):
            try:
                with open(court_json, "r", encoding="utf-8") as cf:
                    cdata = json.load(cf)
                if court_method == "ema" and cdata.get("ema"):
                    court_corners = [(float(x), float(y)) for x, y in cdata["ema"]]
                elif cdata.get("median"):
                    court_corners = [(float(x), float(y)) for x, y in cdata["median"]]
            except Exception:
                court_corners = None
    # Precompute canonical court model size and line template
    Wm, Hm = standard_court_model_size(scale_px_per_meter=100.0)
    model_tpl = build_court_model_template(Wm, Hm, max(1, court_thickness), orientation="vertical")

    # Orientation temporal smoothing + hysteresis
    last_rot = 0  # 0,1,2,3 => 0,90,180,270 deg
    score_ema = np.zeros(4, dtype=np.float32)
    ema_inited = False
    ema_alpha = 0.3  # 0<alpha<=1; higher reacts faster
    improve_eps = 0.05  # require >5% improvement on smoothed score to switch
    switch_patience = 3  # need N consecutive wins to flip
    consecutive_wins = 0
    lock_frames = 10  # after switch, lock orientation for N frames
    lock_left = 0

    i = 0
    last_court_pts: Optional[List[Tuple[float, float]]] = None
    last_court_info: Optional[Dict[str, Any]] = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pred = pred_with_kalman_or_hold(frames_with_pred, best, smoothed, i, hold_mode, hold_ttl) if ball_available else None
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

        # Draw court
        pts: Optional[List[Tuple[float, float]]] = None
        if court_timeseries is not None:
            if i in court_timeseries:
                pts = court_timeseries.get(i)
                last_court_pts = pts
                if court_infos is not None:
                    last_court_info = court_infos.get(i, last_court_info)
            else:
                # Hold previous court corners if current frame missing
                if last_court_pts is not None:
                    pts = last_court_pts
        elif court_corners is not None and len(court_corners) == 4:
            pts = court_corners
        if pts is not None and len(pts) == 4:
            # Orientation-aware model->image homography selection by template precision
            dst = np.array([[pts[0][0], pts[0][1]], [pts[1][0], pts[1][1]], [pts[2][0], pts[2][1]], [pts[3][0], pts[3][1]]], dtype=np.float32)
            src0 = np.array([[0.0, 0.0], [Wm - 1.0, 0.0], [Wm - 1.0, Hm - 1.0], [0.0, Hm - 1.0]], dtype=np.float32)
            # 90/180/270 deg rotations of the model corners
            src90 = np.array([[0.0, Hm - 1.0], [0.0, 0.0], [Wm - 1.0, 0.0], [Wm - 1.0, Hm - 1.0]], dtype=np.float32)
            src180 = np.array([[Wm - 1.0, Hm - 1.0], [0.0, Hm - 1.0], [0.0, 0.0], [Wm - 1.0, 0.0]], dtype=np.float32)
            src270 = np.array([[Wm - 1.0, 0.0], [Wm - 1.0, Hm - 1.0], [0.0, Hm - 1.0], [0.0, 0.0]], dtype=np.float32)

            gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gr, 50, 150)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

            def score_for(src: np.ndarray) -> Tuple[float, np.ndarray]:
                Hmi = cv2.getPerspectiveTransform(src, dst)
                prec = template_precision_score(gr, Hmi, model_tpl)
                return prec, Hmi

            candidates = [src0, src90, src180, src270]
            cand_scores: List[Tuple[float, int, np.ndarray]] = []
            for k, s in enumerate(candidates):
                sc, Hmi = score_for(s)
                cand_scores.append((sc, k, Hmi))
            # Update EMA of scores per orientation
            raw_scores = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            Hcands = {}
            for sc, k, Hmi in cand_scores:
                raw_scores[k] = float(sc)
                Hcands[k] = Hmi
            if not ema_inited:
                # First frame: directly choose the best orientation (no hysteresis)
                score_ema = raw_scores.copy()
                ema_inited = True
                last_rot = int(np.argmax(score_ema))
                rot = last_rot
                H = Hcands[rot]
                consecutive_wins = 0
                lock_left = 0
            else:
                # Smooth scores over time
                score_ema = (1.0 - ema_alpha) * score_ema + ema_alpha * raw_scores
                best_rot = int(np.argmax(score_ema))

                # Orientation locking to prevent rapid flips
                if lock_left > 0:
                    rot = last_rot
                    H = Hcands.get(rot, Hcands.get(best_rot))
                    lock_left -= 1
                else:
                    # Check if best rot truly better than current with margin
                    curr = float(score_ema[last_rot])
                    cand = float(score_ema[best_rot])
                    if best_rot != last_rot and cand > curr * (1.0 + improve_eps):
                        consecutive_wins += 1
                        if consecutive_wins >= switch_patience:
                            rot = best_rot
                            H = Hcands[rot]
                            last_rot = rot
                            consecutive_wins = 0
                            lock_left = lock_frames
                        else:
                            rot = last_rot
                            H = Hcands[rot]
                    else:
                        consecutive_wins = 0
                        rot = last_rot
                        H = Hcands[rot]

            # Draw outer border
            pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
            for a, b in pairs:
                ax, ay = int(round(pts[a][0])), int(round(pts[a][1]))
                bx, by = int(round(pts[b][0])), int(round(pts[b][1]))
                cv2.line(frame, (ax, ay), (bx, by), court_color, court_thickness)

            # Render center/attack lines using chosen H (model->image)
            def proj_vline(x_model: float) -> Tuple[Tuple[int, int], Tuple[int, int]]:
                P = np.array([[x_model, 0.0], [x_model, Hm - 1.0]], dtype=np.float32).reshape(-1, 1, 2)
                Q = cv2.perspectiveTransform(P, H).reshape(-1, 2)
                return (int(round(Q[0, 0])), int(round(Q[0, 1]))), (int(round(Q[1, 0])), int(round(Q[1, 1])))

            x_center = (Wm - 1.0) * 0.5
            c0, c1 = proj_vline(x_center)
            cv2.line(frame, c0, c1, court_center_color, max(1, court_thickness))
            x_a1 = (Wm - 1.0) * (6.0 / 18.0)
            x_a2 = (Wm - 1.0) * (12.0 / 18.0)
            a10, a11 = proj_vline(x_a1)
            a20, a21 = proj_vline(x_a2)
            cv2.line(frame, a10, a11, court_attack_color, max(1, court_thickness))
            cv2.line(frame, a20, a21, court_attack_color, max(1, court_thickness))

            # Optional diagnostics
            if settings.COURT_SHOW_DIAG and last_court_info is not None:
                txt = []
                def add(k):
                    v = last_court_info.get(k)
                    if v is not None:
                        try:
                            txt.append(f"{k}:{float(v):.3f}")
                        except Exception:
                            txt.append(f"{k}:{v}")
                for k in ("inlier_ratio","med_reproj_err","condH","scale_x","scale_y","tpl_prec","roi_scale","matches","inliers"):
                    add(k)
                if txt:
                    s = "  ".join(txt)
                    cv2.putText(frame, s, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)

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
