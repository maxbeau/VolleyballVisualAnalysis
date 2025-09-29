import os
import json
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2

from core.utils import ensure_dir
from court.utils import order_corners, shape_metrics, within_tol
from pipeline.config import CourtConfig
from court.tracker import CourtLKTracker  # canonical implementation
from court.orientation import decide_orientation as decide_court_orientation
from court.io import load_detections


Point = Tuple[float, float]


def _load_detections(detections_jsonl: str) -> List[Dict[str, Any]]:
    return load_detections(detections_jsonl)


def run_tracking(
    *,
    video_path: str,
    detections_jsonl: str,
    tracking_jsonl: str,
    tracking_meta_json: str,
    cfg: CourtConfig,
) -> None:
    """Run tracking using the canonical CourtLKTracker and write JSONL results."""
    ensure_dir(os.path.dirname(tracking_jsonl) or ".")
    dets = _load_detections(detections_jsonl)
    if not dets:
        raise RuntimeError(f"No usable detections found in {detections_jsonl}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracker = CourtLKTracker(cfg=cfg)

    det_idx = 0
    next_key = dets[det_idx]
    next_key_frame = int(next_key["frame"]) if next_key else None
    prev_det_corners: Optional[List[Point]] = None

    # Collect a lightweight in-memory timeseries to compute orientation meta after tracking
    timeseries: Dict[int, List[Point]] = {}

    with open(tracking_jsonl, "w", encoding="utf-8") as out_f:
        frame_i = 0
        last_corners: Optional[List[Point]] = None
        while frame_i < total_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            # If this frame is a keyframe per detections, reset tracker
            if next_key is not None and frame_i == next_key_frame:
                accept = True
                det_corners = order_corners(next_key["corners"])  # TL,TR,BR,BL
                # If we already have a track, validate keyframe against last corners and shape refs
                if tracker.ema_corners is not None:
                    last_arr = np.array(tracker.ema_corners, dtype=np.float32)
                    det_arr = np.array(det_corners, dtype=np.float32)
                    d = np.linalg.norm(det_arr - last_arr, axis=1)
                    med_d = float(np.median(d))
                    # shape metrics
                    r_det, a_det = shape_metrics(det_arr)
                    ok_shape = True
                    if tracker.ref_ratio is not None and not within_tol(r_det, tracker.ref_ratio, max(0.35, tracker.cfg.gates.ratio_tolerance)):
                        ok_shape = False
                    if tracker.ref_area is not None and tracker.ref_area > 1e-6:
                        lo = (1.0 - max(0.6, tracker.cfg.gates.area_tolerance)) * tracker.ref_area
                        hi = (1.0 + max(0.6, tracker.cfg.gates.area_tolerance)) * tracker.ref_area
                        if not (lo <= a_det <= hi):
                            ok_shape = False
                    if med_d > max(12.0, tracker.cfg.gates.max_jump_px * 1.5) or not ok_shape:
                        accept = False
                # Additional fallback: compare to previous detection corners as well
                if accept and prev_det_corners is not None:
                    det_arr = np.array(det_corners, dtype=np.float32)
                    prev_arr = np.array(prev_det_corners, dtype=np.float32)
                    d2 = np.linalg.norm(det_arr - prev_arr, axis=1)
                    med_d2 = float(np.median(d2))
                    if med_d2 > 20.0:  # hard limit across detections
                        accept = False

                if accept:
                    tracker.set_keyframe(frame_i, frame, det_corners)
                    # Write smoothed corners at keyframe to avoid a visual jump
                    if tracker.ema_corners is not None:
                        sm = [(float(x), float(y)) for x, y in order_corners(tracker.ema_corners.tolist())]
                        info = {"keyframe": True, "tpl_prec": getattr(tracker, "last_tpl_prec", None)}
                        out_f.write(json.dumps({"frame": frame_i, "corners": sm, "info": info}, ensure_ascii=False) + "\n")
                        last_corners = sm
                        timeseries[frame_i] = sm
                    else:
                        info = {"keyframe": True, "tpl_prec": getattr(tracker, "last_tpl_prec", None)}
                        out_f.write(json.dumps({"frame": frame_i, "corners": det_corners, "info": info}, ensure_ascii=False) + "\n")
                        last_corners = det_corners
                        timeseries[frame_i] = det_corners
                else:
                    # Reject suspicious keyframe; attempt tracking update instead
                    corners, info = tracker.update(frame)
                    if corners is not None:
                        out_f.write(json.dumps({"frame": frame_i, "corners": corners}, ensure_ascii=False) + "\n")
                        last_corners = corners
                        timeseries[frame_i] = corners
                    elif last_corners is not None:
                        out_f.write(json.dumps({"frame": frame_i, "corners": last_corners}, ensure_ascii=False) + "\n")
                        timeseries[frame_i] = last_corners

                # advance to next detection
                det_idx += 1
                next_key = dets[det_idx] if det_idx < len(dets) else None
                next_key_frame = int(next_key["frame"]) if next_key is not None else None
                prev_det_corners = det_corners
            else:
                # Regular frame: predict via LK+RANSAC
                corners, info = tracker.update(frame)
                if corners is not None:
                    out_f.write(json.dumps({"frame": frame_i, "corners": corners, "info": info}, ensure_ascii=False) + "\n")
                    last_corners = corners
                    timeseries[frame_i] = corners
                else:
                    # If tracker is in hold window and we have last corners, repeat for continuity
                    if info.get("hold") and info.get("hold_left", 0) > 0 and last_corners is not None:
                        out_f.write(json.dumps({"frame": frame_i, "corners": last_corners, "info": info}, ensure_ascii=False) + "\n")
                        timeseries[frame_i] = last_corners

            frame_i += 1

    cap.release()

    # Export orientation meta (avoid recomputing in visualization)
    try:
        # Compact ts for orientation: frame -> {"corners": [(x,y)*4]}
        ts_for_orient: Dict[int, Dict[str, Any]] = {}
        SAMPLE_MAX = 600
        for fi in sorted(timeseries.keys()):
            if fi > SAMPLE_MAX:
                break
            cs = timeseries[fi]
            if cs and len(cs) >= 4:
                ts_for_orient[int(fi)] = {"corners": [(float(cs[0][0]), float(cs[0][1])),
                                                       (float(cs[1][0]), float(cs[1][1])),
                                                       (float(cs[2][0]), float(cs[2][1])),
                                                       (float(cs[3][0]), float(cs[3][1]))]}

        cap2 = cv2.VideoCapture(video_path)
        # standard model size; exact px/m not critical for template voting
        model_W, model_H = (1800, 900)
        orient = decide_court_orientation(cap2, ts_for_orient, (model_W, model_H), mode="template")
        cap2.release()
        meta = {
            "tracking_jsonl": tracking_jsonl,
            "orientation": orient,
        }
        ensure_dir(os.path.dirname(tracking_meta_json) or ".")
        with open(tracking_meta_json, "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
    except Exception:
        pass


__all__ = ["CourtLKTracker", "run_tracking"]
