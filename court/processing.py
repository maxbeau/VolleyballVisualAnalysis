import os
import json
import logging
from collections import deque
from typing import List, Tuple, Optional, Dict, Any, Deque

import numpy as np
import cv2

from core.utils import ensure_dir
from court.utils import order_corners, shape_metrics, within_tol, corners_from_prediction
from court.io import _pick_best_court_pred
from orchestration.config import CourtConfig, DetectionConfig
from court.tracker import CourtLKTracker  # canonical implementation
from detection.factory import create_detection_backend
from court.orientation import decide_orientation as decide_court_orientation
from court.io import load_detections


Point = Tuple[float, float]


def _load_detections(detections_jsonl: str) -> List[Dict[str, Any]]:
    return load_detections(detections_jsonl)


def _prune_detection_buffer(
    buffer: Deque[Dict[str, Any]], *, current_frame: int, max_age: int
) -> None:
    """Drop buffered detections that are too far in the past."""
    while buffer and current_frame - buffer[0]["frame"] > max_age:
        buffer.popleft()


def _consensus_from_buffer(
    buffer: Deque[Dict[str, Any]], *,
    min_support: int,
    max_support: int,
    gate_px: float,
    current_frame: int,
    time_constant: float,
) -> Optional[Tuple[List[Point], Dict[str, Any]]]:
    """Compute a median-based consensus corners estimate from recent detections."""
    if len(buffer) < min_support:
        return None

    window = list(buffer)[-max_support:]
    frames = np.array([int(det["frame"]) for det in window], dtype=np.int32)
    corners_stack = np.array([np.array(det["corners"], dtype=np.float32) for det in window])
    median = np.median(corners_stack, axis=0)
    deviations = np.linalg.norm(corners_stack - median[None, ...], axis=2)
    support_mask = np.median(deviations, axis=1) <= gate_px
    support_idx = np.where(support_mask)[0]
    if support_idx.size < min_support:
        return None

    support_stack = corners_stack[support_idx]
    ages = np.maximum(0.0, float(current_frame)) - frames[support_idx].astype(np.float32)
    tc = max(float(time_constant), 1.0)
    weights = np.exp(-ages / tc)
    weights = np.clip(weights, 1e-3, None)
    weight_tensor = weights[:, None, None]
    weighted_sum = np.sum(support_stack * weight_tensor, axis=0)
    total_weight = float(np.sum(weight_tensor))
    if total_weight > 1e-6:
        consensus = weighted_sum / total_weight
    else:
        consensus = np.median(support_stack, axis=0)
    consensus_pts = [(float(x), float(y)) for x, y in order_corners(consensus.tolist())]
    support_frames = [window[int(idx)]["frame"] for idx in support_idx]
    support_error = float(
        np.median(
            np.median(
                np.linalg.norm(support_stack - consensus[None, ...], axis=2),
                axis=1,
            )
        )
    )
    meta = {
        "support_frames": support_frames,
        "support_error": support_error,
        "support_weights": [float(w) for w in weights.tolist()],
    }
    return consensus_pts, meta


def _consensus_passes_gates(
    consensus: List[Point], *, tracker: CourtLKTracker, cfg: CourtConfig, last_corners: Optional[List[Point]]
) -> bool:
    """Apply tracker-shaped gates to a consensus before resetting the track."""
    if not consensus:
        return False

    consensus_arr = np.array(consensus, dtype=np.float32)
    jump_limit = max(cfg.gates.max_jump_px * 1.5, 12.0)

    if tracker.ema_corners is not None:
        ema_arr = np.array(order_corners(tracker.ema_corners.tolist()), dtype=np.float32)
        med_jump = float(np.median(np.linalg.norm(consensus_arr - ema_arr, axis=1)))
        if med_jump > jump_limit:
            return False

    ratio_cons, area_cons = shape_metrics(consensus_arr)

    if tracker.ref_ratio is not None and not within_tol(
        ratio_cons, tracker.ref_ratio, max(0.35, cfg.gates.ratio_tolerance * 1.5)
    ):
        return False

    if tracker.ref_area is not None and tracker.ref_area > 1e-6:
        area_tol = max(0.6, cfg.gates.area_tolerance * 1.5)
        lo = (1.0 - area_tol) * tracker.ref_area
        hi = (1.0 + area_tol) * tracker.ref_area
        if not (lo <= area_cons <= hi):
            return False

    if last_corners is not None:
        last_arr = np.array(last_corners, dtype=np.float32)
        med_jump_last = float(np.median(np.linalg.norm(consensus_arr - last_arr, axis=1)))
        if med_jump_last > max(cfg.gates.max_jump_px * 2.0, 18.0):
            return False

    return True


def run_tracking(
    *,
    video_path: str,
    detections_jsonl: str,
    tracking_jsonl: str,
    tracking_meta_json: str,
    cfg: CourtConfig,
    detection_cfg: Optional[DetectionConfig] = None,
    min_confidence: float = 0.0,
) -> None:
    """Run tracking using the canonical CourtLKTracker and write JSONL results."""
    ensure_dir(os.path.dirname(tracking_jsonl) or ".")
    dets = _load_detections(detections_jsonl)
    if not dets:
        raise RuntimeError(f"No usable detections found in {detections_jsonl}")
    dets.sort(key=lambda d: int(d["frame"]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    detection_backend = None
    detection_model_id: Optional[str] = None
    if detection_cfg is not None:
        try:
            backend_settings = detection_cfg.model_dump(exclude={"roboflow_api_key"})
            detection_backend = create_detection_backend(backend_settings)
            backend_key = detection_cfg.backend.strip().lower()
            if backend_key == "roboflow":
                detection_model_id = detection_cfg.models_roboflow.get("court")
            elif backend_key == "local-yolo":
                model_path = getattr(detection_cfg.models_yolo, "court", None)
                detection_model_id = str(model_path) if model_path else "court"
            else:
                detection_model_id = detection_cfg.models_roboflow.get("court") or "court"
        except Exception as exc:
            logging.warning("Court on-demand detection disabled: %s", exc)
            detection_backend = None
            detection_model_id = None

    bootstrap_cfg = cfg.bootstrap
    bootstrap_window_frames = int(round(max(0.0, float(bootstrap_cfg.window_sec)) * fps))
    max_span_frames = int(round(max(float(bootstrap_cfg.max_span_sec), float(bootstrap_cfg.window_sec)) * fps))
    bootstrap_min_samples = max(1, int(bootstrap_cfg.min_detections))
    bootstrap_min_inliers = max(1, int(bootstrap_cfg.min_inliers))

    consensus_max_support = 5
    buffer_max_age = max(consensus_max_support * 6, 30, max_span_frames + 5)
    det_buffer_capacity = max(buffer_max_age + consensus_max_support, 64)
    det_buffer: Deque[Dict[str, Any]] = deque(maxlen=det_buffer_capacity)

    tracker = CourtLKTracker(cfg=cfg)

    det_idx = 0
    total_detections = len(dets)
    timeseries: Dict[int, List[Point]] = {}

    def _run_on_demand_detection(frame_idx: int, frame_bgr: np.ndarray) -> Optional[Dict[str, Any]]:
        if detection_backend is None or not detection_model_id:
            return None
        try:
            result = detection_backend.infer(
                frame_bgr,
                frame_idx=frame_idx,
                model_id=detection_model_id,
                confidence=min_confidence,
            )
        except Exception as exc:
            logging.warning("On-demand court detection failed at frame %d: %s", frame_idx, exc)
            return None
        preds = result.get("predictions", []) if isinstance(result, dict) else []
        best_pred = _pick_best_court_pred(preds) if preds else None
        if not isinstance(best_pred, dict):
            return None
        corners = corners_from_prediction(best_pred)
        if not corners:
            return None
        try:
            best_conf = float(best_pred.get("confidence", 0.0))
        except Exception:
            best_conf = 0.0
        ordered = order_corners([(float(x), float(y)) for x, y in corners])
        return {
            "frame": frame_idx,
            "corners": ordered,
            "confidence": best_conf,
        }

    with open(tracking_jsonl, "w", encoding="utf-8") as out_f:
        frame_i = 0
        last_corners: Optional[List[Point]] = None
        bootstrap_done = not bool(bootstrap_cfg.enable)
        bootstrap_samples: List[Dict[str, Any]] = []
        awaiting_reset = False
        reset_reasons: Optional[List[str]] = None
        pending_frames: List[int] = []

        while frame_i < total_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if not bootstrap_done:
                pending_frames.append(frame_i)

            new_detection_added = False
            while det_idx < total_detections and int(dets[det_idx]["frame"]) == frame_i:
                det_entry = dets[det_idx]
                ordered = order_corners(det_entry["corners"])
                record = {
                    "frame": frame_i,
                    "corners": [(float(x), float(y)) for x, y in ordered],
                }
                det_buffer.append(record)
                if not bootstrap_done and frame_i <= bootstrap_window_frames:
                    bootstrap_samples.append(record)
                det_idx += 1
                new_detection_added = True

            _prune_detection_buffer(det_buffer, current_frame=frame_i, max_age=buffer_max_age)

            keyframe_written = False

            if not bootstrap_done:
                enough_samples = len(bootstrap_samples) >= bootstrap_min_samples
                deadline_hit = frame_i >= bootstrap_window_frames
                attempt_samples: List[Dict[str, Any]] = []
                if enough_samples:
                    attempt_samples = [
                        rec for rec in bootstrap_samples
                        if max_span_frames <= 0 or frame_i - rec["frame"] <= max_span_frames
                    ]
                elif deadline_hit and bootstrap_samples:
                    attempt_samples = bootstrap_samples[-bootstrap_min_samples:]

                if attempt_samples:
                    ref = tracker.build_bootstrap_reference(
                        attempt_samples,
                        threshold_px=bootstrap_cfg.ransac_threshold_px,
                        min_inliers=min(len(attempt_samples), bootstrap_min_inliers),
                    )
                    if ref and _consensus_passes_gates(
                        ref["corners"], tracker=tracker, cfg=cfg, last_corners=last_corners
                    ):
                        tracker.set_keyframe(frame_i, frame, ref["corners"])
                        if tracker.ema_corners is not None:
                            write_corners = [
                                (float(x), float(y))
                                for x, y in order_corners(tracker.ema_corners.tolist())
                            ]
                        else:
                            write_corners = ref["corners"]

                        info: Dict[str, Any] = {
                            "keyframe": True,
                            "bootstrap": True,
                            "tpl_prec": getattr(tracker, "last_tpl_prec", None),
                        }
                        meta = ref.get("meta")
                        if meta:
                            info["bootstrap_meta"] = meta

                        if pending_frames:
                            for pf in pending_frames[:-1]:
                                backfill_info = {"backfill": True, "bootstrap": True}
                                out_f.write(
                                    json.dumps(
                                        {"frame": pf, "corners": write_corners, "info": backfill_info},
                                        ensure_ascii=False,
                                    ) + "\n"
                                )
                                timeseries[pf] = write_corners
                            pending_frames = []
                        out_f.write(
                            json.dumps(
                                {"frame": frame_i, "corners": write_corners, "info": info},
                                ensure_ascii=False,
                            ) + "\n"
                        )
                        last_corners = write_corners
                        timeseries[frame_i] = write_corners
                        bootstrap_done = True
                        bootstrap_samples.clear()
                        awaiting_reset = False
                        reset_reasons = None
                        keyframe_written = True

            if not keyframe_written and awaiting_reset:
                candidate_samples = [
                    rec for rec in det_buffer
                    if max_span_frames <= 0 or frame_i - rec["frame"] <= max_span_frames
                ]
                if candidate_samples:
                    ref = tracker.build_bootstrap_reference(
                        candidate_samples,
                        threshold_px=bootstrap_cfg.ransac_threshold_px,
                        min_inliers=min(len(candidate_samples), bootstrap_min_inliers),
                    )
                    if ref and _consensus_passes_gates(
                        ref["corners"], tracker=tracker, cfg=cfg, last_corners=last_corners
                    ):
                        tracker.set_keyframe(frame_i, frame, ref["corners"])
                        if tracker.ema_corners is not None:
                            write_corners = [
                                (float(x), float(y))
                                for x, y in order_corners(tracker.ema_corners.tolist())
                            ]
                        else:
                            write_corners = ref["corners"]

                        info = {
                            "keyframe": True,
                            "tpl_prec": getattr(tracker, "last_tpl_prec", None),
                            "sentinel_reset": True,
                        }
                        if reset_reasons:
                            info["sentinel_reasons"] = reset_reasons
                        meta = ref.get("meta")
                        if meta:
                            info["bootstrap_meta"] = meta

                        out_f.write(
                            json.dumps(
                                {"frame": frame_i, "corners": write_corners, "info": info},
                                ensure_ascii=False,
                            ) + "\n")
                        last_corners = write_corners
                        timeseries[frame_i] = write_corners
                        awaiting_reset = False
                        reset_reasons = None
                        keyframe_written = True

            if keyframe_written:
                frame_i += 1
                continue

            if tracker.keyframe_gray is None:
                frame_i += 1
                continue

            corners, info = tracker.update(frame, frame_index=frame_i)
            info = info or {}

            if info.get("needs_redetect"):
                reset_reasons = info.get("sentinel_reasons")
                record = _run_on_demand_detection(frame_i, frame)
                if record:
                    det_buffer.append(record)
                    if _consensus_passes_gates(
                        record["corners"], tracker=tracker, cfg=cfg, last_corners=last_corners
                    ):
                        tracker.set_keyframe(frame_i, frame, record["corners"])
                        if tracker.ema_corners is not None:
                            write_corners = [
                                (float(x), float(y))
                                for x, y in order_corners(tracker.ema_corners.tolist())
                            ]
                        else:
                            write_corners = record["corners"]
                        info_reset: Dict[str, Any] = {
                            "keyframe": True,
                            "tpl_prec": getattr(tracker, "last_tpl_prec", None),
                            "sentinel_reset": True,
                            "on_demand_detection": True,
                        }
                        if reset_reasons:
                            info_reset["sentinel_reasons"] = reset_reasons
                        if "confidence" in record:
                            info_reset["detection_confidence"] = record["confidence"]
                        out_f.write(
                            json.dumps(
                                {"frame": frame_i, "corners": write_corners, "info": info_reset},
                                ensure_ascii=False,
                            ) + "\n")
                        last_corners = write_corners
                        timeseries[frame_i] = write_corners
                        awaiting_reset = False
                        reset_reasons = None
                        frame_i += 1
                        continue
                awaiting_reset = True
                info.setdefault("hold", True)
                corners = None
            else:
                awaiting_reset = False

            if corners is not None:
                out_f.write(
                    json.dumps(
                        {"frame": frame_i, "corners": corners, "info": info},
                        ensure_ascii=False,
                    ) + "\n")
                last_corners = corners
                timeseries[frame_i] = corners
            elif info.get("hold") and info.get("hold_left", 0) > 0 and last_corners is not None:
                out_f.write(
                    json.dumps(
                        {"frame": frame_i, "corners": last_corners, "info": info},
                        ensure_ascii=False,
                    ) + "\n")
                timeseries[frame_i] = last_corners
            elif new_detection_added and last_corners is not None:
                fallback_info = dict(info)
                fallback_info.setdefault("consensus_stale", True)
                out_f.write(
                    json.dumps(
                        {
                            "frame": frame_i,
                            "corners": last_corners,
                            "info": fallback_info,
                        },
                        ensure_ascii=False,
                    ) + "\n")
                timeseries[frame_i] = last_corners

            frame_i += 1

    cap.release()
    try:
        tracker.close()
    except Exception:
        pass

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
