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


def _load_net_measurements(
    net_jsonl: Optional[str], *, label_aliases: List[str]
) -> Dict[int, Dict[str, Any]]:
    """Load per-frame net detection measurements (height, confidence, etc.)."""
    measurements: Dict[int, Dict[str, Any]] = {}
    if not net_jsonl:
        return measurements
    if not os.path.exists(net_jsonl):
        logging.warning("Net detection file not found: %s", net_jsonl)
        return measurements

    alias_set = {alias.strip().lower() for alias in label_aliases if isinstance(alias, str)}
    if not alias_set:
        alias_set = {"net"}

    def _norm_label(pred: Dict[str, Any]) -> str:
        for key in ("class", "class_name", "label", "name"):
            val = pred.get(key)
            if isinstance(val, str):
                return val.strip().lower()
        return ""

    def _conf(pred: Dict[str, Any]) -> float:
        for key in ("confidence", "score", "probability"):
            val = pred.get(key)
            if val is None:
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
        return 0.0

    def _geometry(pred: Dict[str, Any]) -> Dict[str, Any]:
        corners = corners_from_prediction(pred)
        if corners:
            tl, tr, br, bl = corners[:4]
            top_y = float((tl[1] + tr[1]) * 0.5)
            bottom_y = float((bl[1] + br[1]) * 0.5)
            left_x = float((tl[0] + bl[0]) * 0.5)
            right_x = float((tr[0] + br[0]) * 0.5)
            height = max(0.0, bottom_y - top_y)
            width = max(0.0, right_x - left_x)
            cx = float((left_x + right_x) * 0.5)
            cy = float((top_y + bottom_y) * 0.5)
            return {
                "height": height,
                "width": width,
                "top": top_y,
                "bottom": bottom_y,
                "center": (cx, cy),
                "corners": [(float(x), float(y)) for x, y in corners[:4]],
            }
        try:
            cx = float(pred.get("x", pred.get("cx", 0.0)))
            cy = float(pred.get("y", pred.get("cy", 0.0)))
        except (TypeError, ValueError):
            cx = cy = 0.0
        try:
            width = float(pred.get("width", pred.get("w", 0.0)))
            height = float(pred.get("height", pred.get("h", 0.0)))
        except (TypeError, ValueError):
            width = height = 0.0
        top_y = cy - height * 0.5
        bottom_y = cy + height * 0.5
        return {
            "height": max(0.0, height),
            "width": max(0.0, width),
            "top": float(top_y),
            "bottom": float(bottom_y),
            "center": (float(cx), float(cy)),
            "corners": None,
        }

    with open(net_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                frame_idx = int(rec.get("frame", -1))
            except (TypeError, ValueError):
                continue
            if frame_idx < 0:
                continue
            preds = rec.get("predictions") or []
            best_pred: Optional[Dict[str, Any]] = None
            best_conf = -1.0
            for pred in preds:
                if not isinstance(pred, dict):
                    continue
                label = _norm_label(pred)
                if alias_set and not any(alias in label for alias in alias_set):
                    continue
                conf_val = _conf(pred)
                if conf_val > best_conf:
                    best_pred = pred
                    best_conf = conf_val
            if best_pred is None or best_conf <= 0.0:
                continue
            geom = _geometry(best_pred)
            height = float(geom.get("height", 0.0))
            if height <= 1.0:
                continue
            existing = measurements.get(frame_idx)
            if existing and existing.get("confidence", 0.0) >= best_conf:
                continue
            center_geom = geom.get("center", (0.0, 0.0)) or (0.0, 0.0)
            try:
                center_val = (float(center_geom[0]), float(center_geom[1]))
            except (TypeError, ValueError, IndexError):
                center_val = (0.0, 0.0)
            corners_geom = geom.get("corners")
            if corners_geom:
                corners_val = [(float(p[0]), float(p[1])) for p in corners_geom[:4]]
            else:
                corners_val = None
            measurements[frame_idx] = {
                "height": height,
                "confidence": float(best_conf),
                "top": float(geom.get("top", 0.0)),
                "bottom": float(geom.get("bottom", 0.0)),
                "center": center_val,
                "width": float(geom.get("width", 0.0)),
                "corners": corners_val,
            }
    return measurements


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
    net_detections_jsonl: Optional[str] = None,
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

    net_cfg = getattr(cfg, "net", None)
    net_enabled = bool(net_cfg and getattr(net_cfg, "enable", True))
    net_measurements: Dict[int, Dict[str, Any]] = {}
    if net_enabled:
        primary_net_path = None
        if net_detections_jsonl and os.path.exists(net_detections_jsonl):
            primary_net_path = net_detections_jsonl
        elif os.path.exists(detections_jsonl):
            primary_net_path = detections_jsonl
        if primary_net_path:
            net_measurements = _load_net_measurements(primary_net_path, label_aliases=net_cfg.label_aliases)

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
        last_net: Optional[Dict[str, Any]] = None
        bootstrap_done = not bool(bootstrap_cfg.enable)
        bootstrap_samples: List[Dict[str, Any]] = []
        awaiting_reset = False
        reset_reasons: Optional[List[str]] = None
        pending_frames: List[int] = []

        def _prepare_net_state(state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not state:
                return None

            def _coerce(val: Any) -> Any:
                if isinstance(val, (np.floating, np.integer)):
                    return float(val)
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, (tuple, list)):
                    seq = []
                    for item in val:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            try:
                                seq.append((float(item[0]), float(item[1])))
                                continue
                            except (TypeError, ValueError, IndexError):
                                pass
                        seq.append(_coerce(item))
                    return seq
                if isinstance(val, dict):
                    return {k: _coerce(v) for k, v in val.items()}
                if val is None or isinstance(val, str):
                    return val
                try:
                    return float(val)
                except Exception:
                    return val

            return {k: _coerce(v) for k, v in state.items()}

        def _net_state_from_measurement(frame_idx: int, measurement: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not measurement:
                return None

            corners = measurement.get("corners")
            height_val = measurement.get("height") or measurement.get("height_px") or measurement.get("h")
            try:
                height_px = float(height_val) if height_val is not None else None
            except (TypeError, ValueError):
                height_px = None
            try:
                meas_conf = float(measurement.get("confidence", measurement.get("score", 0.0)) or 0.0)
            except (TypeError, ValueError):
                meas_conf = 0.0
            bottom_y = measurement.get("bottom")
            top_y = measurement.get("top")

            base_pts: Optional[List[Tuple[float, float]]] = None
            top_pts: Optional[List[Tuple[float, float]]] = None
            direction: Tuple[float, float] = (0.0, -1.0)
            vanish: Optional[Tuple[float, float]] = None

            if corners and len(corners) >= 4:
                try:
                    arr = np.array(corners[:4], dtype=np.float64)
                    tl, tr, br, bl = arr
                    base_pts = [(float(bl[0]), float(bl[1])), (float(br[0]), float(br[1]))]
                    top_pts = [(float(tl[0]), float(tl[1])), (float(tr[0]), float(tr[1]))]
                    col_left = tl - bl
                    col_right = tr - br
                    avg_vec = (col_left + col_right) * 0.5
                    norm = float(np.linalg.norm(avg_vec))
                    if norm > 1e-6:
                        direction = (float(avg_vec[0] / norm), float(avg_vec[1] / norm))
                    try:
                        a = np.array([bl[0], bl[1], 1.0], dtype=np.float64)
                        b = np.array([tl[0], tl[1], 1.0], dtype=np.float64)
                        c = np.array([br[0], br[1], 1.0], dtype=np.float64)
                        d = np.array([tr[0], tr[1], 1.0], dtype=np.float64)
                        line1 = np.cross(a, b)
                        line2 = np.cross(c, d)
                        vp = np.cross(line1, line2)
                        if abs(vp[2]) > 1e-9:
                            vanish = (float(vp[0] / vp[2]), float(vp[1] / vp[2]))
                    except Exception:
                        vanish = None
                    if height_px is None:
                        height_px = norm
                except Exception:
                    base_pts = top_pts = None

            if base_pts is None:
                center = measurement.get("center") or (None, None)
                width = measurement.get("width") or measurement.get("w")
                try:
                    cx = float(center[0]) if center[0] is not None else None
                    cy = float(center[1]) if center[1] is not None else None
                except (TypeError, ValueError, IndexError):
                    cx = cy = None
                try:
                    width_val = float(width) if width is not None else None
                except (TypeError, ValueError):
                    width_val = None
                if cx is None or cy is None or width_val is None:
                    return None
                half_w = width_val * 0.5
                y_bottom = float(bottom_y) if bottom_y is not None else cy
                base_pts = [(cx - half_w, y_bottom), (cx + half_w, y_bottom)]
                if height_px is not None:
                    top_y_est = y_bottom - height_px
                else:
                    top_y_est = float(top_y) if top_y is not None else y_bottom - 80.0
                top_pts = [(cx - half_w, top_y_est), (cx + half_w, top_y_est)]
                if height_px is None:
                    height_px = abs(top_y_est - y_bottom)

            if height_px is None:
                return None

            state: Dict[str, Any] = {
                "frame": int(frame_idx),
                "height_px": float(height_px),
                "base": base_pts,
                "top": top_pts,
                "polygon": [base_pts[0], base_pts[1], top_pts[1], top_pts[0]] if base_pts and top_pts else None,
                "confidence": float(np.clip(meas_conf, 0.0, 1.0)),
                "measurement_conf": float(meas_conf),
                "measurement_height_px": float(height_px),
                "measurement_bottom": float(bottom_y) if bottom_y is not None else None,
                "measurement_top": float(top_y) if top_y is not None else None,
                "missing_frames": 0,
                "filter_variance": float(getattr(cfg.net, "kalman_r", 100.0)),
                "direction": [direction[0], direction[1]],
            }
            if vanish is not None:
                state["vanish_point"] = [vanish[0], vanish[1]]
            return state

        def _write_record(frame_idx: int, corners: List[Point], info: Dict[str, Any], net_state: Optional[Dict[str, Any]]) -> None:
            info_payload = dict(info) if info else {}
            safe_net = _prepare_net_state(net_state)
            if safe_net is not None:
                info_payload.setdefault("net", safe_net)
            record = {
                "frame": frame_idx,
                "corners": corners,
                "info": info_payload,
            }
            if safe_net is not None:
                record["net"] = safe_net
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

        while frame_i < total_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            net_measurement = net_measurements.get(frame_i) if net_enabled else None

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
                        tracker.set_keyframe(frame_i, frame, ref["corners"], net_measurement=net_measurement)
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

                        net_state_current = _prepare_net_state(tracker.net_state)
                        if pending_frames:
                            for pf in pending_frames[:-1]:
                                backfill_info = {"backfill": True, "bootstrap": True}
                                meas_state = _net_state_from_measurement(pf, net_measurements.get(pf) if net_enabled else None)
                                net_backfill = meas_state or net_state_current or last_net
                                _write_record(pf, write_corners, backfill_info, net_backfill)
                                timeseries[pf] = write_corners
                            pending_frames = []
                        _write_record(frame_i, write_corners, info, net_state_current)
                        last_corners = write_corners
                        last_net = net_state_current
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
                        tracker.set_keyframe(frame_i, frame, ref["corners"], net_measurement=net_measurement)
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

            corners, info = tracker.update(frame, frame_index=frame_i, net_measurement=net_measurement)
            info = info or {}

            if info.get("needs_redetect"):
                reset_reasons = info.get("sentinel_reasons")
                record = _run_on_demand_detection(frame_i, frame)
                if record:
                    det_buffer.append(record)
                    if _consensus_passes_gates(
                        record["corners"], tracker=tracker, cfg=cfg, last_corners=last_corners
                    ):
                        tracker.set_keyframe(frame_i, frame, record["corners"], net_measurement=net_measurement)
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
                        net_state_current = _prepare_net_state(tracker.net_state)
                        _write_record(frame_i, write_corners, info_reset, net_state_current)
                        last_corners = write_corners
                        last_net = net_state_current
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

            net_state_current = info.get("net") or tracker.net_state
            safe_net_current = _prepare_net_state(net_state_current) if net_state_current else None

            if corners is not None:
                _write_record(frame_i, corners, info, net_state_current)
                last_corners = corners
                last_net = safe_net_current if safe_net_current is not None else last_net
                timeseries[frame_i] = corners
            elif info.get("hold") and info.get("hold_left", 0) > 0 and last_corners is not None:
                _write_record(frame_i, last_corners, info, net_state_current or last_net)
                timeseries[frame_i] = last_corners
            elif new_detection_added and last_corners is not None:
                fallback_info = dict(info)
                fallback_info.setdefault("consensus_stale", True)
                _write_record(frame_i, last_corners, fallback_info, net_state_current or last_net)
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
