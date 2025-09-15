import os
import json
import argparse
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2

try:
    # Prefer unified config path
    from config import settings
except Exception:  # fallback for older import path
    from core.config import settings
from core.utils import ensure_dir
from court.utils import order_corners, shape_metrics, within_tol
from court.config import CourtTrackerConfig
from court.tracker import CourtLKTracker  # canonical implementation
from court.orientation import decide_orientation as decide_court_orientation
from court.io import load_detections


Point = Tuple[float, float]


def _load_detections(detections_jsonl: str) -> List[Dict[str, Any]]:
    return load_detections(detections_jsonl)


def run_tracking(
    video_path: str,
    detections_jsonl: str,
    tracking_jsonl: str,
    use_homography: bool = True,
    ransac_thresh: float = 3.0,
    hold_ttl: int = 8,
    cfg: Optional[CourtTrackerConfig] = None,
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

    # Build tracker with provided cfg if any, otherwise with project-tuned defaults
    tracker_cfg = cfg or CourtTrackerConfig(
        lk_roi_expand_ratio=float(getattr(settings.court, "LK_ROI_EXPAND_RATIO", 0.12)),
        max_scale_change_per_frame=float(getattr(settings.court, "MAX_SCALE_CHANGE_PER_FRAME", 0.10)),
        kf_adaptive_from_template=bool(getattr(settings.court, "KF_ADAPTIVE_FROM_TEMPLATE", True)),
        kf_r_api_min=float(getattr(settings.court, "KF_R_API_MIN", 0.8)),
        kf_r_api_max=float(getattr(settings.court, "KF_R_API_MAX", 2.5)),
        use_homography=use_homography,
        ransac_reproj_thresh=ransac_thresh,
        hold_ttl_frames=hold_ttl,
    )
    tracker = CourtLKTracker(cfg=tracker_cfg)

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
                    if tracker.ref_ratio is not None and not within_tol(r_det, tracker.ref_ratio, max(0.35, tracker.cfg.ratio_tolerance)):
                        ok_shape = False
                    if tracker.ref_area is not None and tracker.ref_area > 1e-6:
                        lo = (1.0 - max(0.6, tracker.cfg.area_tolerance)) * tracker.ref_area
                        hi = (1.0 + max(0.6, tracker.cfg.area_tolerance)) * tracker.ref_area
                        if not (lo <= a_det <= hi):
                            ok_shape = False
                    if med_d > max(12.0, tracker.cfg.max_jump_px * 1.5) or not ok_shape:
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
        orient = decide_court_orientation(cap2, ts_for_orient, (model_W, model_H), mode=getattr(settings.court, "MINI_ORIENT_MODE", "template"))
        cap2.release()
        meta = {
            "tracking_jsonl": tracking_jsonl,
            "orientation": orient,
        }
        ensure_dir(os.path.dirname(getattr(settings.court, "TRACKING_META", tracking_jsonl)) or ".")
        with open(getattr(settings.court, "TRACKING_META", tracking_jsonl), "w", encoding="utf-8") as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Track court corners between low-frequency detections using LK+RANSAC")
    parser.add_argument("--detections-jsonl", default=getattr(settings.court, "DETECTIONS_JSONL"))
    parser.add_argument("--tracking-jsonl", default=getattr(settings.court, "TRACKING_JSONL"))
    # Model choice
    parser.add_argument("--use-homography", action="store_true", help="Use homography (default). If not set, uses affine.")
    parser.add_argument("--affine", action="store_true", help="Force affine instead of homography")

    # Core gating/robustness
    parser.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC reprojection threshold (pixels)")
    parser.add_argument("--hold-ttl", type=int, default=8, help="Frames to hold last result on failure")
    parser.add_argument("--min-inlier-ratio", type=float, default=None, help="Minimum inlier ratio to accept model")
    parser.add_argument("--min-inliers", type=int, default=None, help="Minimum inlier count to accept model")
    parser.add_argument("--fb-reproj-thresh", type=float, default=None, help="Forward-backward reprojection error threshold (px)")
    parser.add_argument("--max-scale-change-per-frame", type=float, default=None, help="Max per-frame scale change (fraction)")

    # Feature tracking / seeding
    parser.add_argument("--feature-max", type=int, default=None, help="Max features to track")
    parser.add_argument("--roi-expand-ratio", type=float, default=None, help="ROI expand ratio for seeding")
    parser.add_argument("--lk-roi-expand-ratio", type=float, default=None, help="LK ROI expand ratio (tracking)")
    parser.add_argument("--reseed-min-tracks", type=int, default=None, help="Reseed when surviving tracks less than this")
    parser.add_argument("--subpix-win", type=int, default=None, help="cornerSubPix half window size")
    parser.add_argument("--subpix-stride", type=int, default=None, help="Refine subpixel corners every N frames")

    # Geometry gating
    parser.add_argument("--max-jump-px", type=float, default=None, help="Max median corner jump vs EMA (px)")
    parser.add_argument("--ratio-tolerance", type=float, default=None, help="Aspect ratio tolerance (fraction)")
    parser.add_argument("--area-tolerance", type=float, default=None, help="Area tolerance (fraction)")

    # Template score
    parser.add_argument("--use-template-score", dest="use_template_score", action="store_true", default=None, help="Enable template precision gating")
    parser.add_argument("--no-template-score", dest="use_template_score", action="store_false", help="Disable template precision gating")
    parser.add_argument("--template-line-px", type=int, default=None, help="Template line thickness (px)")
    parser.add_argument("--template-min-precision", type=float, default=None, help="Minimum template precision to accept frame")
    parser.add_argument("--template-stride", type=int, default=None, help="Compute template precision every N frames")

    # Smoothing / Kalman
    parser.add_argument("--use-kalman", dest="use_kalman", action="store_true", default=None, help="Enable Kalman smoothing")
    parser.add_argument("--no-kalman", dest="use_kalman", action="store_false", help="Disable Kalman smoothing")
    parser.add_argument("--kalman-q-pos", type=float, default=None, help="Kalman process noise (position)")
    parser.add_argument("--kalman-q-vel", type=float, default=None, help="Kalman process noise (velocity)")
    parser.add_argument("--kalman-r-meas", type=float, default=None, help="Kalman measurement noise sigma for API detections (px)")
    parser.add_argument("--kf-adaptive-from-template", dest="kf_adaptive_from_template", action="store_true", default=None, help="Adapt R from template precision")
    parser.add_argument("--no-kf-adaptive-from-template", dest="kf_adaptive_from_template", action="store_false", help="Disable adaptive R from template precision")
    parser.add_argument("--kf-r-api-min", type=float, default=None, help="Lower bound of adaptive measurement sigma")
    parser.add_argument("--kf-r-api-max", type=float, default=None, help="Upper bound of adaptive measurement sigma")
    # New performance/robustness options
    parser.add_argument("--use-roi-downsample", dest="use_roi_downsample", action="store_true", default=None, help="Enable ROI downsample for LK")
    parser.add_argument("--no-roi-downsample", dest="use_roi_downsample", action="store_false", help="Disable ROI downsample for LK")
    parser.add_argument("--roi-downsample-scale", type=float, default=None, help="ROI downsample scale (e.g., 0.5)")
    parser.add_argument("--early-motion-gate", dest="early_motion_gate", action="store_true", default=None, help="Enable early motion gate to skip LK")
    parser.add_argument("--no-early-motion-gate", dest="early_motion_gate", action="store_false", help="Disable early motion gate")
    parser.add_argument("--early-motion-mad-gray-thr", type=float, default=None, help="Early motion gate MAD gray threshold")
    parser.add_argument("--fallback-affine", dest="model_fallback_affine_on_fail", action="store_true", default=None, help="Try affine fallback when homography fails gates")
    parser.add_argument("--no-fallback-affine", dest="model_fallback_affine_on_fail", action="store_false", help="Disable affine fallback")
    parser.add_argument("--kalman-q-scale-from-motion", dest="kalman_q_scale_from_motion", action="store_true", default=None, help="Adapt Kalman Q scale from motion magnitude")
    parser.add_argument("--no-kalman-q-scale-from-motion", dest="kalman_q_scale_from_motion", action="store_false", help="Disable Q adaptation")
    parser.add_argument("--kalman-q-scale-lo", type=float, default=None, help="Lower bound of Kalman Q scale")
    parser.add_argument("--kalman-q-scale-hi", type=float, default=None, help="Upper bound of Kalman Q scale")
    parser.add_argument("--motion-md-ref-px", type=float, default=None, help="Reference median displacement (px) for Q scale mapping")
    args = parser.parse_args()

    # Resolve method flags
    use_h = True
    if args.affine:
        use_h = False
    elif args.use_homography:
        use_h = True

    # Build tracker config with CLI overrides (falling back to dataclass defaults,
    # and to settings for some project-tuned parameters where applicable).
    cfg = CourtTrackerConfig()
    # Method
    cfg.use_homography = use_h
    # Core
    cfg.ransac_reproj_thresh = float(args.ransac_thresh)
    cfg.hold_ttl_frames = int(args.hold_ttl)
    if args.min_inlier_ratio is not None:
        cfg.min_inlier_ratio = float(args.min_inlier_ratio)
    if args.min_inliers is not None:
        cfg.min_inliers = int(args.min_inliers)
    if args.fb_reproj_thresh is not None:
        cfg.fb_reproj_thresh = float(args.fb_reproj_thresh)
    # ROI/Features
    if args.feature_max is not None:
        cfg.feature_max = int(args.feature_max)
    if args.roi_expand_ratio is not None:
        cfg.roi_expand_ratio = float(args.roi_expand_ratio)
    cfg.lk_roi_expand_ratio = float(args.lk_roi_expand_ratio if args.lk_roi_expand_ratio is not None else getattr(settings.court, "LK_ROI_EXPAND_RATIO", cfg.lk_roi_expand_ratio))
    if args.reseed_min_tracks is not None:
        cfg.reseed_min_tracks = int(args.reseed_min_tracks)
    if args.subpix_win is not None:
        cfg.subpix_win = int(args.subpix_win)
    if args.subpix_stride is not None:
        cfg.subpix_stride = int(args.subpix_stride)
    # Geometry gates
    if args.max_jump_px is not None:
        cfg.max_jump_px = float(args.max_jump_px)
    if args.ratio_tolerance is not None:
        cfg.ratio_tolerance = float(args.ratio_tolerance)
    if args.area_tolerance is not None:
        cfg.area_tolerance = float(args.area_tolerance)
    # Template
    if args.use_template_score is not None:
        cfg.use_template_score = bool(args.use_template_score)
    if args.template_line_px is not None:
        cfg.template_line_px = int(args.template_line_px)
    if args.template_min_precision is not None:
        cfg.template_min_precision = float(args.template_min_precision)
    if args.template_stride is not None:
        cfg.template_stride = int(args.template_stride)
    # Kalman
    if args.use_kalman is not None:
        cfg.use_kalman = bool(args.use_kalman)
    if args.kalman_q_pos is not None:
        cfg.kalman_q_pos = float(args.kalman_q_pos)
    if args.kalman_q_vel is not None:
        cfg.kalman_q_vel = float(args.kalman_q_vel)
    if args.kalman_r_meas is not None:
        cfg.kalman_r_meas = float(args.kalman_r_meas)
    if args.kf_adaptive_from_template is not None:
        cfg.kf_adaptive_from_template = bool(args.kf_adaptive_from_template)
    cfg.kf_r_api_min = float(args.kf_r_api_min if args.kf_r_api_min is not None else getattr(settings.court, "KF_R_API_MIN", cfg.kf_r_api_min))
    cfg.kf_r_api_max = float(args.kf_r_api_max if args.kf_r_api_max is not None else getattr(settings.court, "KF_R_API_MAX", cfg.kf_r_api_max))
    if args.max_scale_change_per_frame is not None:
        cfg.max_scale_change_per_frame = float(args.max_scale_change_per_frame)
    # New performance/robustness options
    if args.use_roi_downsample is not None:
        cfg.use_roi_downsample = bool(args.use_roi_downsample)
    if args.roi_downsample_scale is not None:
        cfg.roi_downsample_scale = float(args.roi_downsample_scale)
    if args.early_motion_gate is not None:
        cfg.early_motion_gate = bool(args.early_motion_gate)
    if args.early_motion_mad_gray_thr is not None:
        cfg.early_motion_mad_gray_thr = float(args.early_motion_mad_gray_thr)
    if args.model_fallback_affine_on_fail is not None:
        cfg.model_fallback_affine_on_fail = bool(args.model_fallback_affine_on_fail)
    if args.kalman_q_scale_from_motion is not None:
        cfg.kalman_q_scale_from_motion = bool(args.kalman_q_scale_from_motion)
    if args.kalman_q_scale_lo is not None:
        cfg.kalman_q_scale_lo = float(args.kalman_q_scale_lo)
    if args.kalman_q_scale_hi is not None:
        cfg.kalman_q_scale_hi = float(args.kalman_q_scale_hi)
    if args.motion_md_ref_px is not None:
        cfg.motion_md_ref_px = float(args.motion_md_ref_px)

    run_tracking(
        video_path=getattr(settings.common, "VIDEO_PATH"),
        detections_jsonl=args.detections_jsonl,
        tracking_jsonl=args.tracking_jsonl,
        use_homography=cfg.use_homography,
        ransac_thresh=cfg.ransac_reproj_thresh,
        hold_ttl=cfg.hold_ttl_frames,
        cfg=cfg,
    )

    print(f"Court tracking saved: {args.tracking_jsonl}")


if __name__ == "__main__":
    main()

__all__ = ["CourtLKTracker", "run_tracking"]
