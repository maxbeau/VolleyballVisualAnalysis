import os
import json
from typing import Tuple, Optional

import numpy as np
import cv2

from core.utils import ensure_dir
from court.utils import (
    median_corners_from_tracking,
    compute_homography,
    standard_court_model_size,
    warp_birdseye,
)


def compute_and_save_homography(
    *,
    tracking_jsonl: str,
    output_npy: str,
    output_meta_json: str,
    model_size: Optional[Tuple[int, int]] = None,
    scale_px_per_meter: float = 100.0,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    ensure_dir(os.path.dirname(output_npy) or ".")
    ensure_dir(os.path.dirname(output_meta_json) or ".")

    corners = median_corners_from_tracking(tracking_jsonl)
    if not corners:
        raise RuntimeError(f"No corners found in tracking JSONL: {tracking_jsonl}")

    if model_size is None:
        model_size = standard_court_model_size(scale_px_per_meter)

    H, dst_size = compute_homography(corners, dst_corners=None, dst_size=model_size)
    np.save(output_npy, H)

    existing_meta = {}
    if os.path.exists(output_meta_json):
        try:
            with open(output_meta_json, "r", encoding="utf-8") as f:
                existing_meta = json.load(f) or {}
        except Exception:
            existing_meta = {}

    meta = dict(existing_meta)
    meta.update(
        {
            "tracking_jsonl": tracking_jsonl,
            "dst_size": {"w": int(dst_size[0]), "h": int(dst_size[1])},
            "scale_px_per_meter": float(scale_px_per_meter),
            "src_corners_order": ["TL", "TR", "BR", "BL"],
            "dst_corners": [
                [0.0, 0.0],
                [dst_size[0] - 1.0, 0.0],
                [dst_size[0] - 1.0, dst_size[1] - 1.0],
                [0.0, dst_size[1] - 1.0],
            ],
        }
    )
    with open(output_meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return H, dst_size


def generate_birdseye_image(
    *,
    video_path: str,
    H: np.ndarray,
    dst_size: Tuple[int, int],
    output_jpg: str,
    frame_index: Optional[int] = None,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    if frame_index is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_index = max(0, total_frames // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from video: {video_path}")

    bird = warp_birdseye(frame, H, dst_size)
    ensure_dir(os.path.dirname(output_jpg) or ".")
    cv2.imwrite(output_jpg, bird)
