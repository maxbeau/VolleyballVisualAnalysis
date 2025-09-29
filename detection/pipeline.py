"""Modular detection pipeline supporting pluggable backends."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from core.utils import ensure_dir
from detection.factory import create_detection_backend


class DetectionPipeline:
    """Generic detection runner with caching and backend abstraction."""

    def __init__(
        self,
        *,
        video_path: str,
        target_name: str,
        model_id: str,
        confidence: float,
        fps_sample: float,
        cache_dir: str,
        combined_jsonl: str,
        save_frame_json: bool,
        detection_settings: Dict,
        max_frames: Optional[int] = None,
        pred_extractor: Optional[Callable[[Dict], List[Dict]]] = None,
    ) -> None:
        self.video_path = video_path
        self.target_name = target_name
        self.model_id = model_id
        self.confidence = confidence
        self.fps_sample = fps_sample
        self.max_frames = max_frames
        self.cache_dir = cache_dir
        self.combined_jsonl = combined_jsonl
        self.save_frame_json = save_frame_json
        self.pred_extractor = pred_extractor or self._default_pred_extractor
        self.backend = create_detection_backend(detection_settings)
        self.cache_policy = detection_settings.get("cache_policy", "cache_first")
        self.detection_settings = detection_settings
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> None:
        ensure_dir(self.cache_dir)
        ensure_dir(os.path.dirname(self.combined_jsonl) or ".")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        stride = max(1, int(round(vid_fps / max(0.1, self.fps_sample)))) if self.fps_sample > 0 else 1

        t0 = time.time()

        # 1. Determine which frames need inference based on cache policy
        frames_to_sample = self._get_frames_to_sample(total_frames, stride)
        frames_to_infer = self._plan_inference(frames_to_sample)

        # 2. Run inference only for the frames that need it
        if frames_to_infer:
            self.logger.info(f"[{self.target_name}] Inferring {len(frames_to_infer)} frames...")
            self._run_inference_loop(cap, frames_to_infer)
        else:
            self.logger.info(f"[{self.target_name}] All frames found in cache, skipping inference.")

        # 3. Rebuild the final JSONL from the complete cache
        records = self._rebuild_from_cache(vid_fps, width, height)

        with open(self.combined_jsonl, "w", encoding="utf-8") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        cap.release()
        dt = time.time() - t0
        self.logger.info(
            f"[{self.target_name}] Detection done | model: {self.model_id} | frames: {total_frames} | "
            f"sampled: {len(frames_to_sample)} | inferred: {len(frames_to_infer)} | stride: {stride} | time: {dt:.1f}s"
        )
        self.logger.info(f"Cache dir: {self.cache_dir}")
        self.logger.info(f"Combined detections: {self.combined_jsonl}")

        self._write_manifest(
            {
                "video_path": self.video_path,
                "target": self.target_name,
                "model_id": self.model_id,
                "confidence": self.confidence,
                "backend": self.detection_settings.get("backend", "unknown"),
                "fps_sample": self.fps_sample,
                "stride": stride,
                "cache_policy": self.cache_policy,
                "frames_total": total_frames,
                "frames_sampled": len(frames_to_sample),
                "frames_inferred": len(frames_to_infer),
                "frames_cached": len(records),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _get_frames_to_sample(self, total_frames: int, stride: int) -> List[int]:
        """Get a list of all frame indices that should be sampled."""
        sampled_frames = list(range(0, total_frames, stride))
        if self.max_frames and len(sampled_frames) > self.max_frames:
            return sampled_frames[:self.max_frames]
        return sampled_frames

    def _plan_inference(self, frames_to_sample: List[int]) -> List[int]:
        """Decide which frames to run inference on based on the cache policy."""
        if self.cache_policy == "always_infer":
            return frames_to_sample

        frames_to_infer = []
        for frame_idx in frames_to_sample:
            json_path = os.path.join(self.cache_dir, f"frame_{frame_idx:06d}.json")
            if not os.path.exists(json_path):
                if self.cache_policy == "cache_only":
                    raise FileNotFoundError(
                        f"Cache policy is 'cache_only' but cache file is missing for frame {frame_idx}: {json_path}"
                    )
                frames_to_infer.append(frame_idx)
        return frames_to_infer

    def _rebuild_from_cache(self, vid_fps: float, width: int, height: int) -> List[Dict]:
        """Load all results from individual cache files and compile them."""
        self.logger.info(f"[{self.target_name}] Rebuilding results from cache: {self.cache_dir}...")
        pattern = re.compile(r"frame_(\d{6})\.json$")
        cached_files = []
        for name in os.listdir(self.cache_dir):
            match = pattern.match(name)
            if match:
                cached_files.append((int(match.group(1)), os.path.join(self.cache_dir, name)))

        cached_files.sort(key=lambda item: item[0])

        records: List[Dict] = []
        for idx, json_path in cached_files:
            with open(json_path, "r", encoding="utf-8") as jf:
                result = json.load(jf)
            preds = self.pred_extractor(result)
            records.append(
                {
                    "frame": idx,
                    "time_sec": idx / vid_fps if vid_fps else None,
                    "image_size": {"w": width, "h": height},
                    "model_id": self.model_id,
                    "confidence": self.confidence,
                    "predictions": preds,
                    "backend": self.detection_settings.get("backend", "unknown"),
                }
            )
        return records

    def _run_inference_loop(self, cap, frames_to_infer: List[int]) -> None:
        """Efficiently run inference by performing a single pass over the video."""
        if not frames_to_infer:
            return

        frames_set = set(frames_to_infer)
        max_frame_to_infer = max(frames_to_infer)
        current_frame_idx = 0
        
        # Ensure we start from the beginning of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ok, frame = cap.read()
            if not ok:
                break  # End of video

            if current_frame_idx in frames_set:
                json_path = os.path.join(self.cache_dir, f"frame_{current_frame_idx:06d}.json")
                result = self.backend.infer(
                    frame,
                    frame_idx=current_frame_idx,
                    model_id=self.model_id,
                    confidence=self.confidence,
                )
                if self.save_frame_json:
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(result, jf, ensure_ascii=False)

            current_frame_idx += 1
            
            # Optimization: stop reading frames if we've processed all required ones
            if current_frame_idx > max_frame_to_infer:
                break

    def _default_pred_extractor(self, result: Dict) -> List[Dict]:
        return result.get("predictions", []) if isinstance(result, dict) else []

    def _write_manifest(self, manifest: Dict[str, object]) -> None:
        # Update manifest with backend info from instance settings
        manifest["backend"] = self.detection_settings.get("backend", "unknown")
        
        path = os.path.join(self.cache_dir, "manifest.json")
        ensure_dir(os.path.dirname(path) or ".")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to write manifest file: {e}")


__all__ = ["DetectionPipeline"]
