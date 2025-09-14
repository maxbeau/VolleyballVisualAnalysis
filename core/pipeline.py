import os
import json
import time
import re
import cv2
from typing import Dict, List, Optional, Tuple, Callable

from config import settings
from core.roboflow_client import RoboflowClient
from core.utils import ensure_dir

class DetectionPipeline:
    def __init__(
        self,
        model_id: str,
        confidence: float,
        fps_sample: float,
        cache_dir: str,
        combined_jsonl: str,
        save_frame_json: bool,
        max_frames: Optional[int] = None,
        cache_only: bool = False,
        pred_extractor: Optional[Callable[[Dict], List[Dict]]] = None,
    ):
        self.model_id = model_id
        self.confidence = confidence
        self.fps_sample = fps_sample
        self.max_frames = max_frames
        self.cache_dir = cache_dir
        self.combined_jsonl = combined_jsonl
        self.save_frame_json = save_frame_json
        self.cache_only = cache_only
        self.pred_extractor = pred_extractor or self._default_pred_extractor

    def _default_pred_extractor(self, result: Dict) -> List[Dict]:
        return result.get("predictions", []) if isinstance(result, dict) else []

    def run(self) -> None:
        ensure_dir(self.cache_dir)
        ensure_dir(os.path.dirname(self.combined_jsonl) or ".")

        video_path = settings.common.VIDEO_PATH
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        stride = max(1, int(round(vid_fps / max(0.1, self.fps_sample)))) if self.fps_sample > 0 else 1

        if not settings.common.ROBOFLOW_API_KEY:
            raise EnvironmentError("ROBOFLOW_API_KEY not found in .env or environment")
        client = RoboflowClient(api_key=settings.common.ROBOFLOW_API_KEY)

        t0 = time.time()

        if self.cache_only:
            records = self._rebuild_from_cache(vid_fps, width, height)
            sampled_count = len(records)
        else:
            records, sampled_count = self._run_inference_loop(
                cap, stride, vid_fps, width, height, client, total_frames
            )

        with open(self.combined_jsonl, "w", encoding="utf-8") as out_f:
            for rec in records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        cap.release()
        dt = time.time() - t0
        print(
            f"Detection done for {self.model_id}. Video: {video_path} | frames: {total_frames} | "
            f"sampled: {sampled_count} | stride: {stride} | time: {dt:.1f}s"
        )
        print(f"Per-frame cache: {self.cache_dir}")
        print(f"Combined detections: {self.combined_jsonl}")

    def _rebuild_from_cache(self, vid_fps: float, width: int, height: int) -> List[Dict]:
        print(f"Cache-only mode: rebuilding from {self.cache_dir}...")
        pattern = re.compile(r"frame_(\d{6})\.json$")
        cached_files = []
        for name in os.listdir(self.cache_dir):
            m = pattern.match(name)
            if m:
                cached_files.append((int(m.group(1)), os.path.join(self.cache_dir, name)))

        cached_files.sort(key=lambda x: x[0])

        records: List[Dict] = []
        for idx, jp in cached_files:
            with open(jp, "r", encoding="utf-8") as jf:
                result = json.load(jf)
            preds = self.pred_extractor(result)
            records.append({
                "frame": idx,
                "time_sec": idx / vid_fps if vid_fps else None,
                "image_size": {"w": width, "h": height},
                "model_id": self.model_id,
                "confidence": self.confidence,
                "predictions": preds,
            })
        return records

    def _run_inference_loop(
        self, cap, stride: int, vid_fps: float, width: int, height: int, client, total_frames: int
    ) -> Tuple[List[Dict], int]:
        records: List[Dict] = []
        frame_idx = 0
        sampled_count = 0

        while True:
            ok = cap.grab()
            if not ok:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                frame_idx += 1
                continue

            json_path = os.path.join(self.cache_dir, f"frame_{frame_idx:06d}.json")

            if self.save_frame_json and os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as jf:
                    result = json.load(jf)
            else:
                print(f"Inferring frame {frame_idx}/{total_frames} with model {self.model_id}")
                result = client.infer_frame(frame, model_id=self.model_id, confidence=self.confidence)
                if self.save_frame_json:
                    with open(json_path, "w", encoding="utf-8") as jf:
                        json.dump(result, jf, ensure_ascii=False)

            preds = self.pred_extractor(result)
            records.append({
                "frame": frame_idx,
                "time_sec": frame_idx / vid_fps if vid_fps else None,
                "image_size": {"w": width, "h": height},
                "model_id": self.model_id,
                "confidence": self.confidence,
                "predictions": preds,
            })

            sampled_count += 1
            frame_idx += 1

            if self.max_frames and sampled_count >= self.max_frames:
                break

        return records, sampled_count