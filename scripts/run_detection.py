import argparse
import os
from typing import Dict, List, Any, Callable, Optional

from config import settings, BallSettings, PlayersSettings, ActionsSettings, CourtSettings
from core.pipeline import DetectionPipeline

# --- Prediction Extractors ---

def _default_pred_extractor(result: Dict) -> List[Dict]:
    """Default extractor for object detection models."""
    return result.get("predictions", []) if isinstance(result, dict) else []

def _actions_pred_extractor(result: Dict) -> List[Dict]:
    """Custom extractor for classification models used in actions detection."""
    if isinstance(result, dict):
        preds = result.get("predictions")
        if isinstance(preds, list) and len(preds) > 0:
            return preds
        # classification fallbacks
        top = result.get("top")
        conf = result.get("confidence")
        if isinstance(top, str) and isinstance(conf, (int, float)):
            return [{"class": top, "confidence": float(conf)}]
        labels = result.get("labels") or result.get("classes") or result.get("probabilities")
        if isinstance(labels, dict):
            out = []
            for k, v in labels.items():
                try:
                    out.append({"class": str(k), "confidence": float(v)})
                except Exception:
                    continue
            out.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
            return out
    return []

# --- Target Configuration ---

class TargetConfig:
    def __init__(self, settings_obj: Any, extractor: Callable[[Dict], List[Dict]]):
        self.settings = settings_obj
        self.extractor = extractor

TARGETS: Dict[str, TargetConfig] = {
    "ball": TargetConfig(settings.ball, _default_pred_extractor),
    "players": TargetConfig(settings.players, _default_pred_extractor),
    "actions": TargetConfig(settings.actions, _actions_pred_extractor),
    "court": TargetConfig(settings.court, _default_pred_extractor),
}

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Run detection for a specific target (ball, players, actions) using Roboflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "target",
        choices=TARGETS.keys(),
        help="The detection target to run."
    )
    parser.add_argument("--confidence", type=float, help=f"Confidence threshold. Default is from common settings: {settings.common.OVERLAY_MIN_CONF}")
    parser.add_argument("--infer-fps", type=float, help="Inference FPS. Default is from target-specific settings.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process. 0 means no limit.")
    parser.add_argument("--cache-only", action="store_true", help="Rebuild combined JSONL from existing cache; no API calls.")
    parser.add_argument("--model-id", type=str, help="Override the model ID from settings.")

    args = parser.parse_args()

    target_config = TARGETS[args.target]
    target_settings = target_config.settings

    # Determine final values, using args > target settings > common settings
    confidence = args.confidence if args.confidence is not None else settings.common.OVERLAY_MIN_CONF
    infer_fps = args.infer_fps if args.infer_fps is not None else getattr(target_settings, 'INFER_FPS', settings.common.INFER_FPS)
    model_id = args.model_id if args.model_id is not None else target_settings.MODEL_ID
    
    # Use a common base cache dir and create a subdirectory for the target
    base_cache_dir = settings.common.CACHE_DIR
    cache_dir = os.path.join(base_cache_dir, args.target)

    pipeline = DetectionPipeline(
        model_id=model_id,
        confidence=confidence,
        fps_sample=infer_fps,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        cache_dir=cache_dir,
        combined_jsonl=target_settings.DETECTIONS_JSONL,
        save_frame_json=target_settings.SAVE_FRAME_JSON,
        cache_only=args.cache_only,
        pred_extractor=target_config.extractor,
    )
    pipeline.run()

if __name__ == "__main__":
    main()
