import os
import sys
import argparse

# Ensure project root in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.detect import infer_actions_on_video
from core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Run actions detection (Roboflow volleyball-actions/4)")
    parser.add_argument("--model-id", default=os.getenv("ACTIONS_MODEL_ID", "volleyball-actions/4"))
    parser.add_argument("--confidence", type=float, default=settings.OVERLAY_MIN_CONF)
    parser.add_argument("--infer-fps", type=float, default=settings.ACTIONS_INFER_FPS)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    infer_actions_on_video(
        model_id=args.model_id,
        confidence=args.confidence,
        fps_sample=args.infer_fps,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        cache_dir=settings.ACTIONS_CACHE_DIR,
        combined_jsonl=settings.ACTIONS_DETECTIONS_JSONL,
        save_frame_json=settings.ACTIONS_SAVE_FRAME_JSON,
        cache_only=args.cache_only,
    )


if __name__ == "__main__":
    main()

