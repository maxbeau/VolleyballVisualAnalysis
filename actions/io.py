import json
from typing import Dict, List, Any


def load_actions_by_frame(jsonl_path: str) -> Dict[int, List[dict]]:
    """Load actions detections JSONL into a frame->list[pred] dictionary.
    Returns empty dict if file missing or parse error occurs.
    """
    out: Dict[int, List[dict]] = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                fi = int(rec.get("frame", -1))
                preds = rec.get("predictions")
                if fi >= 0 and isinstance(preds, list) and preds:
                    out.setdefault(fi, []).extend(preds)
    except Exception:
        return {}
    return out

__all__ = ["load_actions_by_frame"]


def load_action_clips(jsonl_path: str) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "class" in rec and "start" in rec and "end" in rec:
                    clips.append(rec)
    except Exception:
        return []
    return clips

__all__ += ["load_action_clips"]
