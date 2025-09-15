import json
from typing import Dict, List, Any


def load_players_tracks_by_frame(jsonl_path: str) -> Dict[int, List[dict]]:
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
                trs = rec.get("tracks")
                if fi >= 0 and isinstance(trs, list):
                    out[fi] = trs
    except Exception:
        return {}
    return out

__all__ = ["load_players_tracks_by_frame"]

