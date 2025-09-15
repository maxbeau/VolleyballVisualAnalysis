from typing import Dict, Any, Tuple


def to_tlbr_from_xywh(x: float, y: float, w: float, h: float) -> Tuple[int, int, int, int]:
    x1 = int(round(x - w / 2))
    y1 = int(round(y - h / 2))
    x2 = int(round(x + w / 2))
    y2 = int(round(y + h / 2))
    return x1, y1, x2, y2


def has_box(p: Dict[str, Any]) -> bool:
    return all(k in p for k in ("x", "y", "width", "height"))


def action_color(cls: str) -> Tuple[int, int, int]:
    # OpenCV BGR palette mapping for common volleyball actions
    cls_l = (cls or "").strip().lower()
    palette = {
        "serve": (0, 165, 255),
        "pass": (0, 255, 255),
        "bump": (0, 255, 255),
        "set": (255, 0, 255),
        "spike": (0, 255, 0),
        "attack": (0, 255, 0),
        "block": (255, 0, 0),
        "dig": (255, 255, 0),
        "celebrate": (180, 105, 255),
        "idle": (200, 200, 200),
        "standby": (200, 200, 200),
    }
    if cls_l in palette:
        return palette[cls_l]
    # Deterministic fallback via hashed hue
    import hashlib, colorsys
    h = int(hashlib.md5(cls_l.encode("utf-8")).hexdigest(), 16) % 360
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.8, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))

