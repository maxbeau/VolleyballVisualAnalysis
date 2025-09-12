from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


Point = Tuple[float, float]


@dataclass
class CourtFrame:
    frame: int
    corners: List[Point]
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CourtMeta:
    tracking_jsonl: str
    orientation: str  # "horizontal" | "vertical"
    extra: Optional[Dict[str, Any]] = None


__all__ = ["CourtFrame", "CourtMeta", "Point"]

