"""Simplified feature flags for FastVid compression in InternNav."""

from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class FastVidConfig:
    """Configuration for FastVid video token compression."""

    enabled: bool = False
    retention_ratio: float = 0.5
    dyseg_c: int = 8
    dyseg_tau: float = 0.9
    stprune_d: float = 0.4
    dtm_p: int = 4
    dtm_beta: float = 0.6
    score_type: str = "attn_proxy"
    min_tokens_per_frame: int = 4  # Safety floor to avoid over-compression

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FastVidConfig":
        cfg = cls()
        for key, value in data.items():
            if hasattr(cfg, key):
                field_type = type(getattr(cfg, key))
                if field_type == bool:
                    setattr(cfg, key, str(value).strip().lower() in {"1", "true", "yes", "on"})
                elif field_type == int:
                    setattr(cfg, key, int(value))
                elif field_type == float:
                    setattr(cfg, key, float(value))
                else:
                    setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
