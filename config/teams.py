from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class TeamsSettings(BaseSettings):
    """Team names and simple HUD preferences."""
    model_config = SettingsConfigDict(env_prefix='TEAMS_', env_file='.env', env_file_encoding='utf-8', extra='ignore')

    TEAM_A_NAME: str = "TeamA"
    TEAM_B_NAME: str = "TeamB"

    # If set to 'left' or 'right', fixes which bird's-eye half is Team A.
    # Otherwise 'auto' assigns Team A to the first observed server's half.
    TEAM_A_SIDE: str = "auto"  # 'auto' | 'left' | 'right'

    # Minimal confidence to consider an action for HUD logic
    ACTION_MIN_CONF: float = 0.25

    # Frames cooldown to treat two serves as separate rallies
    SERVE_COOLDOWN_FRAMES: int = 20

    # How long to flash a score message when a rally outcome is inferred
    SCORE_FLASH_FRAMES: int = 24

    # Binding strategy for side->team mapping
    # 'earliest': use the earliest serve clip's actor_side as Team A side
    # 'majority': unweighted majority over early serves
    # 'weighted_majority': weighted by side_conf
    # 'windowed': per-clip local voting within +/- window frames (falls back to 'earliest')
    BIND_STRATEGY: str = "earliest"
    BIND_WINDOW_FRAMES: int = 240

    # Rally anchoring options
    BIND_BLOCK_TO_SERVE: bool = True
    BIND_RALLY_MAX_GAP_FRAMES: int = 600
    BIND_SET_TO_RECEIVE: bool = True
    # Block attribution heuristics
    BIND_BLOCK_OPPOSE_SPIKE: bool = True
    BIND_BLOCK_OPPOSE_SPIKE_WINDOW_FRAMES: int = 24
    BIND_BLOCK_OPPOSE_SET: bool = True
    BIND_BLOCK_OPPOSE_SET_WINDOW_FRAMES: int = 48

__all__ = ["TeamsSettings"]
