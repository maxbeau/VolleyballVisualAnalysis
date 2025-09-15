from dotenv import load_dotenv

# Load environment variables from .env file at the beginning
load_dotenv()

from .common import CommonSettings
from .actions import ActionsSettings
from .ball import BallSettings
from .court import CourtSettings
from .players import PlayersSettings
from .teams import TeamsSettings

class Settings:
    """Aggregates all module-specific settings after loading .env."""
    def __init__(self):
        self.common = CommonSettings()
        self.actions = ActionsSettings()
        self.ball = BallSettings()
        self.court = CourtSettings()
        self.players = PlayersSettings()
        self.teams = TeamsSettings()

# Create a single, project-wide instance of the settings
settings = Settings()

__all__ = ["settings", "Settings"]
