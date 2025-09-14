from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file at the beginning
load_dotenv()

from .common import CommonSettings
from .actions import ActionsSettings
from .ball import BallSettings
from .court import CourtSettings
from .players import PlayersSettings

class Settings(BaseModel):
    """
    The main settings object, aggregating all module-specific settings.
    """
    common: CommonSettings = CommonSettings()
    actions: ActionsSettings = ActionsSettings()
    ball: BallSettings = BallSettings()
    court: CourtSettings = CourtSettings()
    players: PlayersSettings = PlayersSettings()

# Create a single, project-wide instance of the settings
settings = Settings()

__all__ = ["settings", "Settings"]