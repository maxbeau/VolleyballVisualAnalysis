from pydantic_settings import BaseSettings, SettingsConfigDict

class ActionsSettings(BaseSettings):
    """
    Configuration specific to actions detection.
    """
    model_config = SettingsConfigDict(env_prefix='ACTIONS_', env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Detection
    DETECTIONS_JSONL: str = "outputs/actions_detections.jsonl"
    CACHE_DIR: str = "outputs/actions_preds"
    SAVE_FRAME_JSON: bool = True
    INFER_FPS: int = 12
    MODEL_ID: str = "volleyball-actions/4"

    # Overlay
    OVERLAY_FULL: str = "outputs/actions_overlay.mp4"