# Canvas: core_config_v2
# Path: app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Database Config
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "postgres"
    POSTGRES_PORT: int = 5432

    # Connection Pooling
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # App Config
    APP_NAME: str = "Onyx"
    ENV: str = "development"

    # --- AI Model Config ---

    # 1. Voice Activity Detection (MarbleNet)
    VAD_MODEL_PATH: str = "models/vad_multilingual_marblenet.nemo"
    VAD_VRAM_REQUIREMENT_MB: float = 1200.0

    # 2. Transcription (Faster-Whisper)
    WHISPER_MODEL_PATH: str = "models/faster-whisper-large-v3"
    WHISPER_VRAM_REQUIREMENT_MB: float = 3500.0  # ~3.5GB for INT8 Large-v3

    # 3. Speaker Diarization (TitaNet)
    DIARIZATION_MODEL_PATH: str = "models/titanet_large.nemo"
    DIARIZATION_VRAM_REQUIREMENT_MB: float = 500.0

    # 4. Semantic NLP Router (HuggingFace Transformers FP16)
    NLP_MODEL_PATH: str = "microsoft/Phi-3.5-mini-instruct"
    NLP_VRAM_REQUIREMENT_MB: float = 8500.0  # ~8.5GB VRAM requirement for unquantized Phi-3.5

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )


settings = Settings()