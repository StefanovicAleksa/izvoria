# Canvas: core_config_v10
# Path: app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================================
    # DATABASE CONFIGURATION
    # ==========================================
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "onyx_v2_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"

    # Connection Pooling
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # ==========================================
    # APP SETTINGS
    # ==========================================
    APP_NAME: str = "Onyx AI Appliance"
    ENV: str = "development"

    # ==========================================
    # APPLIANCE DOMAIN ADAPTATION
    # ==========================================
    APPLIANCE_INDUSTRY: str = "GENERAL"
    APPLIANCE_LANGUAGE: str = "en"

    # GPU Models (Discovery Phase / Domain Adaptation)
    DISCOVERY_SCOUT_MODEL_PATH: str = "models/whisper-tiny-en"
    DISCOVERY_SCOUT_VRAM_REQUIREMENT_MB: float = 200.0

    DOMAIN_CLASSIFIER_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    DOMAIN_CLASSIFIER_MODEL_VRAM_REQUIREMENT_MB: float = 3500

    # ==========================================
    # GPU MODELS (Pipeline Phases)
    # ==========================================
    VAD_MODEL_PATH: str = "models/vad_multilingual_marblenet.nemo"
    VAD_VRAM_REQUIREMENT_MB: float = 1200.0

    WHISPER_MODEL_PATH: str = "models/faster-whisper-large-v3"
    WHISPER_VRAM_REQUIREMENT_MB: float = 3500.0

    DIARIZATION_MODEL_PATH: str = "models/titanet_large.nemo"
    DIARIZATION_VRAM_REQUIREMENT_MB: float = 500.0

    NLP_MODEL_PATH: str = "models/Phi-3.5-mini-instruct"
    NLP_VRAM_REQUIREMENT_MB: float = 8000.0

    PUNCTUATION_MODEL_PATH: str = "models/fullstop-punctuation-multilingual-base"
    PUNCTUATION_VRAM_REQUIREMENT_MB: float = 500.0

    ASR_PHASE_1_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    ASR_PHASE_1_VRAM_REQUIREMENT_MB: float = 3500.0

    ASR_PHASE_2_MODEL_PATH: str = "models/flanec-base-cd"
    ASR_PHASE_2_VRAM_REQUIREMENT_MB: float = 2000.0

    ASR_PHASE_3_MODEL_PATH: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    ASR_PHASE_3_VRAM_REQUIREMENT_MB: float = 5000.0

    # Phase 10: Vision Analysis
    VISION_MODEL_PATH: str = "models/llava-1.5-7b/ggml-model-q4_k.gguf"
    VISION_MMPROJ_PATH: str = "models/llava-1.5-7b/mmproj-model-f16.gguf"
    # 4.5GB weights + ~2GB multimodal context allocation buffer
    VISION_VRAM_REQUIREMENT_MB: float = 6500.0

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Instructs Pydantic to read from the .env file and ignore extra variables
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )


settings = Settings()