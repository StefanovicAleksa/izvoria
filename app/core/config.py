# Canvas: core_config_v4
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
    WHISPER_VRAM_REQUIREMENT_MB: float = 3500.0

    # 3. Speaker Diarization (TitaNet)
    DIARIZATION_MODEL_PATH: str = "models/titanet_large.nemo"
    DIARIZATION_VRAM_REQUIREMENT_MB: float = 500.0

    # 4. Semantic NLP Router (llama.cpp GGUF)
    NLP_MODEL_PATH: str = "models/Phi-3.5-mini-instruct"
    NLP_VRAM_REQUIREMENT_MB: float = 8000.0

    # 5. Sentence Reconstruction (wtpsplit / SaT)
    SAT_MODEL_NAME: str = "sat-3l-sm"
    SAT_VRAM_REQUIREMENT_MB: float = 750.0

    # --- ASR Error Correction Models ---

    # Phase 1: Overarching Glossary Extraction (Llama 3.2 3B Q8_0 GGUF)
    ASR_PHASE_1_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    ASR_PHASE_1_VRAM_REQUIREMENT_MB: float = 3500.0

    # Phase 2: Anomaly Detection (T5-ASR-EC Base)
    ASR_PHASE_2_MODEL_PATH: str = "models/flanec-base-cd"
    ASR_PHASE_2_VRAM_REQUIREMENT_MB: float = 2000.0

    # Phase 3: Grounded Correction (Qwen 2.5 7B Q4_K_M GGUF)
    ASR_PHASE_3_MODEL_PATH: str = "models/qwen2.5-7b-instruct-q4_k_m.gguf"
    ASR_PHASE_3_VRAM_REQUIREMENT_MB: float = 5000.0

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()