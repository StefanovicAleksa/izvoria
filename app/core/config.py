# File: app/core/config.py
# Canvas: config_v3

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ENV = os.environ.get("ENV", "development")

if APP_ENV == "test":
    ENV_FILE = ".env.test"
elif APP_ENV == "production":
    ENV_FILE = ".env.prod"
else:
    ENV_FILE = ".env"

class Settings(BaseSettings):
    # ==========================================
    # ENVIRONMENT SETTINGS
    # ==========================================
    # Options: development, test, production
    ENV: str = APP_ENV

    # Project Root: app/core/config.py -> parent(core) -> parent(app) -> parent(root)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # ==========================================
    # DATABASE CONFIGURATION
    # ==========================================
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "onyx_dev_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # ==========================================
    # STORAGE ISOLATION
    # ==========================================
    DATA_DIR: str = "onyx_data"

    @property
    def VAULT_PATH(self) -> str:
        # onyx_data/development/vault or onyx_data/test/vault
        return os.path.join(self.DATA_DIR, self.ENV, "vault")

    @property
    def GLOSSARY_DIR(self) -> Path:
        """Centralized location for industry-specific knowledge bases."""
        path = self.BASE_DIR / "data" / "glossaries"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ==========================================
    # PERFORMANCE TUNING
    # ==========================================
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # ==========================================
    # APPLIANCE DOMAIN ADAPTATION
    # ==========================================
    APPLIANCE_INDUSTRY: str = "GENERAL"
    APPLIANCE_LANGUAGE: str = "en"

    # ==========================================
    # MODEL PATHS & VRAM REQUIREMENTS
    # ==========================================
    DISCOVERY_SCOUT_MODEL_PATH: str = "models/whisper-tiny-en"
    DISCOVERY_SCOUT_VRAM_REQUIREMENT_MB: float = 200.0

    DOMAIN_CLASSIFIER_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    DOMAIN_CLASSIFIER_MODEL_VRAM_REQUIREMENT_MB: float = 3500.0

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

    HALLUCINATION_SCRUBBER_MODEL_PATH: str = "models/deberta-v3-base-zeroshot"
    HALLUCINATION_SCRUBBER_VRAM_REQUIREMENT_MB: float = 800.0

    DOMAIN_PHONETIC_CORRECTION_MODEL_PATH: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    DOMAIN_PHONETIC_CORRECTION_VRAM_REQUIREMENT_MB: float = 2500.0

    CONTEXTUAL_RETRIEVAL_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    CONTEXTUAL_RETRIEVAL_VRAM_REQUIREMENT_MB: float = 3500.0

    VISUAL_CUE_DETECTION_MODEL_PATH: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    VISUAL_CUE_DETECTION_VRAM_REQUIREMENT_MB: float = 5000.0

    VISION_MODEL_PATH: str = "models/qwen2-vl-7b/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
    VISION_MMPROJ_PATH: str = "models/qwen2-vl-7b/Qwen2-VL-7B-Instruct-vision-encoder.gguf"
    VISION_VRAM_REQUIREMENT_MB: float = 7500.0

    VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    VISUAL_HALLUCINATION_CORRECTION_VRAM_REQUIREMENT_MB: float = 2500.0

    VISUAL_CONTEXT_SYNTHESIS_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    VISUAL_CONTEXT_SYNTHESIS_VRAM_REQUIREMENT_MB: float = 3500.0

    RAG_EMBEDDING_MODEL_PATH: str = "models/bge-m3"
    RAG_EMBEDDING_VRAM_REQUIREMENT_MB: float = 2500.0
    RAG_RRF_K_CONSTANT: int = 60

    RETRIEVAL_SCRUBBER_MODEL_PATH: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    RETRIEVAL_SCRUBBER_VRAM_REQUIREMENT_MB: float = 2500.0

    QA_GENERATION_MODEL_PATH: str = "models/google_gemma-4-E4B-it-Q4_K_M.gguf"
    QA_GENERATION_VRAM_REQUIREMENT_MB: float = 4500.0

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_ignore_empty=True,
        extra="ignore"
    )


settings = Settings()