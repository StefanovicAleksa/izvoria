# File: app/core/config.py
# Canvas: config_v3

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
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

    # Overrides the glossary directory. Integration tests point it at a throwaway directory so seeding test
    # glossaries never overwrites the real ones in data/glossaries.
    GLOSSARY_PATH: str = ""

    @property
    def GLOSSARY_DIR(self) -> Path:
        """Centralized location for industry-specific knowledge bases."""
        path = Path(self.GLOSSARY_PATH) if self.GLOSSARY_PATH else self.BASE_DIR / "data" / "glossaries"
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
    # Sub-domains the router scores at least this high (0-10) feed Whisper hotwords and the phonetic dictionary
    DOMAIN_ROUTER_MIN_SCORE: int = 5
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
    # Windows scored per MarbleNet forward pass, and silence padding added around each speech segment
    VAD_BATCH_SIZE: int = 512
    VAD_PAD_SEC: float = 0.3

    WHISPER_MODEL_PATH: str = "models/faster-whisper-large-v3"
    WHISPER_VRAM_REQUIREMENT_MB: float = 3500.0
    # Clips decoded in parallel by faster-whisper's BatchedInferencePipeline (1 = sequential only)
    WHISPER_BATCH_SIZE: int = 8
    # Spans shorter than this are widened (not skipped) so isolated one-word answers are transcribed
    WHISPER_MIN_SLICE_SEC: float = 1.0
    # Whisper repeats hotword/prompt text as if spoken: runs of at least this many consecutive injected words are removed
    WHISPER_PROMPT_ECHO_MIN_TOKENS: int = 5

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
    # Below this many dictionary terms there is nothing to correct against; the pass is skipped.
    PHONETIC_MIN_DICTIONARY_TERMS: int = 1
    # A heard span becomes a candidate for a dictionary term only if its letters or sound key are at least this similar
    PHONETIC_CANDIDATE_MIN_SIMILARITY: float = 0.7
    # Allowed letter-length ratio between the heard span and the term
    PHONETIC_LENGTH_RATIO_MIN: float = 0.6
    PHONETIC_LENGTH_RATIO_MAX: float = 1.67
    # A span whose Whisper words were all heard at least this confidently is never a candidate
    PHONETIC_MAX_WORD_PROBABILITY: float = 0.95

    CONTEXTUAL_RETRIEVAL_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    CONTEXTUAL_RETRIEVAL_VRAM_REQUIREMENT_MB: float = 3500.0
    # Context window of the contextual-retrieval SLM. Transcripts longer than
    # (N_CTX - RESERVED_TOKENS) are situated within overlapping document sections instead.
    CONTEXTUAL_RETRIEVAL_N_CTX: int = 32768
    CONTEXTUAL_RETRIEVAL_RESERVED_TOKENS: int = 1500

    VISUAL_CUE_DETECTION_MODEL_PATH: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    VISUAL_CUE_DETECTION_VRAM_REQUIREMENT_MB: float = 5000.0

    # Native Hugging Face directory, pre-quantized to 4-bit (bitsandbytes/nf4) and
    # saved back to disk - NOT GGUF. llama.cpp's mtmd vision encoder has a confirmed
    # unresolved CUDA crash specific to the Qwen-VL model family (see
    # app/features/visual_analysis/data/vlm_adapter.py for details/upstream links).
    # Pre-quantizing on disk (rather than quantizing at load time from full-precision
    # weights) keeps this directory's size representative of real VRAM usage, which
    # the Model Orchestrator's admission control relies on.
    VISION_MODEL_PATH: str = "models/qwen2-vl-7b-hf-bnb4bit"
    VISION_VRAM_REQUIREMENT_MB: float = 7500.0
    # A chunk's frames are described in multi-image calls of at most this many frames
    VISION_MAX_IMAGES_PER_CALL: int = 4
    # Per-frame pixel cap inside a multi-image call (a lone frame keeps the adapter's larger cap)
    VISION_MULTI_IMAGE_MAX_PIXELS: int = 768 * 28 * 28
    # Generation budget per frame (structured fields: what is shown, legible text, marks, speech link); a 4-frame
    # call may generate up to 4x this
    VISION_MAX_NEW_TOKENS_PER_FRAME: int = 200

    # Frame extraction: Laplacian-variance sharpness floor and the per-chunk cap on visually distinct frames
    FRAME_BLUR_THRESHOLD: float = 100.0
    FRAME_MAX_PER_CHUNK: int = 4
    # Chunks extracted concurrently, each with one FFmpeg process; 0 means max(2, cpu_count // 2)
    FRAME_EXTRACTION_WORKERS: int = 0
    FRAME_FFMPEG_THREADS: int = 2

    VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    VISUAL_HALLUCINATION_CORRECTION_VRAM_REQUIREMENT_MB: float = 2500.0
    # Frames whose quoted on-screen text shares this many topic words (or one specific number such as "7-2") with the
    # chunk are accepted without asking the text-only checker model, which rejected matching slides it couldn't see
    VISUAL_CHECK_MIN_SHARED_TOPIC_WORDS: int = 2

    VISUAL_CONTEXT_SYNTHESIS_MODEL_PATH: str = "models/llama-3.2-3b-instruct-q8_0.gguf"
    VISUAL_CONTEXT_SYNTHESIS_VRAM_REQUIREMENT_MB: float = 3500.0

    RAG_EMBEDDING_MODEL_PATH: str = "models/bge-m3"
    RAG_EMBEDDING_VRAM_REQUIREMENT_MB: float = 2500.0
    RAG_RRF_K_CONSTANT: int = 60
    # Size of the RRF-fused candidate pool handed to the reranker
    RAG_RETRIEVAL_CANDIDATES: int = 40
    # Postgres text search configuration for the sparse (FTS) leg. Used in DDL, so restricted to identifiers.
    FTS_LANGUAGE_CONFIG: str = Field(default="english", pattern=r"^[a-z_]+$")

    RAG_RERANKER_MODEL_PATH: str = "models/bge-reranker-v2-m3"
    RAG_RERANKER_VRAM_REQUIREMENT_MB: float = 1500.0
    # Cross-encoder score floor (sigmoid 0..1): candidates below it never reach the (slow) scrubber; the best
    # candidate is always kept, and unset keeps every top-N candidate. On the dev-DB eval every chunk that answered
    # a question scored >= 0.12, while about 45% of non-answering candidates scored below 0.1.
    RAG_RERANK_MIN_SCORE: Optional[float] = 0.1
    # Chat pipelines add the chunks before and after this many of the best surviving chunks (0 disables)
    RAG_NEIGHBOR_EXPANSION_TOP_K: int = 3

    # Qwen2.5-7B (same weights as the reasoner): on the dev-DB eval (scripts/eval/rag_eval.py) the 3B judge kept the
    # chunk that answers the question for 75% of questions, the 7B for 89%
    RETRIEVAL_SCRUBBER_MODEL_PATH: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    RETRIEVAL_SCRUBBER_VRAM_REQUIREMENT_MB: float = 5000.0

    QA_GENERATION_MODEL_PATH: str = "models/google_gemma-4-E4B-it-Q4_K_M.gguf"
    QA_GENERATION_VRAM_REQUIREMENT_MB: float = 4500.0
    # Context window of the answer model. Retrieved evidence is packed to fit what's left after the prompts,
    # the question and the answer/critique generation budget, so no chunk count can overflow it.
    # 16K measured at ~4.6 GB VRAM on the dev box (8K: ~4.1 GB).
    QA_GENERATION_N_CTX: int = 16384
    QA_GENERATION_MAX_ANSWER_TOKENS: int = 1024

    # Deep Research reasoning loop: reuses the same Qwen2.5-7B weights as ASR_PHASE_3/VISUAL_CUE_DETECTION,
    # but under its own ModelType/n_ctx since residency is keyed on (model_type, model_path).
    QA_REASONER_MODEL_PATH: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    QA_REASONER_VRAM_REQUIREMENT_MB: float = 5000.0
    # Context window of the reasoner; its evidence block is cut to fit alongside the prompt and a 1000-token reply
    QA_REASONER_N_CTX: int = 8192
    DEEP_RESEARCH_MAX_ITERATIONS: int = 3
    DEEP_RESEARCH_MAX_CRITIQUE_RETRIES: int = 1

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_ignore_empty=True,
        extra="ignore"
    )


settings = Settings()