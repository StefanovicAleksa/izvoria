# File: app/core/config.py
# Canvas: config_v3

import logging
import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Annotated, FrozenSet, List, Optional
from pydantic import Field, PrivateAttr, TypeAdapter, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

APP_ENV = os.environ.get("ENV", "development")


@lru_cache(maxsize=8)
def _pack_fts_configs(packs_dir: Path) -> tuple:
    """Each pack's text search configuration, deduplicated in code order. Read once: search asks on every query."""
    configs = []
    for path in sorted(packs_dir.glob("*.toml")):
        with open(path, "rb") as f:
            config = tomllib.load(f).get("search", {}).get("fts_config", "english")
        if config not in configs:
            configs.append(config)
    return tuple(configs)


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
        # The driver is named: SQLAlchemy 2.1 made psycopg (3) the default for a bare postgresql://, and the
        # appliance installs psycopg2. A fresh install picks up 2.1, so without it nothing connects.
        return f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

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
        """
        Centralized location for industry-specific knowledge bases, one directory per language
        (data/glossaries/<APPLIANCE_LANGUAGE>/): a glossary's terms are words of one language.
        """
        base = Path(self.GLOSSARY_PATH) if self.GLOSSARY_PATH else self.BASE_DIR / "data" / "glossaries"
        path = base / self.APPLIANCE_LANGUAGE
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Overrides where finished research reports are written. Integration tests point it at a tmp_path.
    REPORTS_PATH: str = ""

    @property
    def REPORTS_DIR(self) -> Path:
        """Where Deep Research Report mode writes its Markdown deliverables."""
        path = Path(self.REPORTS_PATH) if self.REPORTS_PATH else self.BASE_DIR / self.DATA_DIR / self.ENV / "reports"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Whether a finished report is also written to REPORTS_DIR as Markdown. The hosted GPU box turns it off: it
    # keeps no client data on disk, and its web API renders every download from the database row.
    REPORT_FILES_ENABLED: bool = True

    # What the vault keeps after a recording is indexed. The defaults are the appliance's; the hosted web version
    # turns all three around to store one compact file per recording (docs/web_deploy.md, "Storage").
    # A per-chunk clip repeats its stretch of the recording; without one a citation is cut from the original.
    VIDEO_CLIPS_ENABLED: bool = True
    # The 16 kHz WAV only the audio stages read; dropped once its source is indexed, re-extracted if ever needed.
    KEEP_EXTRACTED_AUDIO: bool = True
    # Replace an indexed recording's original with a smaller copy a browser plays as it is (H.264/AAC MP4, AAC
    # m4a). Ingestion always reads the upload itself; the copy is only what is kept and played back.
    COMPACT_MEDIA_ORIGINALS: bool = False
    COMPACT_VIDEO_MAX_HEIGHT: int = 720
    COMPACT_VIDEO_CRF: int = 26
    COMPACT_AUDIO_BITRATE_KBPS: int = 64
    # The original stays unless the copy is at least this much smaller
    COMPACT_MIN_SAVING: float = 0.2

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

    # ==========================================
    # LANGUAGE
    # ==========================================
    # The ACTIVE language: the one the next ingestion batch is transcribed and indexed in, and the one answers
    # are written in. Switched at runtime with `use_language()`; the index holds every language. Every
    # language-specific rule lives in its pack, data/languages/<code>.toml (see app/core/language).
    APPLIANCE_LANGUAGE: str = "en"
    # Overrides the language pack directory (tests point it at fixture packs).
    LANGUAGE_PACKS_PATH: str = ""
    # Start in the language last switched to (ACTIVE_LANGUAGE_FILE) when APPLIANCE_LANGUAGE is not set
    # explicitly. The root conftest.py turns it off, so a developer's remembered language never reaches a test.
    REMEMBER_LANGUAGE: bool = True

    @property
    def LANGUAGE_PACKS_DIR(self) -> Path:
        return Path(self.LANGUAGE_PACKS_PATH) if self.LANGUAGE_PACKS_PATH else self.BASE_DIR / "data" / "languages"

    @property
    def ACTIVE_LANGUAGE_FILE(self) -> Path:
        """Where the last active language is remembered, so the app starts in the language it was left in."""
        return self.BASE_DIR / self.DATA_DIR / self.ENV / "active_language"

    # ==========================================
    # MODEL RESIDENCY
    # ==========================================
    # Models of these types stay in VRAM when their holder releases them, and are evicted (the fewest megabytes that
    # make room) only when a later load needs the room, so a chat turn or report round reuses them instead of reloading.
    # Measured 2026-10-03 on the 12 GB dev card: BGE-M3 1090 MB + reranker 1088 MB + Qwen3.5-9B at 128K q8_0
    # 7788 MB, with ~1.1 GB left free. Ingestion still evicts everything. False restores the single-slot pool.
    MODEL_KEEP_WARM: bool = True
    MODEL_KEEP_WARM_TYPES: List[str] = ["rag_embedding", "rag_reranking", "qa_generation", "report_authoring"]

    # ==========================================
    # HOSTED / MULTI-USER SERVING (web_deploy/)
    # ==========================================
    # Every default below keeps the single-user appliance exactly as it was. The hosted web API turns them on so
    # several users can chat, search and run reports at once against models that never leave VRAM.
    #
    # Where the Qwen chat adapters (answer, reasoner, report writer, retrieval scrubber) run: "llama_cpp" loads the
    # GGUF in this process; "llama_server" talks to a local llama.cpp `llama-server` that holds the weights once and
    # decodes several requests in parallel slots. Ingestion always runs in-process.
    LLM_BACKEND: str = "llama_cpp"
    LLM_SERVER_URL: str = "http://127.0.0.1:8081"
    # Tokens the server's KV cache holds across all slots; requests are admitted against it so a full cache
    # never fails a request mid-decode. 0 reads it from the server (/props) on first use.
    LLM_SERVER_KV_TOKENS: int = 0
    # One request may run this long (a report section, a 100K-token prefill).
    LLM_SERVER_TIMEOUT_S: float = 1800.0
    # Loaded once (Orchestrator.warm_up) and never released or evicted: the live path's PyTorch models.
    MODEL_PINNED_TYPES: List[str] = []
    # Served outside this process (LLM_BACKEND=llama_server): handed out without budgeting, never unloaded.
    MODEL_EXTERNAL_TYPES: List[str] = []
    # Live calls run concurrently instead of being gated on the engine state; requires every live model type to be
    # pinned or external (the orchestrator refuses any other live acquire).
    ENGINE_CONCURRENT_LIVE: bool = False
    # Research reports allowed at once when ENGINE_CONCURRENT_LIVE is on (the desktop runs one, exclusively).
    REPORT_MAX_CONCURRENT: int = 1
    # Chunks the retrieval scrubber judges at once; > 1 only pays off against llama-server's parallel slots.
    RETRIEVAL_SCRUBBER_CONCURRENCY: int = 1
    # VRAM the orchestrator treats as taken by someone else (the hosted ingestion worker leaves the live models'
    # share of the card alone).
    VRAM_HEADROOM_MB: float = 0.0

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

    # --- Document ingestion (PDF / DOCX) ---
    # RapidOCR PP-OCRv5 models, fetched once by scripts/download_ocr_models.py. OCR runs on the CPU; when these
    # files are missing, scanned pages are reported as unreadable rather than downloaded at runtime.
    DOC_OCR_MODEL_DIR: str = "models/rapidocr"
    # Scanned pages are rendered at this resolution before OCR
    DOC_OCR_DPI: int = 300
    # A PDF page is treated as scanned (and OCR'd) only when BOTH hold: its text layer has fewer letters than
    # this, and images cover at least DOC_SCANNED_IMAGE_COVERAGE of it. A sparse born-digital page (a heading and
    # a table) keeps its exact text layer instead of being replaced by OCR's reading of it.
    DOC_MIN_TEXT_LETTERS_PER_PAGE: int = 40
    DOC_SCANNED_IMAGE_COVERAGE: float = 0.5
    # Chunk budget for documents; a chunk never crosses a top-level clause
    DOC_CHUNK_MAX_WORDS: int = 250
    # A document shorter than this (a receipt, a one-page letter) is indexed as one chunk
    DOC_SHORT_DOCUMENT_WORDS: int = 400

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

    # --- Image ingestion (standalone stills, described by the VLM above) ---
    # A still gets a sentence or two of description and up to 20 legible text items, so more room than a frame
    IMAGE_DESCRIPTION_MAX_NEW_TOKENS: int = 500
    # OCR reads the image with its long side capped at this: a 48 MP phone photo would otherwise take minutes on the
    # CPU, and text a person can read survives the downscale
    IMAGE_OCR_MAX_SIDE: int = 4000
    # A text item the VLM reports is kept only when OCR read at least this share of its words (and every number)
    IMAGE_TEXT_MIN_OCR_COVERAGE: float = 0.8
    # The reverse check. OCR never invents a line, but on handwriting it writes gibberish ("Analbeig aftea long
    # apok"), which would be indexed as the image's text. An OCR line below this confidence is kept only if the
    # vision model read the same words. Measured on real images: misreads sit below 0.7, printed column headers
    # ("Hartf d", "A. M.") at 0.7-0.8, stylised sign lettering at 0.65-0.76 (kept, the vision model reads it too).
    IMAGE_OCR_MIN_LINE_CONFIDENCE: float = 0.7
    # When at least this share of an image's lines (and IMAGE_OCR_UNRELIABLE_MIN_LINES of them) fall below the floor
    # above, the image is handwriting OCR can't read (22 of 26 lines of an 1839 letter), and every line below
    # IMAGE_OCR_UNRELIABLE_LINE_CONFIDENCE needs the vision model's confirmation.
    IMAGE_OCR_UNRELIABLE_SHARE: float = 0.5
    IMAGE_OCR_UNRELIABLE_MIN_LINES: int = 5
    IMAGE_OCR_UNRELIABLE_LINE_CONFIDENCE: float = 0.9

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
    # The dense leg's best this many always join that pool, however they fuse: asked in another language than the
    # material, the keyword leg matches only the question's language, and RRF then buries the right chunk
    RAG_DENSE_CANDIDATES_KEPT: int = 10
    # Postgres text search configuration for the sparse (FTS) leg. Used in DDL, so restricted to identifiers.
    # Taken from the language pack's [search] table unless set explicitly.
    FTS_LANGUAGE_CONFIG: str = Field(default="english", pattern=r"^[a-z_]+$")

    RAG_RERANKER_MODEL_PATH: str = "models/bge-reranker-v2-m3"
    RAG_RERANKER_VRAM_REQUIREMENT_MB: float = 1500.0
    # Cross-encoder score floor (sigmoid 0..1): candidates below it never reach the (slow) scrubber; the best
    # candidate is always kept, and unset keeps every top-N candidate. On the dev-DB eval every chunk that answered
    # a question scored >= 0.12, while about 45% of non-answering candidates scored below 0.1.
    RAG_RERANK_MIN_SCORE: Optional[float] = 0.1
    # Chat pipelines add the chunks before and after this many of the best surviving chunks (0 disables)
    RAG_NEIGHBOR_EXPANSION_TOP_K: int = 3

    # Qwen2.5-7B: on the dev-DB eval (scripts/eval/rag_eval.py) the 3B judge kept the
    # chunk that answers the question for 75% of questions, the 7B for 89%
    RETRIEVAL_SCRUBBER_MODEL_PATH: str = "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    # A chunk the cross-encoder scores at or above this is kept without asking the scrubber. The 7B judge rejected
    # chunks the reranker ranked first for reasons like "the sign says 'STOP' and not 'GIMAr MOntaz MAutino'" or
    # "does not provide the specific year". Measured 2026-09-28 over 62 answerable and 21 unanswerable questions
    # (en/sl/sr benchmark lectures, a book, images): the best candidate for an unanswerable question never scored
    # above 0.01, the relevant one for an answerable question never below 0.29. None judges every chunk again.
    RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE: Optional[float] = 0.5
    RETRIEVAL_SCRUBBER_VRAM_REQUIREMENT_MB: float = 5000.0

    # Qwen3.5-9B: hybrid Gated DeltaNet / Gated Attention, so only 8 of 32 layers keep a KV cache (~32 KiB/token
    # at f16, ~17 KiB at q8_0). 128K context fits a 12 GB card next to the ~5.7 GB Q4_K_M weights.
    QA_GENERATION_MODEL_PATH: str = "models/Qwen3.5-9B-Q4_K_M.gguf"
    QA_GENERATION_VRAM_REQUIREMENT_MB: float = 10000.0
    # Context window of the answer model. Retrieved evidence and the conversation history are packed to fit what's
    # left after the prompts, the question and the answer/critique generation budget, so nothing can overflow it.
    QA_GENERATION_N_CTX: int = 131072
    QA_GENERATION_MAX_ANSWER_TOKENS: int = 1024

    # Reasoned Answer mode's reasoning loop: the same weights as the answer model, but under its own ModelType/n_ctx
    # since residency is keyed on (model_type, model_path) — switching between the two reloads the file.
    QA_REASONER_MODEL_PATH: str = "models/Qwen3.5-9B-Q4_K_M.gguf"
    QA_REASONER_VRAM_REQUIREMENT_MB: float = 10000.0
    # Context window of the reasoner; its evidence block is cut to fit alongside the prompt and a 1000-token reply
    QA_REASONER_N_CTX: int = 65536

    # Shared by the answer model and the reasoner: KV cache precision (f16, q8_0, q4_0) and the chat template's
    # thinking mode. Grounded answers and JSON decisions are generated without a <think> block.
    QA_LLM_KV_CACHE_TYPE: str = "q8_0"
    QA_LLM_ENABLE_THINKING: bool = False
    REASONING_MAX_ITERATIONS: int = 3
    REASONING_MAX_CRITIQUE_RETRIES: int = 1

    # ==========================================
    # DEEP RESEARCH REPORT MODE
    # ==========================================
    # A third ModelType on the same GGUF as QA_GENERATION/QA_REASONER, for the same reason they are separate:
    # residency is keyed on (model_type, model_path). Report mode holds ONE acquisition across outline
    # optimisation and every section write, so the file is loaded once per run rather than once per section.
    REPORT_AUTHORING_MODEL_PATH: str = "models/Qwen3.5-9B-Q4_K_M.gguf"
    REPORT_AUTHORING_VRAM_REQUIREMENT_MB: float = 10000.0
    REPORT_AUTHORING_N_CTX: int = 131072
    # A report section is longer than a chat answer, so it does not reuse QA_GENERATION_MAX_ANSWER_TOKENS (1024).
    REPORT_SECTION_MAX_TOKENS: int = 1600
    REPORT_PLAN_MAX_TOKENS: int = 1500
    # Evidence shown to the writer for ONE section. Deliberately far below n_ctx: prefill is the cost driver on
    # a 3060, and a section that takes the whole window would make an 8-section report unusable.
    REPORT_SECTION_EVIDENCE_TOKENS: int = 8000

    REPORT_MAX_SECTIONS: int = 8
    REPORT_MAX_SUB_QUESTIONS: int = 12
    REPORT_CANDIDATES_PER_QUERY: int = 40
    REPORT_RERANK_TOP_N: int = 12

    # Per-document sweep: every indexed source gets looked at, so a relevant recording is not missed just
    # because no generated sub-question happened to surface it. ANN-gated so the cost does not scale with
    # the corpus.
    REPORT_SWEEP_MAX_SOURCES: int = 50
    REPORT_SWEEP_PER_SOURCE: int = 3
    REPORT_SWEEP_ANN_DEPTH: int = 2000

    # The scrubber is one sequential 7B call per chunk and drops the answering chunk for ~1 question in 9.
    # For a report, recall dominates precision: an irrelevant chunk is simply never cited, but a dropped one
    # cannot be. Off by default, and the report states which way this was set.
    REPORT_SCRUB_ENABLED: bool = False
    REPORT_SCRUB_MAX_CHUNKS: int = 150

    # Exhaustive mode only; Deep mode always passes 1. A round that adds no new chunks stops the loop
    # early, so most runs finish well under this. Each extra round costs one embedder load, one reranker
    # load and one reload of the report model.
    REPORT_MAX_GATHER_ROUNDS: int = 3
    # A section with no assigned documents at all is the gap test (`OutlineAssignment.uncovered`); this
    # caps how many follow-up queries one round may hand back for those gaps.
    REPORT_FOLLOWUP_QUERIES_PER_ROUND: int = 6

    # Multi-turn chat: most recent turns of a session shown to the answer model and the reasoner
    CHAT_HISTORY_MAX_TURNS: int = 20

    # Fields given explicitly (environment, .env file, constructor): a language pack never overrides them.
    _explicit_fields: FrozenSet[str] = PrivateAttr(default=frozenset())
    # Fields the active pack set, reset to their defaults when another language is switched in.
    _pack_fields: FrozenSet[str] = PrivateAttr(default=frozenset())

    @model_validator(mode="after")
    def _apply_language_pack(self) -> "Settings":
        """
        Applies the active language pack's settings overrides (see `_apply_pack_overrides`). The explicit fields
        are snapshotted first: pydantic adds every assigned field to `model_fields_set`, so after one pack is
        applied its fields would otherwise look explicit to the next.

        Without an explicit APPLIANCE_LANGUAGE the app starts in the language it was last switched to, unless
        REMEMBER_LANGUAGE is off.
        """
        self._explicit_fields = frozenset(self.model_fields_set)
        if "APPLIANCE_LANGUAGE" not in self._explicit_fields and self.REMEMBER_LANGUAGE:
            remembered = self._remembered_language()
            if remembered:
                self.APPLIANCE_LANGUAGE = remembered
        self._apply_pack_overrides()
        return self

    def _read_pack(self, code: str) -> dict:
        path = self.LANGUAGE_PACKS_DIR / f"{code}.toml"
        if not path.is_file():
            if (self.LANGUAGE_PACKS_DIR / "legacy" / f"{code}.toml").is_file():
                raise ValueError(f"APPLIANCE_LANGUAGE='{code}' is a legacy language, no longer supported "
                                 f"(see {self.LANGUAGE_PACKS_DIR / 'legacy' / 'README.md'})")
            raise ValueError(f"APPLIANCE_LANGUAGE='{code}' has no language pack at {path}")
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _apply_pack_overrides(self) -> None:
        """
        Applies the active language pack's settings overrides: its [models] table (a language the default small
        models can't handle gets a model that can) and its FTS configuration. A field set explicitly --
        environment, .env file or constructor -- always wins over the pack. Fields the previous pack set go back
        to their defaults first, so switching from Serbian back to English does not keep Serbian's Gemma.

        Only these two tables are read here, straight from the TOML: the full pack is validated by
        app/core/language, which imports this module and so can't be imported by it.
        """
        pack = self._read_pack(self.APPLIANCE_LANGUAGE)

        overrides = dict(pack.get("models", {}))
        if "search" in pack and "fts_config" in pack["search"]:
            overrides["FTS_LANGUAGE_CONFIG"] = pack["search"]["fts_config"]

        fields = type(self).model_fields
        for name in overrides:
            if name not in fields:
                raise ValueError(f"Language pack {self.APPLIANCE_LANGUAGE}.toml overrides unknown setting '{name}'")

        for name in self._pack_fields - self._explicit_fields:
            setattr(self, name, fields[name].get_default(call_default_factory=True))

        applied = set()
        for name, value in overrides.items():
            if name not in self._explicit_fields:
                # Validated against the field's own type and constraints (e.g. the FTS identifier pattern)
                field = fields[name]
                field_type = Annotated[field.annotation, *field.metadata] if field.metadata else field.annotation
                setattr(self, name, TypeAdapter(field_type).validate_python(value))
                applied.add(name)
        self._pack_fields = frozenset(applied)

    def use_language(self, code: str) -> None:
        """
        Makes `code` the active language: the next ingestion batch is indexed in it and answers are written in
        it. Only safe while no engine job runs; the engine is rebuilt afterwards, because model paths are baked
        into its manifests at build time.
        """
        if code not in self.supported_languages():
            raise ValueError(
                f"'{code}' is not a supported language. Available: {', '.join(self.supported_languages())}."
            )
        previous = self.APPLIANCE_LANGUAGE
        self.APPLIANCE_LANGUAGE = code
        try:
            self._apply_pack_overrides()
        except Exception:
            self.APPLIANCE_LANGUAGE = previous
            self._apply_pack_overrides()
            raise

    def supported_languages(self) -> List[str]:
        """
        The language codes this appliance can ingest and answer in: one per pack in LANGUAGE_PACKS_DIR. Packs in its
        `legacy/` subdirectory (Slovenian) are kept but not served.
        """
        return sorted(path.stem for path in self.LANGUAGE_PACKS_DIR.glob("*.toml"))

    @property
    def SEARCH_FTS_CONFIGS(self) -> List[str]:
        """
        Every Postgres text search configuration a chunk in the index may carry: each pack's, plus the active one
        (an explicit FTS_LANGUAGE_CONFIG may name none of the packs'). The keyword half of search builds one
        query per configuration and matches each chunk only against the query built with its own.
        """
        configs = list(_pack_fts_configs(self.LANGUAGE_PACKS_DIR))
        if self.FTS_LANGUAGE_CONFIG not in configs:
            configs.append(self.FTS_LANGUAGE_CONFIG)
        return configs

    def _remembered_language(self) -> Optional[str]:
        """The language last switched to, if it is still a supported one."""
        try:
            code = self.ACTIVE_LANGUAGE_FILE.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return code if code in self.supported_languages() else None

    def remember_language(self, code: str) -> None:
        """Remembers `code` as the language to start in next time. Failing to write it is not worth failing a switch."""
        try:
            self.ACTIVE_LANGUAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.ACTIVE_LANGUAGE_FILE.write_text(code, encoding="utf-8")
        except OSError as e:
            logger.warning(f"Could not remember the active language ({e}); the next start uses the default.")

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_ignore_empty=True,
        extra="ignore"
    )


settings = Settings()