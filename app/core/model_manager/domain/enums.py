# Canvas: model_enums_v5
# Path: app/core/model_manager/domain/enums.py

from enum import Enum


class ModelType(str, Enum):
    """
    The functional role of the model within the Onyx pipeline.
    The Domain layer does not care about the specific architecture (e.g., NeMo vs Silero),
    only what the model is supposed to do.
    """
    DOMAIN_SCOUT = "domain_scout"
    DOMAIN_CLASSIFIER = "domain_classifier"
    VAD = "vad"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    LLM = "llm"
    VISION = "vision"
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction"

    # Legacy Features / Legacy Pipeline Models
    NLP = "nlp"
    INTELLIGENCE_BRIDGE = "intelligence_bridge"
    ASR_CORRECTION = "asr_correction"
    ASR_CORRECTION_PHASE_1 = "asr_correction_phase_1"
    ASR_CORRECTION_PHASE_2 = "asr_correction_phase_2"
    ASR_CORRECTION_PHASE_3 = "asr_correction_phase_3"

    # Modern Isolated Pipeline
    HALLUCINATION_SCRUBBER = "hallucination_scrubber"  # Phase 6.5
    DOMAIN_PHONETIC_CORRECTION = "phonetic_correction"  # Phase 8
    CONTEXTUAL_RETRIEVAL = "contextual_retrieval"  # Phase 9
    VISUAL_CUE_DETECTION = "visual_cue_detection"  # Phase 10
    VISUAL_HALLUCINATION_CORRECTION = "visual_hallucination_correction"
    VISUAL_CONTEXT_SYNTHESIS = "visual_context_synthesis"
    RAG_EMBEDDING = "rag_embedding"  # Phase 16