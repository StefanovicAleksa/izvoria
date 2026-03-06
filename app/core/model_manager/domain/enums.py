from enum import Enum


class ModelType(str, Enum):
    """
    The functional role of the model within the Onyx pipeline.
    The Domain layer does not care about the specific architecture (e.g., NeMo vs Silero),
    only what the model is supposed to do.
    """
    VAD = "vad"
    TRANSCRIPTION = "transcription"
    DIARIZATION = "diarization"
    LLM = "llm"
    VISION = "vision"
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction"

    ASR_CORRECTION_PHASE_1 = "asr_correction_phase_1"  # Extractor (Llama 3.2)
    ASR_CORRECTION_PHASE_2 = "asr_correction_phase_2"  # Detector (T5-ASR-EC)
    ASR_CORRECTION_PHASE_3 = "asr_correction_phase_3"  # Corrector (Qwen 2.5)