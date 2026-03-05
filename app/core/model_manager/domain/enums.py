# Canvas: model_manager_enums_v2
# Path: app/core/model_manager/domain/enums.py

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
    SENTENCE_RECONSTRUCTION = "sentence_reconstruction" # FIXED: Officially adding the new model type