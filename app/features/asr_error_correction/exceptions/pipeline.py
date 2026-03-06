from .base import AsrCorrectionError

class GlossaryJSONParseError(AsrCorrectionError):
    """Raised when Phase 1 (Llama 3.2) hallucinates invalid JSON while building the glossary."""
    pass

class AnomalySyncError(AsrCorrectionError):
    """Raised when Phase 2 (T5) flags an anomaly for a sentence ID that no longer exists in the transcript."""
    pass

class HallucinationTrapError(AsrCorrectionError):
    """
    Raised by the Audit Generator if Phase 3 (Qwen 2.5) completely rewrites
    a sentence instead of only fixing the tagged phonetic anomaly.
    """
    pass