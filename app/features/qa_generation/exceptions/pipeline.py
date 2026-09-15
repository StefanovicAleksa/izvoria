# Canvas: qa_generation_exceptions_pipeline_v1
# Path: app/features/qa_generation/exceptions/pipeline.py

from .base import QAGenerationError

class JSONCritiqueParseError(QAGenerationError):
    """
    Raised when the self-critique call hallucinates invalid JSON instead of adhering to the
    strict AnswerCritique schema (verdict, gap_description).
    """
    pass
