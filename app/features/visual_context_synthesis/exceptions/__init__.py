# Canvas: visual_synthesis_exceptions_init_v1
# Path: app/features/visual_context_synthesis/exceptions/__init__.py

from .base import VisualSynthesisError
from .database import VisualSynthesisDatabaseError, VisualTimelinePersistenceError
from .model import VisualSynthesisModelError, VisualSynthesisOOMError, VisualSLMLoadError, VisualSLMInferenceError
from .process import VisualSynthesisPipelineError, FuzzyGroupingError

__all__ = [
    "VisualSynthesisError",
    "VisualSynthesisDatabaseError",
    "VisualTimelinePersistenceError",
    "VisualSynthesisModelError",
    "VisualSynthesisOOMError",
    "VisualSLMLoadError",
    "VisualSLMInferenceError",
    "VisualSynthesisPipelineError",
    "FuzzyGroupingError"
]