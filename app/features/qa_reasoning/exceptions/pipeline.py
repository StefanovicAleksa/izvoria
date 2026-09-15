# Path: app/features/qa_reasoning/exceptions/pipeline.py

from .base import QAReasoningError

class JSONReasonerParseError(QAReasoningError):
    """
    Raised when the Reasoner hallucinates invalid JSON, or an unrecognized/missing
    `decision` field, instead of adhering to the strict ReasonerDecision schema. There is no
    safe default among the three control-flow branches, so this fails closed (unlike the
    scrubber's fail-open default) rather than silently picking a branch.
    """
    pass
