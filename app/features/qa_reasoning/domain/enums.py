# Path: app/features/qa_reasoning/domain/enums.py

from enum import Enum


class ReasonerDecisionType(str, Enum):
    """The three control-flow branches the Reasoner may choose for one turn."""
    ANSWER_READY = "answer_ready"
    REFINE_QUERY = "refine_query"
    ASK_CLARIFICATION = "ask_clarification"
