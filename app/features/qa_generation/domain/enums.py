# Path: app/features/qa_generation/domain/enums.py

from enum import Enum


class CritiqueVerdict(str, Enum):
    """The self-critique's verdict on a draft answer, checked in the same model acquisition."""
    SUFFICIENT = "sufficient"
    NEEDS_MORE_EVIDENCE = "needs_more_evidence"
