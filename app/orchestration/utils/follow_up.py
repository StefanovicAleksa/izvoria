# Path: app/orchestration/utils/follow_up.py

from typing import Optional


def answer_question(question: str, standalone_query: Optional[str]) -> str:
    """
    The question an answer model is given. A follow-up ("and when did he walk in?") also carries the standalone form
    it was searched with: an answer model reading the conversation still refuses pronoun follow-ups it can't tie to
    a transcript on its own, and one without the conversation can't resolve them at all. The user's words stay first.
    """
    if not standalone_query or standalone_query.strip() == question.strip():
        return question
    return f"{question}\n(Standalone form of this follow-up question: {standalone_query})"
