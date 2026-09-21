# Path: app/features/qa_reasoning/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Dict

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.qa_reasoning.domain.entities import ReasonerDecision


class IReasonerAdapter(IModelLoader, ABC):
    """
    Contract for the Reasoner LLM (e.g., Qwen3.5-9B).
    Inherits from IModelLoader so the Orchestrator manages its VRAM lifecycle.
    """

    @abstractmethod
    def load(self, manifest: ModelManifest) -> LoadedModel:
        """Loads the reasoning LLM into VRAM."""
        pass

    @abstractmethod
    def unload(self, model: LoadedModel) -> bool:
        """Safely removes the LLM from VRAM and triggers garbage collection."""
        pass

    @abstractmethod
    def count_tokens(self, model: LoadedModel, text: str) -> int:
        """Number of tokens the loaded model's tokenizer produces for text."""
        pass

    @abstractmethod
    def get_evidence_token_budget(
        self,
        model: LoadedModel,
        original_question: str,
        decision_history_block: str,
        prior_session_context: str,
        extra_context: str,
    ) -> int:
        """Tokens left for the evidence block once the rest of the reasoning prompt and the reply are counted."""
        pass

    @abstractmethod
    def rewrite_query(self, model: LoadedModel, question: str, prior_session_context: str) -> str:
        """Rewrites a follow-up question into a standalone search query using the prior turn."""
        pass

    @abstractmethod
    def reason(
        self,
        model: LoadedModel,
        original_question: str,
        evidence_block: str,
        decision_history_block: str,
        prior_session_context: str,
        extra_context: str,
        evidence_tags: Dict[str, uuid.UUID],
    ) -> ReasonerDecision:
        """
        Runs one reasoning turn: chain-of-thought over the evidence pool so far, plus exactly
        one of ANSWER_READY / REFINE_QUERY / ASK_CLARIFICATION.

        prior_session_context is the previous turn's Q&A (from persisted session memory), fed
        in only on a question's first iteration. extra_context is the self-critique's gap
        description when looping back after a NEEDS_MORE_EVIDENCE verdict; "" otherwise.
        evidence_tags maps the [E<n>] tags shown in evidence_block to chunk ids; evidence links citing any
        other tag are dropped.
        """
        pass
