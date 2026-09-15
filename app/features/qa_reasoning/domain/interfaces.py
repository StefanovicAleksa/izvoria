# Path: app/features/qa_reasoning/domain/interfaces.py

from abc import ABC, abstractmethod

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.qa_reasoning.domain.entities import ReasonerDecision


class IReasonerAdapter(IModelLoader, ABC):
    """
    Contract for the Reasoner LLM (e.g., Qwen2.5-7B-Instruct).
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
    def reason(
        self,
        model: LoadedModel,
        original_question: str,
        evidence_block: str,
        decision_history_block: str,
        prior_session_context: str,
        extra_context: str,
    ) -> ReasonerDecision:
        """
        Runs one reasoning turn: chain-of-thought over the evidence pool so far, plus exactly
        one of ANSWER_READY / REFINE_QUERY / ASK_CLARIFICATION.

        prior_session_context is the previous turn's Q&A (from persisted session memory), fed
        in only on a question's first iteration. extra_context is the self-critique's gap
        description when looping back after a NEEDS_MORE_EVIDENCE verdict; "" otherwise.
        """
        pass
