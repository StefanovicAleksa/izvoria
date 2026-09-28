# Canvas: qa_generation_interfaces_v1
# Path: app/features/qa_generation/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, List

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.interfaces import IModelLoader
from app.features.qa_generation.domain.entities import AnswerCritique


class IQAGenerationAdapter(IModelLoader, ABC):
    """
    Contract for the Generation LLM (e.g., Qwen3.5-9B).
    Inherits from IModelLoader so the Orchestrator manages its VRAM lifecycle.
    """

    @abstractmethod
    def load(self, manifest: ModelManifest) -> LoadedModel:
        """Loads the generation LLM into VRAM."""
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
    def get_context_token_budget(self, model: LoadedModel, user_query: str, include_critique: bool,
                                 system_prompt: Optional[str] = None) -> int:
        """
        Tokens available for the context block once the prompts, the question and the generation budget(s) are
        accounted for. With include_critique, the block must also fit the self-critique prompt. `system_prompt`
        is the localized answer prompt that will be sent; the transcript prompt when omitted.
        """
        pass

    @abstractmethod
    def get_conversation_token_budget(self, model: LoadedModel, system_prompt: str) -> int:
        """
        Tokens available for all conversation messages (earlier turns, their evidence and the current question)
        once the system prompt and the answer generation budget are accounted for.
        """
        pass

    @abstractmethod
    def generate_answer(
        self,
        model: LoadedModel,
        system_prompt: str,
        context_block: str,
        user_query: str
    ) -> str:
        """
        Executes the inference using the assembled strict context prompt.
        Returns the raw generated string containing citation tags.
        """
        pass

    @abstractmethod
    def generate_conversation_answer(self, model: LoadedModel, messages: List[Dict[str, str]]) -> str:
        """
        Answers the last user message of a multi-turn conversation (system prompt first, then alternating
        user/assistant messages). Returns the raw generated string containing citation tags.
        """
        pass

    @abstractmethod
    def critique_answer(
        self,
        model: LoadedModel,
        user_query: str,
        draft_answer: str,
        evidence_block: str
    ) -> AnswerCritique:
        """
        Asks the same loaded model whether every claim in draft_answer matches the text of the
        chunk it cites, and whether the draft fully addresses user_query. Called on the same
        LoadedModel immediately after generate_answer, within one model acquisition.
        """
        pass