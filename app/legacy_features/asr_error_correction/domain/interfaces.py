import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.model_manager.domain.interfaces import IModelLoader
from app.core.model_manager.domain.entities import LoadedModel

# FIXED: Depending strictly on Phase 7 output
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript

from app.legacy_features.asr_error_correction.domain.entities import (
    DocumentGlossary,
    AnomalyTag,
    DomainCorrectedTranscript
)


# --- Repository Contract ---

class ICorrectionRepository(ABC):
    """Persists the extracted Glossary and the final Corrected Transcript with its Audit Trail."""

    @abstractmethod
    def save_glossary(self, glossary: DocumentGlossary) -> DocumentGlossary:
        pass

    @abstractmethod
    def get_glossary(self, source_id: uuid.UUID) -> Optional[DocumentGlossary]:
        pass

    @abstractmethod
    def save_corrected_transcript(self, transcript: DomainCorrectedTranscript) -> DomainCorrectedTranscript:
        pass

    @abstractmethod
    def get_corrected_transcript(self, source_id: uuid.UUID) -> Optional[DomainCorrectedTranscript]:
        pass


# --- Model Adapter Contracts ---

class ILlamaGlossaryAdapter(IModelLoader, ABC):
    """Phase 1: Llama 3.2 3B (GGUF) - Full document scan for terminology."""

    @abstractmethod
    def extract_glossary(self, model: LoadedModel, transcript: ReconstructedTranscript) -> DocumentGlossary:
        pass


class IT5AnomalyAdapter(IModelLoader, ABC):
    """Phase 2: T5-ASR-EC - Scans reconstructed sentences and flags acoustic/phonetic anomalies."""

    @abstractmethod
    def detect_anomalies(self, model: LoadedModel, transcript: ReconstructedTranscript) -> List[AnomalyTag]:
        pass


class IQwenCorrectionAdapter(IModelLoader, ABC):
    """Phase 3: Qwen 2.5 7B (GGUF) - Grounded resolution of anomalies using the Glossary."""

    @abstractmethod
    def apply_corrections(
            self,
            model: LoadedModel,
            transcript: ReconstructedTranscript,
            anomalies: List[AnomalyTag],
            glossary: DocumentGlossary
    ) -> DomainCorrectedTranscript:
        pass