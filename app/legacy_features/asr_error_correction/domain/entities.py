import uuid
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, ConfigDict

from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence


# --- Phase 1: Overarching Glossary Extraction (Llama 3.2) ---

class GlossaryItem(BaseModel):
    """A specific domain term, name, or acronym extracted by the LLM."""
    term: str
    context: str = Field(description="How the term was used, to help Phase 3 disambiguate.")

    model_config = ConfigDict(frozen=True)


class DocumentGlossary(BaseModel):
    """The complete context of the document used to ground the final correction."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    domain_category: str = Field(description="e.g., 'Medical Oncology', 'Corporate Litigation'")
    items: List[GlossaryItem]

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)


# --- Phase 2: Anomaly Detection (T5) ---

class AnomalyTag(BaseModel):
    """Marks a specific Reconstructed Sentence as highly suspicious without altering it yet."""
    sentence_id: uuid.UUID
    original_start_sec: float
    original_end_sec: float
    original_text: str
    speaker_id: str
    confidence_of_error: float = Field(ge=0.0, le=1.0)
    anomaly_reason: str = Field(description="e.g., 'Phonetic dissonance', 'Medical terminology mismatch'")

    model_config = ConfigDict(frozen=True)


# --- Phase 3: Grounded Correction (Qwen 2.5) & Auditing ---

class CorrectionAuditRecord(BaseModel):
    """Strict Before/After tracking to prove no 'Hallucination of Correctness' occurred."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    anomaly_tag: AnomalyTag
    corrected_text: str
    applied_glossary_terms: List[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class DomainCorrectedTranscript(BaseModel):
    """The aggregate root representing the finalized, post-processed transcript."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID

    sentences: List[ReconstructedSentence] = Field(default_factory=list)
    audit_trail: List[CorrectionAuditRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)