# Canvas: contextualizer_api_v4
# Path: app/features/contextual_retrieval/service/contextualizer_api.py

import logging
import uuid
from typing import Dict, List, Tuple

from app.core.anchors.domain.enums import AnchorKind
from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

from app.features.document_structuring.domain.entities import StructuredChunk, StructuredDocument
from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticCorrectedTranscript
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository, ISLMContextAdapter
from app.features.contextual_retrieval.data.prompts import DOCUMENT_CONTEXTUAL_SYSTEM_PROMPT
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.contextual_retrieval.utils.chunker import build_rag_chunks
from app.features.contextual_retrieval.utils.document_window import build_document_sections, assign_section
from app.features.contextual_retrieval.utils.transcript_formatter import format_speaker_turns
from app.features.contextual_retrieval.utils.error_mapper import map_contextual_error

logger = logging.getLogger(__name__)


class ContextualizerAPI:
    def __init__(
            self,
            repository: IContextualRepository,
            orchestrator: ModelManagerOrchestrator,
            slm_adapter: ISLMContextAdapter
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._slm_adapter = slm_adapter

        # Strict Domain Manifest Isolation
        self._manifest = ModelManifest(
            model_type=ModelType.CONTEXTUAL_RETRIEVAL,
            model_path=settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH,
            required_vram_mb=settings.CONTEXTUAL_RETRIEVAL_VRAM_REQUIREMENT_MB
        )

    def update_transcript(self, transcript: ContextualizedTranscript) -> ContextualizedTranscript:
        try:
            return self._repo.update(transcript)
        except Exception as e:
            raise map_contextual_error(e, context=f"ContextualizerAPI.update_transcript for {transcript.id}") from e

    def execute(
            self,
            transcript: DomainPhoneticCorrectedTranscript,
            release_after: bool = True
    ) -> ContextualizedTranscript:
        try:
            existing_record = self._repo.get_by_source_id(transcript.source_id)
            if existing_record:
                logger.info(f"Contextualized Transcript exists for Source {transcript.source_id}. Skipping.")
                return existing_record

            if not transcript.sentences:
                empty_transcript = ContextualizedTranscript(source_id=transcript.source_id, chunks=[])
                return self._repo.save(empty_transcript)

            sentences = transcript.sentences
            raw_chunks = build_rag_chunks(sentences, max_words=250, overlap_sentences=1)
            sentence_index = {s.sentence_id: idx for idx, s in enumerate(sentences)}

            loaded_model = None
            contextualized_chunks: List[ContextualizedChunk] = []

            try:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._slm_adapter)

                sections = self._plan_document_sections(loaded_model, sentences)
                if len(sections) > 1:
                    logger.info(
                        f"Transcript for Source {transcript.source_id} exceeds the context window; "
                        f"situating chunks within {len(sections)} overlapping document sections."
                    )
                section_texts: Dict[int, str] = {}

                for idx, chunk_sentences in enumerate(raw_chunks):
                    logger.info(f"  -> Generating Context for Chunk {idx + 1}/{len(raw_chunks)}...")
                    # Stored/indexed text keeps full "[Speaker_0 @ 00:12:31]" labels; the SLM reads plain turns
                    chunk_text = format_speaker_turns(chunk_sentences)

                    section_idx = assign_section(
                        sections,
                        sentence_index[chunk_sentences[0].sentence_id],
                        sentence_index[chunk_sentences[-1].sentence_id]
                    )
                    if section_idx not in section_texts:
                        start, end = sections[section_idx]
                        section_texts[section_idx] = format_speaker_turns(sentences[start:end], include_timestamps=False)

                    context_string = self._slm_adapter.generate_context(
                        model=loaded_model,
                        full_document_text=section_texts[section_idx],
                        chunk_text=format_speaker_turns(chunk_sentences, include_timestamps=False)
                    )

                    contextualized_chunks.append(
                        ContextualizedChunk(
                            start_sec=chunk_sentences[0].start_sec,
                            end_sec=chunk_sentences[-1].end_sec,
                            original_text=chunk_text,
                            context_string=context_string,
                            sentences=chunk_sentences
                        )
                    )

            finally:
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._slm_adapter, flush=True)

            final_transcript = ContextualizedTranscript(
                source_id=transcript.source_id,
                chunks=contextualized_chunks
            )

            return self._repo.save(final_transcript)

        except Exception as e:
            raise map_contextual_error(e, context=f"ContextualizerAPI.execute for Source {transcript.source_id}") from e

    def execute_image(self, source_id: uuid.UUID, excerpt: str, context_string: str) -> ContextualizedTranscript:
        """
        A standalone image as a one-chunk transcript, so unification and indexing take it like any other source.
        No model: there is no wider source to situate a lone image in, so its context is the file's own metadata
        (name, date taken, size) the caller renders.
        """
        try:
            existing_record = self._repo.get_by_source_id(source_id)
            if existing_record:
                logger.info(f"Contextualized image exists for Source {source_id}. Skipping.")
                return existing_record

            chunk = ContextualizedChunk(anchor_kind=AnchorKind.IMAGE, original_text=excerpt,
                                        context_string=context_string)
            return self._repo.save(ContextualizedTranscript(source_id=source_id, chunks=[chunk]))

        except Exception as e:
            raise map_contextual_error(e, context=f"ContextualizerAPI.execute_image for Source {source_id}") from e

    def execute_document(
            self,
            document: StructuredDocument,
            release_after: bool = True
    ) -> ContextualizedTranscript:
        """
        Situates each chunk of a structured document within the whole document, as `execute` does for a
        transcript. The chunk text is stored verbatim; the SLM also sees the headings the chunk sits under.
        """
        try:
            existing_record = self._repo.get_by_source_id(document.source_id)
            if existing_record:
                logger.info(f"Contextualized document exists for Source {document.source_id}. Skipping.")
                return existing_record

            if not document.chunks:
                return self._repo.save(ContextualizedTranscript(source_id=document.source_id, chunks=[]))

            loaded_model = None
            contextualized_chunks: List[ContextualizedChunk] = []
            chunk_texts = [chunk.text for chunk in document.chunks]

            try:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._slm_adapter)

                sections = self._plan_sections(loaded_model, "\n\n".join(chunk_texts), chunk_texts)
                if len(sections) > 1:
                    logger.info(
                        f"Document for Source {document.source_id} exceeds the context window; "
                        f"situating chunks within {len(sections)} overlapping document sections."
                    )
                section_texts: Dict[int, str] = {}

                for idx, chunk in enumerate(document.chunks):
                    logger.info(f"  -> Generating Context for Chunk {idx + 1}/{len(document.chunks)}...")
                    section_idx = assign_section(sections, idx, idx)
                    if section_idx not in section_texts:
                        start, end = sections[section_idx]
                        section_texts[section_idx] = "\n\n".join(chunk_texts[start:end])

                    context_string = self._slm_adapter.generate_context(
                        model=loaded_model,
                        full_document_text=section_texts[section_idx],
                        chunk_text=_with_section_path(chunk),
                        system_prompt=DOCUMENT_CONTEXTUAL_SYSTEM_PROMPT
                    )

                    contextualized_chunks.append(
                        ContextualizedChunk(
                            anchor_kind=document.anchor_kind,
                            page_start=_range_start(document, chunk),
                            page_end=_range_end(document, chunk),
                            section_path=chunk.section_path,
                            original_text=chunk.text,
                            context_string=context_string
                        )
                    )

            finally:
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._slm_adapter, flush=True)

            return self._repo.save(ContextualizedTranscript(source_id=document.source_id, chunks=contextualized_chunks))

        except Exception as e:
            raise map_contextual_error(e, context=f"ContextualizerAPI.execute_document for Source {document.source_id}") from e

    def _plan_document_sections(
            self,
            model: LoadedModel,
            sentences: List[ReconstructedSentence]
    ) -> List[Tuple[int, int]]:
        # Per-sentence costs include a speaker label each, which over-estimates merged turns (safe side).
        return self._plan_sections(
            model,
            format_speaker_turns(sentences, include_timestamps=False),
            [format_speaker_turns([s], include_timestamps=False) for s in sentences]
        )

    def _plan_sections(self, model: LoadedModel, whole_text: str, unit_texts: List[str]) -> List[Tuple[int, int]]:
        """
        Uses the whole transcript or document as the context whenever it fits the budget (one stable prompt
        prefix for every chunk). Otherwise splits it into overlapping sections of units (sentences, or document
        chunks); a whole transcript that overflows n_ctx used to fail the source at this phase.
        """
        budget = settings.CONTEXTUAL_RETRIEVAL_N_CTX - settings.CONTEXTUAL_RETRIEVAL_RESERVED_TOKENS

        if self._slm_adapter.count_tokens(model, whole_text) <= budget:
            return [(0, len(unit_texts))]

        token_counts = [self._slm_adapter.count_tokens(model, text) for text in unit_texts]
        return build_document_sections(token_counts, budget)


def _range_start(document: StructuredDocument, chunk: StructuredChunk):
    """A LINE anchor keeps its lines in the page fields (see AnchorKind): the one place they are put there."""
    return chunk.line_start if document.anchor_kind is AnchorKind.LINE else chunk.page_start


def _range_end(document: StructuredDocument, chunk: StructuredChunk):
    return chunk.line_end if document.anchor_kind is AnchorKind.LINE else chunk.page_end


def _with_section_path(chunk: StructuredChunk) -> str:
    """The chunk as the SLM reads it: the headings it sits under, then its text."""
    return f"[{chunk.section_path}]\n{chunk.text}" if chunk.section_path else chunk.text
