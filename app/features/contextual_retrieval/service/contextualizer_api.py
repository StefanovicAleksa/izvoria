# Canvas: contextual_api_v1
# Path: app/features/contextual_retrieval/service/contextualizer_api.py

import logging
from typing import List

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript
from app.features.contextual_retrieval.domain.interfaces import IContextualRepository, ISLMContextAdapter
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.contextual_retrieval.utils.chunker import build_rag_chunks
from app.features.contextual_retrieval.utils.error_mapper import map_contextual_error

logger = logging.getLogger(__name__)


class ContextualizerAPI:
    """
    Orchestrates the Anthropic Contextual Retrieval pattern (Locally).
    Transforms a PhoneticCorrectedTranscript into semantic chunks, generates situational
    context using a cached SLM, and prepares the payload for Hybrid RAG indexing.
    """

    def __init__(
            self,
            repository: IContextualRepository,
            orchestrator: ModelManagerOrchestrator,
            slm_adapter: ISLMContextAdapter
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._slm_adapter = slm_adapter

    def execute(
            self,
            transcript: PhoneticCorrectedTranscript,
            release_after: bool = True
    ) -> ContextualizedTranscript:
        try:
            # 1. Idempotency Check
            existing_record = self._repo.get_by_source_id(transcript.source_id)
            if existing_record:
                logger.info(f"Contextualized Transcript already exists for Source {transcript.source_id}. Skipping.")
                return existing_record

            # Handle empty transcripts gracefully
            if not transcript.sentences:
                empty_transcript = ContextualizedTranscript(source_id=transcript.source_id, chunks=[])
                return self._repo.save(empty_transcript)

            # 2. Extract Full Document Text (Used for KV Caching)
            full_document_text = " ".join([s.text for s in transcript.sentences])

            # 3. Deterministic Chunking (Pure Python)
            # Groups 200-300 words per chunk while respecting sentence boundaries
            raw_chunks = build_rag_chunks(transcript.sentences, max_words=250, overlap_sentences=1)

            # 4. Define SLM Model Requirements
            # Reusing the Llama 3.2 3B model config used in Domain Classification
            # as it is highly instruction-tuned and optimized for this hardware footprint.
            manifest = ModelManifest(
                model_type=ModelType.DOMAIN_CLASSIFIER,
                model_path=settings.DOMAIN_CLASSIFIER_MODEL_PATH,
                required_vram_mb=settings.DOMAIN_CLASSIFIER_MODEL_VRAM_REQUIREMENT_MB
            )

            loaded_model = None
            contextualized_chunks: List[ContextualizedChunk] = []

            try:
                # 5. Acquire GPU Resource
                loaded_model = self._orchestrator.acquire_model(manifest, self._slm_adapter)

                # 6. Generate Context Loop
                for idx, chunk_sentences in enumerate(raw_chunks):
                    logger.info(f"  -> Generating Context for Chunk {idx + 1}/{len(raw_chunks)}...")

                    chunk_text = " ".join([s.text for s in chunk_sentences])

                    # LLM Call
                    context_string = self._slm_adapter.generate_context(
                        model=loaded_model,
                        full_document_text=full_document_text,
                        chunk_text=chunk_text
                    )

                    # Assemble Domain Entity
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
                # 7. Safe GPU Lifecycle Management
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._slm_adapter, flush=True)

            # 8. Create Aggregate Root & Persist
            final_transcript = ContextualizedTranscript(
                source_id=transcript.source_id,
                chunks=contextualized_chunks
            )

            return self._repo.save(final_transcript)

        except Exception as e:
            # Provide specific context for the error mapper
            raise map_contextual_error(e, context=f"ContextualizerAPI.execute for Source {transcript.source_id}") from e