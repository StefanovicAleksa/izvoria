# Canvas: contextualizer_api_v3
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
            transcript: PhoneticCorrectedTranscript,
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

            full_document_text = " ".join([s.text for s in transcript.sentences])
            raw_chunks = build_rag_chunks(transcript.sentences, max_words=250, overlap_sentences=1)

            loaded_model = None
            contextualized_chunks: List[ContextualizedChunk] = []

            try:
                loaded_model = self._orchestrator.acquire_model(self._manifest, self._slm_adapter)

                for idx, chunk_sentences in enumerate(raw_chunks):
                    logger.info(f"  -> Generating Context for Chunk {idx + 1}/{len(raw_chunks)}...")
                    chunk_text = " ".join([s.text for s in chunk_sentences])

                    context_string = self._slm_adapter.generate_context(
                        model=loaded_model,
                        full_document_text=full_document_text,
                        chunk_text=chunk_text
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
