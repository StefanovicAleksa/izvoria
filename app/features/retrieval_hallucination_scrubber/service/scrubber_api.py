# Canvas: retrieval_scrubber_api_v3
# Path: app/features/retrieval_hallucination_scrubber/service/scrubber_api.py

import logging
from typing import List, Tuple

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.retrieval_hallucination_scrubber.domain.interfaces import IRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.domain.entities import ScrubAudit
from app.features.retrieval_hallucination_scrubber.utils.error_mapper import map_scrubber_error

logger = logging.getLogger(__name__)


class RetrievalScrubberAPI:
    """
    Service for applying a strict Zero-Shot SLM evaluation to raw RAG search results,
    dropping context that is hallucinatory or irrelevant to the user's prompt.
    """

    def __init__(
            self,
            adapter: IRetrievalScrubberAdapter,
            orchestrator: ModelManagerOrchestrator
    ):
        self._adapter = adapter
        self._orchestrator = orchestrator

    def scrub_retrieved_chunks(
            self,
            user_prompt: str,
            raw_chunks: List[RetrievedChunk],
            release_after: bool = True
    ) -> Tuple[List[RetrievedChunk], List[ScrubAudit]]:
        if not raw_chunks:
            return [], []

        try:
            # FIX: Point strictly to the new RETRIEVAL_SCRUBBER enum and config paths
            manifest = ModelManifest(
                model_type=ModelType.RETRIEVAL_SCRUBBER,
                model_path=settings.RETRIEVAL_SCRUBBER_MODEL_PATH,
                required_vram_mb=settings.RETRIEVAL_SCRUBBER_VRAM_REQUIREMENT_MB
            )

            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter, wait_for_clear=True)

            surviving_chunks: List[RetrievedChunk] = []
            audits: List[ScrubAudit] = []

            try:
                for chunk in raw_chunks:
                    safe_text = chunk.text_payload[:3000] if len(chunk.text_payload) > 3000 else chunk.text_payload

                    audit = self._adapter.evaluate_relevance(
                        model=loaded_model,
                        chunk_id=chunk.chunk_id,
                        user_prompt=user_prompt,
                        text_chunk=safe_text
                    )

                    audits.append(audit)

                    if audit.is_relevant:
                        surviving_chunks.append(chunk)
                    else:
                        logger.debug(f"Chunk {chunk.chunk_id} rejected by SLM. Reason: {audit.discard_reason}")

                logger.info(
                    f"Retrieval Scrubber complete. Retained {len(surviving_chunks)} out of {len(raw_chunks)} chunks.")
                return surviving_chunks, audits

            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

        except Exception as e:
            raise map_scrubber_error(e, context=f"Scrubbing {len(raw_chunks)} chunks for user prompt") from e