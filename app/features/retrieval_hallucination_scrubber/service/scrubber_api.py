# Canvas: retrieval_scrubber_api_v4
# Path: app/features/retrieval_hallucination_scrubber/service/scrubber_api.py

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.language.service.language_api import submit_in_context
from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.core.model_manager.domain.interfaces import IChatModel
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.retrieval_hallucination_scrubber.domain.interfaces import IRetrievalScrubberAdapter
from app.features.retrieval_hallucination_scrubber.domain.entities import ScrubAudit
from app.features.retrieval_hallucination_scrubber.exceptions.pipeline import JSONScrubParseError
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

            trust = settings.RETRIEVAL_SCRUBBER_TRUST_RERANK_SCORE

            def trusted(chunk: RetrievedChunk) -> bool:
                # The cross-encoder is the better relevance judge; the SLM second-guessing a chunk it
                # ranked this high was how answering evidence got thrown away
                return trust is not None and chunk.rerank_score is not None and chunk.rerank_score >= trust

            if all(trusted(chunk) for chunk in raw_chunks):
                # Nothing for the SLM to judge, so don't load it: on a warm GPU that load would evict the answer model
                logger.info(f"Retrieval Scrubber skipped: all {len(raw_chunks)} chunks are trusted by rerank score.")
                return list(raw_chunks), [
                    ScrubAudit(chunk_id=chunk.chunk_id, is_relevant=True, kept_by_rerank_score=True)
                    for chunk in raw_chunks
                ]

            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter, wait_for_clear=True)

            surviving_chunks: List[RetrievedChunk] = []
            audits: List[ScrubAudit] = []

            try:
                verdicts = self._judge_untrusted(loaded_model, user_prompt,
                                                 [chunk for chunk in raw_chunks if not trusted(chunk)])

                for chunk in raw_chunks:
                    if trusted(chunk):
                        audits.append(ScrubAudit(chunk_id=chunk.chunk_id, is_relevant=True,
                                                 kept_by_rerank_score=True))
                        surviving_chunks.append(chunk)
                        continue

                    audit = verdicts[chunk.chunk_id]
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

    def _judge_untrusted(self, loaded_model: LoadedModel, user_prompt: str,
                         chunks: List[RetrievedChunk]) -> Dict[uuid.UUID, ScrubAudit]:
        """
        A verdict per chunk. With RETRIEVAL_SCRUBBER_CONCURRENCY > 1 and a llama-server judge (the hosted API's
        parallel slots) the chunks are judged at once instead of one after another; each worker carries the
        caller's context, so the request's answer language follows it. An in-process llama.cpp model is never
        called from two threads at once, whatever the setting says: it is not thread-safe.
        """
        workers = min(max(1, settings.RETRIEVAL_SCRUBBER_CONCURRENCY), len(chunks))
        if not isinstance(loaded_model.model_instance, IChatModel):
            workers = 1
        if workers <= 1:
            return {chunk.chunk_id: self._judge(loaded_model, user_prompt, chunk) for chunk in chunks}

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scrubber") as pool:
            futures = {chunk.chunk_id: submit_in_context(pool, self._judge, loaded_model, user_prompt, chunk)
                       for chunk in chunks}
            return {chunk_id: future.result() for chunk_id, future in futures.items()}

    def _judge(self, loaded_model: LoadedModel, user_prompt: str, chunk: RetrievedChunk) -> ScrubAudit:
        safe_text = chunk.content_payload[:3000]

        try:
            return self._adapter.evaluate_relevance(
                model=loaded_model,
                chunk_id=chunk.chunk_id,
                user_prompt=user_prompt,
                text_chunk=safe_text
            )
        except JSONScrubParseError as parse_err:
            # Fail open for this chunk only: one malformed verdict from the SLM used to
            # abort the whole query. The chunk is kept (the prompt already says "when
            # in doubt, accept") and the audit records that no real verdict exists.
            logger.warning(f"Scrubber verdict unparseable for chunk {chunk.chunk_id}; keeping it. {parse_err}")
            return ScrubAudit(
                chunk_id=chunk.chunk_id,
                is_relevant=True,
                discard_reason=None,
                evaluation_failed=True
            )
