# Canvas: generation_api_v2
# Path: app/features/qa_generation/service/generation_api.py

import time
import logging
from typing import List, Tuple

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.interfaces import IQAGenerationAdapter
from app.features.qa_generation.domain.entities import QAResponse, AnswerCritique
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.data.prompts import QA_SYSTEM_PROMPT
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError
from app.features.qa_generation.utils.citation_parser import build_context_block, parse_citations
from app.features.qa_generation.utils.error_mapper import map_qa_error

logger = logging.getLogger(__name__)


class QAGenerationAPI:
    """
    The main service layer for the QA Generation feature.
    Coordinates RAG context compilation, VRAM offloading, inference, and citation mapping.
    """

    def __init__(
            self,
            adapter: IQAGenerationAdapter,
            orchestrator: ModelManagerOrchestrator
    ):
        self._adapter = adapter
        self._orchestrator = orchestrator

    def generate(
            self,
            user_query: str,
            retrieved_chunks: List[RetrievedChunk],
            release_after: bool = True
    ) -> QAResponse:

        logger.info(f"Executing QA Generation for query: '{user_query}'")
        start_time = time.time()

        try:
            # 1. Compile the strict RAG Context Block
            context_block = build_context_block(retrieved_chunks)

            # 2. Define the Model Requirements (Gemma 4 E4B)
            manifest = ModelManifest(
                model_type=ModelType.QA_GENERATION,
                model_path=settings.QA_GENERATION_MODEL_PATH,
                required_vram_mb=settings.QA_GENERATION_VRAM_REQUIREMENT_MB
            )

            # 3. VRAM Orchestration: Safely load model (evicting others if necessary)
            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter, wait_for_clear=True)

            try:
                # 4. Execute Autoregressive Inference
                generated_text = self._adapter.generate_answer(
                    model=loaded_model,
                    system_prompt=QA_SYSTEM_PROMPT,
                    context_block=context_block.formatted_text,
                    user_query=user_query
                )
            finally:
                # 5. Guaranteed VRAM Cleanup
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            # 6. Parse LLM output for citations mapping
            citations = parse_citations(
                generated_text=generated_text,
                chunks=retrieved_chunks,
                doc_mapping=context_block.doc_mapping
            )

            inference_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"QA Generation complete in {inference_time_ms}ms with {len(citations)} citations extracted.")

            # 7. Construct Final Response Payload
            return QAResponse(
                query=user_query,
                generated_answer=generated_text,
                citations=citations,
                inference_time_ms=inference_time_ms
            )

        except Exception as e:
            raise map_qa_error(e, context=f"Processing QA Generation for query: '{user_query}'")

    def generate_with_critique(
            self,
            user_query: str,
            retrieved_chunks: List[RetrievedChunk],
            extra_context: str = "",
            release_after: bool = True
    ) -> Tuple[QAResponse, AnswerCritique]:
        """
        Drafts an answer, then immediately (same model acquisition, no extra GPU swap) asks
        the same model to critique its own draft against the evidence. Independent from
        generate() rather than sharing its body, so generate()'s existing behavior and pinned
        unit tests are never put at risk by this addition.
        """
        logger.info(f"Executing QA Generation with Self-Critique for query: '{user_query}'")
        start_time = time.time()

        try:
            context_block = build_context_block(retrieved_chunks)

            manifest = ModelManifest(
                model_type=ModelType.QA_GENERATION,
                model_path=settings.QA_GENERATION_MODEL_PATH,
                required_vram_mb=settings.QA_GENERATION_VRAM_REQUIREMENT_MB
            )

            loaded_model = self._orchestrator.acquire_model(manifest, self._adapter, wait_for_clear=True)

            try:
                prompt = user_query
                if extra_context:
                    prompt = f"{user_query}\n\n[REASONER ANALYSIS]\n{extra_context}"

                generated_text = self._adapter.generate_answer(
                    model=loaded_model,
                    system_prompt=QA_SYSTEM_PROMPT,
                    context_block=context_block.formatted_text,
                    user_query=prompt
                )

                try:
                    critique = self._adapter.critique_answer(
                        model=loaded_model,
                        user_query=user_query,
                        draft_answer=generated_text,
                        evidence_block=context_block.formatted_text
                    )
                except JSONCritiqueParseError as parse_err:
                    # Fail open toward distrust: never silently ship an unverified answer.
                    logger.warning(f"Critique verdict unparseable; treating draft as unresolved. {parse_err}")
                    critique = AnswerCritique(
                        verdict=CritiqueVerdict.NEEDS_MORE_EVIDENCE,
                        gap_description="Self-critique verdict was unparseable; treating as unresolved."
                    )
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            citations = parse_citations(
                generated_text=generated_text,
                chunks=retrieved_chunks,
                doc_mapping=context_block.doc_mapping
            )

            inference_time_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"QA Generation with Self-Critique complete in {inference_time_ms}ms. Verdict: {critique.verdict.value}"
            )

            response = QAResponse(
                query=user_query,
                generated_answer=generated_text,
                citations=citations,
                inference_time_ms=inference_time_ms
            )
            return response, critique

        except Exception as e:
            raise map_qa_error(e, context=f"Processing QA Generation with Critique for query: '{user_query}'")