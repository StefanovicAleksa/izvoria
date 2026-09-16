# Canvas: generation_api_v3
# Path: app/features/qa_generation/service/generation_api.py

import time
import logging
from typing import List, Tuple

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.interfaces import IQAGenerationAdapter
from app.features.qa_generation.domain.entities import QAResponse, AnswerCritique, RAGContextBlock
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.data.prompts import QA_SYSTEM_PROMPT, NO_EVIDENCE_ANSWER
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError
from app.features.qa_generation.utils.citation_parser import (
    build_context_block, parse_citations, normalize_citation_tags
)
from app.features.qa_generation.utils.context_packer import select_within_budget
from app.features.qa_generation.utils.error_mapper import map_qa_error

logger = logging.getLogger(__name__)


class QAGenerationAPI:
    """
    The main service layer for the QA Generation feature.
    Coordinates RAG context compilation, VRAM offloading, inference, and citation mapping.

    Chunks arrive best-first. They are cut to the model's context budget in that order, then shown in reading
    order. With no evidence to show, the grounded answer is always the refusal, so the model is never loaded.
    """

    def __init__(
            self,
            adapter: IQAGenerationAdapter,
            orchestrator: ModelManagerOrchestrator
    ):
        self._adapter = adapter
        self._orchestrator = orchestrator
        self._manifest = ModelManifest(
            model_type=ModelType.QA_GENERATION,
            model_path=settings.QA_GENERATION_MODEL_PATH,
            required_vram_mb=settings.QA_GENERATION_VRAM_REQUIREMENT_MB
        )

    def _pack_context(
            self,
            loaded_model: LoadedModel,
            user_query: str,
            retrieved_chunks: List[RetrievedChunk],
            include_critique: bool
    ) -> Tuple[List[RetrievedChunk], RAGContextBlock]:
        budget = self._adapter.get_context_token_budget(loaded_model, user_query, include_critique)
        packed = select_within_budget(
            retrieved_chunks, budget, lambda text: self._adapter.count_tokens(loaded_model, text)
        )
        if len(packed) < len(retrieved_chunks):
            logger.info(f"Context budget of {budget} tokens fits {len(packed)} of {len(retrieved_chunks)} chunks.")
        return packed, build_context_block(packed)

    @staticmethod
    def _refusal(user_query: str, start_time: float) -> QAResponse:
        return QAResponse(
            query=user_query,
            generated_answer=NO_EVIDENCE_ANSWER,
            citations=[],
            inference_time_ms=int((time.time() - start_time) * 1000)
        )

    def generate(
            self,
            user_query: str,
            retrieved_chunks: List[RetrievedChunk],
            release_after: bool = True
    ) -> QAResponse:

        logger.info(f"Executing QA Generation for query: '{user_query}'")
        start_time = time.time()

        if not retrieved_chunks:
            logger.info("No evidence survived retrieval; answering with the refusal without loading the model.")
            return self._refusal(user_query, start_time)

        try:
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)

            try:
                packed, context_block = self._pack_context(loaded_model, user_query, retrieved_chunks, False)
                if not packed:
                    logger.warning("No retrieved chunk fits the context budget; answering with the refusal.")
                    return self._refusal(user_query, start_time)

                generated_text = self._adapter.generate_answer(
                    model=loaded_model,
                    system_prompt=QA_SYSTEM_PROMPT,
                    context_block=context_block.formatted_text,
                    user_query=user_query
                )
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            # Grouped tags ([Doc 1, 3]) become [Doc 1][Doc 3] in the returned text too, so it matches the citations
            generated_text = normalize_citation_tags(generated_text, len(context_block.doc_mapping))
            citations = parse_citations(
                generated_text=generated_text,
                chunks=packed,
                doc_mapping=context_block.doc_mapping
            )

            inference_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"QA Generation complete in {inference_time_ms}ms with {len(citations)} citations extracted.")

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
            release_after: bool = True
    ) -> Tuple[QAResponse, AnswerCritique]:
        """
        Drafts an answer, then immediately (same model acquisition, no extra GPU swap) asks
        the same model to critique its own draft against the same packed evidence.

        Only the question and the evidence reach the answer model. Model-written analysis (e.g. the Deep
        Research reasoner's) is not source material, so callers express its judgement by ordering the chunks.
        """
        logger.info(f"Executing QA Generation with Self-Critique for query: '{user_query}'")
        start_time = time.time()

        if not retrieved_chunks:
            # Refusing is the grounded answer when there is no evidence, which the critique accepts by definition
            logger.info("No evidence to answer from; returning the refusal without loading the model.")
            return self._refusal(user_query, start_time), AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        try:
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)

            try:
                packed, context_block = self._pack_context(loaded_model, user_query, retrieved_chunks, True)
                if not packed:
                    logger.warning("No retrieved chunk fits the context budget; answering with the refusal.")
                    return (self._refusal(user_query, start_time),
                            AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT))

                generated_text = self._adapter.generate_answer(
                    model=loaded_model,
                    system_prompt=QA_SYSTEM_PROMPT,
                    context_block=context_block.formatted_text,
                    user_query=user_query
                )
                generated_text = normalize_citation_tags(generated_text, len(context_block.doc_mapping))

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
                chunks=packed,
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
