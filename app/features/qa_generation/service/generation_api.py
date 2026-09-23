# Canvas: generation_api_v3
# Path: app/features/qa_generation/service/generation_api.py

import time
import logging
import re
from typing import Collection, Dict, List, Tuple

from app.core.language.service.language_api import get_language_pack
from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.interfaces import IQAGenerationAdapter
from app.features.qa_generation.domain.entities import (
    QAResponse, AnswerCritique, RAGContextBlock, ConversationHistory, ConversationAnswer, NumberedChunk
)
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.data.prompts import (
    QA_USER_TEMPLATE, QA_CONVERSATION_RULES, QA_CONVERSATION_USER_TEMPLATE, qa_system_prompt,
    NO_NEW_CONTEXT_BLOCK, CITATION_RETRY_PROMPT
)
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError
from app.features.qa_generation.utils.citation_parser import (
    build_context_block, build_numbered_context_block, doc_numbers, parse_citations, normalize_citation_tags
)
from app.features.qa_generation.utils.conversation_packer import pack_conversation, PackedConversation
from app.features.qa_generation.utils.context_packer import has_document_chunks, select_within_budget
from app.features.qa_generation.utils.error_mapper import map_qa_error

logger = logging.getLogger(__name__)


def _localized(prompt: str) -> str:
    """The prompt for this appliance's language: its refusal sentence, and an instruction to answer in the language."""
    return get_language_pack().localize_prompt(prompt)

_CITATION_TAG = re.compile(r"\[Doc \d+]")


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
            include_critique: bool,
            system_prompt: str
    ) -> Tuple[List[RetrievedChunk], RAGContextBlock]:
        budget = self._adapter.get_context_token_budget(loaded_model, user_query, include_critique,
                                                        system_prompt=system_prompt)
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
            generated_answer=get_language_pack().answers.refusal,
            citations=[],
            inference_time_ms=int((time.time() - start_time) * 1000)
        )

    def _cited_or_refused(
            self,
            loaded_model: LoadedModel,
            messages: List[Dict[str, str]],
            generated_text: str,
            valid_numbers: Collection[int]
    ) -> str:
        """
        An answer must cite the evidence it rests on. One that is neither the refusal nor cites anything shown is
        sent back once (same acquisition) to be rewritten with citations; if it still cites nothing, the refusal
        is returned instead of an untraceable answer. Tags are normalized to the shown documents either way.
        """
        text = normalize_citation_tags(generated_text, 0, valid_numbers)
        if self._is_refusal(text) or _CITATION_TAG.search(text):
            return text

        logger.warning("Answer came back without citations; asking the model to cite its claims.")
        retry_messages = messages + [
            {"role": "assistant", "content": generated_text},
            {"role": "user", "content": _localized(CITATION_RETRY_PROMPT)},
        ]
        retried = normalize_citation_tags(
            self._adapter.generate_conversation_answer(loaded_model, retry_messages), 0, valid_numbers
        )
        if self._is_refusal(retried) or _CITATION_TAG.search(retried):
            return retried

        logger.warning("Answer still has no citations after the retry; returning the refusal instead.")
        return get_language_pack().answers.refusal

    @staticmethod
    def _is_refusal(text: str) -> bool:
        return get_language_pack().is_refusal(text)

    @staticmethod
    def _answer_messages(system_prompt: str, context_block: str, user_query: str) -> List[Dict[str, str]]:
        """The messages generate_answer sends, for a follow-up request in the same exchange."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": QA_USER_TEMPLATE.format(context_block=context_block, user_query=user_query)},
        ]

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

        system_prompt = _localized(qa_system_prompt(has_document_chunks(retrieved_chunks)))
        try:
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)

            try:
                packed, context_block = self._pack_context(loaded_model, user_query, retrieved_chunks, False,
                                                           system_prompt)
                if not packed:
                    logger.warning("No retrieved chunk fits the context budget; answering with the refusal.")
                    return self._refusal(user_query, start_time)

                generated_text = self._adapter.generate_answer(
                    model=loaded_model,
                    system_prompt=system_prompt,
                    context_block=context_block.formatted_text,
                    user_query=user_query
                )
                # Grouped tags ([Doc 1, 3]) become [Doc 1][Doc 3] in the returned text too, so it matches the citations
                generated_text = self._cited_or_refused(
                    loaded_model, self._answer_messages(system_prompt, context_block.formatted_text, user_query),
                    generated_text, doc_numbers(context_block.doc_mapping)
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
            logger.info(f"QA Generation complete in {inference_time_ms}ms with {len(citations)} citations extracted.")

            return QAResponse(
                query=user_query,
                generated_answer=generated_text,
                citations=citations,
                inference_time_ms=inference_time_ms
            )

        except Exception as e:
            raise map_qa_error(e, context=f"Processing QA Generation for query: '{user_query}'")

    @staticmethod
    def _context_text(evidence: List[NumberedChunk]) -> str:
        if not evidence:
            return NO_NEW_CONTEXT_BLOCK
        ordered = sorted(evidence, key=lambda n: n.doc_number)
        return build_numbered_context_block([(n.doc_number, n.chunk) for n in ordered]).formatted_text

    def _conversation_messages(
            self, system_prompt: str, packed: PackedConversation, user_query: str
    ) -> List[Dict[str, str]]:
        """
        system, then each retained turn as user (its evidence + question) / assistant (its answer), then the
        current question with its new evidence. Earlier turns only ever get appended, so the prompt of one turn is
        a prefix of the next one's while nothing is trimmed.
        """
        shown_numbers = {n.doc_number for n in packed.shown_evidence}
        messages = [{"role": "system", "content": system_prompt}]
        for turn in packed.turns:
            messages.append({"role": "user", "content": QA_CONVERSATION_USER_TEMPLATE.format(
                context_block=self._context_text(turn.evidence), user_query=turn.question)})
            # A tag whose excerpt was dropped for space would point at nothing the model can see
            messages.append({"role": "assistant",
                             "content": normalize_citation_tags(turn.answer, 0, shown_numbers)})
        messages.append({"role": "user", "content": QA_CONVERSATION_USER_TEMPLATE.format(
            context_block=self._context_text(packed.new_evidence), user_query=user_query)})
        return messages

    def generate_conversation(
            self,
            user_query: str,
            retrieved_chunks: List[RetrievedChunk],
            history: ConversationHistory,
            release_after: bool = True
    ) -> ConversationAnswer:
        """
        Answers a question that is part of a conversation. The model sees the earlier turns with the evidence
        they showed, under document numbers that stay the same for the whole conversation, so a follow-up can cite
        an excerpt an earlier turn found. Earlier answers are shown for context only; every citation must still
        resolve to an excerpt in the prompt.

        With no earlier turns this is generate() with conversation-wide numbering from history.next_doc_number.
        """
        logger.info(f"Executing Conversational QA Generation ({len(history.turns)} earlier turns): '{user_query}'")
        start_time = time.time()

        if not retrieved_chunks and not any(turn.evidence for turn in history.turns):
            logger.info("No evidence in this turn or the conversation; answering with the refusal without a model.")
            return ConversationAnswer(response=self._refusal(user_query, start_time))

        # Earlier turns' evidence is in the prompt too, so it counts towards "does the context hold documents"
        earlier = [n.chunk for turn in history.turns for n in turn.evidence]
        system_prompt = _localized(
            qa_system_prompt(has_document_chunks([*retrieved_chunks, *earlier])) + QA_CONVERSATION_RULES)

        try:
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)

            try:
                def count(text: str) -> int:
                    return self._adapter.count_tokens(loaded_model, text)

                packed = pack_conversation(
                    history=history,
                    current_question=user_query,
                    current_chunks=retrieved_chunks,
                    token_budget=self._adapter.get_conversation_token_budget(loaded_model, system_prompt),
                    turn_template_tokens=count(QA_CONVERSATION_USER_TEMPLATE.format(context_block="", user_query="")),
                    count_tokens=count
                )
                shown = packed.shown_evidence
                if not shown:
                    logger.warning("No evidence fits the conversation budget; answering with the refusal.")
                    return ConversationAnswer(response=self._refusal(user_query, start_time))
                if len(packed.turns) < len(history.turns):
                    logger.info(f"Conversation budget keeps {len(packed.turns)} of {len(history.turns)} earlier turns.")

                messages = self._conversation_messages(system_prompt, packed, user_query)
                generated_text = self._cited_or_refused(
                    loaded_model, messages, self._adapter.generate_conversation_answer(loaded_model, messages),
                    {n.doc_number for n in shown}
                )
            finally:
                if release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

            doc_mapping = packed.doc_mapping
            citations = parse_citations(
                generated_text=generated_text,
                chunks=[n.chunk for n in shown],
                doc_mapping=doc_mapping
            )

            inference_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Conversational QA complete in {inference_time_ms}ms with {len(citations)} citations.")

            return ConversationAnswer(
                response=QAResponse(
                    query=user_query,
                    generated_answer=generated_text,
                    citations=citations,
                    inference_time_ms=inference_time_ms
                ),
                new_chunk_ids=[n.chunk.chunk_id for n in packed.new_evidence]
            )

        except Exception as e:
            raise map_qa_error(e, context=f"Processing Conversational QA Generation for query: '{user_query}'")

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

        system_prompt = _localized(qa_system_prompt(has_document_chunks(retrieved_chunks)))
        try:
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter, wait_for_clear=True)

            try:
                packed, context_block = self._pack_context(loaded_model, user_query, retrieved_chunks, True,
                                                           system_prompt)
                if not packed:
                    logger.warning("No retrieved chunk fits the context budget; answering with the refusal.")
                    return (self._refusal(user_query, start_time),
                            AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT))

                generated_text = self._adapter.generate_answer(
                    model=loaded_model,
                    system_prompt=system_prompt,
                    context_block=context_block.formatted_text,
                    user_query=user_query
                )
                generated_text = self._cited_or_refused(
                    loaded_model, self._answer_messages(system_prompt, context_block.formatted_text, user_query),
                    generated_text, doc_numbers(context_block.doc_mapping)
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
