# Canvas: test_generation_api_v3
# Path: tests/features/qa_generation/service/test_generation_api.py

import pytest
import uuid
from unittest.mock import MagicMock
from app.features.qa_generation.service.generation_api import QAGenerationAPI
from app.features.qa_generation.domain.entities import (
    AnswerCritique, ConversationHistory, ConversationTurn, NumberedChunk
)
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.data.prompts import (
    CITATION_RETRY_PROMPT, NO_EVIDENCE_ANSWER, QA_USER_TEMPLATE, NO_NEW_CONTEXT_BLOCK, QA_CONVERSATION_RULES, QA_SYSTEM_PROMPT
)
from app.features.qa_generation.exceptions.base import QAGenerationError
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError
from app.features.rag_retrieval.domain.entities import RetrievedChunk


def _chunk(text: str = "Context text", start: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=text,
                          start_sec=start, end_sec=start + 5, relevance_score=1.0)


class TestQAGenerationAPI:
    @pytest.fixture
    def mocks(self):
        adapter = MagicMock()
        adapter.get_context_token_budget.return_value = 4000
        adapter.count_tokens.side_effect = lambda model, text: len(text.split())
        adapter.generate_answer.return_value = "Answer [Doc 1]."
        return {
            "adapter": adapter,
            "orchestrator": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return QAGenerationAPI(adapter=mocks["adapter"], orchestrator=mocks["orchestrator"])

    def test_generate_success_flow(self, api, mocks):
        chunk = _chunk()
        object.__setattr__(chunk, 'physical_path', '/vault/vid.mp4')

        mocks["adapter"].generate_answer.return_value = "Answer [Doc 1]"

        response = api.generate("What happened?", [chunk])

        assert response.query == "What happened?"
        assert "[Doc 1]" in response.generated_answer
        assert len(response.citations) == 1

        # Verify VRAM Lifecycle
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once()
        mocks["adapter"].get_context_token_budget.assert_called_once_with(
            mocks["orchestrator"].acquire_model.return_value, "What happened?", False
        )

    def test_generate_error_mapping(self, api, mocks):
        mocks["orchestrator"].acquire_model.side_effect = RuntimeError("OOM")

        with pytest.raises(Exception):  # Caught and re-raised by map_qa_error
            api.generate("query", [_chunk()])

    def test_generate_without_evidence_refuses_without_loading_the_model(self, api, mocks):
        response = api.generate("What happened?", [])

        assert response.generated_answer == NO_EVIDENCE_ANSWER
        assert response.citations == []
        mocks["orchestrator"].acquire_model.assert_not_called()
        mocks["adapter"].generate_answer.assert_not_called()

    def test_generate_drops_chunks_beyond_the_token_budget(self, api, mocks):
        best, too_long, short = _chunk("alpha beta", 0), _chunk("word " * 50, 10), _chunk("gamma", 20)
        # Each document also costs DOC_HEADER_TOKENS (24): best 26, too_long 74, short 25
        mocks["adapter"].get_context_token_budget.return_value = 60
        mocks["adapter"].generate_answer.return_value = "A [Doc 1] B [Doc 2] C [Doc 3]"

        response = api.generate("Q", [best, too_long, short])

        context = mocks["adapter"].generate_answer.call_args.kwargs["context_block"]
        assert "alpha beta" in context and "gamma" in context
        assert "word word" not in context
        # [Doc 3] no longer exists, so it is removed from the answer rather than left dangling
        assert response.generated_answer == "A [Doc 1] B [Doc 2] C "
        assert [c.chunk_id for c in response.citations] == [best.chunk_id, short.chunk_id]

    def test_generate_refuses_when_nothing_fits_and_still_releases(self, api, mocks):
        mocks["adapter"].get_context_token_budget.return_value = 0

        response = api.generate("Q", [_chunk()])

        assert response.generated_answer == NO_EVIDENCE_ANSWER
        mocks["adapter"].generate_answer.assert_not_called()
        mocks["orchestrator"].release_model.assert_called_once()

    def test_generate_expands_grouped_citations_in_answer_and_citations(self, api, mocks):
        chunks = [_chunk("one", 0), _chunk("two", 10), _chunk("three", 20)]
        mocks["adapter"].generate_answer.return_value = "Claim [Doc 1, 3]. Other [Docs 1-2]."

        response = api.generate("Q", chunks)

        assert response.generated_answer == "Claim [Doc 1][Doc 3]. Other [Doc 1][Doc 2]."
        assert [c.doc_id for c in response.citations] == ["[Doc 1]", "[Doc 2]", "[Doc 3]"]

    def test_generate_with_critique_sufficient_acquires_and_releases_once(self, api, mocks):
        chunk = _chunk()
        mocks["adapter"].generate_answer.return_value = "Answer [Doc 1]"
        mocks["adapter"].critique_answer.return_value = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        response, critique = api.generate_with_critique("What happened?", [chunk])

        assert response.query == "What happened?"
        assert "[Doc 1]" in response.generated_answer
        assert critique.verdict == CritiqueVerdict.SUFFICIENT

        # Both calls happen on the same acquisition: one acquire, one release.
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once()
        mocks["adapter"].generate_answer.assert_called_once()
        mocks["adapter"].critique_answer.assert_called_once()
        # The budget must leave room for the critique prompt as well
        mocks["adapter"].get_context_token_budget.assert_called_once_with(
            mocks["orchestrator"].acquire_model.return_value, "What happened?", True
        )

    def test_generate_with_critique_shows_the_model_only_the_question_and_evidence(self, api, mocks):
        chunk = _chunk()
        mocks["adapter"].generate_answer.return_value = "Answer [Doc 1]"
        mocks["adapter"].critique_answer.return_value = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        api.generate_with_critique("What happened?", [chunk])

        draft_kwargs = mocks["adapter"].generate_answer.call_args.kwargs
        critique_kwargs = mocks["adapter"].critique_answer.call_args.kwargs
        assert draft_kwargs["user_query"] == "What happened?"
        assert critique_kwargs["evidence_block"] == draft_kwargs["context_block"]
        assert critique_kwargs["draft_answer"] == "Answer [Doc 1]"

    def test_generate_with_critique_without_evidence_refuses_as_sufficient(self, api, mocks):
        response, critique = api.generate_with_critique("What happened?", [])

        assert response.generated_answer == NO_EVIDENCE_ANSWER
        assert critique.verdict == CritiqueVerdict.SUFFICIENT
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_generate_with_critique_unparseable_verdict_fails_open_toward_distrust(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Answer [Doc 1]."
        mocks["adapter"].critique_answer.side_effect = JSONCritiqueParseError("bad json")

        response, critique = api.generate_with_critique("query", [_chunk()])

        assert critique.verdict == CritiqueVerdict.NEEDS_MORE_EVIDENCE
        assert critique.gap_description is not None
        mocks["orchestrator"].release_model.assert_called_once()

    def test_generate_with_critique_releases_even_on_draft_failure(self, api, mocks):
        mocks["adapter"].generate_answer.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(Exception):
            api.generate_with_critique("query", [_chunk()])

        mocks["orchestrator"].release_model.assert_called_once()

    def test_generate_with_critique_release_after_false_keeps_model_resident(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Answer [Doc 1]."
        mocks["adapter"].critique_answer.return_value = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        api.generate_with_critique("query", [_chunk()], release_after=False)

        mocks["orchestrator"].release_model.assert_not_called()


class TestGenerateConversation:
    @pytest.fixture
    def mocks(self):
        adapter = MagicMock()
        adapter.get_conversation_token_budget.return_value = 100_000
        adapter.count_tokens.side_effect = lambda model, text: len(text.split())
        return {"adapter": adapter, "orchestrator": MagicMock()}

    @pytest.fixture
    def api(self, mocks):
        return QAGenerationAPI(adapter=mocks["adapter"], orchestrator=mocks["orchestrator"])

    @pytest.fixture
    def earlier(self):
        chunk = _chunk("The vote was nine to zero.", 0)
        history = ConversationHistory(
            turns=[ConversationTurn(question="How did they vote?", answer="Nine to zero [Doc 1].",
                                    evidence=[NumberedChunk(doc_number=1, chunk=chunk)])],
            next_doc_number=2
        )
        return chunk, history

    def test_builds_the_conversation_and_numbers_new_evidence_after_the_history(self, api, mocks, earlier):
        old_chunk, history = earlier
        new_chunk = _chunk("Then the chair adjourned.", 60)
        mocks["adapter"].generate_conversation_answer.return_value = "It passed [Doc 1], then adjourned [Doc 2]."

        answer = api.generate_conversation("What happened after that?", [new_chunk], history)

        loaded = mocks["orchestrator"].acquire_model.return_value
        mocks["orchestrator"].acquire_model.assert_called_once_with(api._manifest, mocks["adapter"],
                                                                    wait_for_clear=True)
        mocks["orchestrator"].release_model.assert_called_once_with(loaded, mocks["adapter"], flush=True)
        mocks["adapter"].get_conversation_token_budget.assert_called_once_with(
            loaded, QA_SYSTEM_PROMPT + QA_CONVERSATION_RULES
        )

        model, messages = mocks["adapter"].generate_conversation_answer.call_args[0]
        assert model is loaded
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[0]["content"] == QA_SYSTEM_PROMPT + QA_CONVERSATION_RULES
        assert "[Doc 1]" in messages[1]["content"] and "nine to zero" in messages[1]["content"]
        assert "How did they vote?" in messages[1]["content"]
        assert messages[2]["content"] == "Nine to zero [Doc 1]."
        assert "[Doc 2]" in messages[3]["content"] and "adjourned" in messages[3]["content"]
        assert "What happened after that?" in messages[3]["content"]

        assert answer.new_chunk_ids == [new_chunk.chunk_id]
        assert answer.response.generated_answer == "It passed [Doc 1], then adjourned [Doc 2]."
        assert [(c.doc_id, c.chunk_id) for c in answer.response.citations] == [
            ("[Doc 1]", old_chunk.chunk_id), ("[Doc 2]", new_chunk.chunk_id)
        ]

    def test_follow_up_without_new_evidence_still_answers_from_the_history(self, api, mocks, earlier):
        old_chunk, history = earlier
        mocks["adapter"].generate_conversation_answer.return_value = "Unanimously [Doc 1]."

        answer = api.generate_conversation("Was it unanimous?", [], history)

        messages = mocks["adapter"].generate_conversation_answer.call_args[0][1]
        assert NO_NEW_CONTEXT_BLOCK in messages[-1]["content"]
        assert answer.new_chunk_ids == []
        assert [c.chunk_id for c in answer.response.citations] == [old_chunk.chunk_id]

    def test_no_evidence_anywhere_refuses_without_loading_the_model(self, api, mocks):
        history = ConversationHistory(turns=[ConversationTurn(question="Hi?", answer="Hello.")])

        answer = api.generate_conversation("Anything?", [], history)

        assert answer.response.generated_answer == NO_EVIDENCE_ANSWER
        assert answer.new_chunk_ids == []
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_nothing_fitting_the_budget_refuses_and_releases(self, api, mocks):
        mocks["adapter"].get_conversation_token_budget.return_value = 0

        answer = api.generate_conversation("Q", [_chunk()], ConversationHistory())

        assert answer.response.generated_answer == NO_EVIDENCE_ANSWER
        mocks["adapter"].generate_conversation_answer.assert_not_called()
        mocks["orchestrator"].release_model.assert_called_once()

    def test_tags_to_excerpts_not_shown_are_removed(self, api, mocks, earlier):
        _, history = earlier
        mocks["adapter"].generate_conversation_answer.return_value = "Yes [Doc 1][Doc 7]."

        answer = api.generate_conversation("Q", [], history)

        assert answer.response.generated_answer == "Yes [Doc 1]."

    def test_earlier_answer_tags_pointing_at_dropped_evidence_are_removed(self, api, mocks):
        dropped = _chunk("word " * 300, 0)
        history = ConversationHistory(
            turns=[ConversationTurn(question="q1", answer="Long [Doc 1].",
                                    evidence=[NumberedChunk(doc_number=1, chunk=dropped)])],
            next_doc_number=2
        )
        # Room for both turns' text and the short excerpt (~170 counted words), not the 300-word one
        mocks["adapter"].get_conversation_token_budget.return_value = 200
        mocks["adapter"].generate_conversation_answer.return_value = "Short [Doc 2]."

        api.generate_conversation("q2", [_chunk("short", 5)], history)

        messages = mocks["adapter"].generate_conversation_answer.call_args[0][1]
        assert messages[2]["content"] == "Long ."
        assert "word word" not in messages[1]["content"]

    def test_release_after_false_keeps_the_model(self, api, mocks, earlier):
        _, history = earlier
        mocks["adapter"].generate_conversation_answer.return_value = "A [Doc 1]."

        api.generate_conversation("Q", [], history, release_after=False)

        mocks["orchestrator"].release_model.assert_not_called()

    def test_generation_failure_releases_and_maps_the_error(self, api, mocks, earlier):
        _, history = earlier
        mocks["adapter"].generate_conversation_answer.side_effect = RuntimeError("llama crashed")

        with pytest.raises(QAGenerationError):
            api.generate_conversation("Q", [], history)

        mocks["orchestrator"].release_model.assert_called_once()


class TestCitationGuard:
    @pytest.fixture
    def mocks(self):
        adapter = MagicMock()
        adapter.get_context_token_budget.return_value = 4000
        adapter.get_conversation_token_budget.return_value = 100_000
        adapter.count_tokens.side_effect = lambda model, text: len(text.split())
        return {"adapter": adapter, "orchestrator": MagicMock()}

    @pytest.fixture
    def api(self, mocks):
        return QAGenerationAPI(adapter=mocks["adapter"], orchestrator=mocks["orchestrator"])

    def test_uncited_answer_is_rewritten_with_citations_in_the_same_acquisition(self, api, mocks):
        chunk = _chunk("The stop goes above the high.")
        mocks["adapter"].generate_answer.return_value = "The stop goes above the high."
        mocks["adapter"].generate_conversation_answer.return_value = "The stop goes above the high [Doc 1]."

        response = api.generate("Where is the stop?", [chunk])

        assert response.generated_answer == "The stop goes above the high [Doc 1]."
        assert [c.chunk_id for c in response.citations] == [chunk.chunk_id]
        loaded = mocks["orchestrator"].acquire_model.return_value
        model, messages = mocks["adapter"].generate_conversation_answer.call_args[0]
        assert model is loaded
        assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
        assert messages[0]["content"] == QA_SYSTEM_PROMPT
        assert messages[1]["content"].startswith(QA_USER_TEMPLATE.split("{context_block}")[0])
        assert "Where is the stop?" in messages[1]["content"]
        assert messages[2]["content"] == "The stop goes above the high."
        assert messages[3]["content"] == CITATION_RETRY_PROMPT
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once()

    def test_answer_still_uncited_after_the_retry_becomes_the_refusal(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Something."
        mocks["adapter"].generate_conversation_answer.return_value = "Something else [Doc 9]."

        response = api.generate("Q", [_chunk()])

        assert response.generated_answer == NO_EVIDENCE_ANSWER
        assert response.citations == []

    def test_retry_may_refuse(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Something."
        mocks["adapter"].generate_conversation_answer.return_value = NO_EVIDENCE_ANSWER

        assert api.generate("Q", [_chunk()]).generated_answer == NO_EVIDENCE_ANSWER

    @pytest.mark.parametrize("answer", ["Cited [Doc 1].", NO_EVIDENCE_ANSWER])
    def test_cited_answers_and_refusals_are_not_retried(self, api, mocks, answer):
        mocks["adapter"].generate_answer.return_value = answer

        assert api.generate("Q", [_chunk()]).generated_answer == answer
        mocks["adapter"].generate_conversation_answer.assert_not_called()

    def test_critique_checks_the_cited_rewrite(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Uncited."
        mocks["adapter"].generate_conversation_answer.return_value = "Cited [Doc 1]."
        mocks["adapter"].critique_answer.return_value = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        response, _ = api.generate_with_critique("Q", [_chunk()])

        assert response.generated_answer == "Cited [Doc 1]."
        assert mocks["adapter"].critique_answer.call_args.kwargs["draft_answer"] == "Cited [Doc 1]."

    def test_conversation_answer_is_retried_with_the_whole_conversation(self, api, mocks):
        chunk = _chunk()
        mocks["adapter"].generate_conversation_answer.side_effect = ["Uncited.", "Cited [Doc 1]."]

        answer = api.generate_conversation("Q", [chunk], ConversationHistory())

        assert answer.response.generated_answer == "Cited [Doc 1]."
        first, second = [call[0][1] for call in mocks["adapter"].generate_conversation_answer.call_args_list]
        assert second == first + [{"role": "assistant", "content": "Uncited."},
                                  {"role": "user", "content": CITATION_RETRY_PROMPT}]


class TestAnswersInTheApplianceLanguage:
    """A Serbian appliance answers, refuses and is asked to cite in Serbian; the grounding rules are unchanged."""

    SERBIAN_REFUSAL = "Ne mogu da odgovorim na ovo na osnovu dostupnog konteksta."

    @pytest.fixture
    def mocks(self):
        adapter = MagicMock()
        adapter.get_context_token_budget.return_value = 4000
        adapter.get_conversation_token_budget.return_value = 100_000
        adapter.count_tokens.side_effect = lambda model, text: len(text.split())
        return {"adapter": adapter, "orchestrator": MagicMock()}

    @pytest.fixture
    def api(self, mocks, appliance_language):
        appliance_language("sr")
        return QAGenerationAPI(adapter=mocks["adapter"], orchestrator=mocks["orchestrator"])

    def test_no_evidence_refuses_in_serbian(self, api):
        assert api.generate("Šta je rekao lekar?", []).generated_answer == self.SERBIAN_REFUSAL

    def test_system_prompt_uses_the_serbian_refusal_and_asks_for_serbian(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Lekar je rekao da je stabilan [Doc 1]."

        api.generate("Šta je rekao lekar?", [_chunk()])

        system_prompt = mocks["adapter"].generate_answer.call_args.kwargs["system_prompt"]
        assert self.SERBIAN_REFUSAL in system_prompt
        assert NO_EVIDENCE_ANSWER not in system_prompt
        assert "<OUTPUT_LANGUAGE>" in system_prompt and "in Serbian" in system_prompt

    @pytest.mark.parametrize("refusal", [SERBIAN_REFUSAL, NO_EVIDENCE_ANSWER])
    def test_a_refusal_in_either_language_is_not_sent_back_for_citations(self, api, mocks, refusal):
        mocks["adapter"].generate_answer.return_value = refusal

        assert api.generate("Q", [_chunk()]).generated_answer == refusal
        mocks["adapter"].generate_conversation_answer.assert_not_called()

    def test_citation_retry_is_localized_and_uncited_answer_becomes_the_serbian_refusal(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Nešto."
        mocks["adapter"].generate_conversation_answer.return_value = "Nešto drugo."

        response = api.generate("Q", [_chunk()])

        retry = mocks["adapter"].generate_conversation_answer.call_args[0][1][-1]["content"]
        assert self.SERBIAN_REFUSAL in retry and NO_EVIDENCE_ANSWER not in retry
        assert response.generated_answer == self.SERBIAN_REFUSAL

    def test_conversation_system_prompt_is_localized(self, api, mocks):
        mocks["adapter"].generate_conversation_answer.return_value = "Da [Doc 1]."

        api.generate_conversation("Q", [_chunk()], ConversationHistory())

        system_prompt = mocks["adapter"].generate_conversation_answer.call_args[0][1][0]["content"]
        assert system_prompt.startswith(QA_SYSTEM_PROMPT.replace(NO_EVIDENCE_ANSWER, self.SERBIAN_REFUSAL))
        assert "<CONVERSATION_RULES>" in system_prompt and system_prompt.rstrip().endswith("</OUTPUT_LANGUAGE>")
