# Canvas: test_generation_api_v3
# Path: tests/features/qa_generation/service/test_generation_api.py

import pytest
import uuid
from unittest.mock import MagicMock
from app.features.qa_generation.service.generation_api import QAGenerationAPI
from app.features.qa_generation.domain.entities import AnswerCritique
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.data.prompts import NO_EVIDENCE_ANSWER
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
        mocks["adapter"].generate_answer.return_value = "Answer"
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
        mocks["adapter"].generate_answer.return_value = "Answer"
        mocks["adapter"].critique_answer.return_value = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        api.generate_with_critique("query", [_chunk()], release_after=False)

        mocks["orchestrator"].release_model.assert_not_called()
