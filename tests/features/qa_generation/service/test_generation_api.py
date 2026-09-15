# Canvas: test_generation_api_v2
# Path: tests/features/qa_generation/service/test_generation_api.py

import pytest
import uuid
from unittest.mock import MagicMock
from app.features.qa_generation.service.generation_api import QAGenerationAPI
from app.features.qa_generation.domain.entities import AnswerCritique
from app.features.qa_generation.domain.enums import CritiqueVerdict
from app.features.qa_generation.exceptions.pipeline import JSONCritiqueParseError
from app.features.rag_retrieval.domain.entities import RetrievedChunk


class TestQAGenerationAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "adapter": MagicMock(),
            "orchestrator": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return QAGenerationAPI(adapter=mocks["adapter"], orchestrator=mocks["orchestrator"])

    def test_generate_success_flow(self, api, mocks):
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
            text_payload="Context text", start_sec=0, end_sec=5, relevance_score=1.0
        )
        object.__setattr__(chunk, 'physical_path', '/vault/vid.mp4')

        mocks["adapter"].generate_answer.return_value = "Answer [Doc 1]"

        response = api.generate("What happened?", [chunk])

        assert response.query == "What happened?"
        assert "[Doc 1]" in response.generated_answer
        assert len(response.citations) == 1

        # Verify VRAM Lifecycle
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once()

    def test_generate_error_mapping(self, api, mocks):
        mocks["orchestrator"].acquire_model.side_effect = RuntimeError("OOM")

        with pytest.raises(Exception):  # Caught and re-raised by map_qa_error
            api.generate("query", [])

    def test_generate_with_critique_sufficient_acquires_and_releases_once(self, api, mocks):
        chunk = RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="Context text",
                               start_sec=0, end_sec=5, relevance_score=1.0)
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

    def test_generate_with_critique_appends_extra_context_to_prompt(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Answer"
        mocks["adapter"].critique_answer.return_value = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        api.generate_with_critique("What happened?", [], extra_context="Reasoner says X matters")

        call_kwargs = mocks["adapter"].generate_answer.call_args.kwargs
        assert "Reasoner says X matters" in call_kwargs["user_query"]
        assert "What happened?" in call_kwargs["user_query"]

    def test_generate_with_critique_unparseable_verdict_fails_open_toward_distrust(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Answer"
        mocks["adapter"].critique_answer.side_effect = JSONCritiqueParseError("bad json")

        response, critique = api.generate_with_critique("query", [])

        assert critique.verdict == CritiqueVerdict.NEEDS_MORE_EVIDENCE
        assert critique.gap_description is not None
        mocks["orchestrator"].release_model.assert_called_once()

    def test_generate_with_critique_releases_even_on_draft_failure(self, api, mocks):
        mocks["adapter"].generate_answer.side_effect = RuntimeError("CUDA out of memory")

        with pytest.raises(Exception):
            api.generate_with_critique("query", [])

        mocks["orchestrator"].release_model.assert_called_once()

    def test_generate_with_critique_release_after_false_keeps_model_resident(self, api, mocks):
        mocks["adapter"].generate_answer.return_value = "Answer"
        mocks["adapter"].critique_answer.return_value = AnswerCritique(verdict=CritiqueVerdict.SUFFICIENT)

        api.generate_with_critique("query", [], release_after=False)

        mocks["orchestrator"].release_model.assert_not_called()