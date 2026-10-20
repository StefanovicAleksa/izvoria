# Path: tests/orchestration/phases/test_source_scope_phase_unit.py

import uuid
from unittest.mock import MagicMock, patch

from app.features.qa_generation.domain.entities import QAResponse
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.source_profile.domain.entities import SourceScope
from app.orchestration.service.phases.source_scope_phase import SourceScopePhase


def _chunk(text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload=text, raw_text=text,
                          start_sec=0, end_sec=1, relevance_score=0.0)


class TestSourceScopePhase:
    def test_resolve_returns_the_api_scope(self):
        api = MagicMock()
        scope = SourceScope(source_id=uuid.uuid4(), title="Lecture 4")
        api.resolve_scope.return_value = scope

        assert SourceScopePhase(api).resolve(uuid.uuid4(), "lecture 4?") is scope
        api.resolve_scope.assert_called_once_with("lecture 4?")

    def test_a_failed_resolution_searches_everything(self):
        api = MagicMock()
        api.resolve_scope.side_effect = RuntimeError("db down")

        assert SourceScopePhase(api).resolve(uuid.uuid4(), "lecture 4?") is None

    def test_fits_whole_estimates_tokens_against_the_setting(self):
        with patch("app.orchestration.service.phases.source_scope_phase.settings") as settings:
            settings.SOURCE_SCOPE_WHOLE_READ_MAX_TOKENS = 10
            assert SourceScopePhase.fits_whole([_chunk("a" * 15), _chunk("b" * 15)]) is True
            assert SourceScopePhase.fits_whole([_chunk("a" * 31)]) is False
            assert SourceScopePhase.fits_whole([]) is False

    def test_mark_names_the_source_on_the_answer(self):
        response = QAResponse(query="q", generated_answer="a")
        scope = SourceScope(source_id=uuid.uuid4(), title="Lecture 4")

        assert SourceScopePhase.mark(response, scope).scoped_source_title == "Lecture 4"
        assert SourceScopePhase.mark(response, None) is response
        assert SourceScopePhase.mark(None, scope) is None
