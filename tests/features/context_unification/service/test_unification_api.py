# Canvas: test_context_unification_api_v1
# Path: tests/features/context_unification/service/test_unification_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.visual_context_synthesis.domain.entities import VisualTimeline
from app.features.context_unification.service.unification_api import UnificationAPI
from app.features.context_unification.domain.entities import UnifiedTranscript
from app.features.context_unification.exceptions.process import TranscriptMismatchError


class TestUnificationAPI:
    @pytest.fixture
    def mock_repo(self):
        return MagicMock()

    @pytest.fixture
    def api(self, mock_repo):
        return UnificationAPI(repository=mock_repo)

    def test_execute_idempotent_return(self, api, mock_repo):
        source_id = uuid.uuid4()
        mock_existing = UnifiedTranscript(source_id=source_id, chunks=[])
        mock_repo.get_by_source_id.return_value = mock_existing

        dummy_audio = ContextualizedTranscript(source_id=source_id, chunks=[])
        result = api.execute(audio_transcript=dummy_audio)

        assert result == mock_existing
        mock_repo.get_by_source_id.assert_called_once_with(source_id)
        mock_repo.save.assert_not_called()

    def test_execute_raises_mismatch_error(self, api, mock_repo):
        mock_repo.get_by_source_id.return_value = None

        audio_source = uuid.uuid4()
        visual_source = uuid.uuid4()

        dummy_audio = ContextualizedTranscript(source_id=audio_source, chunks=[])
        dummy_visual = VisualTimeline(source_id=visual_source, visual_chunks=[])

        with pytest.raises(TranscriptMismatchError):
            api.execute(audio_transcript=dummy_audio, visual_timeline=dummy_visual)

    @patch("app.features.context_unification.service.unification_api.assemble_unified_chunks")
    def test_execute_success_flow(self, mock_assemble, api, mock_repo):
        source_id = uuid.uuid4()
        mock_repo.get_by_source_id.return_value = None

        dummy_audio = ContextualizedTranscript(source_id=source_id, chunks=[])
        mock_assemble.return_value = []
        mock_repo.save.side_effect = lambda x: x

        result = api.execute(audio_transcript=dummy_audio, visual_timeline=None)

        mock_assemble.assert_called_once_with(audio_transcript=dummy_audio, visual_timeline=None)
        mock_repo.save.assert_called_once()
        assert result.source_id == source_id