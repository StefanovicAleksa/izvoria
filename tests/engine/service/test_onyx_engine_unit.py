# Canvas: test_onyx_engine_unit_v5
# Path: tests/engine/service/test_onyx_engine_unit.py

import pytest
import uuid
from unittest.mock import Mock, patch

from app.engine.service.onyx_engine import OnyxEngine
from app.engine.domain.enums import EngineState
from app.engine.domain.entities import IngestionReport
from app.engine.exceptions.lifecycle import EngineOfflineError
from app.features.rag_retrieval.domain.entities import SearchQuery
from app.features.qa_generation.domain.entities import QAResponse, QACitation
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult
from app.features.storage.domain.entities import Source, SourceType


class TestOnyxEngineUnit:
    @pytest.fixture
    def mock_async_service(self): return Mock()

    @pytest.fixture
    def mock_media_pipeline(self): return Mock()

    @pytest.fixture
    def mock_audio_pipeline(self): return Mock()

    @pytest.fixture
    def mock_live_pipeline(self): return Mock()

    @pytest.fixture
    def mock_live_verified_pipeline(self): return Mock()

    @pytest.fixture
    def mock_live_qa_pipeline(self): return Mock()

    @pytest.fixture
    def mock_storage_api(self): return Mock()

    @pytest.fixture
    def engine(self, mock_async_service, mock_media_pipeline, mock_audio_pipeline, mock_live_pipeline,
               mock_live_verified_pipeline, mock_live_qa_pipeline, mock_storage_api):
        return OnyxEngine(
            async_model_service=mock_async_service,
            media_pipeline=mock_media_pipeline,
            audio_pipeline=mock_audio_pipeline,
            live_pipeline=mock_live_pipeline,
            live_verified_pipeline=mock_live_verified_pipeline,
            live_qa_pipeline=mock_live_qa_pipeline,
            storage_api=mock_storage_api
        )

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_start_and_stop_lifecycle(self, mock_db_manager, engine, mock_async_service):
        assert engine.get_state() == EngineState.OFFLINE

        # Test Start
        engine.start()
        assert engine.get_state() == EngineState.ONLINE
        mock_db_manager.session.assert_called_once()

        # Test Stop
        engine.stop()
        assert engine.get_state() == EngineState.OFFLINE
        mock_async_service.shutdown.assert_called_once()
        mock_db_manager.dispose.assert_called_once()

    def test_ingest_videos_raises_if_offline(self, engine):
        with pytest.raises(EngineOfflineError):
            engine.ingest_videos("/dummy/path")

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_ingest_videos_happy_path(self, _, engine, mock_media_pipeline):
        engine.start()

        mock_media_pipeline.process_directory.return_value = {
            PipelinePhase.SCAN: PhaseBatchResult(phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
                                                 successful_count=1, items=[]),
            PipelinePhase.CLIP_EXTRACTION: PhaseBatchResult(phase=PipelinePhase.CLIP_EXTRACTION,
                                                            status=PhaseStatus.COMPLETED,
                                                            items=[Mock(success=True, extracted_clips_count=5)])
        }

        report = engine.ingest_videos("/dummy/path")

        assert isinstance(report, IngestionReport)
        assert report.total_files_scanned == 1
        assert report.clips_extracted == 5
        assert engine.get_state() == EngineState.ONLINE

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_search_standard_resolves_physical_paths(self, _, engine, mock_live_pipeline, mock_storage_api):
        engine.start()

        mock_chunk_video = Mock(
            source_id=uuid.uuid4(),
            extracted_file_id=uuid.uuid4(),
            text_payload="video text",
            start_sec=0,
            end_sec=5,
            relevance_score=0.9
        )
        mock_chunk_audio = Mock(
            source_id=uuid.uuid4(),
            extracted_file_id=None,
            text_payload="audio text",
            start_sec=0,
            end_sec=5,
            relevance_score=0.8
        )

        mock_search_result = Mock()
        mock_search_result.results = [mock_chunk_video, mock_chunk_audio]
        mock_search_result.execution_time_ms = 100

        mock_live_pipeline.execute_search.return_value = mock_search_result

        mock_storage_api.get_file_path.return_value = "/vault/video_clip.mp4"

        mock_source = Source(
            id=mock_chunk_audio.source_id,
            name="audio",
            source_type=SourceType.AUDIO,
            content_hash="hash",
            file_id=uuid.uuid4()
        )
        mock_source.file = Mock(file_path="/vault/audio_source.wav")
        mock_storage_api.get_source.return_value = mock_source

        result = engine.search_standard(SearchQuery(prompt="test"))

        assert len(result.results) == 2
        assert getattr(result.results[0], 'physical_path') == "/vault/video_clip.mp4"
        assert getattr(result.results[1], 'physical_path') == "/vault/audio_source.wav"

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_search_verified_resolves_physical_paths(self, _, engine, mock_live_verified_pipeline, mock_storage_api):
        engine.start()

        mock_chunk_audio = Mock(
            source_id=uuid.uuid4(),
            extracted_file_id=None,
            text_payload="audio text",
            start_sec=0,
            end_sec=5,
            relevance_score=0.8
        )

        mock_search_result = Mock()
        mock_search_result.results = [mock_chunk_audio]

        mock_live_verified_pipeline.execute_verified_search.return_value = mock_search_result

        mock_source = Source(
            id=mock_chunk_audio.source_id,
            name="audio",
            source_type=SourceType.AUDIO,
            content_hash="hash",
            file_id=uuid.uuid4()
        )
        mock_source.file = Mock(file_path="/vault/audio_source.wav")
        mock_storage_api.get_source.return_value = mock_source

        result = engine.search_verified(SearchQuery(prompt="test"))

        assert len(result.results) == 1
        assert getattr(result.results[0], 'physical_path') == "/vault/audio_source.wav"

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_chat_resolves_citation_paths(self, _, engine, mock_live_qa_pipeline, mock_storage_api):
        engine.start()

        # An audio-only chunk has no extracted clip, so its citation points at the source file
        mock_source_id = uuid.uuid4()
        real_citation = QACitation(
            doc_id="[Doc 1]",
            chunk_id=uuid.uuid4(),
            source_id=mock_source_id,
            start_sec=0.0,
            end_sec=5.0
        )

        mock_qa_response = QAResponse(
            query="test",
            generated_answer="This is a test answer.",
            citations=[real_citation],
            inference_time_ms=100
        )

        mock_live_qa_pipeline.execute_chat.return_value = mock_qa_response

        mock_source = Source(
            id=mock_source_id,
            name="video",
            source_type=SourceType.VIDEO,
            content_hash="hash",
            file_id=uuid.uuid4()
        )
        mock_source.file = Mock(file_path="/vault/video_clip.mp4")
        mock_storage_api.get_source.return_value = mock_source

        result = engine.chat(SearchQuery(prompt="test"))

        assert len(result.citations) == 1
        assert result.citations[0].physical_path == "/vault/video_clip.mp4"
        mock_storage_api.get_source.assert_called_once_with(mock_source_id)

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_chat_resolves_citations_to_their_extracted_clip_like_search(self, _, engine, mock_live_qa_pipeline,
                                                                         mock_storage_api):
        engine.start()
        clip_id = uuid.uuid4()
        citation = QACitation(doc_id="[Doc 1]", chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
                              extracted_file_id=clip_id, start_sec=1415.1, end_sec=1510.8)
        mock_live_qa_pipeline.execute_chat.return_value = QAResponse(
            query="test", generated_answer="Answer [Doc 1].", citations=[citation]
        )
        mock_storage_api.get_file_path.return_value = "/vault/clips/clip_1415.mp4"

        result = engine.chat(SearchQuery(prompt="test"))

        assert result.citations[0].physical_path == "/vault/clips/clip_1415.mp4"
        mock_storage_api.get_file_path.assert_called_once_with(clip_id)
        mock_storage_api.get_source.assert_not_called()

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_ingestion_cleanup_releases_all_models(self, _, engine, mock_media_pipeline, mock_async_service):
        engine.start()
        mock_media_pipeline.process_directory.return_value = {}

        engine.ingest_videos("/dummy/path")

        mock_async_service.release_all.assert_called_once_with()
        assert engine.get_state() == EngineState.ONLINE

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_failed_chat_releases_all_models(self, _, engine, mock_live_qa_pipeline, mock_async_service):
        engine.start()
        mock_live_qa_pipeline.execute_chat.side_effect = RuntimeError("Scrubber crashed")

        with pytest.raises(Exception):
            engine.chat(SearchQuery(prompt="test"))

        mock_async_service.release_all.assert_called_once_with()

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_failed_search_still_raises_if_release_fails(self, _, engine, mock_live_pipeline, mock_async_service):
        engine.start()
        mock_live_pipeline.execute_search.side_effect = RuntimeError("DB down")
        mock_async_service.release_all.side_effect = RuntimeError("release failed")

        with pytest.raises(Exception, match="DB down"):
            engine.search_standard(SearchQuery(prompt="test"))
