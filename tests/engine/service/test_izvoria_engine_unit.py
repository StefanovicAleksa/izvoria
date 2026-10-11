# Canvas: test_izvoria_engine_unit_v5
# Path: tests/engine/service/test_izvoria_engine_unit.py

import pytest
import uuid
from unittest.mock import Mock, call, patch

from app.engine.service.izvoria_engine import IzvoriaEngine
from app.engine.domain.enums import EngineState
from app.engine.exceptions.base import EngineError
from app.core.database.exceptions import DBMigrationError
from app.engine.domain.entities import IngestionReport, StorageCompactionReport
from app.features.rag_retrieval.domain.entities import IndexedSource
from app.engine.exceptions.lifecycle import EngineBusyError, EngineOfflineError
from app.features.rag_retrieval.domain.entities import SearchQuery
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import QAResponse, QACitation, ConversationHistory
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.domain.enums import SessionTurnMode
from app.features.search_history.domain.enums import SearchHistoryMode
from app.features.deep_research_report.domain.entities import (
    OutlineSection, ResearchPlan, SubQuestion,
)
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import (
    DocumentConversionJobResult, PhaseBatchResult, ReasonedChatResponse, ConversationChatResult,
    StorageCompactionJobResult,
)
from app.features.storage.domain.entities import Source, SourceType


@pytest.fixture(autouse=True)
def mock_upgrade_to_head():
    """start() migrates the database; every test here runs against a mocked one."""
    with patch('app.engine.service.izvoria_engine.upgrade_to_head') as upgrade:
        yield upgrade


class TestIzvoriaEngineUnit:
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
    def mock_live_reasoning_pipeline(self): return Mock()

    @pytest.fixture
    def mock_session_memory_api(self):
        api = Mock()
        api.get_history.return_value = []
        api.format_prior_context.return_value = None
        return api

    @pytest.fixture
    def mock_retrieval_api(self): return Mock()

    @pytest.fixture
    def engine(self, mock_async_service, mock_media_pipeline, mock_audio_pipeline, mock_live_pipeline,
               mock_live_verified_pipeline, mock_live_qa_pipeline, mock_storage_api,
               mock_live_reasoning_pipeline, mock_session_memory_api, mock_retrieval_api):
        return IzvoriaEngine(
            async_model_service=mock_async_service,
            media_pipeline=mock_media_pipeline,
            audio_pipeline=mock_audio_pipeline,
            live_pipeline=mock_live_pipeline,
            live_verified_pipeline=mock_live_verified_pipeline,
            live_qa_pipeline=mock_live_qa_pipeline,
            storage_api=mock_storage_api,
            live_reasoning_pipeline=mock_live_reasoning_pipeline,
            session_memory_api=mock_session_memory_api,
            retrieval_api=mock_retrieval_api
        )

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_start_and_stop_lifecycle(self, mock_db_manager, engine, mock_async_service):
        assert engine.get_state() == EngineState.OFFLINE

        # Test Start
        engine.start()
        assert engine.get_state() == EngineState.ONLINE
        mock_db_manager.get_engine.assert_called_once_with()

        # Test Stop
        engine.stop()
        assert engine.get_state() == EngineState.OFFLINE
        mock_async_service.shutdown.assert_called_once()
        mock_db_manager.dispose.assert_called_once()

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_stop_frees_every_resident_model_after_the_worker_has_drained(self, mock_db_manager, engine,
                                                                          mock_async_service):
        order = []
        mock_async_service.shutdown.side_effect = lambda: order.append("shutdown")
        mock_async_service.release_all.side_effect = lambda: order.append("release_all")
        engine.start()

        engine.stop()

        assert order == ["shutdown", "release_all"]

    def test_active_language_is_the_settings_language(self, engine, appliance_language):
        appliance_language("sr")

        assert engine.active_language == "sr"

    @pytest.mark.parametrize("state", [EngineState.BOOTING, EngineState.INGESTING, EngineState.RESEARCHING])
    def test_require_idle_refuses_while_a_job_holds_the_engine(self, engine, state):
        engine._state = state

        with pytest.raises(EngineBusyError, match=state.value):
            engine.require_idle()

    @pytest.mark.parametrize("state", [EngineState.OFFLINE, EngineState.ONLINE, EngineState.ERROR])
    def test_require_idle_passes_otherwise(self, engine, state):
        engine._state = state

        engine.require_idle()

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_start_fails_when_the_migration_fails(self, mock_db_manager, engine, mock_upgrade_to_head):
        mock_upgrade_to_head.side_effect = DBMigrationError("column already exists")

        with pytest.raises(EngineError):
            engine.start()

        assert engine.get_state() == EngineState.ERROR

    def test_ingest_videos_raises_if_offline(self, engine):
        with pytest.raises(EngineOfflineError):
            engine.ingest_videos("/dummy/path")

    @patch('app.engine.service.izvoria_engine.db_manager')
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

    @patch('app.engine.service.izvoria_engine.db_manager')
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

    @patch('app.engine.service.izvoria_engine.db_manager')
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

    @patch('app.engine.service.izvoria_engine.db_manager')
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

    @patch('app.engine.service.izvoria_engine.db_manager')
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

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_ingestion_cleanup_releases_all_models(self, _, engine, mock_media_pipeline, mock_async_service):
        engine.start()
        mock_media_pipeline.process_directory.return_value = {}

        engine.ingest_videos("/dummy/path")

        # Once before (warm chat models leave the GPU) and once after
        assert mock_async_service.release_all.call_args_list == [call(), call()]
        assert engine.get_state() == EngineState.ONLINE

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_ingestion_evicts_warm_models_before_the_pipeline_runs(
            self, _, engine, mock_media_pipeline, mock_async_service
    ):
        engine.start()
        released_before_run = []
        mock_media_pipeline.process_directory.side_effect = \
            lambda *a, **k: released_before_run.append(mock_async_service.release_all.call_count) or {}

        engine.ingest_videos("/dummy/path")

        assert released_before_run == [1]

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_failed_chat_releases_all_models(self, _, engine, mock_live_qa_pipeline, mock_async_service):
        engine.start()
        mock_live_qa_pipeline.execute_chat.side_effect = RuntimeError("Scrubber crashed")

        with pytest.raises(Exception):
            engine.chat(SearchQuery(prompt="test"))

        mock_async_service.release_all.assert_called_once_with()

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_failed_search_still_raises_if_release_fails(self, _, engine, mock_live_pipeline, mock_async_service):
        engine.start()
        mock_live_pipeline.execute_search.side_effect = RuntimeError("DB down")
        mock_async_service.release_all.side_effect = RuntimeError("release failed")

        with pytest.raises(Exception, match="DB down"):
            engine.search_standard(SearchQuery(prompt="test"))

    @staticmethod
    def _chunk(chunk_id=None):
        return RetrievedChunk(chunk_id=chunk_id or uuid.uuid4(), source_id=uuid.uuid4(), text_payload="t",
                              start_sec=0.0, end_sec=1.0, relevance_score=0.0)

    @staticmethod
    def _turn(session_id, first_doc_number, evidence_ids, mode=SessionTurnMode.CHAT):
        return SessionTurn(id=uuid.uuid4(), session_id=session_id, question="Earlier?", answer_text="Earlier [Doc 1].",
                           mode=mode, evidence_chunk_ids=evidence_ids, first_doc_number=first_doc_number,
                           created_at="2026-01-01T00:00:00Z")

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_chat_with_session_runs_a_conversation_turn_and_records_it(
            self, _, engine, mock_live_qa_pipeline, mock_session_memory_api, mock_retrieval_api
    ):
        engine.start()
        session_id = uuid.uuid4()
        earlier = self._chunk()
        turns = [self._turn(session_id, 1, [earlier.chunk_id])]
        mock_session_memory_api.get_history.return_value = turns
        mock_session_memory_api.format_prior_context.return_value = "prior"
        mock_retrieval_api.get_chunks.return_value = [earlier]
        new_id = uuid.uuid4()
        qa_response = QAResponse(query="And then?", generated_answer="Then [Doc 2].", citations=[])
        mock_live_qa_pipeline.execute_conversation_chat.return_value = ConversationChatResult(
            qa_response=qa_response, standalone_query="What happened after the vote?", new_chunk_ids=[new_id]
        )

        result = engine.chat(SearchQuery(prompt="And then?"), session_id=session_id)

        assert result == qa_response
        mock_session_memory_api.get_history.assert_called_once_with(session_id, owner_id=None)
        mock_retrieval_api.get_chunks.assert_called_once_with([earlier.chunk_id])
        _, query, history, prior = mock_live_qa_pipeline.execute_conversation_chat.call_args[0]
        assert query.prompt == "And then?"
        assert prior == "prior"
        assert history.next_doc_number == 2
        assert [(n.doc_number, n.chunk.chunk_id) for n in history.turns[0].evidence] == [(1, earlier.chunk_id)]
        mock_session_memory_api.record_turn.assert_called_once_with(
            session_id, "And then?", "Then [Doc 2].",
            mode=SessionTurnMode.CHAT,
            standalone_query="What happened after the vote?",
            evidence_chunk_ids=[new_id],
            first_doc_number=2,
            owner_id=None
        )
        mock_live_qa_pipeline.execute_chat.assert_not_called()

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_chat_without_session_does_not_touch_conversation_memory(
            self, _, engine, mock_live_qa_pipeline, mock_session_memory_api
    ):
        engine.start()
        mock_live_qa_pipeline.execute_chat.return_value = QAResponse(query="q", generated_answer="a", citations=[])

        engine.chat(SearchQuery(prompt="q"))

        mock_session_memory_api.get_history.assert_not_called()
        mock_session_memory_api.record_turn.assert_not_called()

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_chat_with_session_raises_if_memory_not_wired(self, _, mock_async_service, mock_media_pipeline,
                                                          mock_audio_pipeline, mock_live_pipeline,
                                                          mock_live_verified_pipeline, mock_live_qa_pipeline,
                                                          mock_storage_api):
        bare_engine = IzvoriaEngine(
            async_model_service=mock_async_service, media_pipeline=mock_media_pipeline,
            audio_pipeline=mock_audio_pipeline, live_pipeline=mock_live_pipeline,
            live_verified_pipeline=mock_live_verified_pipeline, live_qa_pipeline=mock_live_qa_pipeline,
            storage_api=mock_storage_api
        )
        bare_engine.start()

        with pytest.raises(EngineOfflineError):
            bare_engine.chat(SearchQuery(prompt="test"), session_id=uuid.uuid4())

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_failed_conversation_chat_releases_all_models_and_records_nothing(
            self, _, engine, mock_live_qa_pipeline, mock_async_service, mock_session_memory_api
    ):
        engine.start()
        mock_live_qa_pipeline.execute_conversation_chat.side_effect = RuntimeError("Scrubber crashed")

        with pytest.raises(Exception):
            engine.chat(SearchQuery(prompt="test"), session_id=uuid.uuid4())

        mock_async_service.release_all.assert_called_once_with()
        mock_session_memory_api.record_turn.assert_not_called()

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_chat_reasoned_happy_path_records_turn(
            self, _, engine, mock_live_reasoning_pipeline, mock_session_memory_api
    ):
        engine.start()
        session_id = uuid.uuid4()
        pool_id = uuid.uuid4()
        qa_response = QAResponse(query="test", generated_answer="Grounded answer.", citations=[])
        deep_result = ReasonedChatResponse(qa_response=qa_response, total_iterations=2,
                                           standalone_query="test", evidence_chunk_ids=[pool_id])
        mock_live_reasoning_pipeline.execute_reasoning.return_value = deep_result
        mock_session_memory_api.format_prior_context.return_value = "prior context"

        result = engine.chat_reasoned(session_id, SearchQuery(prompt="test"))

        assert result.qa_response.generated_answer == "Grounded answer."
        mock_session_memory_api.get_history.assert_called_once_with(session_id, owner_id=None)
        mock_live_reasoning_pipeline.execute_reasoning.assert_called_once()
        call = mock_live_reasoning_pipeline.execute_reasoning.call_args
        assert call.kwargs == {"prior_session_context": "prior context", "seed_chunks": []}
        mock_session_memory_api.record_turn.assert_called_once_with(
            session_id, "test", "Grounded answer.",
            mode=SessionTurnMode.REASONING,
            standalone_query="test",
            evidence_chunk_ids=[pool_id],
            first_doc_number=1,
            owner_id=None
        )

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_chat_reasoned_seeds_history_evidence_and_numbers_only_new_chunks(
            self, _, engine, mock_live_reasoning_pipeline, mock_session_memory_api, mock_retrieval_api
    ):
        engine.start()
        session_id = uuid.uuid4()
        old_a, old_b = self._chunk(), self._chunk()
        mock_session_memory_api.get_history.return_value = [
            self._turn(session_id, 1, [old_a.chunk_id]),
            self._turn(session_id, 2, [old_b.chunk_id], mode=SessionTurnMode.REASONING),
        ]
        mock_retrieval_api.get_chunks.return_value = [old_a, old_b]
        fresh_id = uuid.uuid4()
        mock_live_reasoning_pipeline.execute_reasoning.return_value = ReasonedChatResponse(
            clarification_question="Which vote?", evidence_chunk_ids=[fresh_id, old_a.chunk_id]
        )

        engine.chat_reasoned(session_id, SearchQuery(prompt="test"))

        seeds = mock_live_reasoning_pipeline.execute_reasoning.call_args.kwargs["seed_chunks"]
        assert [c.chunk_id for c in seeds] == [old_b.chunk_id, old_a.chunk_id]
        mock_session_memory_api.record_turn.assert_called_once_with(
            session_id, "test", "[Clarification requested] Which vote?",
            mode=SessionTurnMode.REASONING,
            standalone_query=None,
            evidence_chunk_ids=[fresh_id],
            first_doc_number=3,
            owner_id=None
        )

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_chat_reasoned_raises_if_not_wired(self, _, mock_async_service, mock_media_pipeline,
                                                     mock_audio_pipeline, mock_live_pipeline,
                                                     mock_live_verified_pipeline, mock_live_qa_pipeline,
                                                     mock_storage_api):
        bare_engine = IzvoriaEngine(
            async_model_service=mock_async_service, media_pipeline=mock_media_pipeline,
            audio_pipeline=mock_audio_pipeline, live_pipeline=mock_live_pipeline,
            live_verified_pipeline=mock_live_verified_pipeline, live_qa_pipeline=mock_live_qa_pipeline,
            storage_api=mock_storage_api
        )
        bare_engine.start()

        with pytest.raises(Exception):
            bare_engine.chat_reasoned(uuid.uuid4(), SearchQuery(prompt="test"))

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_failed_chat_reasoned_releases_all_models(
            self, _, engine, mock_live_reasoning_pipeline, mock_async_service
    ):
        engine.start()
        mock_live_reasoning_pipeline.execute_reasoning.side_effect = RuntimeError("Reasoner crashed")

        with pytest.raises(Exception):
            engine.chat_reasoned(uuid.uuid4(), SearchQuery(prompt="test"))

        mock_async_service.release_all.assert_called_once_with()


class TestIzvoriaEngineReadSideAccess:
    """
    get_chunks / list_sources exist so a UI can show the verbatim text behind a citation and the
    contents of the vault. A QACitation carries ids and timestamps but no text, so without
    get_chunks an answer's evidence cannot be read alongside its clip.

    Standalone rather than a subclass of TestIzvoriaEngineUnit: inheriting would re-run every test in
    that class under this name.
    """

    @pytest.fixture
    def mock_storage_api(self):
        return Mock()

    @pytest.fixture
    def mock_retrieval_api(self):
        return Mock()

    @pytest.fixture
    def engine(self, mock_storage_api, mock_retrieval_api):
        return IzvoriaEngine(
            async_model_service=Mock(),
            media_pipeline=Mock(),
            audio_pipeline=Mock(),
            live_pipeline=Mock(),
            live_verified_pipeline=Mock(),
            live_qa_pipeline=Mock(),
            storage_api=mock_storage_api,
            retrieval_api=mock_retrieval_api,
        )

    @pytest.fixture
    def online_engine(self, engine):
        engine._state = EngineState.ONLINE
        return engine

    @staticmethod
    def _read_chunk(chunk_id, source_id, extracted_file_id=None):
        return RetrievedChunk(
            chunk_id=chunk_id,
            source_id=source_id,
            text_payload="contextualized payload",
            raw_text="SPEAKER_01: verbatim words.",
            speaker_labels=["SPEAKER_01"],
            start_sec=10.0,
            end_sec=20.0,
            relevance_score=0.0,
            extracted_file_id=extracted_file_id,
        )

    def test_get_chunks_delegates_to_the_retrieval_api(self, online_engine, mock_retrieval_api):
        chunk_id, source_id = uuid.uuid4(), uuid.uuid4()
        mock_retrieval_api.get_chunks.return_value = [self._read_chunk(chunk_id, source_id)]

        result = online_engine.get_chunks([chunk_id])

        mock_retrieval_api.get_chunks.assert_called_once_with([chunk_id])
        assert [c.chunk_id for c in result] == [chunk_id]

    def test_get_chunks_resolves_the_extracted_clip_path(
        self, online_engine, mock_retrieval_api, mock_storage_api
    ):
        """get_chunks does not resolve paths upstream, so the facade must inject them itself."""
        chunk_id, source_id, clip_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        mock_retrieval_api.get_chunks.return_value = [
            self._read_chunk(chunk_id, source_id, extracted_file_id=clip_id)
        ]
        mock_storage_api.get_file_path.return_value = "/vault/clips/abc.mp4"

        chunk = online_engine.get_chunks([chunk_id])[0]

        mock_storage_api.get_file_path.assert_called_once_with(clip_id)
        assert getattr(chunk, "physical_path") == "/vault/clips/abc.mp4"

    def test_get_chunks_falls_back_to_the_source_file_when_no_clip_exists(
        self, online_engine, mock_retrieval_api, mock_storage_api
    ):
        """The audio-only pipeline extracts no clips, so the whole source is the only source."""
        chunk_id, source_id = uuid.uuid4(), uuid.uuid4()
        mock_retrieval_api.get_chunks.return_value = [self._read_chunk(chunk_id, source_id)]
        source = Mock()
        source.file.file_path = "/vault/raw/whole.wav"
        mock_storage_api.get_source.return_value = source

        chunk = online_engine.get_chunks([chunk_id])[0]

        mock_storage_api.get_source.assert_called_once_with(source_id)
        assert getattr(chunk, "physical_path") == "/vault/raw/whole.wav"

    def test_get_chunks_survives_an_unresolvable_path(
        self, online_engine, mock_retrieval_api, mock_storage_api
    ):
        """A missing file must not lose the excerpt text, which is still worth reading."""
        chunk_id, source_id = uuid.uuid4(), uuid.uuid4()
        mock_retrieval_api.get_chunks.return_value = [self._read_chunk(chunk_id, source_id)]
        mock_storage_api.get_source.side_effect = RuntimeError("file vanished")

        chunk = online_engine.get_chunks([chunk_id])[0]

        assert getattr(chunk, "physical_path", None) is None
        assert chunk.raw_text == "SPEAKER_01: verbatim words."

    def test_get_chunks_is_refused_while_offline(self, engine):
        with pytest.raises(EngineOfflineError):
            engine.get_chunks([uuid.uuid4()])

    def test_get_chunks_without_a_retrieval_api_raises(self, online_engine):
        online_engine._retrieval_api = None

        with pytest.raises(EngineOfflineError):
            online_engine.get_chunks([uuid.uuid4()])

    def test_list_sources_delegates_to_storage(self, online_engine, mock_storage_api):
        source = Source(
            id=uuid.uuid4(), name="deposition.mp4", source_type=SourceType.VIDEO,
            content_hash="abc123", file_id=uuid.uuid4(),
        )
        mock_storage_api.list_sources.return_value = [source]

        assert online_engine.list_sources() == [source]
        mock_storage_api.list_sources.assert_called_once_with()

    def test_list_sources_is_refused_while_offline(self, engine):
        with pytest.raises(EngineOfflineError):
            engine.list_sources()


class TestProposeResearchPlanJobCount:
    """
    `expected_job_count` is the Report view's progress-bar denominator, so it has to count the same
    job types the view ticks on (`_REPORT_PHASES`). The rerank fan-out is the easy term to miss: the
    reranking phase runs one `run_job` PER SUB-QUESTION, not one per round.

    Standalone rather than a subclass of TestIzvoriaEngineUnit, which would re-run that whole class here.
    """

    @pytest.fixture
    def pipeline(self):
        return Mock()

    @pytest.fixture
    def engine(self, pipeline):
        engine = IzvoriaEngine(
            async_model_service=Mock(),
            media_pipeline=Mock(),
            audio_pipeline=Mock(),
            live_pipeline=Mock(),
            live_verified_pipeline=Mock(),
            live_qa_pipeline=Mock(),
            storage_api=Mock(),
            research_report_pipeline=pipeline,
            report_api=Mock(),
        )
        engine._state = EngineState.ONLINE
        return engine

    @staticmethod
    def _proposed(sections: int, sub_questions: int):
        report_id = uuid.uuid4()
        keys = [f"s{index}" for index in range(sections)]
        plan = ResearchPlan(
            report_id=report_id,
            topic="ECB rate decisions",
            sections=[
                OutlineSection(key=key, title=f"Section {index}", order_index=index)
                for index, key in enumerate(keys)
            ],
            sub_questions=[
                SubQuestion(text=f"question {index}", section_key=keys[index % len(keys)], order_index=index)
                for index in range(sub_questions)
            ],
        )
        report = Mock()
        report.id = report_id
        report.plan = plan
        report.corpus = None
        report.indexed_source_count = 3
        return report

    def test_the_denominator_covers_one_rerank_job_per_sub_question(self, engine, pipeline):
        pipeline.propose_plan.return_value = self._proposed(sections=8, sub_questions=12)

        with patch("app.engine.service.izvoria_engine.settings") as mock_settings:
            mock_settings.REPORT_MAX_GATHER_ROUNDS = 3
            proposal = engine.propose_research_plan("ECB rate decisions")

        # per round: 1 batched search + 12 rerank jobs + 1 outline pass; then 8 sections + 1 summary.
        assert proposal.expected_job_count == 3 * (2 + 12) + 8 + 1

    def test_the_denominator_is_never_below_a_single_deep_run(self, engine, pipeline):
        """
        A deep run is one round. The bar may finish early (exhaustive's ceiling is used) but must never
        under-count a deep run's jobs, or it would pin at 100% while the sections are still being written.
        """
        pipeline.propose_plan.return_value = self._proposed(sections=8, sub_questions=12)

        with patch("app.engine.service.izvoria_engine.settings") as mock_settings:
            mock_settings.REPORT_MAX_GATHER_ROUNDS = 3
            proposal = engine.propose_research_plan("ECB rate decisions")

        deep_run_jobs = 1 + 12 + 1 + 8 + 1
        assert proposal.expected_job_count >= deep_run_jobs


class TestIngestDocuments:
    @pytest.fixture
    def document_pipeline(self):
        return Mock()

    @pytest.fixture
    def engine(self, document_pipeline):
        return IzvoriaEngine(
            async_model_service=Mock(), media_pipeline=Mock(), audio_pipeline=Mock(), live_pipeline=Mock(),
            live_verified_pipeline=Mock(), live_qa_pipeline=Mock(), storage_api=Mock(),
            document_pipeline=document_pipeline,
        )

    def test_raises_if_offline(self, engine):
        with pytest.raises(EngineOfflineError):
            engine.ingest_documents("/dummy/path")

    def test_raises_if_not_wired(self):
        engine = IzvoriaEngine(async_model_service=Mock(), media_pipeline=Mock(), audio_pipeline=Mock(),
                            live_pipeline=Mock(), live_verified_pipeline=Mock(), live_qa_pipeline=Mock(),
                            storage_api=Mock())
        engine._state = EngineState.ONLINE

        with pytest.raises(EngineOfflineError, match="Document ingestion"):
            engine.ingest_documents("/dummy/path")

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_reports_page_counts_and_lists_documents_with_missing_text(self, _, engine, document_pipeline):
        engine.start()
        clean, scanned = uuid.uuid4(), uuid.uuid4()
        document_pipeline.process_directory.return_value = {
            PipelinePhase.SCAN: PhaseBatchResult(phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
                                                 successful_count=2, items=[]),
            PipelinePhase.DOCUMENT_CONVERSION: PhaseBatchResult(
                phase=PipelinePhase.DOCUMENT_CONVERSION, status=PhaseStatus.COMPLETED, successful_count=2, items=[
                    DocumentConversionJobResult(source_id=clean, success=True, pages_total=12),
                    DocumentConversionJobResult(source_id=scanned, success=True, pages_total=3, pages_ocr=3,
                                                pages_ocr_incomplete=1, pages_unreadable=1),
                ]),
            PipelinePhase.RAG_INDEXING: PhaseBatchResult(
                phase=PipelinePhase.RAG_INDEXING, status=PhaseStatus.COMPLETED, successful_count=2,
                items=[Mock(success=True, dense_vectors_saved=9), Mock(success=True, dense_vectors_saved=2)]),
        }

        report = engine.ingest_documents("/docs")

        document_pipeline.process_directory.assert_called_once_with("/docs")
        assert (report.pages_converted, report.pages_ocr, report.pages_ocr_incomplete, report.pages_unreadable) == (
            15, 3, 1, 1)
        assert report.vectors_created == 11 and report.successful_ingestions == 2
        assert list(report.errors) == [f"document_pages_{scanned}"]
        assert "1 scanned page(s) could not be read" in report.errors[f"document_pages_{scanned}"]
        assert engine.get_state() == EngineState.ONLINE


class TestStorageCompaction:
    @pytest.fixture
    def media_pipeline(self):
        return Mock()

    @pytest.fixture
    def retrieval_api(self):
        return Mock()

    @pytest.fixture
    def clip_api(self):
        api = Mock()
        api.discard_clips.return_value = 600
        return api

    @pytest.fixture
    def compaction_phase(self):
        return Mock()

    @pytest.fixture
    def engine(self, media_pipeline, retrieval_api, clip_api, compaction_phase):
        return IzvoriaEngine(
            async_model_service=Mock(), media_pipeline=media_pipeline, audio_pipeline=Mock(), live_pipeline=Mock(),
            live_verified_pipeline=Mock(), live_qa_pipeline=Mock(), storage_api=Mock(),
            retrieval_api=retrieval_api, clip_extractor_api=clip_api, storage_compaction_phase=compaction_phase,
        )

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_the_ingestion_report_carries_the_bytes_saved(self, _, engine, media_pipeline):
        engine.start()
        media_pipeline.process_directory.return_value = {
            PipelinePhase.STORAGE_COMPACTION: PhaseBatchResult(
                phase=PipelinePhase.STORAGE_COMPACTION, status=PhaseStatus.PARTIAL_SUCCESS, items=[
                    StorageCompactionJobResult(source_id=uuid.uuid4(), success=True, bytes_saved=700),
                    StorageCompactionJobResult(source_id=uuid.uuid4(), success=False, error_message="x"),
                ], errors=["Failed Storage Compaction for Source s: x"]),
        }

        report = engine.ingest_videos("/videos")

        assert report.storage_bytes_saved == 700
        # The recording is indexed either way: a failed compaction is not a file that was not added
        assert report.errors == {}

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_compact_storage_applies_the_settings_to_every_indexed_source(self, _, engine, retrieval_api, clip_api,
                                                                          compaction_phase, monkeypatch):
        monkeypatch.setattr("app.engine.service.izvoria_engine.settings.VIDEO_CLIPS_ENABLED", False)
        engine.start()
        first, second = uuid.uuid4(), uuid.uuid4()
        retrieval_api.list_indexed_sources.return_value = [
            IndexedSource(source_id=first, name="a.mp4", chunk_count=3),
            IndexedSource(source_id=second, name="b.wav", chunk_count=2),
        ]
        compaction_phase.compact_source.side_effect = [
            {"bytes_saved": 900, "audio_bytes_freed": 100, "original": "replaced", "original_bytes": 1000,
             "original_bytes_saved": 800},
            {"bytes_saved": 50, "audio_bytes_freed": 50, "original": "already_compact", "original_bytes": 40,
             "original_bytes_saved": 0},
        ]

        report = engine.compact_storage()

        assert clip_api.discard_clips.call_args_list == [call(first, dry_run=False), call(second, dry_run=False)]
        assert compaction_phase.compact_source.call_args_list == [call(first, dry_run=False),
                                                                 call(second, dry_run=False)]
        assert report == StorageCompactionReport(sources_checked=2, clip_bytes=1200, audio_bytes=150,
                                                 originals_replaced=1, original_bytes_saved=800)
        assert report.bytes_saved == 2150
        assert engine.get_state() == EngineState.ONLINE

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_a_dry_run_counts_the_originals_it_would_reencode(self, _, engine, retrieval_api, clip_api,
                                                              compaction_phase, monkeypatch):
        monkeypatch.setattr("app.engine.service.izvoria_engine.settings.VIDEO_CLIPS_ENABLED", True)
        engine.start()
        source_id = uuid.uuid4()
        retrieval_api.list_indexed_sources.return_value = [IndexedSource(source_id=source_id, name="a.mkv",
                                                                         chunk_count=1)]
        compaction_phase.compact_source.return_value = {"bytes_saved": 0, "original": "pending",
                                                        "original_bytes": 5000, "original_bytes_saved": 0}

        report = engine.compact_storage(dry_run=True)

        clip_api.discard_clips.assert_not_called()  # clips are wanted on this engine
        compaction_phase.compact_source.assert_called_once_with(source_id, dry_run=True)
        assert (report.dry_run, report.originals_to_compact, report.original_bytes_to_compact) == (True, 1, 5000)

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_one_source_failing_is_listed_and_the_rest_continue(self, _, engine, retrieval_api, compaction_phase):
        engine.start()
        retrieval_api.list_indexed_sources.return_value = [
            IndexedSource(source_id=uuid.uuid4(), name="bad.mp4", chunk_count=1),
            IndexedSource(source_id=uuid.uuid4(), name="good.mp4", chunk_count=1),
        ]
        compaction_phase.compact_source.side_effect = [RuntimeError("corrupt"), {"bytes_saved": 0}]

        report = engine.compact_storage()

        assert report.errors == {"bad.mp4": "corrupt"}
        assert report.sources_checked == 2

    def test_compact_storage_holds_the_engine_like_an_ingestion(self, engine):
        engine._state = EngineState.INGESTING

        with pytest.raises(EngineBusyError):
            engine.compact_storage()

    def test_compact_storage_needs_its_parts(self):
        engine = IzvoriaEngine(async_model_service=Mock(), media_pipeline=Mock(), audio_pipeline=Mock(),
                            live_pipeline=Mock(), live_verified_pipeline=Mock(), live_qa_pipeline=Mock(),
                            storage_api=Mock())

        with pytest.raises(EngineError, match="not wired"):
            engine.compact_storage()


class TestIngestImages:
    @pytest.fixture
    def image_pipeline(self):
        return Mock()

    @pytest.fixture
    def engine(self, image_pipeline):
        return IzvoriaEngine(
            async_model_service=Mock(), media_pipeline=Mock(), audio_pipeline=Mock(), live_pipeline=Mock(),
            live_verified_pipeline=Mock(), live_qa_pipeline=Mock(), storage_api=Mock(),
            image_pipeline=image_pipeline,
        )

    def test_raises_if_offline(self, engine):
        with pytest.raises(EngineOfflineError):
            engine.ingest_images("/dummy/path")

    def test_raises_if_not_wired(self):
        engine = IzvoriaEngine(async_model_service=Mock(), media_pipeline=Mock(), audio_pipeline=Mock(),
                            live_pipeline=Mock(), live_verified_pipeline=Mock(), live_qa_pipeline=Mock(),
                            storage_api=Mock())
        engine._state = EngineState.ONLINE

        with pytest.raises(EngineOfflineError, match="Image ingestion"):
            engine.ingest_images("/dummy/path")

    @patch('app.engine.service.izvoria_engine.db_manager')
    def test_reports_image_counts_and_lists_images_with_unconfirmed_text(self, _, engine, image_pipeline):
        from app.orchestration.domain.entities import ImageDescriptionJobResult
        engine.start()
        clean, invented, serbian, letter = uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        image_pipeline.process_directory.return_value = {
            PipelinePhase.SCAN: PhaseBatchResult(phase=PipelinePhase.SCAN, status=PhaseStatus.COMPLETED,
                                                 successful_count=4, items=[]),
            PipelinePhase.IMAGE_DESCRIPTION: PhaseBatchResult(
                phase=PipelinePhase.IMAGE_DESCRIPTION, status=PhaseStatus.PARTIAL_SUCCESS, successful_count=3,
                failed_count=1, errors=["Failed Image Description for Source x: decode failed"], items=[
                    ImageDescriptionJobResult(source_id=clean, success=True, ocr_checked=True),
                    ImageDescriptionJobResult(source_id=invented, success=True, ocr_checked=True,
                                              text_items_dropped=2),
                    ImageDescriptionJobResult(source_id=serbian, success=True, ocr_checked=False),
                    ImageDescriptionJobResult(source_id=letter, success=True, ocr_checked=True, ocr_lines_withheld=26),
                    ImageDescriptionJobResult(source_id=uuid.uuid4(), success=False, text_items_dropped=5),
                ]),
            PipelinePhase.RAG_INDEXING: PhaseBatchResult(
                phase=PipelinePhase.RAG_INDEXING, status=PhaseStatus.COMPLETED, successful_count=3,
                items=[Mock(success=True, dense_vectors_saved=1)] * 3),
        }

        report = engine.ingest_images("/photos")

        image_pipeline.process_directory.assert_called_once_with("/photos")
        assert (report.images_described, report.images_text_unverified, report.image_text_items_dropped,
                report.image_ocr_lines_withheld) == (4, 1, 2, 26)
        assert report.vectors_created == 3 and report.clips_extracted == 0
        assert f"image_text_{invented}" in report.errors and f"image_text_{clean}" not in report.errors
        # Only what happened is listed: no "0 text item(s)" when nothing was dropped
        assert report.errors[f"image_text_{invented}"] == (
            f"Source {invented}: 2 text item(s) the vision model read were not found by OCR. That text is not indexed.")
        assert report.errors[f"image_text_{letter}"] == (
            f"Source {letter}: 26 line(s) OCR read with low confidence were not confirmed by the vision model "
            f"(handwriting?). That text is not indexed.")
        assert "image_description_error_0" in report.errors
        assert engine.get_state() == EngineState.ONLINE


class TestIzvoriaEngineHistory:
    """
    Searches are recorded as shown, and every history read touches only the database: it must work while
    ingestion or a report holds the GPU, and a failure to record must never cost the user their search.
    """

    @pytest.fixture
    def search_history(self):
        return Mock()

    @pytest.fixture
    def session_memory(self):
        return Mock()

    @pytest.fixture
    def live_pipeline(self):
        return Mock()

    @pytest.fixture
    def verified_pipeline(self):
        return Mock()

    @pytest.fixture
    def engine(self, search_history, session_memory, live_pipeline, verified_pipeline):
        engine = IzvoriaEngine(
            async_model_service=Mock(), media_pipeline=Mock(), audio_pipeline=Mock(),
            live_pipeline=live_pipeline, live_verified_pipeline=verified_pipeline, live_qa_pipeline=Mock(),
            storage_api=Mock(), session_memory_api=session_memory, retrieval_api=Mock(),
            search_history_api=search_history,
        )
        engine._state = EngineState.ONLINE
        return engine

    @staticmethod
    def _chunk(score):
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=uuid.uuid4(), text_payload="p", raw_text="r",
                              speaker_labels=[], start_sec=0.0, end_sec=1.0, relevance_score=score)

    def _result(self, *scores):
        result = Mock()
        result.results = [self._chunk(s) for s in scores]
        result.execution_time_ms = 640
        return result

    @pytest.mark.parametrize("method, pipeline_attr, execute, mode", [
        ("search_standard", "live_pipeline", "execute_search", SearchHistoryMode.STANDARD),
        ("search_verified", "verified_pipeline", "execute_verified_search", SearchHistoryMode.VERIFIED),
    ])
    def test_a_search_is_recorded_with_the_hits_it_returned(
            self, request, engine, search_history, method, pipeline_attr, execute, mode):
        result = self._result(0.9, 0.3)
        getattr(request.getfixturevalue(pipeline_attr), execute).return_value = result

        assert getattr(engine, method)(SearchQuery(prompt="notice period", limit=7)) is result

        search_history.record.assert_called_once_with(
            query="notice period", mode=mode, result_limit=7,
            chunk_ids=[c.chunk_id for c in result.results], scores=[0.9, 0.3], execution_time_ms=640.0,
            owner_id=None,
        )

    def test_a_failure_to_record_does_not_fail_the_search(self, engine, search_history, live_pipeline):
        live_pipeline.execute_search.return_value = self._result(0.5)
        search_history.record.side_effect = RuntimeError("history table missing")

        assert engine.search_standard(SearchQuery(prompt="q", limit=5)).results

    def test_a_failed_search_records_nothing(self, engine, search_history, live_pipeline):
        live_pipeline.execute_search.side_effect = RuntimeError("model crashed")
        with pytest.raises(EngineError):
            engine.search_standard(SearchQuery(prompt="q", limit=5))
        search_history.record.assert_not_called()

    @pytest.mark.parametrize("state", [EngineState.INGESTING, EngineState.RESEARCHING, EngineState.ONLINE])
    @pytest.mark.parametrize("call, api_attr, api_method, expected_args", [
        (lambda e: e.list_conversations(10), "session_memory", "list_sessions", (10,)),
        (lambda e: e.get_conversation(SESSION), "session_memory", "get_all_turns", ("SESSION",)),
        (lambda e: e.delete_conversation(SESSION), "session_memory", "delete_session", ("SESSION",)),
        (lambda e: e.list_searches(10), "search_history", "list", (10,)),
        (lambda e: e.get_search(SESSION), "search_history", "get", ("SESSION",)),
        (lambda e: e.delete_search(SESSION), "search_history", "delete", ("SESSION",)),
    ])
    def test_history_reads_work_while_the_gpu_is_busy(
            self, request, engine, state, call, api_attr, api_method, expected_args):
        engine._state = state
        api = request.getfixturevalue(api_attr)
        getattr(api, api_method).return_value = "value"

        assert call(engine) == "value"
        getattr(api, api_method).assert_called_once_with(*[SESSION if a == "SESSION" else a for a in expected_args],
                                                         owner_id=None)

    def test_history_is_refused_while_offline(self, engine):
        engine._state = EngineState.OFFLINE
        with pytest.raises(EngineOfflineError):
            engine.list_searches()

    def test_history_errors_are_engine_errors(self, engine, session_memory):
        session_memory.list_sessions.side_effect = RuntimeError("DB down")
        with pytest.raises(EngineError):
            engine.list_conversations()

    def test_get_chunks_is_allowed_mid_ingestion(self, engine):
        engine._state = EngineState.INGESTING
        engine._retrieval_api.get_chunks.return_value = []
        assert engine.get_chunks([uuid.uuid4()]) == []

    @pytest.mark.parametrize("call, api_attr, api_method, expected_args", [
        (lambda e: e.list_conversations(10, "u7"), "session_memory", "list_sessions", (10,)),
        (lambda e: e.get_conversation(SESSION, "u7"), "session_memory", "get_all_turns", ("SESSION",)),
        (lambda e: e.delete_conversation(SESSION, "u7"), "session_memory", "delete_session", ("SESSION",)),
        (lambda e: e.list_searches(10, "u7"), "search_history", "list", (10,)),
        (lambda e: e.get_search(SESSION, "u7"), "search_history", "get", ("SESSION",)),
        (lambda e: e.delete_search(SESSION, "u7"), "search_history", "delete", ("SESSION",)),
    ])
    def test_history_calls_are_scoped_to_the_owner_given(self, request, engine, call, api_attr, api_method,
                                                         expected_args):
        api = request.getfixturevalue(api_attr)

        call(engine)

        getattr(api, api_method).assert_called_once_with(*[SESSION if a == "SESSION" else a for a in expected_args],
                                                         owner_id="u7")

    @pytest.mark.parametrize("method, pipeline_attr, execute", [
        ("search_standard", "live_pipeline", "execute_search"),
        ("search_verified", "verified_pipeline", "execute_verified_search"),
    ])
    def test_a_search_is_recorded_under_its_owner(self, request, engine, search_history, method, pipeline_attr,
                                                  execute):
        getattr(request.getfixturevalue(pipeline_attr), execute).return_value = self._result(0.4)

        getattr(engine, method)(SearchQuery(prompt="q", limit=3), owner_id="u7")

        assert search_history.record.call_args.kwargs["owner_id"] == "u7"


SESSION = uuid.uuid4()


class TestIzvoriaEngineConcurrentLive:
    """The hosted API's engine: live calls and capped reports run at once against shared, never-released models."""

    @pytest.fixture
    def async_service(self):
        return Mock()

    @pytest.fixture
    def report_pipeline(self):
        return Mock()

    @pytest.fixture
    def report_api(self):
        return Mock()

    @pytest.fixture
    def database(self):
        return Mock()

    @pytest.fixture
    def engine(self, async_service, report_pipeline, report_api, database):
        engine = IzvoriaEngine(
            async_model_service=async_service, media_pipeline=Mock(), audio_pipeline=Mock(),
            live_pipeline=Mock(), live_verified_pipeline=Mock(), live_qa_pipeline=Mock(),
            storage_api=Mock(), session_memory_api=Mock(), retrieval_api=Mock(),
            research_report_pipeline=report_pipeline, report_api=report_api, search_history_api=Mock(),
            database=database, owns_model_service=False, concurrent_live=True, report_max_concurrent=2,
        )
        engine._state = EngineState.ONLINE
        return engine

    def test_start_migrates_and_stop_disposes_the_firms_database_but_keeps_the_shared_models(
            self, mock_upgrade_to_head, async_service, database):
        engine = IzvoriaEngine(
            async_model_service=async_service, media_pipeline=Mock(), audio_pipeline=Mock(), live_pipeline=Mock(),
            live_verified_pipeline=Mock(), live_qa_pipeline=Mock(), storage_api=Mock(),
            database=database, owns_model_service=False, concurrent_live=True,
        )

        engine.start()
        engine.stop()

        mock_upgrade_to_head.assert_called_once_with(database.get_engine.return_value)
        database.dispose.assert_called_once_with()
        async_service.shutdown.assert_not_called()
        async_service.release_all.assert_not_called()

    def test_live_calls_run_while_a_report_runs_and_are_counted(self, engine, report_pipeline):
        import threading
        inside = threading.Event()
        release = threading.Event()

        def slow_report(report_id, plan, token):
            inside.set()
            release.wait(5)
            return "report"

        report_pipeline.execute_report.side_effect = slow_report
        runner = threading.Thread(target=engine.execute_research_report, args=(uuid.uuid4(), Mock()))
        runner.start()
        assert inside.wait(5)

        seen = {}
        engine._live_pipeline.execute_search.side_effect = lambda *a: seen.update(engine.busy_snapshot()) or Mock(results=[], execution_time_ms=1)
        engine.search_standard(SearchQuery(prompt="q", limit=3))
        release.set()
        runner.join(5)

        assert seen == {"inflight": 1, "reports_running": 1}
        assert engine.busy_snapshot() == {"inflight": 0, "reports_running": 0}

    def test_reports_beyond_the_cap_are_refused(self, engine):
        engine._research_cancels = {uuid.uuid4(): Mock(), uuid.uuid4(): Mock()}

        with pytest.raises(EngineBusyError, match="2 research report"):
            engine.execute_research_report(uuid.uuid4(), Mock())

    def test_the_same_report_cannot_run_twice(self, engine):
        report_id = uuid.uuid4()
        engine._research_cancels = {report_id: Mock()}

        with pytest.raises(EngineBusyError, match="already running"):
            engine.resume_research_report(report_id)

    def test_a_report_never_swaps_the_shared_models(self, engine, async_service, report_pipeline):
        report_pipeline.execute_report.return_value = "report"

        assert engine.execute_research_report(uuid.uuid4(), Mock()) == "report"

        async_service.release_all.assert_not_called()
        assert engine.get_state() == EngineState.ONLINE

    def test_a_failed_live_call_releases_nothing(self, engine, async_service):
        engine._live_pipeline.execute_search.side_effect = RuntimeError("boom")

        with pytest.raises(EngineError):
            engine.search_standard(SearchQuery(prompt="q", limit=3))

        async_service.release_all.assert_not_called()
        assert engine.busy_snapshot()["inflight"] == 0

    def test_ingestion_is_refused_and_left_to_the_worker(self, engine):
        with pytest.raises(EngineBusyError, match="ingestion worker"):
            engine.ingest_videos("/tmp/batch")

    def test_require_idle_refuses_while_a_call_is_in_flight(self, engine):
        engine._inflight = 1
        with pytest.raises(EngineBusyError):
            engine.require_idle()

    def test_the_owner_scopes_the_conversation_and_owns_a_proposed_plan(self, engine, report_pipeline, report_api):
        session_id = uuid.uuid4()
        engine._session_memory_api.get_history.return_value = []
        engine._session_memory_api.format_prior_context.return_value = None
        engine._retrieval_api.get_chunks.return_value = []
        chat_result = Mock(new_chunk_ids=[], standalone_query="q")
        chat_result.qa_response.generated_answer = "a"
        chat_result.qa_response.citations = []
        engine._live_qa_pipeline.execute_conversation_chat.return_value = chat_result
        report = Mock(plan=Mock(sub_questions=[], sections=[]), corpus=Mock())
        report_pipeline.propose_plan.return_value = report

        engine.chat(SearchQuery(prompt="q", limit=3), session_id, owner_id="u7")
        with patch("app.engine.service.izvoria_engine.ResearchPlanProposal"):
            engine.propose_research_plan("topic", owner_id="u7")

        engine._session_memory_api.get_history.assert_called_once_with(session_id, owner_id="u7")
        assert engine._session_memory_api.record_turn.call_args.kwargs["owner_id"] == "u7"
        report_api.set_owner.assert_called_once_with(report.id, "u7")

    def test_a_reasoned_turn_is_scoped_to_its_owner(self, engine):
        session_id = uuid.uuid4()
        engine._live_reasoning_pipeline = Mock()
        engine._session_memory_api.get_history.return_value = []
        engine._retrieval_api.get_chunks.return_value = []
        result = engine._live_reasoning_pipeline.execute_reasoning.return_value
        result.evidence_chunk_ids = []
        result.qa_response.citations = []

        engine.chat_reasoned(session_id, SearchQuery(prompt="q", limit=3), owner_id="u7")

        engine._session_memory_api.get_history.assert_called_once_with(session_id, owner_id="u7")
        assert engine._session_memory_api.record_turn.call_args.kwargs["owner_id"] == "u7"

    def test_reports_are_read_and_listed_for_their_owner(self, engine, report_api):
        report_id = uuid.uuid4()

        engine.get_report(report_id, owner_id="u7")
        engine.list_reports(None, 20, owner_id="u7")

        report_api.get_report.assert_called_once_with(report_id, owner_id="u7")
        report_api.list_reports.assert_called_once_with(None, 20, owner_id="u7")

    def test_a_report_is_renamed_only_by_its_owner(self, engine, report_api):
        report_id = uuid.uuid4()
        report_api.get_report.return_value = None

        assert engine.rename_research_report(report_id, "New", owner_id="u7") is None
        report_api.get_report.assert_called_once_with(report_id, owner_id="u7")
        report_api.rename_report.assert_not_called()

        report_api.get_report.return_value = Mock()
        engine.rename_research_report(report_id, "New", owner_id="u7")
        report_api.rename_report.assert_called_once_with(report_id, "New")

    def test_the_appliance_renames_without_an_owner_check(self, engine, report_api):
        report_id = uuid.uuid4()

        engine.rename_research_report(report_id, "New")

        report_api.get_report.assert_not_called()
        report_api.rename_report.assert_called_once_with(report_id, "New")

    def test_a_plan_proposed_without_an_owner_is_nobodys(self, engine, report_pipeline, report_api):
        report_pipeline.propose_plan.return_value = Mock(plan=Mock(sub_questions=[], sections=[]), corpus=Mock())

        with patch("app.engine.service.izvoria_engine.ResearchPlanProposal"):
            engine.propose_research_plan("topic")

        report_api.set_owner.assert_not_called()

    def test_a_live_call_is_counted_while_it_runs_and_not_after_it_fails(self, engine):
        seen = []
        engine._live_pipeline.execute_search.side_effect = lambda *a: seen.append(engine.busy_snapshot()) or 1 / 0

        with pytest.raises(EngineError):
            engine.search_standard(SearchQuery(prompt="q", limit=3))

        assert seen == [{"inflight": 1, "reports_running": 0}]
        assert engine.busy_snapshot() == {"inflight": 0, "reports_running": 0}

    def test_a_live_call_on_an_offline_engine_is_refused_without_being_counted(self, engine):
        engine._state = EngineState.OFFLINE

        with pytest.raises(EngineOfflineError):
            engine.search_standard(SearchQuery(prompt="q", limit=3))

        assert engine.busy_snapshot()["inflight"] == 0

    def test_require_idle_refuses_while_a_report_runs(self, engine):
        engine._research_cancels = {uuid.uuid4(): Mock()}

        with pytest.raises(EngineBusyError):
            engine.require_idle()

    def test_the_active_language_follows_the_requests_scope(self, engine, appliance_language):
        from app.core.language.service.language_api import language_scope
        appliance_language("en")

        with language_scope("sr"):
            assert engine.active_language == "sr"
        assert engine.active_language == "en"
