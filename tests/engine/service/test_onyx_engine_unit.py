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
from app.features.rag_retrieval.domain.entities import RetrievedChunk
from app.features.qa_generation.domain.entities import QAResponse, QACitation, ConversationHistory
from app.features.session_memory.domain.entities import SessionTurn
from app.features.session_memory.domain.enums import SessionTurnMode
from app.core.jobs.data.models import JOBS_UPGRADE_DDL
from app.features.deep_research_report.data.models import RESEARCH_REPORT_UPGRADE_DDL
from app.features.session_memory.data.models import SESSION_TURNS_UPGRADE_DDL
from app.orchestration.domain.enums import PipelinePhase, PhaseStatus
from app.orchestration.domain.entities import PhaseBatchResult, ReasonedChatResponse, ConversationChatResult
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
        return OnyxEngine(
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

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_start_adds_multi_turn_columns_to_an_existing_session_turns_table(self, mock_db_manager, engine):
        session = mock_db_manager.session.return_value.__enter__.return_value

        engine.start()

        executed = [str(call.args[0]) for call in session.execute.call_args_list]
        assert executed == [*SESSION_TURNS_UPGRADE_DDL, *JOBS_UPGRADE_DDL, *RESEARCH_REPORT_UPGRADE_DDL]
        assert all(
            "ADD COLUMN IF NOT EXISTS" in statement
            for statement in executed
            if statement.startswith("ALTER TABLE")
        )

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_start_backfills_the_renamed_reasoning_mode(self, mock_db_manager, engine):
        """
        SessionTurn parses `mode` into SessionTurnMode, so a row still on the pre-rename 'deep_research'
        value fails validation and breaks the whole conversation. There is no migration framework, so the
        backfill lives in the upgrade DDL that start() runs -- if it is ever dropped, this fails.
        """
        session = mock_db_manager.session.return_value.__enter__.return_value

        engine.start()

        executed = [str(call.args[0]) for call in session.execute.call_args_list]
        assert "UPDATE session_turns SET mode = 'reasoning' WHERE mode = 'deep_research'" in executed
        assert "UPDATE jobs SET job_type = 'reasoning_turn' WHERE job_type = 'deep_research_reasoning'" in executed

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

    @staticmethod
    def _chunk(chunk_id=None):
        return RetrievedChunk(chunk_id=chunk_id or uuid.uuid4(), source_id=uuid.uuid4(), text_payload="t",
                              start_sec=0.0, end_sec=1.0, relevance_score=0.0)

    @staticmethod
    def _turn(session_id, first_doc_number, evidence_ids, mode=SessionTurnMode.CHAT):
        return SessionTurn(id=uuid.uuid4(), session_id=session_id, question="Earlier?", answer_text="Earlier [Doc 1].",
                           mode=mode, evidence_chunk_ids=evidence_ids, first_doc_number=first_doc_number,
                           created_at="2026-01-01T00:00:00Z")

    @patch('app.engine.service.onyx_engine.db_manager')
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
        mock_session_memory_api.get_history.assert_called_once_with(session_id)
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
            first_doc_number=2
        )
        mock_live_qa_pipeline.execute_chat.assert_not_called()

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_chat_without_session_does_not_touch_conversation_memory(
            self, _, engine, mock_live_qa_pipeline, mock_session_memory_api
    ):
        engine.start()
        mock_live_qa_pipeline.execute_chat.return_value = QAResponse(query="q", generated_answer="a", citations=[])

        engine.chat(SearchQuery(prompt="q"))

        mock_session_memory_api.get_history.assert_not_called()
        mock_session_memory_api.record_turn.assert_not_called()

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_chat_with_session_raises_if_memory_not_wired(self, _, mock_async_service, mock_media_pipeline,
                                                          mock_audio_pipeline, mock_live_pipeline,
                                                          mock_live_verified_pipeline, mock_live_qa_pipeline,
                                                          mock_storage_api):
        bare_engine = OnyxEngine(
            async_model_service=mock_async_service, media_pipeline=mock_media_pipeline,
            audio_pipeline=mock_audio_pipeline, live_pipeline=mock_live_pipeline,
            live_verified_pipeline=mock_live_verified_pipeline, live_qa_pipeline=mock_live_qa_pipeline,
            storage_api=mock_storage_api
        )
        bare_engine.start()

        with pytest.raises(EngineOfflineError):
            bare_engine.chat(SearchQuery(prompt="test"), session_id=uuid.uuid4())

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_failed_conversation_chat_releases_all_models_and_records_nothing(
            self, _, engine, mock_live_qa_pipeline, mock_async_service, mock_session_memory_api
    ):
        engine.start()
        mock_live_qa_pipeline.execute_conversation_chat.side_effect = RuntimeError("Scrubber crashed")

        with pytest.raises(Exception):
            engine.chat(SearchQuery(prompt="test"), session_id=uuid.uuid4())

        mock_async_service.release_all.assert_called_once_with()
        mock_session_memory_api.record_turn.assert_not_called()

    @patch('app.engine.service.onyx_engine.db_manager')
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
        mock_session_memory_api.get_history.assert_called_once_with(session_id)
        mock_live_reasoning_pipeline.execute_reasoning.assert_called_once()
        call = mock_live_reasoning_pipeline.execute_reasoning.call_args
        assert call.kwargs == {"prior_session_context": "prior context", "seed_chunks": []}
        mock_session_memory_api.record_turn.assert_called_once_with(
            session_id, "test", "Grounded answer.",
            mode=SessionTurnMode.REASONING,
            standalone_query="test",
            evidence_chunk_ids=[pool_id],
            first_doc_number=1
        )

    @patch('app.engine.service.onyx_engine.db_manager')
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
            first_doc_number=3
        )

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_chat_reasoned_raises_if_not_wired(self, _, mock_async_service, mock_media_pipeline,
                                                     mock_audio_pipeline, mock_live_pipeline,
                                                     mock_live_verified_pipeline, mock_live_qa_pipeline,
                                                     mock_storage_api):
        bare_engine = OnyxEngine(
            async_model_service=mock_async_service, media_pipeline=mock_media_pipeline,
            audio_pipeline=mock_audio_pipeline, live_pipeline=mock_live_pipeline,
            live_verified_pipeline=mock_live_verified_pipeline, live_qa_pipeline=mock_live_qa_pipeline,
            storage_api=mock_storage_api
        )
        bare_engine.start()

        with pytest.raises(Exception):
            bare_engine.chat_reasoned(uuid.uuid4(), SearchQuery(prompt="test"))

    @patch('app.engine.service.onyx_engine.db_manager')
    def test_failed_chat_reasoned_releases_all_models(
            self, _, engine, mock_live_reasoning_pipeline, mock_async_service
    ):
        engine.start()
        mock_live_reasoning_pipeline.execute_reasoning.side_effect = RuntimeError("Reasoner crashed")

        with pytest.raises(Exception):
            engine.chat_reasoned(uuid.uuid4(), SearchQuery(prompt="test"))

        mock_async_service.release_all.assert_called_once_with()


class TestOnyxEngineReadSideAccess:
    """
    get_chunks / list_sources exist so a UI can show the verbatim text behind a citation and the
    contents of the vault. A QACitation carries ids and timestamps but no text, so without
    get_chunks an answer's evidence cannot be read alongside its clip.

    Standalone rather than a subclass of TestOnyxEngineUnit: inheriting would re-run every test in
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
        return OnyxEngine(
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
