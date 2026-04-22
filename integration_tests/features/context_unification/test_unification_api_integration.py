# Canvas: test_context_unification_api_integration_v1
# Path: integration_tests/features/context_unification/test_unification_api_integration.py

import uuid
import pytest
from sqlalchemy import text as sa_text

from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk

from app.features.visual_context_synthesis.data.repositories import VisualSynthesisRepository
from app.features.visual_context_synthesis.domain.entities import VisualTimeline, VisualTimelineChunk, VisualContextFrame

from app.features.context_unification.data.repositories import ContextUnificationRepository
from app.features.context_unification.service.unification_api import UnificationAPI
from app.features.context_unification.domain.entities import UnifiedTranscript


@pytest.mark.integration
class TestContextUnificationAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Tests the purely deterministic Phase 15 Context Unification pipeline:
      1. Fetching Phase 9 and Phase 14 Data via real PostgreSQL Repositories.
      2. Deterministic Template Fusion.
      3. JSONB Database Persistence in the `unified_transcripts` table.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture
    def isolated_setup(self) -> uuid.UUID:
        source_id = uuid.uuid4()
        with db_manager.session() as session:
            session.execute(sa_text(
                "ALTER TABLE unified_transcripts DROP CONSTRAINT IF EXISTS unified_transcripts_source_id_fkey;"
            ))
            session.execute(sa_text(
                "ALTER TABLE rag_visual_timelines DROP CONSTRAINT IF EXISTS rag_visual_timelines_source_id_fkey;"
            ))
            session.execute(sa_text(
                "ALTER TABLE contextualized_transcripts DROP CONSTRAINT IF EXISTS contextualized_transcripts_source_id_fkey;"
            ))
            session.commit()
        return source_id

    @pytest.fixture
    def seed_data(self, isolated_setup):
        source_id = isolated_setup
        chunk_1_id = uuid.uuid4()
        chunk_2_id = uuid.uuid4()

        contextual_repo = ContextualRepository(db_manager)
        synthesis_repo = VisualSynthesisRepository(db_manager)

        # 1. Build and Save Phase 9 Contextual Transcript (Audio)
        audio_transcript = ContextualizedTranscript(
            source_id=source_id,
            chunks=[
                ContextualizedChunk(
                    chunk_id=chunk_1_id, start_sec=0.0, end_sec=5.0,
                    original_text="As you can see on the screen, revenue is up.",
                    context_string="Speaker discusses Q3 financials.",
                    needs_visual=True
                ),
                ContextualizedChunk(
                    chunk_id=chunk_2_id, start_sec=5.0, end_sec=10.0,
                    original_text="This is great news for all of us.",
                    context_string="Speaker expresses optimism.",
                    needs_visual=False
                )
            ]
        )
        contextual_repo.save(audio_transcript)

        # 2. Build and Save Phase 14 Visual Timeline (Visuals)
        # Note: Only Chunk 1 has visual data
        frame_id = uuid.uuid4()
        visual_timeline = VisualTimeline(
            source_id=source_id,
            visual_chunks=[
                VisualTimelineChunk(
                    chunk_id=chunk_1_id,
                    visual_context_string="A bar chart indicating a 20% revenue spike.",
                    visual_evidence=[VisualContextFrame(frame_id=frame_id, timestamp_sec=2.5)]
                )
            ]
        )
        synthesis_repo.save(visual_timeline)

        return audio_transcript, visual_timeline, frame_id

    @pytest.fixture
    def live_api(self) -> UnificationAPI:
        repo = ContextUnificationRepository(db_manager)
        return UnificationAPI(repository=repo)

    def test_execute_end_to_end_context_unification(self, seed_data, live_api: UnificationAPI):
        audio_transcript, visual_timeline, expected_frame_id = seed_data
        source_id = audio_transcript.source_id

        print(f"\n[INTEGRATION] Starting Context Unification for Source {source_id}")

        # --- ACT ---
        unified_transcript = live_api.execute(
            audio_transcript=audio_transcript,
            visual_timeline=visual_timeline
        )

        # --- ASSERT Domain Object Structure ---
        assert isinstance(unified_transcript, UnifiedTranscript)
        assert unified_transcript.source_id == source_id
        assert len(unified_transcript.chunks) == 2

        # Verify Chunk 1 (Fused Audio + Visual)
        chunk_1 = unified_transcript.chunks[0]
        assert "Audio Context: Speaker discusses Q3 financials." in chunk_1.text_payload
        assert "Visual Context: A bar chart indicating a 20% revenue spike." in chunk_1.text_payload
        assert "As you can see on the screen, revenue is up." in chunk_1.text_payload
        assert len(chunk_1.visual_evidence) == 1
        assert chunk_1.visual_evidence[0].frame_id == expected_frame_id

        # Verify Chunk 2 (Audio Only)
        chunk_2 = unified_transcript.chunks[1]
        assert "Audio Context: Speaker expresses optimism." in chunk_2.text_payload
        assert "Visual Context:" not in chunk_2.text_payload
        assert "This is great news for all of us." in chunk_2.text_payload
        assert len(chunk_2.visual_evidence) == 0

        # --- ASSERT Database Persistence ---
        db_transcript = live_api._repo.get_by_source_id(source_id)
        assert db_transcript is not None
        assert len(db_transcript.chunks) == 2
        assert db_transcript.chunks[0].text_payload == chunk_1.text_payload

        print(f"\n[INTEGRATION SUCCESS] Successfully stitched and persisted Unified Transcript.")