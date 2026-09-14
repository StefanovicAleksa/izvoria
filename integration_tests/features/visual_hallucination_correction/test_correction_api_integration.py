import os
import uuid
import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk

from app.features.visual_analysis.data.repositories import VisualAnalysisRepository
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameAnalysis

from app.features.visual_hallucination_correction.data.repositories import VisualHallucinationRepository
from app.features.visual_hallucination_correction.data.slm_adapter import QwenHallucinationAdapter
from app.features.visual_hallucination_correction.service.correction_api import VisualHallucinationCorrectionAPI
from app.features.visual_hallucination_correction.domain.entities import VisualHallucinationEvaluatedTranscript


class TestVisualHallucinationCorrectionAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL and downloaded SLM weights.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def isolated_setup(self, setup_database):
        # Seeded rows use random source ids with no `sources` row; without this the test only passed
        # when an earlier test in the session had already dropped these constraints.
        with db_manager.session() as session:
            for table in ("contextualized_transcripts", "chunk_visual_analyses", "visual_hallucination_evaluations"):
                session.execute(sa_text(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_source_id_fkey;"))
            session.commit()

    @pytest.fixture(scope="module", autouse=True)
    def check_model_downloaded(self):
        if not os.path.exists(settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH):
            pytest.skip(f"Integration test skipped: Model not found at {settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH}")

    @pytest.fixture
    def contextual_repo(self): return ContextualRepository(db_manager)

    @pytest.fixture
    def visual_repo(self): return VisualAnalysisRepository(db_manager)

    @pytest.fixture
    def hallucination_repo(self): return VisualHallucinationRepository(db_manager)

    @pytest.fixture
    def live_api(self, contextual_repo, visual_repo, hallucination_repo) -> VisualHallucinationCorrectionAPI:
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        adapter = QwenHallucinationAdapter()

        return VisualHallucinationCorrectionAPI(
            repository=hallucination_repo,
            adapter=adapter,
            contextual_repo=contextual_repo,
            visual_analysis_repo=visual_repo,
            orchestrator=orchestrator
        )

    @pytest.fixture
    def seed_data(self, contextual_repo, visual_repo) -> uuid.UUID:
        """Seeds standard valid/hallucinated data using dictionary unpacking."""
        source_id = uuid.uuid4()
        chunk_1_id = uuid.uuid4()
        chunk_2_id = uuid.uuid4()

        # 1. Save Transcript
        transcript = ContextualizedTranscript(**{
            "source_id": source_id,
            "chunks": [
                ContextualizedChunk(**{
                    "chunk_id": chunk_1_id,
                    "start_sec": 0.0,
                    "end_sec": 5.0,
                    "original_text": "A man in a blue suit is sitting at the desk.",
                    "needs_visual": True
                }),
                ContextualizedChunk(**{
                    "chunk_id": chunk_2_id,
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "original_text": "There is a massive fire in the room.",
                    "needs_visual": True
                })
            ]
        })
        contextual_repo.save(transcript)

        # 2. Save Visual Analysis Results
        visual_repo.save(ChunkVisualAnalysis(**{
            "source_id": source_id,
            "chunk_id": chunk_1_id,
            "analyses": [
                FrameAnalysis(**{"frame_id": uuid.uuid4(), "description": "A man in a blue suit is at a desk."})
            ]
        }))

        visual_repo.save(ChunkVisualAnalysis(**{
            "source_id": source_id,
            "chunk_id": chunk_2_id,
            "analyses": [
                FrameAnalysis(**{"frame_id": uuid.uuid4(), "description": "A quiet office. No fire visible."})
            ]
        }))

        return source_id

    def test_execute_standard_hallucination_audit(self, live_api, seed_data):
        """Verifies standard accept/reject logic against real SLM."""
        source_id = seed_data
        result = live_api.execute(source_id, release_after=True)

        assert len(result.audits) == 2
        assert any(a.is_visual_accepted is True for a in result.audits)
        assert any(a.is_visual_accepted is False for a in result.audits)

    def test_execute_truncates_massive_visual_evidence_integration(
        self, live_api, contextual_repo, visual_repo, hallucination_repo
    ):
        """Integration test verifying that massive frame lists are safely truncated."""
        source_id = uuid.uuid4()
        chunk_id = uuid.uuid4()

        # 1. Seed massive payload
        transcript = ContextualizedTranscript(**{
            "source_id": source_id,
            "chunks": [
                ContextualizedChunk(**{
                    "chunk_id": chunk_id, "start_sec": 0.0, "end_sec": 30.0, 
                    "original_text": "Verify massive truncation.", "needs_visual": True
                })
            ]
        })
        contextual_repo.save(transcript)

        huge_description = "Repetitive visual context. " * 500  # ~25k characters
        visual_result = ChunkVisualAnalysis(**{
            "source_id": source_id,
            "chunk_id": chunk_id,
            "analyses": [
                FrameAnalysis(**{"frame_id": uuid.uuid4(), "description": huge_description}),
                FrameAnalysis(**{"frame_id": uuid.uuid4(), "description": huge_description})
            ]
        })
        visual_repo.save(visual_result)

        # 2. Execute
        result = live_api.execute(source_id, release_after=True)

        # 3. Assertions
        assert isinstance(result, VisualHallucinationEvaluatedTranscript)
        assert len(result.audits) == 1
        assert hallucination_repo.get_by_source_id(source_id) is not None