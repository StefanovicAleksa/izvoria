# Canvas: test_visual_synthesis_integration_v5
# Path: integration_tests/features/visual_context_synthesis/test_synthesis_api_integration.py

import json
import uuid
import pytest
from pathlib import Path
from typing import Generator
from sqlalchemy import text as sa_text

from app.core.database.service.manager import db_manager
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor

from app.features.contextual_retrieval.data.repositories import ContextualRepository
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk

from app.features.visual_analysis.data.repositories import VisualAnalysisRepository
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis, FrameAnalysis

from app.features.visual_hallucination_correction.data.repositories import VisualHallucinationRepository
from app.features.visual_hallucination_correction.domain.entities import (
    VisualHallucinationEvaluatedTranscript,
    VisualHallucinationAudit
)

from app.features.visual_context_synthesis.service.synthesis_api import VisualSynthesisAPI
from app.features.visual_context_synthesis.data.repositories import VisualSynthesisRepository
from app.features.visual_context_synthesis.data.slm_adapter import VisualSLMAdapter
from app.features.visual_context_synthesis.domain.entities import VisualTimeline


@pytest.mark.integration
class TestVisualSynthesisAPIIntegration:
    @pytest.fixture(scope="class")
    def api_setup(self):
        """Sets up the real database, orchestrator, adapter, and repositories."""
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        slm_adapter = VisualSLMAdapter()

        synthesis_repo = VisualSynthesisRepository(db_manager)
        contextual_repo = ContextualRepository(db_manager)
        visual_repo = VisualAnalysisRepository(db_manager)
        hallucination_repo = VisualHallucinationRepository(db_manager)

        api = VisualSynthesisAPI(
            repository=synthesis_repo,
            orchestrator=orchestrator,
            slm_adapter=slm_adapter,
            visual_analysis_repo=visual_repo,
            hallucination_repo=hallucination_repo
        )

        return api, contextual_repo, visual_repo, hallucination_repo

    @pytest.fixture
    def isolated_setup(self) -> Generator[uuid.UUID, None, None]:
        """
        Isolates the DB environment by dropping foreign keys to sources.
        Includes a strict teardown to prevent hardcoded UUID collisions.
        """
        source_id = uuid.uuid4()
        with db_manager.session() as session:
            # Drop constraints
            session.execute(sa_text(
                "ALTER TABLE rag_visual_timelines DROP CONSTRAINT IF EXISTS rag_visual_timelines_source_id_fkey;"))
            session.execute(sa_text(
                "ALTER TABLE visual_hallucination_evaluations DROP CONSTRAINT IF EXISTS visual_hallucination_evaluations_source_id_fkey;"))
            session.execute(sa_text(
                "ALTER TABLE chunk_visual_analyses DROP CONSTRAINT IF EXISTS chunk_visual_analyses_source_id_fkey;"))
            session.execute(sa_text(
                "ALTER TABLE contextualized_transcripts DROP CONSTRAINT IF EXISTS contextualized_transcripts_source_id_fkey;"))
            session.commit()

        yield source_id

        # STRICT TEARDOWN: Wipe the hardcoded DB footprints
        with db_manager.session() as session:
            session.execute(sa_text("DELETE FROM rag_visual_timelines WHERE source_id = :sid"), {"sid": source_id})
            session.execute(sa_text("DELETE FROM visual_hallucination_evaluations WHERE source_id = :sid"),
                            {"sid": source_id})
            session.execute(sa_text("DELETE FROM chunk_visual_analyses WHERE source_id = :sid"), {"sid": source_id})
            session.execute(sa_text("DELETE FROM contextualized_transcripts WHERE source_id = :sid"),
                            {"sid": source_id})
            session.commit()

    @pytest.fixture
    def seed_data(self, api_setup, isolated_setup):
        """Reads the JSON resource file and seeds the PostgreSQL database."""
        api, contextual_repo, visual_repo, hallucination_repo = api_setup
        source_id = isolated_setup

        # 1. Load Resource File
        resource_path = Path("integration_tests/resources/visual_synthesis_seed.json")
        assert resource_path.exists(), f"Resource file missing at {resource_path}"
        with open(resource_path, "r") as f:
            seed = json.load(f)

        # 2. Build Phase 9 Contextual Transcript
        chunks = [ContextualizedChunk(**c) for c in seed["contextualized_transcript"]["chunks"]]
        transcript = ContextualizedTranscript(source_id=source_id, chunks=chunks)
        contextual_repo.save(transcript)

        # 3. Build Phase 12 Visual Analysis Records
        for visual_data in seed["chunk_visual_analyses"]:
            analyses = [FrameAnalysis(**f) for f in visual_data["analyses"]]
            chunk_analysis = ChunkVisualAnalysis(
                source_id=source_id,
                chunk_id=uuid.UUID(visual_data["chunk_id"]),
                analyses=analyses
            )
            visual_repo.save(chunk_analysis)

        # 4. Build Phase 13 Hallucination Ledger
        audits = [VisualHallucinationAudit(**a) for a in seed["visual_hallucination_evaluation"]["audits"]]
        ledger = VisualHallucinationEvaluatedTranscript(source_id=source_id, audits=audits)
        hallucination_repo.save(ledger)

        yield transcript

    def test_execute_full_synthesis_pipeline(self, api_setup, seed_data):
        """Executes the VisualSynthesisAPI."""
        api, _, _, _ = api_setup
        transcript = seed_data

        print(f"\n[INTEGRATION] Starting Visual Context Synthesis for Source {transcript.source_id}")

        # --- ACT ---
        final_timeline = api.execute(transcript, release_after=True)

        # --- ASSERT ---
        assert isinstance(final_timeline, VisualTimeline)
        assert final_timeline.source_id == transcript.source_id
        assert len(final_timeline.visual_chunks) == 2

        chunk_1 = final_timeline.visual_chunks[0]
        assert chunk_1.chunk_id == transcript.chunks[0].chunk_id
        assert len(chunk_1.visual_context_string) > 10
        assert len(chunk_1.visual_evidence) > 0

        chunk_2 = final_timeline.visual_chunks[1]
        assert chunk_2.chunk_id == transcript.chunks[1].chunk_id
        assert chunk_2.visual_context_string == ""
        assert len(chunk_2.visual_evidence) == 0