# Canvas: test_hallucination_correction_api_integration_v2
# Path: integration_tests/features/visual_hallucination_correction/test_correction_api_integration.py

import os
import json
import uuid
import pytest
from pathlib import Path
from typing import Generator
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
    Requires: Live PostgreSQL, NVIDIA GPU, and the Qwen 2.5 3B Instruct GGUF model downloaded locally.
    Tests the full Phase 13 Pipeline by loading test data from JSON resources.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self) -> Generator[None, None, None]:
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self) -> None:
        if not os.path.exists(settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH):
            pytest.skip(
                f"Integration test skipped due to missing SLM at: {settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH}")

    @pytest.fixture
    def isolated_setup(self) -> None:
        """Isolates the DB environment by dropping foreign keys to sources."""
        with db_manager.session() as session:
            session.execute(sa_text(
                "ALTER TABLE visual_hallucination_evaluations DROP CONSTRAINT IF EXISTS visual_hallucination_evaluations_source_id_fkey;"
            ))
            session.execute(sa_text(
                "ALTER TABLE chunk_visual_analyses DROP CONSTRAINT IF EXISTS chunk_visual_analyses_source_id_fkey;"
            ))
            session.execute(sa_text(
                "ALTER TABLE contextualized_transcripts DROP CONSTRAINT IF EXISTS contextualized_transcripts_source_id_fkey;"
            ))
            session.commit()

    @pytest.fixture
    def live_api(self) -> VisualHallucinationCorrectionAPI:
        repo = VisualHallucinationRepository(db_manager)
        contextual_repo = ContextualRepository(db_manager)
        visual_analysis_repo = VisualAnalysisRepository(db_manager)

        slm_adapter = QwenHallucinationAdapter()
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        return VisualHallucinationCorrectionAPI(
            repository=repo,
            adapter=slm_adapter,
            contextual_repo=contextual_repo,
            visual_analysis_repo=visual_analysis_repo,
            orchestrator=orchestrator
        )

    @pytest.fixture
    def seed_data_from_json(self, live_api: VisualHallucinationCorrectionAPI) -> uuid.UUID:
        """Loads the resource JSON, creates the domain entities, and saves them to the live DB."""
        integration_tests_dir = Path(__file__).resolve().parent.parent.parent
        json_path = integration_tests_dir / "resources" / "visual_hallucination_transcript.json"

        if not json_path.exists():
            pytest.skip(f"Test transcript missing at {json_path}. Please create it.")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_id = uuid.UUID(data["source_id"])
        chunks = []

        for chunk_data in data["chunks"]:
            chunk_id = uuid.UUID(chunk_data["chunk_id"])
            chunks.append(
                ContextualizedChunk(
                    chunk_id=chunk_id,
                    original_text=chunk_data["original_text"],
                    needs_visual=chunk_data["needs_visual"]
                )
            )

            # If it needs a visual, seed the mocked VLM output into the visual_analysis table
            if chunk_data["needs_visual"]:
                analysis = ChunkVisualAnalysis(
                    source_id=source_id,
                    chunk_id=chunk_id,
                    analyses=[FrameAnalysis(
                        frame_id=uuid.uuid4(),
                        description=chunk_data["visual_analysis_description"]
                    )]
                )
                live_api._visual_analysis_repo.save(analysis)

        transcript = ContextualizedTranscript(source_id=source_id, chunks=chunks)
        live_api._contextual_repo.save(transcript)

        return source_id

    def test_execute_end_to_end_hallucination_audit(
            self,
            isolated_setup: None,
            live_api: VisualHallucinationCorrectionAPI,
            seed_data_from_json: uuid.UUID
    ) -> None:

        source_id = seed_data_from_json

        # --- 1. EXECUTE THE PIPELINE ---
        try:
            print("\n[DEBUG] Spinning up Qwen 2.5 for Visual Audit... This may take a few seconds.")
            evaluated_transcript = live_api.execute(source_id, release_after=True)
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        # --- 2. VERIFY BUSINESS LOGIC ---
        assert isinstance(evaluated_transcript, VisualHallucinationEvaluatedTranscript)
        assert evaluated_transcript.source_id == source_id

        # We sent 3 chunks, but Chunk 3 had needs_visual=False.
        # The API must only generate audits for Chunks 1 and 2.
        assert len(evaluated_transcript.audits) == 2

        # Extract audits mapped by chunk ID
        audit_map = {str(a.chunk_id): a for a in evaluated_transcript.audits}

        # Verify Chunk 1 (The Match)
        chunk_1_id = "11111111-1111-1111-1111-111111111111"
        assert chunk_1_id in audit_map
        assert audit_map[chunk_1_id].is_visual_accepted is True
        assert audit_map[chunk_1_id].discard_reason is None

        # Verify Chunk 2 (The Hallucination Mismatch)
        chunk_2_id = "22222222-2222-2222-2222-222222222222"
        assert chunk_2_id in audit_map
        assert audit_map[chunk_2_id].is_visual_accepted is False
        assert audit_map[chunk_2_id].discard_reason is not None
        assert len(audit_map[chunk_2_id].discard_reason) > 5

        print(
            f"\n[DEBUG] SLM correctly caught the hallucination. Discard Reason: {audit_map[chunk_2_id].discard_reason}")

        # --- 3. VERIFY DATABASE PERSISTENCE ---
        db_transcript = live_api._repo.get_by_source_id(source_id)
        assert db_transcript is not None
        assert len(db_transcript.audits) == 2