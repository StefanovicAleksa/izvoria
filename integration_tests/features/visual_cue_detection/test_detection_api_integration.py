# Canvas: test_detection_api_integration_no_mocks_v5
# Path: integration_tests/features/visual_cue_detection/test_detection_api_integration_no_mocks.py

import os
import uuid
import pytest
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript, ContextualizedChunk
from app.features.contextual_retrieval.data.repositories import ContextualRepository

from app.features.visual_cue_detection.data.qwen_detector_adapter import QwenDetectorAdapter
from app.features.visual_cue_detection.service.detection_api import VisualCueDetectionAPI


class TestVisualCueDetectionAPIIntegrationNoMocks:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Verifies the Visual Cue Detection API can correctly analyze text using the local SLM,
    parse the JSON boolean, and persist the updated flags to the Database directly.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.DOMAIN_CLASSIFIER_MODEL_PATH):
            pytest.skip(f"Integration test skipped due to missing SLM at: {settings.DOMAIN_CLASSIFIER_MODEL_PATH}")

    @pytest.fixture
    def isolated_setup(self) -> uuid.UUID:
        source_id = uuid.uuid4()
        with db_manager.session() as session:
            session.execute(sa_text(
                "ALTER TABLE contextualized_transcripts DROP CONSTRAINT IF EXISTS contextualized_transcripts_source_id_fkey;"
            ))
            session.commit()
        return source_id

    @pytest.fixture
    def detection_api(self) -> VisualCueDetectionAPI:
        repo = ContextualRepository(db_manager)
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        adapter = QwenDetectorAdapter()

        # FIX: Added repository dependency and removed dead contextualizer_api code
        return VisualCueDetectionAPI(
            orchestrator=orchestrator,
            adapter=adapter,
            repository=repo
        )

    def test_process_transcript_detects_cues_and_saves_to_db(self, isolated_setup: uuid.UUID,
                                                             detection_api: VisualCueDetectionAPI):
        source_id = isolated_setup
        repo = ContextualRepository(db_manager)

        # 1. Create a dummy transcript saved in the DB
        chunk_visual = ContextualizedChunk(
            original_text="If you look closely at this x-ray here, you can see the fracture.",
            context_string="Medical diagnosis.",
            needs_visual=False  # Initially false
        )
        chunk_no_visual = ContextualizedChunk(
            original_text="We arrived at the hospital at 9 AM.",
            context_string="Timeline of events.",
            needs_visual=False
        )

        transcript = ContextualizedTranscript(
            source_id=source_id,
            chunks=[chunk_visual, chunk_no_visual]
        )

        # Save it to DB directly using the repo for setup
        repo.save(transcript)

        # 2. Setup the SLM Model Loader
        manifest = ModelManifest(
            model_type=ModelType.DOMAIN_CLASSIFIER,
            model_path=settings.DOMAIN_CLASSIFIER_MODEL_PATH,
            required_vram_mb=settings.DOMAIN_CLASSIFIER_MODEL_VRAM_REQUIREMENT_MB
        )

        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)

        # Define and instantiate the dummy loader BEFORE the try block to satisfy PyCharm
        class DummyLoader:
            @staticmethod
            def load(m):
                from llama_cpp import Llama
                llm = Llama(model_path=m.model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)
                from app.core.model_manager.domain.entities import LoadedModel
                return LoadedModel(manifest=m, model_instance=llm)

            @staticmethod
            def unload(m):
                del m.model_instance

        loader = DummyLoader()
        loaded_model = None

        try:
            print("\n[DEBUG] Loading SLM for Visual Cue Detection...")
            loaded_model = orchestrator.acquire_model(manifest, loader)  # type: ignore

            # 3. Execute Detection API
            print("[DEBUG] Running visual cue detection...")

            # CRITICAL FIX: Match the phase loop by explicitly passing release_after=False
            updated_transcript = detection_api.process_transcript(
                transcript=transcript,
                model=loaded_model,
                release_after=False
            )

        finally:
            if loaded_model:
                orchestrator.release_model(loaded_model, loader, flush=True)  # type: ignore

        # 4. Verify the AI Logic worked exactly as intended
        assert len(updated_transcript.chunks) == 2

        # The x-ray reference should be flagged TRUE
        assert updated_transcript.chunks[0].needs_visual is True

        # The general timeline statement should remain FALSE
        assert updated_transcript.chunks[1].needs_visual is False

        # 5. Verify the API successfully saved it to the DB
        db_transcript = repo.get_by_source_id(source_id)
        assert db_transcript is not None
        assert db_transcript.chunks[0].needs_visual is True
        assert db_transcript.chunks[1].needs_visual is False