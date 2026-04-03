# Canvas: test_visual_analysis_integration_v3
# Path: integration_tests/features/visual_analysis/test_analysis_api_integration.py

import os
import uuid
import pytest
from pathlib import Path
from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.data.engine import Base
from app.core.database.service.manager import db_manager

from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.data.repositories import StorageRepository

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.domain.entities import ExtractedChunkFrames, ExtractedFrame

from app.features.visual_analysis.data.repositories import VisualAnalysisRepository
from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI
from app.features.visual_analysis.domain.entities import ChunkVisualAnalysis


class TestVisualAnalysisAPIIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Requires: Live PostgreSQL, NVIDIA GPU, and LLaVA 1.5 7B GGUF.
    """

    @pytest.fixture(scope="module", autouse=True)
    def setup_database(self):
        engine = db_manager._engine_wrapper._engine
        Base.metadata.create_all(bind=engine)
        yield

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.VISION_MODEL_PATH):
            pytest.skip(f"Integration test skipped due to missing VLM at: {settings.VISION_MODEL_PATH}")

    @pytest.fixture
    def real_image_file(self) -> str:
        """Points to the real medical X-ray test image."""
        integration_tests_dir = Path(__file__).resolve().parent.parent.parent
        image_path = integration_tests_dir / "resources" / "test_medical_xray.jpg"

        if not image_path.exists():
            pytest.skip(f"Test image not found at {image_path}. Please place the X-ray image here.")

        return str(image_path)

    @pytest.fixture
    def isolated_setup(self) -> uuid.UUID:
        source_id = uuid.uuid4()
        with db_manager.session() as session:
            # Drop constraints just for the test environment to avoid needing to seed the full Source tree
            session.execute(sa_text(
                "ALTER TABLE chunk_visual_analyses DROP CONSTRAINT IF EXISTS chunk_visual_analyses_source_id_fkey;"
            ))
            session.commit()
        return source_id

    @pytest.fixture
    def live_api(self, tmp_path) -> VisualAnalysisAPI:
        # 1. Setup Storage API with a temporary vault for this test
        storage_repo = StorageRepository(db_manager)
        storage_api = StorageAPI(storage_repo)
        storage_api.file_manager.base_path = tmp_path / "vault"
        storage_api.file_manager.base_path.mkdir()

        # 2. Setup Visual Analysis feature components
        repo = VisualAnalysisRepository(db_manager)
        vlm_adapter = QwenVLMAdapter()

        return VisualAnalysisAPI(
            repository=repo,
            vlm_adapter=vlm_adapter,
            storage_api=storage_api
        )

    def test_process_chunk_end_to_end_vlm_inference(
            self,
            isolated_setup: uuid.UUID,
            real_image_file: str,
            live_api: VisualAnalysisAPI
    ):
        source_id = isolated_setup
        chunk_id = uuid.uuid4()

        # 1. Vault the physical image using the Storage feature
        vaulted_file = live_api._storage.ingest_raw_file(real_image_file, delete_original=False)

        # 2. Construct the logical domain entities simulating Phase 9 and Phase 9.5
        chunk = ContextualizedChunk(
            chunk_id=chunk_id,
            original_text="As you can see right here, there appears to be a shadow in the lower lobe.",
            context_string="Two medical professionals are reviewing a patient's chest scan.",
            needs_visual=True
        )

        extracted_frames = ExtractedChunkFrames(
            source_id=source_id,
            chunk_id=chunk_id,
            frames=[
                ExtractedFrame(
                    frame_id=uuid.uuid4(),
                    file_id=vaulted_file.id,
                    timestamp_sec=15.5,
                    phash_value="medical_hash_01",
                    quality_score=100.0
                )
            ]
        )

        # 3. Setup the Orchestrator and Manifest [cite: 73, 94]
        manifest = ModelManifest(
            model_type=ModelType.VISION,  # [cite: 74]
            model_path=settings.VISION_MODEL_PATH,
            required_vram_mb=settings.VISION_VRAM_REQUIREMENT_MB
        )

        # FIX: Move orchestrator outside try block to resolve "might be referenced before assignment"
        # FIX: Ensure PyTorchVramMonitor is correctly utilized [cite: 63]
        vram_monitor = PyTorchVramMonitor()
        orchestrator = ModelManagerOrchestrator(vram_monitor)
        loaded_model = None

        try:
            print("\n[DEBUG] Loading LLaVA 1.5 7B into VRAM... This will take a few seconds.")
            # This uses the underlying QwenVLMAdapter we built
            loaded_model = orchestrator.acquire_model(manifest, live_api._vlm_adapter)

            # 4. Execute the actual pipeline orchestration
            print("[DEBUG] Executing Multimodal Inference on the vaulted frame...")
            visual_analysis = live_api.process_chunk(
                source_id=source_id,
                chunk=chunk,
                extracted_frames=extracted_frames,
                vlm_model=loaded_model
            )

            # 5. Verify the Domain Object
            assert isinstance(visual_analysis, ChunkVisualAnalysis)
            assert visual_analysis.chunk_id == chunk_id
            assert len(visual_analysis.analyses) == 1

            description = visual_analysis.analyses[0].description
            print(f"\n[DEBUG] LLaVA Generated Description:\n{description}")

            # Ensure the model actually generated text
            assert len(description) > 5

            # 6. Verify AI Intelligence (Did it actually see the medical context?)
            description_lower = description.lower()
            has_medical_context = any(
                word in description_lower for word in ["x-ray", "xray", "chest", "scan", "lung", "doctor", "pointing"])
            assert has_medical_context is True, f"LLaVA failed to identify medical context. Output: {description}"

            # 7. Verify Database Persistence
            db_record = live_api._repo.get_by_chunk_id(chunk_id)
            assert db_record is not None
            assert db_record.chunk_id == chunk_id
            assert db_record.analyses[0].description == description

        finally:
            # Now safely accessible regardless of where the error occurred
            if loaded_model:
                orchestrator.release_model(loaded_model, live_api._vlm_adapter, flush=True)