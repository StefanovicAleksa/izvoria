# Canvas: test_visual_analysis_integration_v4
# Path: integration_tests/features/visual_analysis/test_analysis_api_integration.py

import os
import re
import uuid
import pytest
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
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
    Requires: Live PostgreSQL, NVIDIA GPU, and Qwen2-VL 7B GGUF.
    FIXED: Updated to test native Qwen multimodal routing instead of LLaVA.
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

    def _run_single_frame(self, live_api: VisualAnalysisAPI, source_id: uuid.UUID, image_path: str,
                          original_text: str, context_string: str) -> ChunkVisualAnalysis:
        chunk_id = uuid.uuid4()
        vaulted_file = live_api._storage.ingest_raw_file(image_path, delete_original=False)
        chunk = ContextualizedChunk(chunk_id=chunk_id, original_text=original_text, context_string=context_string,
                                    needs_visual=True)
        extracted_frames = ExtractedChunkFrames(
            source_id=source_id,
            chunk_id=chunk_id,
            frames=[ExtractedFrame(frame_id=uuid.uuid4(), file_id=vaulted_file.id, timestamp_sec=15.5,
                                   phash_value="frame_hash_01", quality_score=100.0)]
        )
        manifest = ModelManifest(
            model_type=ModelType.VISION,
            model_path=settings.VISION_MODEL_PATH,
            required_vram_mb=settings.VISION_VRAM_REQUIREMENT_MB
        )
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        loaded_model = None
        try:
            loaded_model = orchestrator.acquire_model(manifest, live_api._vlm_adapter)
            visual_analysis = live_api.process_chunk(
                source_id=source_id, chunk=chunk, extracted_frames=extracted_frames, vlm_model=loaded_model
            )
        finally:
            if loaded_model:
                orchestrator.release_model(loaded_model, live_api._vlm_adapter, flush=True)

        assert isinstance(visual_analysis, ChunkVisualAnalysis)
        assert visual_analysis.chunk_id == chunk_id
        db_record = live_api._repo.get_by_chunk_id(chunk_id)
        assert db_record is not None
        assert [a.description for a in db_record.analyses] == [a.description for a in visual_analysis.analyses]
        for analysis in visual_analysis.analyses:
            print(f"\n[DEBUG] Qwen-VL description: {analysis.description}")
        return visual_analysis

    def test_speech_is_never_stored_as_on_screen_text(
            self,
            isolated_setup: uuid.UUID,
            real_image_file: str,
            live_api: VisualAnalysisAPI
    ):
        """
        An X-ray held up by a doctor has no legible text. The frame may get no description at all (nothing legible
        and no specific mark reported), but the spoken sentence must never be stored as text read off the frame.
        """
        spoken = "As you can see right here, there appears to be a shadow in the lower lobe."
        visual_analysis = self._run_single_frame(
            live_api, isolated_setup, real_image_file, spoken,
            "Two medical professionals are reviewing a patient's chest scan."
        )

        assert len(visual_analysis.analyses) <= 1
        for analysis in visual_analysis.analyses:
            on_screen = " ".join(re.findall(r'"([^"]+)"', analysis.description.split("Speaker's reference:")[0]))
            assert "shadow in the lower lobe" not in on_screen.lower(), analysis.description
            assert any(word in analysis.description.lower() for word in ["x-ray", "xray", "chest", "scan", "lung"])

    def test_legible_slide_text_is_read_and_rendered_as_on_screen_text(
            self,
            isolated_setup: uuid.UUID,
            live_api: VisualAnalysisAPI,
            tmp_path: Path
    ):
        """A rendered slide with large legible text: its title and figures must come back quoted as on-screen text."""
        font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if not font_path.exists():
            pytest.skip(f"Font for the synthetic slide missing at {font_path}")

        image = Image.new("RGB", (1280, 720), "white")
        draw = ImageDraw.Draw(image)
        draw.text((80, 80), "EXHIBIT 12", fill="black", font=ImageFont.truetype(str(font_path), 96))
        body_font = ImageFont.truetype(str(font_path), 56)
        draw.text((80, 300), "Contract signed 14 March 2019", fill="black", font=body_font)
        draw.text((80, 420), "Late delivery penalty: 4.5%", fill="black", font=body_font)
        image_path = tmp_path / "exhibit_slide.jpg"
        image.save(image_path)

        visual_analysis = self._run_single_frame(
            live_api, isolated_setup, str(image_path),
            "If you look at this exhibit, the penalty for late delivery is spelled out in the contract.",
            "Counsel is walking the court through the supply contract."
        )

        assert len(visual_analysis.analyses) == 1
        description = visual_analysis.analyses[0].description
        assert "on-screen text" in description
        assert "exhibit 12" in description.lower()
        assert "4.5%" in description or "14 march 2019" in description.lower()
