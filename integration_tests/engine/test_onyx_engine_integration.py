# Canvas: test_onyx_engine_integration_v4
# Path: integration_tests/engine/test_onyx_engine_integration.py

import os
import shutil
import pytest
import json
from pathlib import Path

from app.core.config import settings
from app.engine.service.bootstrap import OnyxBootstrapper
from app.engine.domain.enums import EngineState
from app.features.rag_retrieval.domain.entities import SearchQuery


class TestOnyxEngineIntegration:
    @pytest.fixture(autouse=True)
    def seed_glossary(self):
        """Ensure the glossary exists for the phonetic correction phase."""
        settings.GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
        glossary_file = settings.GLOSSARY_DIR / "general.json"

        if not glossary_file.exists():
            mock_data = {
                "base_terms": ["Onyx", "AI"],
                "sub_domains": {"legal": {"anchor_words": [], "terms": []}}
            }
            with open(glossary_file, 'w') as f:
                json.dump(mock_data, f)

    @pytest.fixture
    def active_engine(self, tmp_path):
        """
        Builds the real engine using the Bootstrapper, starts it,
        and ensures it cleanly shuts down after the test.
        """
        if not os.path.exists(settings.RAG_EMBEDDING_MODEL_PATH):
            pytest.skip("Integration test skipped due to missing Embedding model.")

        # Create temporary vault for this test
        os.environ["DATA_DIR"] = str(tmp_path / "onyx_data")

        # 1. Bootstrapper Build (Real DI)
        engine = OnyxBootstrapper.build()

        # 2. Engine Start (Real DB connection and Model worker)
        engine.start()
        assert engine.get_state() == EngineState.ONLINE

        yield engine

        # 3. Engine Stop (Teardown)
        engine.stop()
        assert engine.get_state() == EngineState.OFFLINE

    @pytest.mark.integration
    def test_engine_full_lifecycle(self, active_engine, tmp_path):
        """
        Tests the full Appliance lifecycle without a single mock:
        Boot -> Scan -> Route to Video Pipeline -> Ingest -> Extract Clips -> Search -> Teardown.
        """
        # 1. Setup real test file using the video from the orchestration tests
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()

        original_vid = Path("integration_tests/resources/test_video.mp4").resolve()
        if not original_vid.exists():
            pytest.skip(f"Required test file not found at: {original_vid}")

        test_file = source_dir / "test_video.mp4"
        shutil.copy2(original_vid, test_file)

        # 2. Run Ingestion (Blocking)
        try:
            report = active_engine.ingest(str(source_dir))
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        # Assert successful processing through the full Video Pipeline
        # The scan phase runs twice (once for audio pipeline pass, once for media pipeline pass)
        assert report.total_files_scanned == 2

        assert report.successful_ingestions == 1
        assert report.failed_ingestions == 0
        assert report.vectors_created > 0
        # Because it's a video, FFmpeg should have successfully sliced it into chunks
        assert report.clips_extracted > 0

        # 3. Run Search
        # Using a generic prompt that should match something in the test video
        query = SearchQuery(prompt="test", limit=1)
        search_result = active_engine.search(query)

        assert len(search_result.results) > 0

        # Ensure the engine successfully resolved the physical path
        first_result = search_result.results[0]
        physical_path = getattr(first_result, 'physical_path', None)

        # Fix for type checker: Explicitly assert type before doing string methods
        assert isinstance(physical_path, str), "Engine failed to resolve the physical path string."

        # Because we used the video pipeline, the path should point to the newly extracted .mp4 clip in the vault
        assert physical_path.endswith(".mp4")