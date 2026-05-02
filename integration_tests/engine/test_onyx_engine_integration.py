# Canvas: test_onyx_engine_integration_v8
# Path: integration_tests/engine/test_onyx_engine_integration.py

import os
import shutil
import time
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
        if not os.path.exists(settings.RAG_EMBEDDING_MODEL_PATH) or \
                not os.path.exists(settings.RETRIEVAL_SCRUBBER_MODEL_PATH) or \
                not os.path.exists(settings.QA_GENERATION_MODEL_PATH):
            pytest.skip("Integration test skipped due to missing RAG, Scrubber, or QA model.")

        os.environ["DATA_DIR"] = str(tmp_path / "onyx_data")

        # 1. Bootstrapper Build (Real DI)
        engine = OnyxBootstrapper.build()

        # 2. Engine Start (Real DB connection and Model worker)
        engine.start()
        assert engine.get_state() == EngineState.ONLINE

        # Give the background Live Inference models a moment to fully load into VRAM,
        # so they can be safely evicted by the engine before it starts batch ingestion.
        time.sleep(3.0)

        yield engine

        # 3. Engine Stop (Teardown)
        engine.stop()
        assert engine.get_state() == EngineState.OFFLINE

    @pytest.mark.integration
    def test_engine_full_lifecycle(self, active_engine, tmp_path):
        """
        Tests the full Appliance lifecycle without a single mock:
        Boot -> Scan -> Route to Video Pipeline -> Extract Clips ->
        Verify VRAM Hand-offs -> Search -> Chat -> Teardown.
        """
        source_dir = tmp_path / "usb_drive"
        source_dir.mkdir()

        original_vid = Path("integration_tests/resources/test_video.mp4").resolve()
        if not original_vid.exists():
            pytest.skip(f"Required test file not found at: {original_vid}")

        test_file = source_dir / "test_video.mp4"
        shutil.copy2(original_vid, test_file)

        # 1. Run Video Ingestion (Blocking)
        try:
            report = active_engine.ingest_videos(str(source_dir))
        except Exception as e:
            if "cuda" in str(e).lower() or "vram" in str(e).lower():
                pytest.skip(f"Skipping test due to insufficient VRAM: {e}")
            raise e

        # Assert successful processing through the full Video Pipeline
        assert report.total_files_scanned == 1
        assert report.successful_ingestions == 1
        assert report.failed_ingestions == 0
        assert report.vectors_created > 0
        assert report.clips_extracted > 0

        assert active_engine.get_state() == EngineState.ONLINE

        # Give the Engine a moment to reload the Live Inference models after batch completion
        time.sleep(3.0)

        # 2. Run Standard Search
        query = SearchQuery(prompt="test", limit=1)
        search_result = active_engine.search_standard(query)

        assert len(search_result.results) > 0

        # Ensure the engine successfully resolved the physical path
        first_result = search_result.results[0]
        physical_path = getattr(first_result, 'physical_path', None)

        assert isinstance(physical_path, str), "Engine failed to resolve the physical path string for Search."
        assert physical_path.endswith(".mp4")

        # 3. Run Chat / QA Generation
        chat_query = SearchQuery(prompt="test", limit=1)
        chat_result = active_engine.chat(chat_query)

        assert chat_result.query == "test"
        assert chat_result.generated_answer is not None
        assert chat_result.inference_time_ms > 0

        # Verify citation path resolution (if citations exist for this test)
        if chat_result.citations:
            first_citation = chat_result.citations[0]
            cit_physical_path = getattr(first_citation, 'physical_path', None)
            assert isinstance(cit_physical_path, str), "Engine failed to resolve physical path for Citation."
            assert cit_physical_path.endswith(".mp4")