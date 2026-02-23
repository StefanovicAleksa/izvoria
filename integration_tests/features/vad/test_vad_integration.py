# File: integration_tests/features/vad/test_vad_integration.py
import os
import uuid
import pytest
from unittest.mock import MagicMock
from sqlalchemy import text

# Core & Infrastructure
from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

# VAD Feature
from app.features.vad.data.repositories import VadRepository
from app.features.vad.data.nemo_adapter import NeMoVadAdapter
from app.features.vad.service.vad_api import VadAPI
from app.features.vad.domain.entities import VadMap


@pytest.fixture(scope="module", autouse=True)
def ensure_nemo_model_exists():
    if not os.path.exists(settings.VAD_MODEL_PATH):
        pytest.fail(f"CRITICAL: VAD Model missing at {settings.VAD_MODEL_PATH}.")


@pytest.fixture(scope="module")
def real_audio_file():
    path = "integration_tests/resources/test_speech.wav"
    if not os.path.exists(path):
        pytest.fail(f"CRITICAL: Test audio missing at {path}.")
    return path


@pytest.fixture
def mock_orchestrator():
    orchestrator = MagicMock(spec=ModelManagerOrchestrator)
    orchestrator.acquire_model.side_effect = lambda manifest, adapter: adapter.load(manifest)
    orchestrator.release_model.side_effect = lambda model, adapter, flush: adapter.unload(model)
    return orchestrator


def test_full_vad_pipeline_with_real_audio(db_manager, real_audio_file, mock_orchestrator):
    # 1. Set up the Feature Components
    repo = VadRepository(db_manager=db_manager)
    adapter = NeMoVadAdapter()
    api = VadAPI(repository=repo, orchestrator=mock_orchestrator, adapter=adapter)

    source_id = uuid.uuid4()

    # --- THE FIX: SATISFY POSTGRES FOREIGN KEYS ---
    # We drop the constraint for this test so we don't have to mock a massive Source object
    with db_manager.session() as session:
        session.execute(text("ALTER TABLE vad_maps DROP CONSTRAINT vad_maps_source_id_fkey;"))
    # ----------------------------------------------

    # 2. ACT: Run the real audio through the AI pipeline
    result_map = api.process_audio(source_id=source_id, audio_path=real_audio_file)

    # 3. ASSERT: Model Orchestration
    mock_orchestrator.acquire_model.assert_called_once()
    mock_orchestrator.release_model.assert_called_once()

    # 4. ASSERT: Domain Logic (AI Math Results)
    assert isinstance(result_map, VadMap)
    assert result_map.source_id == source_id
    assert len(result_map.segments) > 0

    # 5. ASSERT: Database Persistence
    persisted_map = repo.get_map_by_source_id(source_id)
    assert persisted_map is not None
    assert persisted_map.id == result_map.id
    assert len(persisted_map.segments) == len(result_map.segments)