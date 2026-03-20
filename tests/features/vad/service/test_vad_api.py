# File: tests/features/vad/test_vad_api.py
import uuid
import pytest
from unittest.mock import MagicMock

from app.features.vad.service.vad_api import VadAPI
from app.features.vad.domain.entities import VadMap
from app.features.vad.exceptions.model import VadInferenceError


@pytest.fixture
def mocks():
    return {
        "repo": MagicMock(),
        "orchestrator": MagicMock(),
        "adapter": MagicMock()
    }


@pytest.fixture
def api(mocks):
    return VadAPI(
        repository=mocks["repo"],
        orchestrator=mocks["orchestrator"],
        adapter=mocks["adapter"]
    )


def test_process_audio_idempotency_skips_processing(api, mocks):
    source_id = uuid.uuid4()
    existing_map = VadMap(source_id=source_id, segments=[])

    # Mock repository to return an existing map
    mocks["repo"].get_map_by_source_id.return_value = existing_map

    result = api.process_audio(source_id, "dummy.wav")

    assert result == existing_map
    mocks["orchestrator"].acquire_model.assert_not_called()
    mocks["adapter"].generate_vad_map.assert_not_called()


def test_process_audio_success_flow(api, mocks):
    source_id = uuid.uuid4()
    mocks["repo"].get_map_by_source_id.return_value = None

    loaded_model_mock = MagicMock()
    mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

    generated_map = VadMap(source_id=source_id, segments=[])
    mocks["adapter"].generate_vad_map.return_value = generated_map
    mocks["repo"].save_map.return_value = generated_map

    result = api.process_audio(source_id, "dummy.wav")

    # Assertions
    assert result == generated_map
    mocks["orchestrator"].acquire_model.assert_called_once()
    mocks["adapter"].generate_vad_map.assert_called_once_with(
        model=loaded_model_mock, audio_path="dummy.wav", source_id=source_id
    )
    mocks["repo"].save_map.assert_called_once_with(generated_map)
    # Ensure VRAM is flushed!
    mocks["orchestrator"].release_model.assert_called_once_with(
        loaded_model_mock, mocks["adapter"], flush=True
    )


def test_process_audio_guarantees_vram_release_on_crash(api, mocks):
    source_id = uuid.uuid4()
    mocks["repo"].get_map_by_source_id.return_value = None

    loaded_model_mock = MagicMock()
    mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

    # Simulate a CUDA OutOfMemory crash inside the adapter
    mocks["adapter"].generate_vad_map.side_effect = RuntimeError("CUDA out of memory")

    with pytest.raises(VadInferenceError):
        api.process_audio(source_id, "dummy.wav")

    # The crash happened, but the finally block MUST still release the VRAM
    mocks["orchestrator"].release_model.assert_called_once_with(
        loaded_model_mock, mocks["adapter"], flush=True
    )
    mocks["repo"].save_map.assert_not_called()