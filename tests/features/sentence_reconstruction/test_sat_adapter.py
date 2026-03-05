# Canvas: test_sat_adapter_v3
# Path: tests/features/sentence_reconstruction/test_sat_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.sentence_reconstruction.data.wtp_adapter import SaTAdapter
from app.features.sentence_reconstruction.exceptions.model import SaTModelLoadError


@pytest.fixture
def manifest():
    return ModelManifest(
        model_type=ModelType.SENTENCE_RECONSTRUCTION,
        model_path="sat-3l-sm",
        required_vram_mb=750.0
    )


@patch("app.features.sentence_reconstruction.data.wtp_adapter.SaT")
def test_load_success(mock_sat_cls, manifest):
    mock_instance = MagicMock()
    mock_sat_cls.return_value = mock_instance
    mock_instance.half.return_value = mock_instance
    mock_instance.to.return_value = mock_instance

    adapter = SaTAdapter()
    loaded_model = adapter.load(manifest)

    assert loaded_model.manifest == manifest
    assert loaded_model.model_instance == mock_instance
    mock_sat_cls.assert_called_once_with("sat-3l-sm")


@patch("app.features.sentence_reconstruction.data.wtp_adapter.SaT")
def test_load_missing_file_raises_error(mock_sat_cls, manifest):
    mock_sat_cls.side_effect = OSError("Model not found")
    adapter = SaTAdapter()

    with pytest.raises(SaTModelLoadError):
        adapter.load(manifest)


def test_segment_text_success(manifest):  # FIXED: Injected manifest fixture
    adapter = SaTAdapter()

    mock_instance = MagicMock()
    mock_instance.split.return_value = ["Sentence one.", " Sentence two. "]

    # FIXED: Supply real manifest instead of MagicMock to satisfy Pydantic Model validation
    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)

    result = adapter.segment_text(loaded_model, "Sentence one. Sentence two.")

    assert result == ["Sentence one.", "Sentence two."]
    mock_instance.split.assert_called_once()