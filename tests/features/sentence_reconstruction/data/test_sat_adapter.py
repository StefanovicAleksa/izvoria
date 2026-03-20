# Canvas: test_sat_adapter_v4
# Path: tests/features/sentence_reconstruction/test_sat_adapter.py

import os
import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType

# FIXED: Import from sat_adapter instead of the old wtp_adapter name
from app.features.sentence_reconstruction.data.sat_adapter import SaTAdapter
from app.features.sentence_reconstruction.exceptions.model import SaTModelLoadError


@pytest.fixture
def manifest():
    return ModelManifest(
        model_type=ModelType.SENTENCE_RECONSTRUCTION,
        model_path="sat-3l-sm",
        required_vram_mb=750.0
    )


# FIXED: Update patch path to sat_adapter
@patch("app.features.sentence_reconstruction.data.sat_adapter.SaT")
def test_load_success(mock_sat_cls, manifest):
    mock_instance = MagicMock()
    mock_sat_cls.return_value = mock_instance
    mock_instance.half.return_value = mock_instance
    mock_instance.to.return_value = mock_instance

    adapter = SaTAdapter()
    loaded_model = adapter.load(manifest)

    assert loaded_model.manifest == manifest
    assert loaded_model.model_instance == mock_instance

    # Verify SaT was called exactly with the model path and no invalid kwargs
    mock_sat_cls.assert_called_once_with("sat-3l-sm")

    # Verify the offline environment variable was set correctly
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


@patch("app.features.sentence_reconstruction.data.sat_adapter.SaT")
def test_load_missing_file_raises_error(mock_sat_cls, manifest):
    mock_sat_cls.side_effect = OSError("Model not found")
    adapter = SaTAdapter()

    with pytest.raises(SaTModelLoadError):
        adapter.load(manifest)


def test_segment_text_success(manifest):
    adapter = SaTAdapter()

    mock_instance = MagicMock()
    mock_instance.split.return_value = ["Sentence one.", " Sentence two. "]

    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_instance)

    result = adapter.segment_text(loaded_model, "Sentence one. Sentence two.")

    assert result == ["Sentence one.", "Sentence two."]
    mock_instance.split.assert_called_once()