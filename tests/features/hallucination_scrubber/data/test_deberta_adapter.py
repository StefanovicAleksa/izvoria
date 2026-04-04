# Canvas: test_deberta_adapter_v2
# Path: tests/features/hallucination_scrubber/data/test_deberta_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.hallucination_scrubber.data.deberta_adapter import DebertaAdapter


class TestDebertaAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.HALLUCINATION_SCRUBBER,
            model_path="/fake/deberta-v3-base-zeroshot",
            required_vram_mb=800.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(DebertaAdapter, "_get_pipeline")
    def test_load_success(self, mock_get_pipeline, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_pipeline.return_value = mock_instance

        adapter = DebertaAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_pipeline.assert_called_once_with(manifest.model_path)

    def test_classify_success(self, manifest):
        adapter = DebertaAdapter()
        mock_pipeline = MagicMock()

        mock_pipeline.return_value = {
            "labels": ["negative_label", "positive_label"],
            "scores": [0.95, 0.05]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_pipeline)

        result = adapter.classify(loaded_model, "Please subscribe", ["positive_label", "negative_label"])

        assert result["negative_label"] == 0.95
        assert result["positive_label"] == 0.05

        # VERIFY FIX: Ensures multi_label is passed as True
        mock_pipeline.assert_called_once_with("Please subscribe", ["positive_label", "negative_label"],
                                              multi_label=True)

    @patch("app.features.hallucination_scrubber.data.deberta_adapter.gc.collect")
    def test_unload_safely_deletes_pipeline(self, mock_gc, manifest):
        adapter = DebertaAdapter()
        mock_pipeline = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_pipeline)

        result = adapter.unload(loaded_model)

        assert result is True
        assert loaded_model.model_instance is None
        mock_gc.assert_called_once()