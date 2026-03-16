# Canvas: test_deberta_adapter_v4
# Path: tests/features/domain_adaptation/test_deberta_adapter.py

import pytest
from unittest.mock import MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.domain_adaptation.data.deberta_adapter import DebertaClassifierAdapter


def test_classify_domain(mocker):
    adapter = DebertaClassifierAdapter()
    manifest = ModelManifest(model_type=ModelType.DOMAIN_CLASSIFIER, model_path="mock/path", required_vram_mb=200.0)
    mock_pipeline = MagicMock()

    mock_pipeline.return_value = {"labels": ["OPTIONS_TRADING", "UK_MARKETS"], "scores": [0.95, 0.05]}
    loaded_model = LoadedModel(manifest=manifest, model_instance=mock_pipeline)

    best_label = adapter.classify_domain(loaded_model, "Let's talk about strike prices.",
                                         ["UK_MARKETS", "OPTIONS_TRADING"])

    assert best_label == "OPTIONS_TRADING"
    mock_pipeline.assert_called_once()
    assert mock_pipeline.call_args[0][0] == "Let's talk about strike prices."