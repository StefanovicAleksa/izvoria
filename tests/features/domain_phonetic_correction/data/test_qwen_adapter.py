# Canvas: test_domain_qwen_adapter_v16
# Path: tests/features/domain_phonetic_correction/data/test_qwen_adapter.py

import pytest
from unittest.mock import patch, MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter


class TestQwenDomainAdapter:
    @pytest.fixture
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_type=ModelType.DOMAIN_PHONETIC_CORRECTION,
            model_path="/fake/qwen.gguf",
            required_vram_mb=2500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(QwenDomainAdapter, "_get_qwen_model")
    def test_load_success(self, mock_get_qwen: MagicMock, mock_exists: MagicMock, manifest: ModelManifest) -> None:
        mock_instance = MagicMock()
        mock_get_qwen.return_value = mock_instance

        adapter = QwenDomainAdapter()
        loaded = adapter.load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_qwen.assert_called_once_with(manifest.model_path)

    def test_generate_patches_success_with_xml_schema(self, manifest: ModelManifest) -> None:
        adapter = QwenDomainAdapter()
        mock_llm = MagicMock()

        xml_payload = "<patch><orig>calculate the eat bit da margin</orig><corr>calculate the EBITDA margin</corr></patch>"
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": xml_payload}, "finish_reason": "stop"}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        patches = adapter.generate_patches(loaded_model, "text", "dictionary")

        assert len(patches) == 1
        assert patches[0].original == "calculate the eat bit da margin"
        assert patches[0].correction == "calculate the EBITDA margin"

    def test_generate_patches_handles_truncation_without_crashing(self, manifest: ModelManifest) -> None:
        adapter = QwenDomainAdapter()
        mock_llm = MagicMock()

        # Simulate hitting token limit (length) with one complete patch and one torn patch at the end
        xml_payload = "<patch><orig>eat bit da</orig><corr>EBITDA</corr></patch>\n<patch><orig>with a cute angina</orig><corr>with "
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": xml_payload}, "finish_reason": "length"}]
        }

        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)
        patches = adapter.generate_patches(loaded_model, "text", "dictionary")

        # It should successfully parse the finished EBITDA tag and the Regex Firewall will drop the torn angina tag
        # We assert call_count is 1, proving we removed the buggy chat-continuation loop
        assert mock_llm.create_chat_completion.call_count == 1
        assert len(patches) == 1
        assert patches[0].original == "eat bit da"
        assert patches[0].correction == "EBITDA"

    def test_generate_patches_graceful_degradation(self, manifest: ModelManifest) -> None:
        adapter = QwenDomainAdapter()
        mock_llm = MagicMock()

        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "I couldn't find any patches!"}, "finish_reason": "stop"}]
        }
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        patches = adapter.generate_patches(loaded_model, "text", "dictionary")

        assert len(patches) == 0

    @patch("app.features.domain_phonetic_correction.data.qwen_adapter.gc.collect")
    def test_unload_explicitly_closes_c_bindings(self, mock_gc: MagicMock, manifest: ModelManifest) -> None:
        adapter = QwenDomainAdapter()
        mock_llm = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = adapter.unload(loaded_model)

        assert result is True
        mock_llm.close.assert_called_once()
        assert loaded_model.model_instance is None
        mock_gc.assert_called_once()