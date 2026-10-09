# Path: tests/core/model_manager/utils/test_llama_cpp_chat.py

import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.utils.llama_cpp_chat import (
    build_llama, chat_completion, kv_cache_type_id, strip_reasoning
)


class TestKvCacheTypeId:
    @pytest.mark.parametrize("name, expected", [("f16", 1), ("q4_0", 2), ("q8_0", 8), ("Q8_0", 8)])
    def test_known_types(self, name, expected):
        assert kv_cache_type_id(name) == expected

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="q5_1"):
            kv_cache_type_id("q5_1")


class TestBuildLlama:
    @patch("llama_cpp.Llama")
    def test_offloads_everything_with_flash_attention_and_the_kv_cache_type(self, mock_llama):
        result = build_llama("/models/m.gguf", 131072, "q8_0")

        assert result == mock_llama.return_value
        mock_llama.assert_called_once_with(
            model_path="/models/m.gguf", n_gpu_layers=-1, n_ctx=131072, flash_attn=True,
            type_k=8, type_v=8, verbose=False
        )

    @patch("llama_cpp.Llama")
    def test_server_backend_returns_a_client_without_loading_weights(self, mock_llama):
        from app.core.model_manager.data.llama_server_chat_model import LlamaServerChatModel
        with patch("app.core.config.settings.LLM_BACKEND", "llama_server"), \
                patch("app.core.config.settings.LLM_SERVER_URL", "http://127.0.0.1:9999"):
            result = build_llama("/models/m.gguf", 65536, "q8_0")

        assert isinstance(result, LlamaServerChatModel)
        mock_llama.assert_not_called()

    def test_unknown_backend_raises(self):
        with patch("app.core.config.settings.LLM_BACKEND", "vllm"), pytest.raises(ValueError, match="vllm"):
            build_llama("/models/m.gguf", 4096, "f16")


class TestChatCompletion:
    def test_calls_the_model_chat_handler_with_the_thinking_switch(self):
        llm = MagicMock()
        handler = MagicMock(return_value={"choices": []})
        llm.chat_handler = handler
        messages = [{"role": "user", "content": "hi"}]

        result = chat_completion(llm, messages, enable_thinking=False, temperature=0.0, max_tokens=5)

        assert result == {"choices": []}
        handler.assert_called_once_with(
            llama=llm, messages=messages, enable_thinking=False, temperature=0.0, max_tokens=5, repeat_penalty=1.0
        )

    def test_falls_back_to_the_gguf_template_handler(self):
        llm = MagicMock()
        llm.chat_handler = None
        llm.chat_format = "chat_template.default"
        handler = MagicMock(return_value={})
        llm._chat_handlers = {"chat_template.default": handler}

        chat_completion(llm, [], enable_thinking=True, repeat_penalty=1.2)

        handler.assert_called_once_with(llama=llm, messages=[], enable_thinking=True, repeat_penalty=1.2)


class TestStripReasoning:
    @pytest.mark.parametrize("text, expected", [
        ("Answer.", "Answer."),
        ("<think>\n\n</think>\n\nAnswer [Doc 1].", "Answer [Doc 1]."),
        ("<think>long reasoning</think>Answer.", "Answer."),
        ("<think>cut off by max_tokens", ""),
        ("Answer.\x00", "Answer."),
        ("Thinking Process:\n1. Analyze.\n</think>\n\nParis.", "Paris."),
    ])
    def test_removes_think_blocks(self, text, expected):
        assert strip_reasoning(text) == expected


class TestChatCompletionOnAChatModel:
    def test_an_ichatmodel_is_asked_directly(self):
        from app.core.model_manager.domain.interfaces import IChatModel
        llm = MagicMock(spec=IChatModel)
        llm.chat.return_value = {"choices": [{"message": {"content": "ok"}}]}
        messages = [{"role": "user", "content": "hi"}]

        result = chat_completion(llm, messages, enable_thinking=True, temperature=0.1, max_tokens=7)

        assert result == llm.chat.return_value
        llm.chat.assert_called_once_with(messages, enable_thinking=True, temperature=0.1, max_tokens=7)
