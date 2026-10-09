# Path: app/core/model_manager/utils/llama_cpp_chat.py

import re
from typing import Any, Dict, List

from app.core.model_manager.domain.interfaces import IChatModel

# llama.cpp ggml_type ids accepted for the KV cache (llama_cpp.GGML_TYPE_*)
_KV_CACHE_TYPES: Dict[str, int] = {"f16": 1, "q4_0": 2, "q8_0": 8}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Thinking cut off by max_tokens never closes its tag
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.DOTALL)
# With thinking enabled, the chat template itself opens <think>, so the reply holds only the closing tag
_TEMPLATE_OPENED_THINK = re.compile(r"\A.*?</think>", re.DOTALL)


def kv_cache_type_id(name: str) -> int:
    try:
        return _KV_CACHE_TYPES[name.lower()]
    except KeyError:
        raise ValueError(f"Unsupported KV cache type {name!r}; expected one of {sorted(_KV_CACHE_TYPES)}")


def build_llama(model_path: str, n_ctx: int, kv_cache_type: str) -> Any:
    """
    A fully GPU-offloaded llama.cpp model. A quantized value cache needs flash attention, which is always on;
    the hybrid Qwen3.5 models keep a KV cache in only a quarter of their layers, so long contexts stay small.

    With LLM_BACKEND=llama_server the model is instead a client of the local llama-server, which already holds
    the weights (and its own KV cache type): `n_ctx` then caps what this role sends, not what is allocated.
    """
    from app.core.config import settings
    if settings.LLM_BACKEND == "llama_server":
        from app.core.model_manager.data.llama_server_chat_model import LlamaServerChatModel
        return LlamaServerChatModel(settings.LLM_SERVER_URL, n_ctx, settings.LLM_SERVER_TIMEOUT_S,
                                    kv_tokens=settings.LLM_SERVER_KV_TOKENS)
    if settings.LLM_BACKEND != "llama_cpp":
        raise ValueError(f"Unsupported LLM_BACKEND {settings.LLM_BACKEND!r}; expected 'llama_cpp' or 'llama_server'")

    from llama_cpp import Llama
    type_id = kv_cache_type_id(kv_cache_type)
    return Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=n_ctx,
        flash_attn=True,
        type_k=type_id,
        type_v=type_id,
        verbose=False
    )


def chat_completion(llm: Any, messages: List[Dict[str, str]], enable_thinking: bool, **kwargs: Any) -> Dict[str, Any]:
    """
    create_chat_completion with `enable_thinking` passed to the GGUF's chat template.

    Llama.create_chat_completion has a fixed signature and drops template variables, but the chat handler it
    delegates to forwards extra keyword arguments to the Jinja template, so the handler is called directly.
    Sampling defaults match create_chat_completion's (the handler's own repeat_penalty default differs).
    A llama-server client (any IChatModel) passes the flag to the server's template itself.
    """
    if isinstance(llm, IChatModel):
        return llm.chat(messages, enable_thinking=enable_thinking, **kwargs)

    from llama_cpp import llama_chat_format
    handler = (
        llm.chat_handler
        or getattr(llm, "_chat_handlers", {}).get(llm.chat_format)
        or llama_chat_format.get_chat_completion_handler(llm.chat_format)
    )
    kwargs.setdefault("repeat_penalty", 1.0)
    return handler(llama=llm, messages=messages, enable_thinking=enable_thinking, **kwargs)


def strip_reasoning(text: str) -> str:
    """
    Removes the model's <think> block (empty when thinking is disabled) from a reply, including reasoning whose
    opening tag was part of the prompt.
    """
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    text = _TEMPLATE_OPENED_THINK.sub("", text)
    return text.replace("\x00", "").strip()
