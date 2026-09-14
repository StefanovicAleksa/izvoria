# Canvas: test_vlm_multimodal_diagnostics_v1
# Path: integration_tests/features/visual_analysis/test_vlm_multimodal_diagnostics.py

"""
HISTORICAL DIAGNOSTIC TESTS - documents why Visual Analysis no longer uses
llama.cpp/GGUF for Qwen-VL. The adapter (app/features/visual_analysis/data/
vlm_adapter.py) was rewritten to use native Hugging Face Transformers instead;
these tests are kept as a permanent regression/reference record of the original
'?????' garbage-output bug and the deeper crash beneath it, so nobody re-attempts
the GGUF path without rediscovering this from scratch.

These tests exercise the raw `llama_cpp` library directly against the GGUF files
still present in models/qwen2-vl-7b/ (NOT settings.VISION_MODEL_PATH, which now
points at the HF-format directory the current adapter actually uses).

Findings (see each test's docstring for the isolated evidence):

1. Passing `clip_model_path=` directly to `Llama(...)` (the app's original approach)
   is not a real constructor parameter - `Llama.__init__` swallows it into an unused
   **kwargs catch-all - so the vision encoder is NEVER invoked. The model receives the
   literal chat-template placeholder text for the image (no real vision embeddings) and
   produces degenerate output as a result. This is confirmed both statically (test 1)
   and at runtime by the suspiciously tiny prompt-eval token count (test 2).

2. Wiring the image path *correctly*, via `chat_handler=MTMDChatHandler(clip_model_path=...)`
   (the documented way to attach a CLIP/mmproj vision encoder to this llama-cpp-python
   version), causes a native SIGSEGV inside the mtmd vision encoder during
   "encoding image slice". Confirmed as a genuine, unresolved UPSTREAM llama.cpp bug
   (SIGSEGV in ggml_gallocr_alloc_graph, inside clip_image_batch_encode) specific to the
   Qwen-VL model family's vision graph construction - resolution-independent and
   reproduced across NVIDIA/AMD/Intel GPUs by multiple people over several months, with
   no fix as of this writing. See:
     https://github.com/ggml-org/llama.cpp/issues/23407
     https://github.com/ggml-org/llama.cpp/issues/17012
     https://github.com/ggml-org/llama.cpp/issues/17422
   This is why the original (wrong) app code "worked" without crashing: it never
   actually reached the real vision-encoding path at all. Test 3 isolates this in a
   subprocess so a real segfault cannot take down the whole pytest run.
"""

import inspect
import json
import os
import subprocess
import sys

import pytest


def test_llama_init_does_not_accept_clip_model_path():
    """
    Static/API-contract check: `clip_model_path` is not a real `Llama.__init__`
    parameter in the installed llama-cpp-python version.

    This is the root cause of the '?????' bug: app/features/visual_analysis/data/
    vlm_adapter.py's `_get_qwen_model` passes `clip_model_path=mmproj_path` straight
    into `Llama(...)`, believing (per its own comment) that "modern llama.cpp detects
    the architecture and uses the correct internal multimodal logic". It does not -
    the kwarg is silently absorbed by `Llama.__init__`'s `**kwargs` and never used to
    load a vision encoder.

    If this test ever starts FAILING (i.e. `clip_model_path` becomes a real recognized
    parameter), that means a llama-cpp-python upgrade added native support for it, and
    `vlm_adapter.py` can most likely be simplified back to the direct-kwarg form.
    Until then, the fix is to wrap it in `MTMDChatHandler` and pass it via
    `chat_handler=` instead (see test_mtmd_chat_handler_image_encoding_crash below for
    the caveat on doing that today).
    """
    from llama_cpp import Llama

    params = inspect.signature(Llama.__init__).parameters
    assert "clip_model_path" not in params, (
        "Llama.__init__ now has a native 'clip_model_path' parameter - "
        "vlm_adapter.py's current direct-kwarg approach may actually work now. "
        "Re-investigate the original '?????' bug against the fixed handler wiring."
    )
    assert "chat_handler" in params, (
        "Llama.__init__ no longer accepts 'chat_handler' - the MTMDChatHandler "
        "wiring approach documented here is no longer valid either."
    )


def _run_vlm_probe(script: str, timeout: int = 120) -> subprocess.CompletedProcess:
    """
    Runs a small llama-cpp-python probe script in a fresh subprocess.

    Required because test_mtmd_chat_handler_image_encoding_crash reproduces a real
    native SIGSEGV - running that inline would kill (and likely corrupt reporting for)
    the entire pytest process, not just the one test.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_LOAD_IMAGE_AND_CHAT_TEMPLATE = """
import base64, json, sys
from llama_cpp import Llama
{handler_import}

with open({image_path!r}, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

{handler_setup}

llm = Llama(
    model_path={model_path!r},
    {llama_kwarg}
    n_ctx=4096,
    n_gpu_layers=-1,
    flash_attn=False,
    verbose=False,
)
resp = llm.create_chat_completion(
    messages=[
        {{"role": "user", "content": [
            {{"type": "text", "text": "Describe this image in one sentence."}},
            {{"type": "image_url", "image_url": {{"url": f"data:image/jpeg;base64,{{b64}}"}}}}
        ]}}
    ],
    max_tokens=64,
    temperature=0.1,
)
print(json.dumps({{
    "content": resp["choices"][0]["message"]["content"],
    "prompt_tokens": resp["usage"]["prompt_tokens"],
}}))
llm.close()
"""


# Hardcoded on purpose: these tests exercise the abandoned GGUF/llama.cpp path
# directly, not settings.VISION_MODEL_PATH (which now points at the HF-format
# directory the current, transformers-based adapter actually uses).
_GGUF_MODEL_PATH = "models/qwen2-vl-7b/Qwen2-VL-7B-Instruct-Q4_K_M.gguf"
_GGUF_MMPROJ_PATH = "models/qwen2-vl-7b/Qwen2-VL-7B-Instruct-vision-encoder.gguf"


@pytest.fixture(scope="module", autouse=True)
def ensure_vlm_assets_exist():
    if not os.path.exists(_GGUF_MODEL_PATH):
        pytest.skip(f"VLM GGUF base model missing at: {_GGUF_MODEL_PATH}")
    if not os.path.exists(_GGUF_MMPROJ_PATH):
        pytest.skip(f"VLM GGUF mmproj file missing at: {_GGUF_MMPROJ_PATH}")


@pytest.fixture
def xray_image_path() -> str:
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent.parent / "resources" / "test_medical_xray.jpg"
    if not path.exists():
        pytest.skip(f"Test image not found at {path}")
    return str(path)


def test_current_adapter_wiring_never_encodes_the_image(xray_image_path):
    """
    Runtime reproduction of the app's CURRENT (buggy) wiring: `clip_model_path=`
    passed straight to `Llama(...)`, exactly like
    app/features/visual_analysis/data/vlm_adapter.py:_get_qwen_model does today.

    Evidence this never processes the image: a real Qwen2-VL image at typical
    resolution expands to hundreds of vision-patch tokens (this exact image produces
    805 real patch tokens - see test_mtmd_chat_handler_image_encoding_crash). Here,
    with the broken wiring, `usage.prompt_tokens` stays tiny (observed: 29) because
    the image was never actually tokenized into vision embeddings - the model only
    saw the handful of literal text/template tokens.
    """
    script = _LOAD_IMAGE_AND_CHAT_TEMPLATE.format(
        handler_import="",
        handler_setup="",
        model_path=_GGUF_MODEL_PATH,
        llama_kwarg=f"clip_model_path={_GGUF_MMPROJ_PATH!r},",
        image_path=xray_image_path,
    )
    result = _run_vlm_probe(script)

    assert result.returncode == 0, (
        f"Probe crashed unexpectedly (stdout={result.stdout!r}, stderr tail={result.stderr[-2000:]!r})"
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    # This is the actual bug: the image is silently dropped, so almost no prompt
    # tokens are consumed and the output is degenerate.
    assert payload["prompt_tokens"] < 50, (
        "Prompt token count is no longer suspiciously small - the buggy wiring "
        "(clip_model_path= passed directly to Llama()) may have started working. "
        f"Full response: {payload}"
    )


def test_mtmd_chat_handler_image_encoding_crash(xray_image_path):
    """
    Runtime check of the CORRECT wiring: `chat_handler=MTMDChatHandler(clip_model_path=...)`.

    This does reach the real vision encoder (verbose logs show `image_tokens->nx=35,
    ny=23` => 805 real patch tokens, vs. 29 total tokens with the broken wiring above),
    but the native mtmd vision encoder then SIGSEGVs during "encoding image slice"
    with this llama-cpp-python build + Qwen2-VL mmproj GGUF + CUDA 13.4 combination.

    Run in a subprocess so the segfault can't take down the whole pytest session.

    This test currently documents a KNOWN, UNRESOLVED crash - it is expected to fail
    (marked xfail) until one of the following is tried and confirmed working:
      - a different llama-cpp-python version/build (this one is 0.3.35, self-compiled
        with CMAKE_ARGS="-DGGML_CUDA=on" against CUDA 13.4 - try upstream prebuilt
        wheels, or an older/newer release, to bisect),
      - a freshly re-converted/re-quantized mmproj GGUF for this model,
      - CPU-only mtmd encoding (n_gpu_layers=0) as a workaround to confirm whether the
        crash is CUDA-specific.
    """
    script = _LOAD_IMAGE_AND_CHAT_TEMPLATE.format(
        handler_import="from llama_cpp.llama_chat_format import MTMDChatHandler",
        handler_setup=(
            f"handler = MTMDChatHandler(clip_model_path={_GGUF_MMPROJ_PATH!r}, verbose=False)"
        ),
        model_path=_GGUF_MODEL_PATH,
        llama_kwarg="chat_handler=handler,",
        image_path=xray_image_path,
    )
    result = _run_vlm_probe(script)

    if result.returncode == -11:
        pytest.xfail(
            "KNOWN ISSUE: MTMDChatHandler segfaults (SIGSEGV) during native vision "
            "encoding for this llama-cpp-python/CUDA/model combination. See this "
            "test's docstring for next debugging steps. "
            f"stderr tail: {result.stderr[-1000:]}"
        )

    # If it didn't segfault, the crash may already be fixed (e.g. after a
    # llama-cpp-python upgrade) - verify it actually produced a real description
    # this time, matching what test_analysis_api_integration.py expects end-to-end.
    assert result.returncode == 0, (
        f"Probe failed with an unexpected error (not the known SIGSEGV): "
        f"returncode={result.returncode}, stderr tail={result.stderr[-2000:]!r}"
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["prompt_tokens"] > 100, (
        "Expected hundreds of real vision-patch tokens once mtmd actually encodes "
        f"the image. Full response: {payload}"
    )
    assert "?" * 10 not in payload["content"], (
        f"Still producing degenerate output even with correct handler wiring: {payload}"
    )
