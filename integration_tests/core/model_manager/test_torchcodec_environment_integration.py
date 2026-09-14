# Canvas: torchcodec_environment_integration_v1
# Path: integration_tests/core/model_manager/test_torchcodec_environment_integration.py

"""
Regression coverage for a cross-feature global-state bug.

app/features/rag_storage/data/hf_embedder.py:_get_embedding_model() does
`sys.modules["torchcodec"] = None` ("THE SILVER BULLET FIX") to stop torchvision
from touching a broken C++ video-codec path while SentenceTransformer loads the
text embedding model. Setting a sys.modules entry to None makes Python treat that
module as permanently unavailable: any later `import torchcodec` anywhere else in
the SAME process then raises ImportError, whether or not torchcodec is actually
installed - and the hack is never undone in unload(), so the poisoning outlives
the embedding model's own lifecycle for as long as the process runs.

app/features/vad/data/nemo_adapter.py calls torchaudio.load(), which (as of
torchaudio 2.9+) internally does `from torchcodec.decoders import AudioDecoder`.
If the embedding model was ever loaded earlier in the same process, that import
hits the poisoned sys.modules entry and raises ImportError, surfacing as:
    VadError: An unexpected error occurred in the VAD feature:
    TorchCodec is required for load_with_torchcodec. Please install torchcodec
    to use this function.
This is exactly the failure seen running the real app: Standard Search loads the
embedding model, then Ingest boots the VAD phase in the same process.

Neither existing test catches this:
  - tests/features/vad/data/test_nemo_adapter.py mocks torchaudio entirely, so it
    never touches real torchcodec regardless of process state.
  - integration_tests/features/vad/test_vad_integration.py exercises real
    torchaudio.load() on real audio, but that pytest process never loads the
    embedding model, so sys.modules is never poisoned there either.
Each feature is only ever tested in its own isolated process, so the interaction
between them has no coverage.

Since sys.modules poisoning is real, process-global state, every scenario below
runs in its own subprocess - both to keep cases from contaminating each other and
to mirror the fact that this only reproduces when both adapters share one process,
exactly like the real app does.

NOT related to llama.cpp: the visual_analysis adapter's earlier llama.cpp/GGUF
removal (see ../../features/visual_analysis/test_vlm_multimodal_diagnostics.py)
fixed an unrelated native SIGSEGV in llama.cpp's own vision graph allocator. That
bug and this one share no code path - this is a pure-Python sys.modules hack in
hf_embedder.py, and would behave identically under any llama.cpp version.
"""

import json
import os
import subprocess
import sys
import uuid

import pytest
import torch

from app.core.config import settings

_REAL_AUDIO_PATH = "integration_tests/resources/test_speech.wav"


@pytest.fixture(scope="module", autouse=True)
def ensure_hardware_and_assets():
    if not torch.cuda.is_available():
        pytest.skip("Integration test requires an active NVIDIA GPU.")
    if not os.path.exists(_REAL_AUDIO_PATH):
        pytest.skip(f"Test audio missing at {_REAL_AUDIO_PATH}.")
    for label, path in [
        ("VAD", settings.VAD_MODEL_PATH),
        ("Vision", settings.VISION_MODEL_PATH),
        ("RAG embedding", settings.RAG_EMBEDDING_MODEL_PATH),
    ]:
        if not os.path.exists(path):
            pytest.skip(f"{label} model missing at {path}.")


def _run_probe(script: str, timeout: int = 300) -> dict:
    """
    Runs a probe script in a fresh subprocess (own sys.modules, own CUDA context)
    from the repo root so `import app...` resolves the same way it does for pytest.
    """
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(settings.BASE_DIR),
    )
    assert result.returncode == 0, (
        f"Probe subprocess crashed unexpectedly "
        f"(returncode={result.returncode}, stderr tail={result.stderr[-3000:]!r})"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


_CHECK_TORCHCODEC_TAIL = """
import json
import torch, torchaudio
try:
    wf, sr = torchaudio.load({audio_path!r})
    print(json.dumps({{"torchcodec_works": True, "error": None}}))
except Exception as e:
    print(json.dumps({{"torchcodec_works": False, "error": f"{{type(e).__name__}}: {{e}}"}}))
"""


def test_torchcodec_works_standalone():
    """Baseline: torchaudio.load() works when no Onyx model adapter has run yet."""
    script = _CHECK_TORCHCODEC_TAIL.format(audio_path=_REAL_AUDIO_PATH)
    payload = _run_probe(script)
    assert payload["torchcodec_works"], f"torchcodec is broken even standalone: {payload['error']}"


def test_torchcodec_works_after_vad_model_load():
    """torchaudio.load() must keep working after the VAD/NeMo adapter has loaded its model."""
    script = """
import json
from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.vad.data.nemo_adapter import NeMoVadAdapter

manifest = ModelManifest(
    model_type=ModelType.VAD,
    model_path=settings.VAD_MODEL_PATH,
    required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB,
)
NeMoVadAdapter().load(manifest)
""" + _CHECK_TORCHCODEC_TAIL.format(audio_path=_REAL_AUDIO_PATH)

    payload = _run_probe(script)
    assert payload["torchcodec_works"], (
        f"torchcodec broke after loading the VAD/NeMo model: {payload['error']}"
    )


def test_torchcodec_works_after_vlm_adapter_load():
    """torchaudio.load() must keep working after the (transformers-based) VLM adapter has loaded its model."""
    script = """
import json
from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter

manifest = ModelManifest(
    model_type=ModelType.VISION,
    model_path=settings.VISION_MODEL_PATH,
    required_vram_mb=settings.VISION_VRAM_REQUIREMENT_MB,
)
QwenVLMAdapter().load(manifest)
""" + _CHECK_TORCHCODEC_TAIL.format(audio_path=_REAL_AUDIO_PATH)

    payload = _run_probe(script)
    assert payload["torchcodec_works"], (
        f"torchcodec broke after loading the VLM (Qwen2-VL) adapter: {payload['error']}"
    )


def test_torchcodec_breaks_after_embedding_model_load_then_vad_runs():
    """
    KNOWN BUG - real end-to-end reproduction of the production crash: load the
    HuggingFace embedding model (as "Standard Search" does), then run the real
    VAD pipeline (as "Ingest" does) in the same process.

    Root cause: app/features/rag_storage/data/hf_embedder.py's
    `sys.modules["torchcodec"] = None` hack leaks past the embedding model's own
    lifecycle. If this test ever starts passing, the hack has been scoped/fixed
    (or the underlying torchvision/torchcodec conflict it worked around no longer
    exists) - remove the xfail and keep the passing assertion as the regression
    guard.
    """
    script = """
import json
import uuid
from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.rag_storage.data.hf_embedder import HuggingFaceEmbeddingAdapter
from app.features.vad.data.nemo_adapter import NeMoVadAdapter

embed_manifest = ModelManifest(
    model_type=ModelType.RAG_EMBEDDING,
    model_path=settings.RAG_EMBEDDING_MODEL_PATH,
    required_vram_mb=settings.RAG_EMBEDDING_VRAM_REQUIREMENT_MB,
)
# This is what poisons sys.modules["torchcodec"] for the rest of the process.
HuggingFaceEmbeddingAdapter().load(embed_manifest)

vad_manifest = ModelManifest(
    model_type=ModelType.VAD,
    model_path=settings.VAD_MODEL_PATH,
    required_vram_mb=settings.VAD_VRAM_REQUIREMENT_MB,
)
vad_adapter = NeMoVadAdapter()
loaded_vad = vad_adapter.load(vad_manifest)

try:
    vad_map = vad_adapter.generate_vad_map(loaded_vad, {audio_path!r}, uuid.uuid4())
    print(json.dumps({{"torchcodec_works": True, "segments": len(vad_map.segments), "error": None}}))
except Exception as e:
    print(json.dumps({{"torchcodec_works": False, "error": f"{{type(e).__name__}}: {{e}}"}}))
""".format(audio_path=_REAL_AUDIO_PATH)

    payload = _run_probe(script)

    if not payload["torchcodec_works"]:
        pytest.xfail(
            "KNOWN ISSUE: app/features/rag_storage/data/hf_embedder.py sets "
            "sys.modules['torchcodec'] = None to load the embedding model, and never "
            "restores it. Any later torchaudio.load() call in the same process (e.g. "
            "NeMoVadAdapter.generate_vad_map) then fails with an ImportError, exactly "
            "reproducing the real 'Standard Search then Ingest' crash. "
            f"Observed error: {payload['error']}"
        )

    # If it didn't fail, the hack has been fixed/scoped - lock in the real behavior.
    assert payload["segments"] > 0, f"Expected VAD to detect speech segments: {payload}"
