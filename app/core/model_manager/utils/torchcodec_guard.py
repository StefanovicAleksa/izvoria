# Path: app/core/model_manager/utils/torchcodec_guard.py

import sys
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def block_torchcodec_import() -> Iterator[None]:
    """
    Temporarily blocks `import torchcodec` while a sentence-transformers model is imported/loaded.

    torchvision probes torchcodec's broken C++ codec bindings during that import and crashes the
    load. Poisoning `sys.modules["torchcodec"]` makes torchvision fall back gracefully. The entry is
    restored on exit: leaving it in place permanently broke every later `torchaudio.load()` in the
    same process (e.g. the VAD adapter). See
    integration_tests/core/model_manager/test_torchcodec_environment_integration.py for the
    remaining cross-feature leak this scoping does not fully eliminate.
    """
    had_torchcodec = "torchcodec" in sys.modules
    original_torchcodec = sys.modules.get("torchcodec")
    sys.modules["torchcodec"] = None  # type: ignore

    try:
        yield
    finally:
        if had_torchcodec:
            sys.modules["torchcodec"] = original_torchcodec
        else:
            sys.modules.pop("torchcodec", None)
