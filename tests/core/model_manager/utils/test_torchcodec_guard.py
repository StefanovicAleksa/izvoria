# Path: tests/core/model_manager/utils/test_torchcodec_guard.py

import sys
import pytest

from app.core.model_manager.utils.torchcodec_guard import block_torchcodec_import


class TestTorchcodecGuard:
    def test_blocks_inside_and_removes_entry_it_created(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torchcodec", raising=False)

        with block_torchcodec_import():
            assert sys.modules["torchcodec"] is None

        assert "torchcodec" not in sys.modules

    def test_restores_previous_module(self, monkeypatch):
        sentinel = object()
        monkeypatch.setitem(sys.modules, "torchcodec", sentinel)

        with block_torchcodec_import():
            assert sys.modules["torchcodec"] is None

        assert sys.modules["torchcodec"] is sentinel

    def test_restores_even_when_body_raises(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torchcodec", raising=False)

        with pytest.raises(RuntimeError):
            with block_torchcodec_import():
                raise RuntimeError("load failed")

        assert "torchcodec" not in sys.modules
