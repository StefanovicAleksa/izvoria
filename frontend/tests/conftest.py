# Canvas: desktop_tests_conftest_v1
# Path: frontend/tests/conftest.py

"""
frontend/tests/ inherits nothing: there is no conftest.py at the repo root or in tests/, and
pytest.ini's `pythonpath = .` only puts the repo root on the path, not `frontend/`.

Everything here that touches os.environ must run at *import* time, before PySide6 or app.* is
imported by any test module.
"""

import logging
import os
import sys
import uuid
from pathlib import Path

import pytest

FRONTEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRONTEND_ROOT.parent

# `onyx_desktop` lives under frontend/; `app` lives at the repo root.
for _path in (str(FRONTEND_ROOT), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# OVERWRITE, not setdefault: this machine's shell exports QT_QPA_PLATFORM="wayland;xcb", and the
# platform plugin is bound when the first QApplication is constructed. Offscreen needs no display,
# so GUI tests run anywhere rather than skipping.
os.environ["QT_QPA_PLATFORM"] = os.environ.get("ONYX_DESKTOP_TEST_PLATFORM", "offscreen")

# If a test ever does import app.core.config, it must not read the developer's real .env.
os.environ.setdefault("ENV", "test")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CHUNK_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SOURCE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
CLIP_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")
SESSION_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "gui: needs PySide6 and an offscreen QApplication"
    )


@pytest.fixture(autouse=True)
def _isolate_logging():
    """
    LogRelay attaches a handler to the `app` logger and raises its level. A handler left behind by
    one test would receive the next test's records and emit a Qt signal on a deleted QObject, which
    crashes somewhere unrelated.
    """
    target = logging.getLogger("app")
    before_handlers = list(target.handlers)
    before_level = target.level
    yield
    for handler in list(target.handlers):
        if handler not in before_handlers:
            target.removeHandler(handler)
    target.setLevel(before_level)


@pytest.fixture
def stub_gateway():
    from onyx_desktop.data.stub_gateway import StubEngineGateway

    return StubEngineGateway(latency_seconds=0.0)


@pytest.fixture(scope="session")
def qt_app():
    """
    One QApplication for the whole session.

    It must be a QApplication rather than a QCoreApplication even for the non-widget tests: shiboken
    refuses to replace an existing QCoreApplication singleton, so whichever module ran first would
    otherwise decide whether widget tests can run at all.
    """
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication

    existing = QCoreApplication.instance()
    if existing is not None:
        yield existing
        return

    app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path):
    """
    The light/dark choice is saved to QSettings. Pointed at a per-test directory, so no test reads the
    developer's real preference or writes one into it.
    """
    from PySide6.QtCore import QSettings

    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    yield


@pytest.fixture(autouse=True)
def _english_display_language():
    """Tests assert on English text, so every test starts and ends with the app in English."""
    from onyx_desktop.utils import i18n

    i18n.use(i18n.BASE)
    yield
    i18n.use(i18n.BASE)
