# Canvas: desktop_test_engine_bridge_v1
# Path: frontend/tests/service/test_engine_bridge_unit.py

"""
The bridge's job is serialization and interlocking, so these tests drive a real QThread and assert
on what crosses back. They use a plain nested event loop with a hard timeout rather than
pytest-qt, to avoid adding a dependency for four tests -- and to make a never-fired signal fail
instead of hanging forever.
"""

import uuid

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from onyx_desktop.domain.enums import (  # noqa: E402
    ApplianceStatus,
    ChatMode,
    IngestMode,
    SearchMode,
)
from onyx_desktop.service.engine_bridge import EngineBridge  # noqa: E402

pytestmark = pytest.mark.gui

SESSION_ID = uuid.UUID("44444444-4444-4444-4444-444444444444")


def wait_for(signal, timeout_ms=5000):
    """Spins an event loop until `signal` fires. Returns the emitted args, or fails on timeout."""
    loop = QEventLoop()
    captured = {}

    def on_signal(*args):
        captured["args"] = args
        loop.quit()

    signal.connect(on_signal)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    signal.disconnect(on_signal)

    if "args" not in captured:
        pytest.fail(f"signal did not fire within {timeout_ms}ms")
    return captured["args"]


@pytest.fixture
def bridge(qt_app, stub_gateway):
    instance = EngineBridge(stub_gateway)
    yield instance
    instance.stop(timeout_ms=5000)


class TestBoot:
    def test_boot_moves_the_status_to_ready(self, bridge):
        assert bridge.status is ApplianceStatus.OFFLINE
        bridge.boot()
        wait_for(bridge.booted)
        assert bridge.status is ApplianceStatus.READY
        assert bridge.is_busy is False

    def test_work_before_boot_is_refused_with_a_message(self, bridge):
        # This rejection is synchronous -- it never reaches the worker -- so the listener has to be
        # attached before the call rather than waited on afterwards.
        refusals = []
        bridge.failed.connect(lambda rid, msg: refusals.append((rid, msg)))

        assert bridge.run_search("q", SearchMode.STANDARD) is None

        assert len(refusals) == 1
        request_id, message = refusals[0]
        assert request_id == 0
        assert "not running" in message


class TestSerialization:
    def test_a_second_request_while_busy_is_refused_not_queued(self, bridge):
        # Queueing a second GPU call would collide over a single-slot VRAM pool, and the user
        # would see it fail minutes later for no visible reason.
        bridge.boot()
        wait_for(bridge.booted)

        first = bridge.send_chat("one", SESSION_ID, ChatMode.SINGLE_PASS, 5)
        assert first is not None
        assert bridge.is_busy is True

        second = bridge.send_chat("two", SESSION_ID, ChatMode.SINGLE_PASS, 5)
        assert second is None

        wait_for(bridge.chatFinished)
        assert bridge.is_busy is False

    def test_busy_toggles_around_a_request(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        states = []
        bridge.busyChanged.connect(states.append)
        bridge.run_search("q", SearchMode.STANDARD, 5)
        wait_for(bridge.searchFinished)

        assert states == [True, False]

    def test_a_reply_carries_the_request_id_that_asked(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        request_id = bridge.run_search("q", SearchMode.STANDARD, 5)
        replied_id, outcome = wait_for(bridge.searchFinished)

        assert replied_id == request_id
        assert outcome.query == "q"


class TestStatus:
    def test_ingestion_reports_the_ingesting_status(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        seen = []
        bridge.statusChanged.connect(lambda snapshot: seen.append(snapshot.status))
        bridge.start_ingest("/tmp/media", IngestMode.AUDIO)
        assert bridge.is_ingesting is True

        wait_for(bridge.ingestFinished, timeout_ms=15000)
        assert ApplianceStatus.INGESTING in seen
        assert bridge.is_ingesting is False

    def test_progress_ticks_reach_the_gui_thread(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        ticks = []
        bridge.progressTicked.connect(ticks.append)
        bridge.start_ingest("/tmp/media", IngestMode.AUDIO)
        wait_for(bridge.ingestFinished, timeout_ms=15000)

        assert ticks, "the JobManager progress callback produced no ticks"
        assert all(tick.phase for tick in ticks)


class TestFailure:
    def test_a_failing_call_reports_a_user_message_and_clears_busy(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        # The stub raises on a prompt containing "fail".
        bridge.send_chat("please fail", SESSION_ID, ChatMode.SINGLE_PASS, 5)
        _request_id, message = wait_for(bridge.failed)

        assert message
        assert bridge.is_busy is False
        assert bridge.status is ApplianceStatus.READY


class TestShutdown:
    def test_stop_joins_the_worker_thread(self, qt_app, stub_gateway):
        instance = EngineBridge(stub_gateway)
        instance.boot()
        wait_for(instance.booted)

        assert instance.stop(timeout_ms=5000) is True
        assert instance._thread.isRunning() is False
        assert instance.status is ApplianceStatus.OFFLINE

    def test_stop_is_idempotent(self, qt_app, stub_gateway):
        instance = EngineBridge(stub_gateway)
        assert instance.stop(timeout_ms=5000) is True
        assert instance.stop(timeout_ms=5000) is True
