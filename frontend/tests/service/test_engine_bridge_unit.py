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
    ReportMode,
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
        bridge.start_ingest("/tmp/media", IngestMode.AUDIO, "en")
        assert bridge.is_ingesting is True

        wait_for(bridge.ingestFinished, timeout_ms=15000)
        assert ApplianceStatus.INGESTING in seen
        assert bridge.is_ingesting is False

    def test_progress_ticks_reach_the_gui_thread(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        ticks = []
        bridge.progressTicked.connect(ticks.append)
        bridge.start_ingest("/tmp/media", IngestMode.AUDIO, "en")
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


class TestReportRequests:
    def test_planning_and_running_go_through_the_worker_queue(self, bridge):
        bridge._status = ApplianceStatus.READY

        assert bridge.plan_report("a topic", ReportMode.DEEP) is not None
        bridge._pending.clear()
        assert bridge.run_report(object(), ReportMode.EXHAUSTIVE) is not None

    def test_a_running_report_marks_the_bridge_as_researching(self, bridge):
        bridge._status = ApplianceStatus.READY

        bridge.run_report(object(), ReportMode.DEEP)

        assert bridge.is_researching is True
        assert bridge.is_ingesting is False

    def test_cancel_calls_the_gateway_directly_not_through_the_worker(self, bridge, stub_gateway):
        """
        The one deliberate exception to "everything through the worker".

        The worker thread is busy running the very report being cancelled, so a queued call would not
        be delivered until that run had already finished -- which is exactly when it is useless. Safe
        only because the engine just sets a threading.Event here.
        """
        bridge._status = ApplianceStatus.READY
        report_id = uuid.uuid4()
        bridge.run_report(object(), ReportMode.DEEP)

        calls = []
        stub_gateway.request_report_cancel = lambda rid: calls.append(rid) or True

        assert bridge.request_report_cancel(report_id) is True
        # Delivered immediately, while the run is still pending -- not queued behind it.
        assert calls == [report_id]
        assert bridge.is_researching is True

    def test_a_second_request_while_researching_is_refused_not_queued(self, bridge):
        bridge._status = ApplianceStatus.READY
        bridge.run_report(object(), ReportMode.DEEP)

        assert bridge.send_chat("hello", uuid.uuid4(), ChatMode.SINGLE_PASS) is None


class TestHistoryReads:
    """
    History runs on its own thread: an ingestion or report holds the worker for hours, and picking up an
    old conversation must not wait behind it.
    """

    @pytest.fixture
    def blocking_gateway(self):
        import threading

        from onyx_desktop.data.stub_gateway import StubEngineGateway

        class BlockingIngest(StubEngineGateway):
            def __init__(self):
                super().__init__(latency_seconds=0.0)
                self.release = threading.Event()
                self.history_threads = []

            def ingest(self, path, mode, language):
                self.release.wait(10)
                return super().ingest(path, mode, language)

            def list_conversations(self):
                from PySide6.QtCore import QThread
                self.history_threads.append(QThread.currentThread().objectName())
                return super().list_conversations()

        return BlockingIngest()

    def test_history_answers_while_an_ingestion_holds_the_worker(self, qt_app, blocking_gateway):
        bridge = EngineBridge(blocking_gateway)
        try:
            bridge.boot()
            wait_for(bridge.booted)
            assert bridge.start_ingest("/media", IngestMode.VIDEO, "en") is not None

            bridge.list_conversations()
            (conversations,) = wait_for(bridge.conversationsListed)

            assert bridge.is_ingesting
            assert len(conversations) == 1
            assert blocking_gateway.history_threads == ["onyx-history"]
        finally:
            blocking_gateway.release.set()
            bridge.stop(timeout_ms=10000)

    def test_history_does_not_touch_the_busy_interlock(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)
        seen = []
        bridge.busyChanged.connect(seen.append)

        bridge.list_searches()
        wait_for(bridge.searchesListed)

        assert seen == [] and not bridge.is_busy

    def test_history_is_refused_before_the_engine_has_started(self, bridge):
        assert bridge.history_available is False
        assert bridge.list_reports() is None

    def test_deleting_re_lists(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)
        bridge.list_searches()
        (before,) = wait_for(bridge.searchesListed)

        bridge.delete_search(before[0].record_id)
        (after,) = wait_for(bridge.searchesListed)

        assert len(after) == len(before) - 1

    def test_a_loaded_conversation_carries_its_session_id(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)
        bridge.list_conversations()
        (conversations,) = wait_for(bridge.conversationsListed)

        bridge.load_conversation(conversations[0].session_id)
        session_id, messages = wait_for(bridge.conversationLoaded)

        assert session_id == conversations[0].session_id and messages

    def test_a_failed_read_names_the_call(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)
        bridge.load_search(uuid.uuid4())
        method, message = wait_for(bridge.historyFailed)
        assert method == "load_search" and "no longer" in message

    def test_resume_is_a_gpu_job_on_the_worker(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)
        bridge.list_reports()
        (reports,) = wait_for(bridge.reportsListed)
        stopped = next(r for r in reports if r.resumable)

        assert bridge.resume_report(stopped.report_id) is not None
        assert bridge.is_researching
        _, document = wait_for(bridge.reportFinished)
        assert document.status is not None and not bridge.is_busy


class TestHistoryReader:
    def test_it_refuses_anything_that_is_not_a_history_call(self, stub_gateway):
        from onyx_desktop.service.history_reader import HistoryReader
        with pytest.raises(ValueError):
            HistoryReader(stub_gateway).run(1, "ingest", ("/media", IngestMode.VIDEO))


class TestLanguage:
    def test_languages_and_the_active_one_are_known_before_boot(self, bridge):
        assert [option.code for option in bridge.languages] == ["en", "sr"]
        assert bridge.active_language == "en"
        assert bridge.language_name("sr") == "Srpski"

    def test_switching_runs_on_the_worker_and_announces_the_new_language(self, bridge, stub_gateway):
        bridge.boot()
        wait_for(bridge.booted)
        changes = []
        bridge.languageChanged.connect(changes.append)

        assert bridge.switch_language("sr") is not None
        (_rid, code) = wait_for(bridge.languageSwitchFinished)

        assert code == "sr" and changes == ["sr"]
        assert bridge.active_language == stub_gateway.active_language() == "sr"
        assert bridge.is_busy is False

    def test_switching_to_the_active_language_is_not_a_request(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        assert bridge.switch_language("en") is None
        assert bridge.is_busy is False

    def test_switching_is_refused_before_boot_and_while_busy(self, bridge):
        assert bridge.switch_language("sr") is None

        bridge.boot()
        wait_for(bridge.booted)
        bridge.start_ingest("/tmp/media", IngestMode.AUDIO, "en")
        assert bridge.switch_language("sr") is None
        wait_for(bridge.ingestFinished, timeout_ms=15000)

    def test_history_is_closed_while_the_engine_is_rebuilt(self, bridge):
        bridge.boot()
        wait_for(bridge.booted)

        bridge.switch_language("sr")
        assert bridge.history_available is False
        wait_for(bridge.languageSwitchFinished)
        assert bridge.history_available is True

    def test_a_batch_in_another_language_switches_the_appliance_to_it(self, bridge, stub_gateway):
        bridge.boot()
        wait_for(bridge.booted)
        changes = []
        bridge.languageChanged.connect(changes.append)

        bridge.start_ingest("/tmp/media", IngestMode.AUDIO, "sr")
        wait_for(bridge.ingestFinished, timeout_ms=15000)

        assert stub_gateway.ingested_languages == ["sr"]
        assert changes == ["sr"] and bridge.active_language == "sr"
