# Canvas: desktop_test_main_window_smoke_v1
# Path: frontend/tests/ui/test_main_window_smoke.py

"""
Smoke tests for the shell.

Asserting on layout internals would break on every design tweak and prove nothing, so these only
cover the failures that are otherwise found by launching the app: a bad Signal signature, a
connect() to a slot that does not exist, and the close path -- which hung twice while being built,
once on a queued quit() behind a blocked event loop and once on a reentrant close().
"""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402

from onyx_desktop.domain.entities import ChatMessage  # noqa: E402
from onyx_desktop.domain.enums import (  # noqa: E402
    ApplianceStatus,
    ChatMode,
    MessageRole,
    SearchMode,
    TurnOutcome,
)
from onyx_desktop.service.engine_bridge import EngineBridge  # noqa: E402
from onyx_desktop.service.log_relay import LogRelay  # noqa: E402
from onyx_desktop.ui.main_window import MainWindow  # noqa: E402

pytestmark = pytest.mark.gui


def pump(ms: int = 250) -> None:
    """Runs the event loop briefly so queued cross-thread signals are delivered."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


@pytest.fixture
def window(qt_app, stub_gateway, session_store):
    bridge = EngineBridge(stub_gateway)
    instance = MainWindow(bridge, LogRelay(), session_store, stub_mode=True)
    yield instance
    if not instance._shutdown_done:
        bridge.stop(timeout_ms=5000)
    instance.deleteLater()


class TestMainWindowSmoke:
    def test_it_constructs_and_shows_with_three_views(self, window):
        window.show()
        assert window._views.count() == 3
        assert window.windowTitle().startswith("Onyx")

    def test_every_signal_connection_is_valid(self, window):
        """
        MainWindow._connect() wires ~12 signals. A wrong signature fails at connect time, so simply
        having reached this point proves them all -- this asserts the wiring is reachable end to end.
        """
        window.show()
        window._bridge.boot()
        pump(600)
        assert window._bridge.status is ApplianceStatus.READY

    def test_a_chat_turn_reaches_the_transcript(self, window):
        window.show()
        window._bridge.boot()
        pump(600)

        before = window._chat._transcript_layout.count()
        window._on_chat_requested("who signed it?", window._chat.session_id,
                                  ChatMode.SINGLE_PASS, 5)
        pump(600)

        assert window._chat._transcript_layout.count() > before

    def test_a_chat_turn_is_recorded_in_the_session_index(self, window, session_store):
        window.show()
        window._bridge.boot()
        pump(400)

        window._on_chat_requested("first question", window._chat.session_id,
                                  ChatMode.SINGLE_PASS, 5)
        pump(400)

        sessions = session_store.list_sessions()
        assert [s.title for s in sessions] == ["first question"]

    def test_a_search_populates_the_results_list(self, window):
        window.show()
        window._bridge.boot()
        pump(600)

        window._on_search_requested("retention schedule", SearchMode.STANDARD, 5)
        pump(600)

        # The stretch item plus one card per hit.
        assert window._search._results_layout.count() > 1

    def test_an_ungrounded_answer_renders_without_raising(self, window):
        """The warning path has its own widgets; make sure they actually build."""
        window.show()
        window._chat.add_message(ChatMessage(
            role=MessageRole.ASSISTANT,
            text="I could not find that in the material.",
            outcome=TurnOutcome.ANSWERED,
        ))
        assert window._chat._transcript_layout.count() >= 2

    def test_a_clarification_turn_renders_and_arms_the_clarification_flow(self, window):
        window.show()
        window._chat.add_message(ChatMessage(role=MessageRole.USER, text="tell me about it"))
        window._chat.add_message(ChatMessage(
            role=MessageRole.ASSISTANT,
            text="Which schedule do you mean?",
            outcome=TurnOutcome.NEEDS_CLARIFICATION,
        ))
        assert window._chat._clarification_hint.text() != ""


class TestMainWindowShutdown:
    def test_closing_stops_the_engine_and_lets_the_window_close(self, qt_app, stub_gateway,
                                                                session_store):
        bridge = EngineBridge(stub_gateway)
        window = MainWindow(bridge, LogRelay(), session_store, stub_mode=True)
        window.show()
        bridge.boot()
        pump(500)

        window.close()
        # The first close is refused and deferred to the event loop, so it needs a turn to finish.
        pump(800)

        assert window._shutdown_done is True
        assert bridge.status is ApplianceStatus.OFFLINE
        assert bridge._thread.isRunning() is False
        assert window.isVisible() is False
