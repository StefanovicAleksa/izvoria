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

from izvoria_desktop.domain.entities import ChatMessage  # noqa: E402
from izvoria_desktop.domain.enums import (  # noqa: E402
    ApplianceStatus,
    ChatMode,
    MessageRole,
    SearchMode,
    TurnOutcome,
)
from izvoria_desktop.service.engine_bridge import EngineBridge  # noqa: E402
from izvoria_desktop.service.log_relay import LogRelay  # noqa: E402
from izvoria_desktop.ui.main_window import MainWindow  # noqa: E402

pytestmark = pytest.mark.gui


def pump(ms: int = 250) -> None:
    """Runs the event loop briefly so queued cross-thread signals are delivered."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


@pytest.fixture
def window(qt_app, stub_gateway):
    bridge = EngineBridge(stub_gateway)
    instance = MainWindow(bridge, LogRelay(), stub_mode=True)
    yield instance
    if not instance._shutdown_done:
        bridge.stop(timeout_ms=5000)
    instance.deleteLater()


class TestMainWindowSmoke:
    def test_it_constructs_and_shows_with_four_views(self, window):
        window.show()
        # Chat, Search, Ingest, Report. The nav button id IS the stack index, so this count and the
        # label tuple in _sidebar() must stay in step.
        assert window._views.count() == 4
        assert window.windowTitle().startswith("Izvoria")

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

    def test_a_chat_turn_appears_in_the_conversation_history(self, window):
        window.show()
        window._bridge.boot()
        pump(400)
        before = len(window._chat._history.keys())

        window._on_chat_requested("first question", window._chat.session_id,
                                  ChatMode.SINGLE_PASS, 5)
        pump(600)

        assert window._chat.session_id in window._chat._history.keys()
        assert len(window._chat._history.keys()) == before + 1

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


class TestApplianceControls:
    def test_only_start_shows_before_boot(self, window):
        window.show()
        assert window._start_button.isVisible() is True
        assert window._stop_button.isVisible() is False

    def test_stop_replaces_start_once_running(self, window):
        window.show()
        window._bridge.boot()
        pump(600)
        assert window._start_button.isVisible() is False
        assert window._stop_button.isVisible() is True
        assert window._stop_button.isEnabled() is True

    def test_stop_cannot_be_pressed_while_a_job_runs(self, window):
        window.show()
        window._bridge.boot()
        pump(600)

        window._bridge.run_search("q", SearchMode.STANDARD, 5)
        assert window._stop_button.isEnabled() is False

        pump(600)
        assert window._stop_button.isEnabled() is True

    def test_stopping_brings_start_back_and_it_starts_again(self, window):
        window.show()
        window._bridge.boot()
        pump(600)

        window._stop_button.click()
        pump(600)
        assert window._bridge.status is ApplianceStatus.OFFLINE
        assert window._start_button.isVisible() is True
        assert window._stop_button.isVisible() is False
        assert window._language.isEnabled() is False

        window._start_button.click()
        pump(600)
        assert window._bridge.status is ApplianceStatus.READY
        assert window._stop_button.isEnabled() is True


class TestThemes:
    @pytest.mark.parametrize("mode", ["dark", "light"])
    def test_the_window_builds_in_each_mode(self, qt_app, stub_gateway, mode):
        from izvoria_desktop.ui import theme

        try:
            theme.use(mode)
            theme.apply_to(qt_app)
            bridge = EngineBridge(stub_gateway)
            window = MainWindow(bridge, LogRelay(), stub_mode=True)
            window.show()
            pump(100)
            assert window._views.count() == 4
            bridge.stop(timeout_ms=5000)
            window.deleteLater()
        finally:
            theme.use()
            theme.apply_to(qt_app)


class TestModeSwitch:
    """Light/dark is switched in place, at any time, and remembered."""

    @pytest.fixture(autouse=True)
    def _restore_theme(self, qt_app):
        from izvoria_desktop.ui import theme

        yield
        theme.use()
        theme.apply_to(qt_app)

    def test_the_toggle_switches_the_whole_app_and_back(self, window, qt_app):
        from izvoria_desktop.ui import appearance, theme

        window.show()
        assert theme.MODE == "dark"
        assert window._mode_toggle.toolTip() == "Switch to light mode"

        window._mode_toggle.click()

        assert theme.MODE == "light"
        assert qt_app.styleSheet() == theme.STYLESHEET
        assert window._mode_toggle.toolTip() == "Switch to dark mode"
        assert appearance.saved_mode() == "light"

        window._mode_toggle.click()
        assert theme.MODE == "dark"
        assert appearance.saved_mode() == "dark"

    def test_it_works_while_the_appliance_is_busy(self, window):
        from izvoria_desktop.ui import theme

        window.show()
        window._bridge.boot()
        pump(600)
        window._bridge.run_search("q", SearchMode.STANDARD, 5)
        assert window._bridge.is_busy is True

        window._mode_toggle.click()
        assert theme.MODE == "light"
        pump(600)

    def test_citations_already_on_screen_take_the_new_colour(self, window):
        from izvoria_desktop.ui import theme
        from izvoria_desktop.ui.widgets.message_bubble import MessageBubble

        window.show()
        window._chat.add_message(ChatMessage(role=MessageRole.ASSISTANT, text="Signed on the 14th [Doc 1]."))
        (bubble,) = window.findChildren(MessageBubble)
        dark_citation = theme.CITATION
        assert dark_citation in bubble._body_label.text()

        window.set_mode("light")

        assert theme.CITATION != dark_citation
        assert theme.CITATION in bubble._body_label.text()
        assert dark_citation not in bubble._body_label.text()

    def test_an_answer_scoped_to_one_source_names_it(self, window):
        from PySide6.QtWidgets import QLabel
        from izvoria_desktop.ui.widgets.message_bubble import MessageBubble

        window._chat.add_message(ChatMessage(role=MessageRole.ASSISTANT, text="Bayes [Doc 1].",
                                             scoped_source="Lecture 4 Conditional Probability"))
        (bubble,) = window.findChildren(MessageBubble)

        texts = [label.text() for label in bubble.findChildren(QLabel)]
        assert "Answered from: Lecture 4 Conditional Probability" in texts


class TestDisplayLanguage:
    """The app's own text switches language in place, separately from the language answers are written in."""

    def test_switching_relabels_the_whole_window_and_back(self, window):
        from izvoria_desktop.ui import appearance
        from izvoria_desktop.utils import i18n

        window.show()
        window._bridge.boot()
        pump(600)
        answer_language = window._bridge.active_language

        window.set_display_language("sr")

        assert i18n.current() == "sr"
        assert window._nav.button(0).text() == "Razgovor"
        assert window._stop_button.text() == "Zaustavi sistem"
        assert window._card_state.text() == "Spreman"
        assert window._display_language.text() == "SR"
        assert appearance.saved_display_language() == "sr"
        # The appliance's language is its own setting and does not move
        assert window._bridge.active_language == answer_language

        window.set_display_language("en")
        assert window._nav.button(0).text() == "Chat"

    def test_dynamic_text_already_on_screen_is_re_said(self, window):
        window.show()
        window._bridge.boot()
        pump(600)
        window._bridge.run_search("q", SearchMode.STANDARD, 5)
        pump(600)
        assert window._search._summary.text().startswith("5 results for")

        window.set_display_language("sr")

        assert window._search._summary.text().startswith("5 rezultata za")
        assert "rezultat" in window._search._history._list.item(0).text()

    def test_it_works_while_the_appliance_is_busy(self, window):
        window.show()
        window._bridge.boot()
        pump(600)
        window._bridge.run_search("q", SearchMode.STANDARD, 5)
        assert window._bridge.is_busy

        window.set_display_language("sr")
        assert window._nav.button(1).text() == "Pretraga"
        pump(600)

    def test_the_picker_lists_every_display_language_by_its_own_name(self, window):
        names = [action.text() for action in window._display_language.menu().actions()]
        assert names[:2] == ["English", "Srpski"]


class TestMainWindowShutdown:
    def test_closing_stops_the_engine_and_lets_the_window_close(self, qt_app, stub_gateway):
        bridge = EngineBridge(stub_gateway)
        window = MainWindow(bridge, LogRelay(), stub_mode=True)
        window.show()
        bridge.boot()
        pump(500)

        window.close()
        # The first close is refused and deferred to the event loop, so it needs a turn to finish.
        pump(800)

        assert window._shutdown_done is True
        assert bridge.status is ApplianceStatus.OFFLINE
        assert bridge._thread.isRunning() is False
        assert bridge._reader_thread.isRunning() is False
        assert window.isVisible() is False


class TestMainWindowHistory:
    """The stub seeds one of everything, so each reopen path runs through the real bridge and views."""

    @pytest.fixture
    def booted(self, window):
        window.show()
        window._bridge.boot()
        pump(600)
        return window

    def test_history_is_listed_after_boot_and_not_before(self, qt_app, stub_gateway):
        bridge = EngineBridge(stub_gateway)
        window = MainWindow(bridge, LogRelay(), stub_mode=True)
        assert window._chat._history.keys() == []
        assert bridge.list_conversations() is None  # no database until the engine runs
        bridge.boot()
        pump(600)
        assert len(window._chat._history.keys()) == 1
        assert len(window._search._history.keys()) == 1
        assert len(window._report._history.keys()) == 3
        bridge.stop(timeout_ms=5000)

    def test_reopening_a_conversation_renders_it_and_continues_it(self, booted):
        (session_id,) = booted._chat._history.keys()
        booted._chat._on_history_chosen(session_id)
        pump(400)

        assert booted._chat.session_id == session_id
        # Two turns (four bubbles), a note for the answer whose source is gone, and the trailing stretch.
        assert booted._chat._transcript_layout.count() == 6

    def test_reopening_a_search_shows_its_saved_hits(self, booted):
        (record_id,) = booted._search._history.keys()
        booted._search.savedSearchChosen.emit(record_id)
        pump(400)

        assert booted._search._saved_row.isVisibleTo(booted._search)
        assert "no longer available" in booted._search._saved_banner.text()
        assert booted._search._input.text() == "retention schedule cover letter"

    def test_an_unapproved_report_reopens_in_the_approval_step(self, booted):
        ref = next(r for r in booted._report._refs.values() if r.awaiting_approval)
        booted._report._on_history_chosen(ref.report_id)
        pump(400)

        assert booted._report._outline.isVisibleTo(booted._report)
        assert booted._report._topic.text() == ref.topic

    def test_a_stopped_report_can_be_resumed_to_completion(self, booted):
        ref = next(r for r in booted._report._refs.values() if r.resumable)
        booted._report._on_history_chosen(ref.report_id)
        pump(400)
        assert booted._report._resume.isVisibleTo(booted._report)

        booted._report._on_resume()
        pump(800)

        assert booted._report._document.status.value == "completed"
        assert not booted._report._resume.isVisibleTo(booted._report)
        assert booted._report._refs[ref.report_id].status.value == "completed"

    def test_deleting_the_open_conversation_starts_a_new_one(self, booted, monkeypatch):
        (session_id,) = booted._chat._history.keys()
        booted._chat._on_history_chosen(session_id)
        pump(400)

        booted._on_conversation_delete_requested(session_id)
        pump(400)

        assert booted._chat.session_id != session_id
        assert booted._chat._history.keys() == []


class TestMainWindowLanguage:
    @pytest.fixture
    def booted(self, window):
        window.show()
        window._bridge.boot()
        pump(600)
        return window

    def test_the_answer_language_picker_waits_for_the_appliance(self, window):
        assert not window._language.isEnabled()
        assert window._language.currentData() == "en"
        assert window._chat._answer_language.text() == "Answers in English"

    def test_choosing_a_language_switches_the_appliance_and_every_header(self, booted, stub_gateway):
        assert booted._language.isEnabled()

        booted._language.setCurrentIndex(booted._language.findData("sr"))
        pump(600)

        assert stub_gateway.active_language() == "sr"
        for view in (booted._chat, booted._search, booted._report):
            assert view._answer_language.text() == "Answers in Srpski"
        assert booted._ingest._language.currentData() == "sr"

    def test_a_refused_switch_puts_the_picker_back(self, booted, monkeypatch):
        monkeypatch.setattr(booted._bridge, "switch_language", lambda code: None)

        booted._language.setCurrentIndex(booted._language.findData("sr"))

        assert booted._language.currentData() == "en"

    def test_ingesting_in_another_language_moves_the_answer_picker_too(self, booted, stub_gateway):
        booted._ingest._path.setText("/media/sr")
        booted._ingest.select_language("sr")
        booted._ingest._on_start()
        pump(3000)

        assert stub_gateway.ingested_languages == ["sr"]
        assert booted._language.currentData() == "sr"
