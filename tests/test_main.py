# Path: tests/test_main.py

from unittest.mock import Mock, patch

import pytest

from app import main
from app.engine.exceptions.lifecycle import EngineBusyError


def _engine(language="en"):
    engine = Mock()
    engine.active_language = language
    return engine


class TestChooseLanguage:
    @pytest.mark.parametrize("answer, expected", [("sr", "sr"), (" SR ", "sr"), ("", "en"), ("de", None), ("sl", None)])
    def test_answers(self, answer, expected):
        with patch("builtins.input", return_value=answer):
            assert main.choose_language(_engine("en"), "Language of these files") == expected

    def test_offers_every_supported_language_by_its_own_name(self):
        with patch("builtins.input", return_value="") as ask:
            main.choose_language(_engine(), "Answer in")

        assert "en = English, sr = Srpski" in ask.call_args[0][0]


class TestSwitchLanguage:
    def test_the_active_language_keeps_the_engine(self):
        engine = _engine("sr")

        with patch.object(main.IzvoriaBootstrapper, "switch_language") as switch:
            assert main.switch_language(engine, "sr") is engine

        switch.assert_not_called()

    def test_another_language_rebuilds_the_engine(self):
        engine = _engine("en")

        with patch.object(main.IzvoriaBootstrapper, "switch_language") as switch:
            assert main.switch_language(engine, "sr") is switch.return_value

        switch.assert_called_once_with(engine, "sr")

    def test_a_busy_engine_is_reported_not_replaced(self):
        with patch.object(main.IzvoriaBootstrapper, "switch_language", side_effect=EngineBusyError("ingesting")), \
                pytest.raises(EngineBusyError):
            main.switch_language(_engine("en"), "sr")

    def test_a_failed_rebuild_restarts_in_the_previous_language(self):
        with patch.object(main.IzvoriaBootstrapper, "switch_language", side_effect=RuntimeError("no GPU")), \
                patch.object(main.IzvoriaBootstrapper, "build") as build:
            restarted = main.switch_language(_engine("en"), "sr")

        build.return_value.start.assert_called_once_with()
        assert restarted is build.return_value


class TestIngestBatch:
    def test_switches_to_the_batch_language_before_ingesting(self):
        engine, rebuilt, run = _engine("en"), _engine("sr"), Mock()
        with patch("builtins.input", side_effect=["/media/sr", "sr"]), \
                patch.object(main, "switch_language", return_value=rebuilt) as switch, \
                patch.object(main, "print_report"):
            result = main.ingest_batch(engine, "Folder", "Audio Pipeline", run)

        switch.assert_called_once_with(engine, "sr")
        run.assert_called_once_with(rebuilt, "/media/sr")
        assert result is rebuilt

    def test_an_unknown_language_ingests_nothing(self):
        run = Mock()
        with patch("builtins.input", side_effect=["/media/x", "de"]):
            main.ingest_batch(_engine(), "Folder", "Audio Pipeline", run)

        run.assert_not_called()
