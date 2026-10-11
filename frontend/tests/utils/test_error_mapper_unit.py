# Canvas: desktop_test_error_mapper_v1
# Path: frontend/tests/utils/test_error_mapper_unit.py

import pytest

from izvoria_desktop.exceptions.gateway import (
    ApplianceBootError,
    ApplianceBusyError,
    ApplianceOfflineError,
    EngineCallFailedError,
    InvalidRequestError,
    OutOfMemoryError,
)
from izvoria_desktop.utils.error_mapper import file_label, file_problem, map_engine_error


# Stand-ins named exactly as the engine's classes are. The mapper matches on class name walked
# over the MRO, precisely so it needs neither torch nor a database to be tested.
class EngineError(Exception):
    pass


class EngineOfflineError(EngineError):
    pass


class EngineBusyError(EngineError):
    pass


class EngineBootError(EngineError):
    pass


class OutOfVramError(Exception):
    pass


class GenerationOOMError(Exception):
    pass


class CudaCrashError(Exception):
    pass


class ContextLengthExceededError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


class MissingClipError(Exception):
    pass


class ValidationError(Exception):
    pass


class TestMapEngineError:
    @pytest.mark.parametrize(
        "raised, expected",
        [
            (EngineOfflineError("x"), ApplianceOfflineError),
            (EngineBusyError("x"), ApplianceBusyError),
            (EngineBootError("x"), ApplianceBootError),
            (OutOfVramError("x"), OutOfMemoryError),
            (GenerationOOMError("x"), OutOfMemoryError),
            (CudaCrashError("x"), EngineCallFailedError),
            (ContextLengthExceededError("x"), InvalidRequestError),
            (ModelNotFoundError("x"), ApplianceBootError),
            (MissingClipError("x"), EngineCallFailedError),
            (ValidationError("x"), InvalidRequestError),
            (RuntimeError("something else"), EngineCallFailedError),
        ],
    )
    def test_maps_to_expected_desktop_exception(self, raised, expected):
        mapped = map_engine_error(raised)
        assert isinstance(mapped, expected)
        assert mapped.original_exception is raised

    def test_every_mapped_error_carries_a_user_facing_sentence(self):
        for raised in (EngineBusyError("x"), OutOfVramError("x"), ModelNotFoundError("x")):
            mapped = map_engine_error(raised)
            # The message is shown verbatim in the UI, so it must be prose, not a class name.
            assert mapped.user_message
            assert mapped.user_message[0].isupper()
            assert mapped.user_message.endswith(".")
            assert "Error" not in mapped.user_message

    def test_a_subclass_of_a_known_error_still_maps(self):
        class WeirdBusyError(EngineBusyError):
            pass

        assert isinstance(map_engine_error(WeirdBusyError("x")), ApplianceBusyError)

    def test_an_already_mapped_error_passes_through_untouched(self):
        original = ApplianceBusyError("Already mapped.")
        assert map_engine_error(original) is original

    def test_the_operation_name_is_never_shown(self):
        # It names an internal operation; the worker logs it with the original exception instead
        assert map_engine_error(RuntimeError("boom"), context="Video ingestion").user_message == (
            "The appliance could not complete that request.")
        assert "Chat" not in map_engine_error(EngineBusyError("x"), context="Chat").user_message

    @pytest.mark.parametrize("name", [
        "EngineBusyError", "EngineBootError", "OutOfVramError", "GenerationOOMError", "CudaCrashError",
        "ContextLengthExceededError", "ModelNotFoundError", "ModelLoadError", "DatabaseError",
    ])
    def test_messages_do_not_name_the_appliances_components(self, name):
        message = map_engine_error(type(name, (Exception,), {})("x")).user_message.lower()
        for word in ("gpu", "vram", "model", "weights", "database", "container", "cuda", "context window"):
            assert word not in message


class TestFileProblems:
    @pytest.mark.parametrize("engine_message,expected", [
        ("Whisper heard 'de' (probability 0.97) in source 1234, not English, so it was not transcribed: ...",
         "Its spoken language does not match the one selected."),
        ("talk.mp4 was already ingested as 'sr'. Ingest it with 'sr' selected, or reset the index.",
         "It was added before in another language."),
        ("CUDA error: an illegal memory access in transcribe", "It could not be processed."),
    ])
    def test_engine_wording_becomes_one_plain_sentence(self, engine_message, expected):
        assert file_problem(engine_message) == expected

    @pytest.mark.parametrize("key,expected", [
        ("/media/batch/talk.mp4", "talk.mp4"),
        ("C:\\media\\talk.mp4", "talk.mp4"),
        ("talk.mp4", "talk.mp4"),
    ])
    def test_a_path_key_shows_only_the_file_name(self, key, expected):
        assert file_label(key) == expected
