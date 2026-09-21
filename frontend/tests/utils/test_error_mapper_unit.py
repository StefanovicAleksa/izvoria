# Canvas: desktop_test_error_mapper_v1
# Path: frontend/tests/utils/test_error_mapper_unit.py

import pytest

from onyx_desktop.exceptions.gateway import (
    ApplianceBootError,
    ApplianceBusyError,
    ApplianceOfflineError,
    EngineCallFailedError,
    InvalidRequestError,
    OutOfMemoryError,
)
from onyx_desktop.utils.error_mapper import map_engine_error


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

    def test_context_is_included_only_in_the_generic_message(self):
        generic = map_engine_error(RuntimeError("boom"), context="Chat")
        assert "(Chat)" in generic.user_message

        specific = map_engine_error(EngineBusyError("x"), context="Chat")
        assert "Chat" not in specific.user_message
