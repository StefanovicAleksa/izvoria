import inspect

import pytest

from app.engine.domain.interfaces import IOnyxEngine
from app.engine.service.onyx_engine import OnyxEngine

PORT_METHODS = [name for name, _ in inspect.getmembers(IOnyxEngine, predicate=inspect.isfunction)]


class TestIOnyxEngine:
    """The port is what the desktop app, the CLI and the hosted API program against; it must say what the engine takes."""

    @pytest.mark.parametrize("name", PORT_METHODS)
    def test_the_engine_takes_exactly_the_parameters_the_port_declares(self, name):
        declared = inspect.signature(getattr(IOnyxEngine, name)).parameters
        implemented = inspect.signature(getattr(OnyxEngine, name)).parameters

        assert list(implemented) == list(declared)
        assert [p.default for p in implemented.values()] == [p.default for p in declared.values()]

    @pytest.mark.parametrize("name", [
        "search_standard", "search_verified", "chat", "chat_reasoned", "propose_research_plan", "get_report",
        "list_reports", "list_conversations", "get_conversation", "delete_conversation", "list_searches",
        "get_search", "delete_search",
    ])
    def test_every_per_user_call_takes_an_optional_owner(self, name):
        owner = inspect.signature(getattr(IOnyxEngine, name)).parameters["owner_id"]

        assert owner.default is None

    def test_declares_the_busy_snapshot_the_hosted_api_reads(self):
        assert "busy_snapshot" in PORT_METHODS
