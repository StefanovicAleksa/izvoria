import json
from unittest.mock import AsyncMock

import pytest

from deploy.gateway.agent_client import AgentUnavailable
from deploy.gateway.sources import LONG_POLL_SECONDS, resume_point, source_events

ADDRESS = "127.0.0.1:8090"


def offer(seq):
    return {"id": f"{seq:022d}", "seq": seq, "kind": "pdf"}


async def take(events, n):
    out = []
    async for event in events:
        out.append(event)
        if len(out) == n:
            break
    await events.aclose()
    return out


def parsed(event):
    fields = dict(line.split(": ", 1) for line in event.strip().split("\n"))
    return fields


@pytest.fixture
def agent():
    return AsyncMock()


async def address():
    return ADDRESS


class TestResumePoint:
    @pytest.mark.parametrize("raw, expected", [(None, 0), ("", 0), ("12", 12), ("x", 0), ("-3", 0)])
    def test_only_a_plain_number_resumes(self, raw, expected):
        assert resume_point(raw) == expected


class TestSourceEvents:
    @pytest.mark.asyncio
    async def test_a_fresh_page_gets_the_backlog_then_ready_then_live_offers(self, agent):
        agent.offers.side_effect = [
            {"latest_seq": 2, "offers": [offer(1), offer(2)]},
            {"latest_seq": 3, "offers": [offer(3)]},
        ]
        events = await take(source_events(agent, address, None, sleep=AsyncMock()), 5)

        assert events[0] == "retry: 5000\n\n"
        first, second = parsed(events[1]), parsed(events[2])
        assert (first["id"], first["event"]) == ("1", "source")
        assert json.loads(first["data"])["live"] is False and json.loads(second["data"])["live"] is False
        assert events[3] == "event: ready\ndata: {}\n\n"
        assert json.loads(parsed(events[4])["data"]) == {**offer(3), "live": True}
        # The backlog doesn't wait; after it, each call long-polls from the last seq shown
        assert [c.args for c in agent.offers.await_args_list] == [
            (ADDRESS, 0, 0.0), (ADDRESS, 2, LONG_POLL_SECONDS)]

    @pytest.mark.asyncio
    async def test_a_reconnect_resumes_after_its_last_event_id(self, agent):
        agent.offers.return_value = {"latest_seq": 8, "offers": [offer(8)]}
        events = await take(source_events(agent, address, "7", sleep=AsyncMock()), 2)
        assert json.loads(parsed(events[1])["data"])["live"] is True
        agent.offers.assert_awaited_once_with(ADDRESS, 7, LONG_POLL_SECONDS)

    @pytest.mark.asyncio
    async def test_an_agent_restart_starts_counting_again(self, agent):
        agent.offers.side_effect = [
            {"latest_seq": 0, "offers": []},
            {"latest_seq": 1, "offers": [offer(1)]},
        ]
        events = await take(source_events(agent, address, "57", sleep=AsyncMock()), 3)
        assert events[1] == ": ping\n\n"
        assert parsed(events[2])["id"] == "1"
        assert agent.offers.await_args_list[1].args == (ADDRESS, 0, LONG_POLL_SECONDS)

    @pytest.mark.asyncio
    async def test_waits_while_the_appliance_is_off_or_unreachable(self, agent):
        sleep = AsyncMock()
        addresses = iter([None, ADDRESS, ADDRESS])

        async def flaky_address():
            return next(addresses)

        agent.offers.side_effect = [AgentUnavailable("down"), {"latest_seq": 0, "offers": []}]
        events = await take(source_events(agent, flaky_address, None, sleep=sleep), 4)
        assert events[1:] == [": appliance off\n\n", ": waiting for the appliance\n\n", "event: ready\ndata: {}\n\n"]
        assert sleep.await_count == 2
