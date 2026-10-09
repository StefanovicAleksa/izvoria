"""
The browser's Sources panel feed: the agent's offers, long-polled and relayed as Server-Sent Events.

A reconnect resumes from the browser's Last-Event-ID (the offer's seq), so nothing is shown twice or missed. A
fresh page first gets what's already on offer, marked `live: false` so it is listed but doesn't jump into view,
then an `event: ready`, then each new click as it happens.
"""

import asyncio
import json
import logging
import re
from typing import AsyncIterator, Awaitable, Callable, Optional

from deploy.gateway.agent_client import AgentClient, AgentRefused, AgentUnavailable

logger = logging.getLogger(__name__)

OFFER_ID = re.compile(r"^[A-Za-z0-9_-]{22}$")
VARIANTS = frozenset({"media", "excerpt", "text", "original"})
# Under Caddy's and most browsers' idle cut-offs; a quiet poll ends in a comment line that keeps the stream open.
LONG_POLL_SECONDS = 20.0
RETRY_SECONDS = 5.0


def resume_point(last_event_id: Optional[str]) -> int:
    return int(last_event_id) if last_event_id and last_event_id.isdigit() else 0


async def source_events(
    agent: AgentClient,
    agent_address: Callable[[], Awaitable[Optional[str]]],
    last_event_id: Optional[str],
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> AsyncIterator[str]:
    after = resume_point(last_event_id)
    # A reconnecting browser already listed everything up to its Last-Event-ID; whatever follows is new.
    backlog = last_event_id is None
    yield f"retry: {int(RETRY_SECONDS * 1000)}\n\n"
    while True:
        address = await agent_address()
        if not address:
            yield ": appliance off\n\n"
            await sleep(RETRY_SECONDS)
            continue
        try:
            body = await agent.offers(address, after, 0.0 if backlog else LONG_POLL_SECONDS)
        except (AgentUnavailable, AgentRefused) as exc:
            logger.debug(f"Source feed waiting for the agent: {exc}")
            yield ": waiting for the appliance\n\n"
            await sleep(RETRY_SECONDS)
            continue

        if int(body.get("latest_seq", 0)) < after:
            after = 0  # the agent restarted and numbers from 1 again; what it holds now is all new
        offers = body.get("offers") or []
        for offer in offers:
            offer["live"] = not backlog
            after = max(after, int(offer["seq"]))
            yield f"id: {offer['seq']}\nevent: source\ndata: {json.dumps(offer)}\n\n"
        if backlog:
            backlog = False
            yield "event: ready\ndata: {}\n\n"
        elif not offers:
            yield ": ping\n\n"
