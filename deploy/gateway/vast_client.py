"""
The three Vast.ai calls the gateway makes, shaped after vast-ai/vast-cli (start_instance / stop_instance /
show__instance): PUT /api/v0/instances/{id}/ {"state": ...} and GET /api/v0/instances/{id}/?owner=me.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

import httpx


class VastError(RuntimeError):
    pass


@dataclass(frozen=True)
class InstanceInfo:
    actual_status: str  # running | loading | scheduling | created | exited | stopped | offline
    intended_status: str
    public_ip: Optional[str] = None
    ports: Dict[int, int] = field(default_factory=dict)  # container port -> host port
    status_msg: str = ""

    @property
    def running(self) -> bool:
        return self.actual_status == "running"

    def address(self, container_port: int) -> Optional[str]:
        host_port = self.ports.get(container_port)
        if not self.public_ip or host_port is None:
            return None
        return f"{self.public_ip}:{host_port}"


class IVastClient(ABC):
    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def info(self) -> InstanceInfo: ...


class VastClient(IVastClient):
    def __init__(self, api_key: str, instance_id: int, base_url: str = "https://console.vast.ai",
                 transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        self._url = f"{base_url.rstrip('/')}/api/v0/instances/{instance_id}/"
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=30.0, transport=transport,
        )

    async def start(self) -> None:
        await self._set_state("running")

    async def stop(self) -> None:
        await self._set_state("stopped")

    async def info(self) -> InstanceInfo:
        response = await self._request("GET", self._url, params={"owner": "me"})
        row = response.json().get("instances")
        if not row:
            raise VastError("Vast reports no such instance; it may have been destroyed.")
        return InstanceInfo(
            actual_status=str(row.get("actual_status") or "unknown"),
            intended_status=str(row.get("intended_status") or "unknown"),
            public_ip=row.get("public_ipaddr") or None,
            ports=_parse_ports(row.get("ports") or {}),
            status_msg=str(row.get("status_msg") or ""),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _set_state(self, state: str) -> None:
        response = await self._request("PUT", self._url, json={"state": state})
        body = response.json()
        if not body.get("success", False):
            raise VastError(body.get("msg") or f"Vast refused to set the instance {state}.")

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VastError(f"Vast API call failed: {exc}") from exc
        return response


def _parse_ports(raw: dict) -> Dict[int, int]:
    """Docker-style {"8090/tcp": [{"HostIp": "0.0.0.0", "HostPort": "41234"}]} -> {8090: 41234}."""
    ports = {}
    for key, bindings in raw.items():
        container, _, proto = key.partition("/")
        if proto not in ("", "tcp") or not bindings:
            continue
        try:
            ports[int(container)] = int(bindings[0]["HostPort"])
        except (ValueError, KeyError, TypeError, IndexError):
            continue
    return ports


class FakeVastClient(IVastClient):
    """For a local end-to-end run: the "instance" is this host, its ports are unmapped, start/stop only flip state."""

    def __init__(self, host: str, container_ports: tuple) -> None:
        self._host = host
        self._ports = {port: port for port in container_ports}
        self._status = "stopped"
        self.stops = 0

    async def start(self) -> None:
        self._status = "running"

    async def stop(self) -> None:
        self._status = "stopped"
        self.stops += 1

    async def info(self) -> InstanceInfo:
        return InstanceInfo(
            actual_status=self._status, intended_status=self._status,
            public_ip=self._host, ports=dict(self._ports),
        )
