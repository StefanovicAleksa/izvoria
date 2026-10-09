from dataclasses import dataclass
from typing import AsyncIterator, Optional
from urllib.parse import quote

import httpx

SECRET_HEADER = "x-agent-secret"


class AgentUnavailable(RuntimeError):
    pass


class AgentRefused(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class AgentHealth:
    postgres: bool
    display: bool
    app_language: Optional[str]
    busy: bool

    @property
    def ready(self) -> bool:
        return self.postgres and self.display


class AgentClient:
    def __init__(self, secret: str, transport: Optional[httpx.AsyncBaseTransport] = None) -> None:
        # Uploads are multi-gigabyte videos over a residential link: no read/write timeout, only a connect one.
        self._client = httpx.AsyncClient(
            headers={SECRET_HEADER: secret},
            timeout=httpx.Timeout(10.0, read=None, write=None),
            transport=transport,
        )

    async def health(self, address: str) -> AgentHealth:
        body = (await self._call("GET", address, "/health", timeout=5.0)).json()
        app = body.get("app") or {}
        return AgentHealth(
            postgres=bool(body.get("postgres")),
            display=bool(body.get("display")),
            app_language=app.get("language"),
            busy=bool(body.get("busy")),
        )

    async def session(self, address: str, language: str) -> None:
        await self._call("POST", address, "/session", json={"language": language}, timeout=60.0)

    async def shutdown(self, address: str) -> None:
        await self._call("POST", address, "/shutdown", timeout=60.0)

    async def upload(self, address: str, language: str, batch: str, filename: str,
                     body: AsyncIterator[bytes]) -> dict:
        path = f"/upload/{language}/{batch}/{quote(filename, safe='')}"
        return (await self._call("PUT", address, path, content=body)).json()

    async def offers(self, address: str, after: int, wait: float) -> dict:
        """Sources the desktop app offered after `after`, long-polling up to `wait` seconds for a new one."""
        return (await self._call("GET", address, "/offers", params={"after": after, "wait": wait},
                                 timeout=wait + 10.0)).json()

    async def open_source(self, address: str, offer_id: str, variant: str, range_header: Optional[str]) -> httpx.Response:
        """
        The agent's response for one offered file, unread: the caller streams it to the browser and closes it. An
        error status is returned too, not raised, so the browser sees the agent's own reason.
        """
        request = self._client.build_request(
            "GET", f"http://{address}/offers/{quote(offer_id, safe='')}/{quote(variant, safe='')}",
            headers={"range": range_header} if range_header else {},
        )
        try:
            return await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            raise AgentUnavailable(f"The appliance box isn't answering yet ({exc.__class__.__name__}).") from exc

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _call(self, method: str, address: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, f"http://{address}{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise AgentUnavailable(f"The appliance box isn't answering yet ({exc.__class__.__name__}).") from exc
        if response.status_code >= 400:
            try:
                message = response.json().get("error", response.text)
            except ValueError:
                message = response.text
            raise AgentRefused(response.status_code, message)
        return response
