"""
The hosted gateway as the firm's uploader talks to it: sign in as a person of the firm, then the same calls the web
app's Ingest page makes (a batch, resumable pieces, the batch queued). Standard library only.
"""

import http.cookiejar
import json
import random
import time
import urllib.error
import urllib.request
from typing import Callable, Optional

MAX_ATTEMPTS = 10


class GatewayError(RuntimeError):
    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        """The network, or the server busy or briefly away; a refusal (4xx) is final."""
        return self.status == 0 or self.status in (408, 429) or self.status >= 500


class Gateway:
    def __init__(self, base_url: str, timeout: float = 300.0,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._sleep = sleep
        self._cookies = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._cookies))

    # ---------- signing in: the firm, then the person ----------

    def sign_in(self, firm: str, firm_password: str, email: str, password: str) -> None:
        self.call("POST", "/auth/firm", {"firm": firm, "password": firm_password, "remember": False}, retry=False)
        self.call("POST", "/auth/login", {"email": email, "password": password}, retry=False)

    # ---------- a batch ----------

    def new_batch(self) -> str:
        return self.call("POST", "/api/uploads", {})["batch"]

    def status(self, batch: str, name: str, size: int, chunk: int) -> dict:
        return self.call("POST", f"/api/uploads/{_q(batch)}/{_q(name)}/chunks", {"size": size, "chunk": chunk})

    def put_chunk(self, batch: str, name: str, index: int, size: int, chunk: int, data: bytes) -> dict:
        return self.call("PUT", f"/api/uploads/{_q(batch)}/{_q(name)}/chunks/{index}?size={size}&chunk={chunk}",
                         body=data)

    def record(self, batch: str, files: dict) -> dict:
        return self.call("POST", f"/api/uploads/{_q(batch)}/record", {"files": files})

    def enqueue(self, batch: str, mode: str, language: str) -> dict:
        return self.call("POST", "/api/ingest", {"batch": batch, "mode": mode, "language": language})

    # ---------- plumbing ----------

    def call(self, method: str, path: str, payload: Optional[dict] = None, body: Optional[bytes] = None,
             retry: bool = True) -> dict:
        data = json.dumps(payload).encode() if payload is not None else body
        headers = {"Origin": self.base_url, "Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
            try:
                with self._opener.open(request, timeout=self._timeout) as response:
                    text = response.read().decode() or "{}"
                    return json.loads(text)
            except urllib.error.HTTPError as exc:
                error = GatewayError(_message(exc), exc.code)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                error = GatewayError(f"The connection failed: {getattr(exc, 'reason', exc)}")
            if not (retry and error.retryable) or attempt == MAX_ATTEMPTS:
                raise error
            self._sleep(min(60.0, 2 ** (attempt - 1)) * (0.8 + 0.4 * random.random()))
        raise AssertionError("unreachable")


def _message(exc: urllib.error.HTTPError) -> str:
    try:
        return json.loads(exc.read().decode()).get("error") or exc.reason
    except Exception:
        return str(exc.reason)


def _q(value: str) -> str:
    return urllib.request.quote(value, safe="")
