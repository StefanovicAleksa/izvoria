"""
The two HTTP clients the spike drives, on the standard library only: llama-server's own API, and the box's web API
signed exactly as the gateway signs it (so the test needs no gateway, no VPS and no accounts).
"""

import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple
from urllib.parse import urlsplit

from web_deploy.common.signing import Principal, sign


@dataclass
class Completion:
    ok: bool
    seconds: float
    status: int = 0
    content: str = ""
    completion_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""


def _request(url: str, payload: Optional[dict] = None, headers: Optional[dict] = None, timeout: float = 1800,
             context: Optional[ssl.SSLContext] = None, method: Optional[str] = None):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"),
                                     headers={"Content-Type": "application/json", **(headers or {})})
    return urllib.request.urlopen(request, timeout=timeout, context=context)


class LlamaServer:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")

    def props(self) -> dict:
        with _request(f"{self.base}/props", timeout=30) as response:
            return json.loads(response.read())

    def count_tokens(self, text: str) -> int:
        with _request(f"{self.base}/tokenize", {"content": text}, timeout=120) as response:
            return len(json.loads(response.read())["tokens"])

    def chat(self, prompt: str, max_tokens: int = 128, grammar: Optional[str] = None,
             timeout: float = 1800) -> Completion:
        payload = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0.2,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if grammar:
            payload["grammar"] = grammar
        started = time.monotonic()
        try:
            with _request(f"{self.base}/v1/chat/completions", payload, timeout=timeout) as response:
                body = json.loads(response.read())
            usage = body.get("usage") or {}
            return Completion(ok=True, seconds=time.monotonic() - started, status=200,
                              content=body["choices"][0]["message"].get("content") or "",
                              completion_tokens=int(usage.get("completion_tokens", 0)),
                              prompt_tokens=int(usage.get("prompt_tokens", 0)))
        except urllib.error.HTTPError as exc:
            return Completion(ok=False, seconds=time.monotonic() - started, status=exc.code,
                              error=exc.read().decode(errors="replace")[:500])
        except Exception as exc:
            return Completion(ok=False, seconds=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}")


def sse_events(lines: Iterator[bytes]) -> Iterator[Tuple[str, str]]:
    """(event, data) pairs from a text/event-stream."""
    event, data = "message", []
    for raw in lines:
        line = raw.decode(errors="replace").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, "\n".join(data)


class BoxApi:
    """The GPU box's web API (https, self-signed), signed for one test user of one test firm."""

    def __init__(self, base_url: str, secret: str, principal: Principal) -> None:
        self.base = base_url.rstrip("/")
        self._secret = secret
        self._principal = principal
        self._context = ssl.create_default_context()
        self._context.check_hostname = False
        self._context.verify_mode = ssl.CERT_NONE  # the box's own certificate, on the box itself

    def _signed(self, method: str, path: str) -> dict:
        parts = urlsplit(path)
        return sign(self._secret, method, parts.path, parts.query, self._principal)

    def health(self) -> dict:
        with _request(f"{self.base}/health", timeout=10, context=self._context) as response:
            return json.loads(response.read())

    def ask(self, prompt: str, language: str = "en", timeout: float = 900) -> Completion:
        """One chat turn, start to answer, through the same run + event stream the web app uses."""
        started = time.monotonic()
        try:
            with _request(f"{self.base}/api/chat", {"prompt": prompt, "language": language},
                          self._signed("POST", "/api/chat"), timeout=60, context=self._context) as response:
                run = json.loads(response.read())
            path = run["events"]
            with _request(f"{self.base}{path}", headers=self._signed("GET", path), timeout=timeout,
                          context=self._context) as stream:
                for event, data in sse_events(stream):
                    if event == "result":
                        message = json.loads(data).get("message") or {}
                        return Completion(ok=True, seconds=time.monotonic() - started, status=200,
                                          content=str(message.get("text") or message.get("answer") or data))
                    if event == "error":
                        return Completion(ok=False, seconds=time.monotonic() - started, error=data[:500])
            return Completion(ok=False, seconds=time.monotonic() - started, error="the event stream ended early")
        except urllib.error.HTTPError as exc:
            return Completion(ok=False, seconds=time.monotonic() - started, status=exc.code,
                              error=exc.read().decode(errors="replace")[:500])
        except Exception as exc:
            return Completion(ok=False, seconds=time.monotonic() - started, error=f"{type(exc).__name__}: {exc}")


def filler(paragraphs: int) -> str:
    """Varied text of a predictable size, for prompts near a slot's whole context."""
    out: List[str] = []
    for i in range(paragraphs):
        out.append(f"Section {i}. The committee reviewed item {i * 7 % 101} of the ledger, noting that the "
                   f"quarterly figure moved from {i * 13 % 997} to {i * 17 % 991} units after the audit of "
                   f"warehouse {i % 23}, and asked the treasurer to confirm the variance by the next meeting.")
    return "\n".join(out)
