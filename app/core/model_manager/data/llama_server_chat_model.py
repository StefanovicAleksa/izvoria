# File: app/core/model_manager/data/llama_server_chat_model.py

import itertools
import json
import logging
import threading
import urllib.error
import urllib.request
from collections import deque
from typing import Any, Dict, List, Optional

from app.core.model_manager.domain.interfaces import IChatModel
from app.core.model_manager.exceptions.server import LlamaServerError

logger = logging.getLogger(__name__)

# llama-cpp-python's create_chat_completion defaults. The server's own defaults differ (temperature 0.8, other
# samplers on), so they are sent explicitly: a request must decode the same on either backend.
_SAMPLING_DEFAULTS: Dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "min_p": 0.05,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

# Chat-template role markers per message, on top of the content's own tokens
_TEMPLATE_TOKENS_PER_MESSAGE = 8

# llama.cpp's own grammars/json.gbnf. `response_format: json_object` is sent as this raw grammar instead: with
# --jinja, the server builds its JSON grammar around the chat template's reasoning block, and for Qwen3.5 that fails
# with "Failed to initialize samplers" (llama.cpp 2e7c58c, measured 2026-10-04). A raw grammar constrains only what
# the model writes after the template's empty <think></think>, which strip_reasoning removes as before.
JSON_GRAMMAR = r"""root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws ( string ":" ws value ("," ws string ":" ws value)* )? "}" ws
array  ::= "[" ws ( value ("," ws value)* )? "]" ws
string ::= "\"" ( [^"\\\x7F\x00-\x1F] | "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"""


class KvTokenBudget:
    """
    First-come, first-served admission against the server's KV cache, counted in tokens. With a unified cache the
    slots share one pool, so four long requests admitted at once could exhaust it mid-decode; here a request
    waits until its prompt plus its longest reply fits. A request larger than the whole pool is clamped to it
    (it then runs alone).
    """

    def __init__(self, capacity_tokens: int):
        if capacity_tokens <= 0:
            raise ValueError("KV token capacity must be positive")
        self._capacity = capacity_tokens
        self._in_use = 0
        self._queue: deque = deque()
        self._tickets = itertools.count()
        self._cond = threading.Condition()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_use(self) -> int:
        with self._cond:
            return self._in_use

    def acquire(self, tokens: int) -> int:
        """Blocks until `tokens` fit, in arrival order. Returns the tokens actually reserved (pass to release)."""
        tokens = max(1, min(tokens, self._capacity))
        with self._cond:
            ticket = next(self._tickets)
            self._queue.append(ticket)
            try:
                self._cond.wait_for(lambda: self._queue[0] == ticket and self._in_use + tokens <= self._capacity)
            except BaseException:
                self._queue.remove(ticket)
                self._cond.notify_all()
                raise
            self._queue.popleft()
            self._in_use += tokens
            # The next in line may fit too
            self._cond.notify_all()
            return tokens

    def release(self, tokens: int) -> None:
        with self._cond:
            self._in_use = max(0, self._in_use - tokens)
            self._cond.notify_all()


# One budget per server, shared by every role (answer, reasoner, report, scrubber) that talks to it
_budgets: Dict[str, KvTokenBudget] = {}
_budgets_lock = threading.Lock()


def _budget_for(base_url: str, capacity_tokens: int) -> KvTokenBudget:
    with _budgets_lock:
        budget = _budgets.get(base_url)
        if budget is None or budget.capacity != capacity_tokens:
            budget = _budgets[base_url] = KvTokenBudget(capacity_tokens)
        return budget


class LlamaServerChatModel(IChatModel):
    """
    A chat GGUF served by a local llama.cpp `llama-server` (OpenAI-compatible HTTP), standing in for an in-process
    llama_cpp.Llama. The server holds the weights once and decodes several requests in parallel slots, which is
    what lets several users be answered at once from one copy of the model.

    The server must run with `--jinja --reasoning-format none`, so `enable_thinking` reaches the GGUF's chat
    template and a <think> block comes back in the content, where strip_reasoning removes it as before.
    Nothing here is loaded or freed: `close()` is a no-op, the server owns the weights.
    """

    def __init__(self, base_url: str, n_ctx: int, timeout_s: float, kv_tokens: int = 0,
                 opener: Optional[Any] = None):
        self._base_url = base_url.rstrip("/")
        self._role_n_ctx = n_ctx
        self._timeout_s = timeout_s
        self._kv_tokens = kv_tokens
        self._urlopen = opener or urllib.request.urlopen
        self._server_n_ctx: Optional[int] = None
        self._budget: Optional[KvTokenBudget] = None
        self._props_lock = threading.Lock()

    # -- HTTP --------------------------------------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: Optional[dict] = None,
                 timeout_s: Optional[float] = None) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self._base_url}{path}", data=data, method=method,
            headers={"Content-Type": "application/json"} if data is not None else {}
        )
        try:
            with self._urlopen(request, timeout=timeout_s or self._timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise LlamaServerError(f"llama-server {method} {path} returned {e.code}: {detail}", e) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LlamaServerError(f"llama-server at {self._base_url} unreachable ({method} {path}): {e}", e) from e
        except ValueError as e:
            raise LlamaServerError(f"llama-server {method} {path} returned invalid JSON", e) from e

    def _ensure_props(self) -> None:
        with self._props_lock:
            if self._server_n_ctx is not None:
                return
            props = self._request("GET", "/props", timeout_s=30)
            server_n_ctx = int(props.get("default_generation_settings", {}).get("n_ctx") or 0)
            if server_n_ctx <= 0:
                raise LlamaServerError(f"llama-server /props reports no context size: {props!r:.300}")
            self._server_n_ctx = server_n_ctx
            # With --kv-unified one slot may use the whole cache, so a slot's n_ctx is the pool size
            self._budget = _budget_for(self._base_url, self._kv_tokens or server_n_ctx)

    def health(self) -> bool:
        try:
            return self._request("GET", "/health", timeout_s=5).get("status") == "ok"
        except LlamaServerError:
            return False

    # -- IChatModel --------------------------------------------------------------------------------------------

    def tokenize(self, text: bytes, add_bos: bool = True, special: bool = False) -> List[int]:
        content = text.decode("utf-8") if isinstance(text, (bytes, bytearray)) else text
        response = self._request("POST", "/tokenize", {
            "content": content, "add_special": add_bos, "parse_special": special
        }, timeout_s=60)
        return list(response.get("tokens", []))

    def n_ctx(self) -> int:
        """The role's own window, never more than one request may use on the server."""
        self._ensure_props()
        return min(self._role_n_ctx, self._server_n_ctx)

    def _estimate_prompt_tokens(self, messages: List[Dict[str, str]]) -> int:
        text = "\n".join(str(m.get("content") or "") for m in messages)
        return len(self.tokenize(text.encode("utf-8"), add_bos=False, special=True)) \
            + _TEMPLATE_TOKENS_PER_MESSAGE * len(messages)

    def chat(self, messages: List[Dict[str, str]], enable_thinking: bool, **kwargs: Any) -> Dict[str, Any]:
        self._ensure_props()
        max_tokens = int(kwargs.get("max_tokens") or 512)
        payload: Dict[str, Any] = dict(_SAMPLING_DEFAULTS)
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        response_format = payload.pop("response_format", None)
        if isinstance(response_format, dict) and response_format.get("type") in ("json_object", "json_schema"):
            payload.setdefault("grammar", JSON_GRAMMAR)
        payload["messages"] = messages
        payload["max_tokens"] = max_tokens
        payload["stream"] = False
        payload["cache_prompt"] = True
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        reserved = self._budget.acquire(self._estimate_prompt_tokens(messages) + max_tokens)
        try:
            response = self._request("POST", "/v1/chat/completions", payload)
        finally:
            self._budget.release(reserved)

        if not response.get("choices"):
            raise LlamaServerError(f"llama-server returned no choices: {response!r:.300}")
        return response

    def close(self) -> None:
        """The server owns the weights; a client has nothing to free."""
        return None
