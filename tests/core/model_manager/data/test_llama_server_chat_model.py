# Path: tests/core/model_manager/data/test_llama_server_chat_model.py

import io
import json
import threading
import time
import urllib.error
from unittest.mock import MagicMock

import pytest

from app.core.model_manager.data import llama_server_chat_model as module
from app.core.model_manager.data.llama_server_chat_model import KvTokenBudget, LlamaServerChatModel
from app.core.model_manager.exceptions.server import LlamaServerError


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """Answers the three endpoints the client calls and records every request."""

    def __init__(self, n_ctx=131072, completion=None):
        self.n_ctx = n_ctx
        self.completion = completion or {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]}
        self.requests = []

    def __call__(self, request, timeout):
        body = json.loads(request.data.decode()) if request.data else None
        path = request.full_url.split("8081", 1)[1]
        self.requests.append((request.get_method(), path, body, timeout))
        if path == "/props":
            payload = {"default_generation_settings": {"n_ctx": self.n_ctx}, "total_slots": 4}
        elif path == "/tokenize":
            payload = {"tokens": list(range(len(body["content"].split())))}
        elif path == "/health":
            payload = {"status": "ok"}
        else:
            payload = self.completion
        return _Response(json.dumps(payload).encode())

    def bodies(self, path):
        return [body for _, p, body, _ in self.requests if p == path]


@pytest.fixture(autouse=True)
def fresh_budgets():
    module._budgets.clear()
    yield
    module._budgets.clear()


def make_client(server, n_ctx=131072, kv_tokens=0):
    return LlamaServerChatModel("http://127.0.0.1:8081/", n_ctx, timeout_s=900, kv_tokens=kv_tokens, opener=server)


class TestLlamaServerChatModel:
    def test_chat_sends_thinking_switch_sampling_defaults_and_caller_arguments(self):
        server = FakeServer()
        client = make_client(server)
        messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "what happened"}]

        result = client.chat(messages, enable_thinking=False, temperature=0.1, max_tokens=1024,
                             response_format={"type": "json_object"})

        assert result == server.completion
        assert server.bodies("/v1/chat/completions") == [{
            "temperature": 0.1, "top_p": 0.95, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "messages": messages, "max_tokens": 1024, "stream": False, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            # json_object becomes llama.cpp's raw JSON grammar (the server's own json_object fails on Qwen3.5)
            "grammar": module.JSON_GRAMMAR,
        }]
        method, _, _, timeout = server.requests[-1]
        assert (method, timeout) == ("POST", 900)

    def test_tokenize_matches_llama_cpp_python_arguments(self):
        server = FakeServer()
        tokens = make_client(server).tokenize("one two three".encode(), add_bos=False, special=True)

        assert tokens == [0, 1, 2]
        assert server.bodies("/tokenize") == [{"content": "one two three", "add_special": False, "parse_special": True}]

    def test_n_ctx_is_the_role_window_capped_by_the_server(self):
        assert make_client(FakeServer(n_ctx=196608), n_ctx=131072).n_ctx() == 131072
        assert make_client(FakeServer(n_ctx=65536), n_ctx=131072).n_ctx() == 65536

    def test_props_are_read_once(self):
        server = FakeServer()
        client = make_client(server)
        client.n_ctx()
        client.n_ctx()

        assert [p for _, p, _, _ in server.requests].count("/props") == 1

    def test_kv_budget_comes_from_the_setting_when_given(self):
        server = FakeServer(n_ctx=131072)
        client = make_client(server, kv_tokens=200000)
        client.n_ctx()

        assert module._budgets["http://127.0.0.1:8081"].capacity == 200000

    def test_kv_reservation_is_returned_after_the_request(self):
        server = FakeServer()
        client = make_client(server)
        client.chat([{"role": "user", "content": "a b c"}], enable_thinking=False, max_tokens=10)

        assert module._budgets["http://127.0.0.1:8081"].in_use == 0

    def test_a_request_the_shared_cache_cannot_hold_yet_waits_instead_of_reaching_the_server(self):
        # llama-server fails requests that overflow its shared cache ("Context size has been exceeded") instead of
        # queueing them (measured, docs/web_deploy.md "M0 spike"): the budget is the only thing that makes them wait
        entered, release_first = threading.Event(), threading.Event()
        completions = []

        class SlowServer(FakeServer):
            def __call__(self, request, timeout):
                if request.full_url.endswith("/v1/chat/completions"):
                    completions.append(time.monotonic())
                    if len(completions) == 1:
                        entered.set()
                        release_first.wait(5)
                return super().__call__(request, timeout)

        server = SlowServer(n_ctx=1000)
        client = make_client(server, kv_tokens=1000)
        first = threading.Thread(target=client.chat, args=([{"role": "user", "content": "a " * 400}],),
                                 kwargs={"enable_thinking": False, "max_tokens": 200})
        second = threading.Thread(target=client.chat, args=([{"role": "user", "content": "b " * 400}],),
                                  kwargs={"enable_thinking": False, "max_tokens": 200})
        first.start()
        assert entered.wait(5)
        second.start()
        time.sleep(0.2)

        assert len(completions) == 1  # 600 + 600 tokens > 1000: the second is held back, not sent

        release_first.set()
        first.join(5)
        second.join(5)
        assert len(completions) == 2
        assert module._budgets["http://127.0.0.1:8081"].in_use == 0

    def test_http_error_becomes_a_llama_server_error(self):
        def failing(request, timeout):
            if request.full_url.endswith("/props"):
                return _Response(json.dumps({"default_generation_settings": {"n_ctx": 4096}}).encode())
            raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, io.BytesIO(b"slot unavailable"))

        client = make_client(failing)
        with pytest.raises(LlamaServerError, match="500: slot unavailable"):
            client.chat([{"role": "user", "content": "x"}], enable_thinking=False, max_tokens=5)
        assert module._budgets["http://127.0.0.1:8081"].in_use == 0

    def test_unreachable_server_is_a_llama_server_error(self):
        opener = MagicMock(side_effect=urllib.error.URLError("connection refused"))
        with pytest.raises(LlamaServerError, match="unreachable"):
            make_client(opener).n_ctx()

    def test_reply_without_choices_is_an_error(self):
        with pytest.raises(LlamaServerError, match="no choices"):
            make_client(FakeServer(completion={"error": "x"})).chat(
                [{"role": "user", "content": "x"}], enable_thinking=False, max_tokens=5)

    def test_close_frees_nothing(self):
        server = FakeServer()
        make_client(server).close()
        assert server.requests == []

    def test_health(self):
        assert make_client(FakeServer()).health() is True
        assert make_client(MagicMock(side_effect=urllib.error.URLError("down"))).health() is False


class TestKvTokenBudget:
    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            KvTokenBudget(0)

    def test_request_larger_than_the_pool_is_clamped(self):
        budget = KvTokenBudget(100)
        assert budget.acquire(500) == 100
        assert budget.in_use == 100

    def test_waits_until_room_is_released(self):
        budget = KvTokenBudget(100)
        first = budget.acquire(80)
        admitted = threading.Event()

        def second():
            budget.acquire(50)
            admitted.set()

        thread = threading.Thread(target=second)
        thread.start()
        assert not admitted.wait(0.1)
        budget.release(first)
        assert admitted.wait(2)
        thread.join()
        assert budget.in_use == 50

    def test_admission_is_first_come_first_served(self):
        budget = KvTokenBudget(100)
        held = budget.acquire(90)
        order = []
        big_queued = threading.Event()

        def big():
            big_queued.set()
            budget.acquire(60)
            order.append("big")

        def small():
            budget.acquire(5)
            order.append("small")

        t_big = threading.Thread(target=big)
        t_big.start()
        big_queued.wait(1)
        time.sleep(0.05)
        t_small = threading.Thread(target=small)
        t_small.start()
        time.sleep(0.1)
        # The small request would fit now, but the big one arrived first
        assert order == []
        budget.release(held)
        t_big.join(2)
        t_small.join(2)
        assert order == ["big", "small"]


class TestJsonGrammar:
    def test_a_caller_grammar_is_kept(self):
        server = FakeServer()
        make_client(server).chat([{"role": "user", "content": "x"}], enable_thinking=False, max_tokens=5,
                                 response_format={"type": "json_object"}, grammar="root ::= \"a\"")
        assert server.bodies("/v1/chat/completions")[0]["grammar"] == 'root ::= "a"'

    def test_the_grammar_is_llama_cpps_json_grammar(self):
        assert module.JSON_GRAMMAR.startswith("root   ::= object")
        assert "string ::=" in module.JSON_GRAMMAR and "number ::=" in module.JSON_GRAMMAR
