import json
from pathlib import Path

import pytest

from web_deploy.common.signing import Principal, verify
from web_deploy.spike.__main__ import apply_to_env_file, parse_grid
from web_deploy.spike.clients import BoxApi, Completion, filler, sse_events
from web_deploy.spike.gpu import VramSampler
from web_deploy.spike.report import render
from web_deploy.spike.stages import MIN_KV_TOKENS, recommend


def _row(parallel, kv, vram, started=True, slot_ctx=None, ok=True):
    return {"parallel": parallel, "kv_tokens": kv, "started": started, "vram_mb": vram,
            "slot_ctx": slot_ctx if slot_ctx is not None else kv, "first_request_ok": ok}


class TestRecommend:
    def test_most_slots_then_the_biggest_cache_within_the_live_budget(self):
        rows = [_row(2, 131072, 7000), _row(4, 131072, 7600), _row(4, 196608, 8400), _row(6, 262144, 11500)]

        assert recommend(rows) == {"LLM_SERVER_PARALLEL": 4, "LLM_SERVER_KV_TOKENS": 196608,
                                   "live_estimate_mb": 11400}

    def test_a_layout_that_cannot_give_one_request_128k_is_never_picked(self):
        rows = [_row(4, 98304, 6000), _row(4, 131072, 8000, slot_ctx=32768)]

        assert recommend(rows) is None

    def test_layouts_that_failed_to_start_or_answer_are_skipped(self):
        rows = [_row(6, 196608, 0, started=False), _row(4, 196608, 8000, ok=False), _row(2, MIN_KV_TOKENS, 7000)]

        assert recommend(rows)["LLM_SERVER_PARALLEL"] == 2


class TestParseGrid:
    def test_pairs(self):
        assert parse_grid("1:131072, 4:196608") == [(1, 131072), (4, 196608)]


class TestApplyToEnvFile:
    def test_rewrites_the_keys_and_keeps_a_copy(self, tmp_path):
        env = tmp_path / "api.env"
        env.write_text("WEB_API_SECRET=s\nLLM_SERVER_PARALLEL=4\nLLM_SERVER_KV_TOKENS=196608\n")

        backup = apply_to_env_file(env, {"LLM_SERVER_PARALLEL": 6, "LLM_SERVER_KV_TOKENS": 262144})

        assert env.read_text() == "WEB_API_SECRET=s\nLLM_SERVER_PARALLEL=6\nLLM_SERVER_KV_TOKENS=262144\n"
        assert backup.read_text().endswith("LLM_SERVER_KV_TOKENS=196608\n")

    def test_the_first_copy_is_never_overwritten(self, tmp_path):
        env = tmp_path / "api.env"
        env.write_text("LLM_SERVER_PARALLEL=4\n")
        apply_to_env_file(env, {"LLM_SERVER_PARALLEL": 6})
        backup = apply_to_env_file(env, {"LLM_SERVER_PARALLEL": 2, "NEW_KEY": 1})

        assert backup.read_text() == "LLM_SERVER_PARALLEL=4\n"
        assert env.read_text() == "LLM_SERVER_PARALLEL=2\nNEW_KEY=1\n"


class TestSseEvents:
    def test_events_comments_and_multiline_data(self):
        stream = [b"retry: 3000\n", b"\n", b": keepalive\n", b"\n", b"id: 1\n", b"event: progress\n",
                  b"data: {\"a\": 1}\n", b"\n", b"event: result\n", b"data: line one\n", b"data: line two\n", b"\n"]

        events = list(sse_events(iter(stream)))

        assert events == [("progress", '{"a": 1}'), ("result", "line one\nline two")]


class TestBoxApi:
    def test_requests_are_signed_as_the_gateway_signs_them(self):
        api = BoxApi("https://127.0.0.1:8095", "s" * 32, Principal(firm="spike-test", user_id="spike", role="admin"))

        headers = api._signed("GET", "/api/runs/abc/events")

        principal = verify("s" * 32, "GET", "/api/runs/abc/events", "", headers)
        assert (principal.firm, principal.user_id, principal.role) == ("spike-test", "spike", "admin")


class TestVramSampler:
    def test_catches_the_peak_between_readings(self):
        readings = iter([(1000, 24576), (1000, 24576), (9000, 24576), (3000, 24576)] + [(3000, 24576)] * 100)
        sampler = VramSampler(interval=0.01, reader=lambda: next(readings))

        with sampler:
            import time
            time.sleep(0.1)

        assert sampler.peak == 9000
        assert sampler.total == 24576


class TestFiller:
    def test_grows_with_the_paragraph_count(self):
        assert len(filler(200)) > 10 * len(filler(10)) > 0


class TestRender:
    def test_a_full_run_reads_as_verdicts_then_details(self):
        results = {
            "gpu_name": "NVIDIA GeForce RTX 3090", "gpu_total_mb": 24576, "started_at": "t0", "finished_at": "t1",
            "sweep": [_row(4, 196608, 8400), {"parallel": 6, "kv_tokens": 262144, "started": False,
                                              "error": "RuntimeError: llama-server exited while loading"}],
            "recommendation": {"LLM_SERVER_PARALLEL": 4, "LLM_SERVER_KV_TOKENS": 196608, "live_estimate_mb": 11400},
            "stress": {"parallel": 4, "kv_tokens": 196608,
                       "json_grammar": {"ok": True, "content": "{}"},
                       "throughput": {"sequential_tokens_per_s": 60, "concurrent_tokens_per_s": 150},
                       "long_beside_short": {"ok": 4, "failed": 0, "long_prompt_tokens": 119000},
                       "cache_exhaustion": {"ok": 0, "failed": 2, "requests": 2, "server_alive_after": True},
                       "cache_exhaustion_via_client": {"ok": 2, "failed": 0, "requests": 2}},
            "shared": {"vram_total_mb": 24576, "video": "/v.mp4",
                       "chat_alone": {"ok": 3, "failed": 0, "notice_period_right": True},
                       "video_job": {"status": "done", "minutes": 9.5,
                                     "report": {"succeeded": 1, "vectors_created": 40, "errors": {}}},
                       "chat_during_ingestion": {"ok": 30, "failed": 0, "peak_vram_mb": 22000,
                                                 "notice_period_right": "10 of 10"}},
        }

        text = render(results)

        assert text.startswith("# M0 spike: NVIDIA GeForce RTX 3090 (24.0 GB)")
        assert text.index("## Verdict") < text.index("## 1. llama-server")
        assert "**Use:** `LLM_SERVER_PARALLEL=4` and `LLM_SERVER_KV_TOKENS=196608`" in text
        assert "FAIL" not in text
        assert "PASS: a video ingests while users chat" in text
        assert "| 6 | 262,144 | **no**" in text

    def test_failures_are_named(self):
        text = render({"sweep": [_row(4, 98304, 7000)], "recommendation": None,
                       "shared": {"error": "the API never became ready"}})

        assert "FAIL: no layout with 128K per request fits the live budget" in text
        assert "FAIL: the shared test did not run to the end" in text


class TestRenderPriority:
    def test_the_scheduler_patch_is_judged_by_the_chats_beside_a_long_request(self):
        results = {"answers_before_long": {
            "off": {"settings": {"ONYX_SHORT_PROMPT_TOKENS": "0"}, "chat_alone_s": 5.2,
                    "chats_beside_long_s": [42.9, 43.4], "long_s": 40.9, "all_ok": True},
            "on": {"settings": {"ONYX_SHORT_PROMPT_TOKENS": "16384"}, "chat_alone_s": 5.2,
                   "chats_beside_long_s": [12.8, 13.4], "long_s": 44.5, "all_ok": True},
        }}

        text = render(results)

        assert "## 2b. Answers before long prompts" in text
        assert "| on | 5.2 s | [12.8, 13.4] s | 44.5 s | True |" in text
        assert ("PASS: chats beside a long request answer in 13.4 s instead of 43.4 s "
                "(long request 40.9 -> 44.5 s)") in text

    def test_no_gain_is_a_failure(self):
        results = {"answers_before_long": {
            "off": {"chats_beside_long_s": [40.0], "long_s": 40.0, "all_ok": True},
            "on": {"chats_beside_long_s": [41.0], "long_s": 40.0, "all_ok": True},
        }}

        assert "FAIL: the scheduler patch did not speed chats up" in render(results)


class TestRenderVideoVerdict:
    def test_a_finished_job_that_indexed_nothing_is_a_failure(self):
        results = {"shared": {"vram_total_mb": 24576, "video": "/v.mp4", "chat_alone": {"ok": 3, "failed": 0},
                              "video_job": {"status": "done", "minutes": 0.3, "report": {
                                  "succeeded": 0, "vectors_created": 0,
                                  "errors": {"vad_error_0": "Could not load libtorchcodec_image.so"}}},
                              "chat_during_ingestion": {"ok": 12, "failed": 0}}}

        text = render(results)

        assert "FAIL: the video was not indexed beside live chat: Could not load libtorchcodec_image.so" in text
        assert "0 file(s) indexed" in text
