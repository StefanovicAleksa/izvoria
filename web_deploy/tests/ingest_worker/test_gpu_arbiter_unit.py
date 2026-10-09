import io
import urllib.error
from pathlib import Path

import pytest

from web_deploy.common.signing import verify_service
from web_deploy.ingest_worker.gpu_arbiter import (
    ExclusiveGpu, GpuArbiter, HandoffClient, LlamaServerProcess, arbiter_from_env, llama_server_command,
)
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue
from web_deploy.ingest_worker.supervisor import Supervisor

SECRET = "k" * 40


class Recorder:
    def __init__(self):
        self.calls = []


class FakeApi:
    def __init__(self, log, fail_pauses=0):
        self.log = log
        self.fail_pauses = fail_pauses

    def pause(self):
        if self.fail_pauses:
            self.fail_pauses -= 1
            raise urllib.error.URLError("connection refused")
        self.log.append("api.pause")

    def resume(self):
        self.log.append("api.resume")


class FakeLlama:
    def __init__(self, log):
        self.log = log

    def start(self):
        self.log.append("llama.start")

    def wait_healthy(self):
        self.log.append("llama.healthy")

    def stop(self):
        self.log.append("llama.stop")


class TestExclusiveGpu:
    def test_the_gpu_changes_hands_once_per_queue_in_the_right_order(self):
        log = []
        gpu = ExclusiveGpu(FakeApi(log), FakeLlama(log), sleep=lambda s: None)
        gpu.start()
        gpu.before_job()
        gpu.before_job()  # a second batch in the same queue: already the worker's
        gpu.when_idle()
        gpu.when_idle()  # still idle: already given back

        assert log == ["llama.start", "api.pause", "llama.stop", "llama.start", "llama.healthy", "api.resume"]

    def test_waits_for_an_api_that_is_still_starting(self):
        log = []
        gpu = ExclusiveGpu(FakeApi(log, fail_pauses=2), FakeLlama(log), sleep=lambda s: log.append("wait"))
        gpu.start()
        gpu.before_job()
        assert log == ["llama.start", "wait", "wait", "api.pause", "llama.stop"]

    def test_a_failed_give_back_is_retried_on_the_next_idle_tick(self):
        log = []

        class Flaky(FakeApi):
            tries = 0

            def resume(self):
                Flaky.tries += 1
                if Flaky.tries == 1:
                    raise urllib.error.URLError("down")
                super().resume()

        gpu = ExclusiveGpu(Flaky(log), FakeLlama(log), sleep=lambda s: None)
        gpu.start()
        gpu.before_job()
        gpu.when_idle()
        gpu.when_idle()
        assert log.count("api.resume") == 1


class TestSupervisorWithTheArbiter:
    def test_hands_the_gpu_over_before_a_job_and_back_when_the_queue_is_empty(self):
        log = []
        queue = IngestQueue.connect("sqlite://")
        queue.enqueue("legal-demo", "u", IngestMode.DOCUMENT, "en", "/b", 1)
        queue.enqueue("legal-demo", "u", IngestMode.DOCUMENT, "en", "/c", 1)
        gpu = ExclusiveGpu(FakeApi(log), FakeLlama(log), sleep=lambda s: None)
        gpu.start()

        def spawn(job, env):
            log.append(f"ingest {job.batch_dir}")
            queue.finish(job.id, {})
            return 0

        supervisor = Supervisor(queue, Path("/t"), spawn=spawn, base_env={}, arbiter=gpu)
        while supervisor.run_once():
            pass

        assert log == ["llama.start", "api.pause", "llama.stop", "ingest /b", "ingest /c",
                       "llama.start", "llama.healthy", "api.resume"]


class TestHandoffClient:
    def test_requests_are_service_signed(self):
        seen = []

        def opener(request, timeout):
            seen.append((request.full_url, request.get_method(), dict(request.header_items()), timeout))
            return io.BytesIO(b"{}")

        HandoffClient("http://127.0.0.1:8095/", SECRET, opener=opener).pause()
        url, method, headers, timeout = seen[0]
        assert (url, method, timeout) == ("http://127.0.0.1:8095/internal/gpu/pause", "POST", None)
        verify_service(SECRET, "POST", "/internal/gpu/pause", headers)


class TestLlamaServerProcess:
    def test_start_stop_and_health(self, tmp_path):
        class Proc:
            def __init__(self, *a, **k):
                self.alive = True
                self.terminated = False

            def poll(self):
                return None if self.alive else 0

            def terminate(self):
                self.terminated = True
                self.alive = False

            def wait(self, timeout=None):
                return 0

        started = []

        def popen(command, **kwargs):
            started.append(command)
            return Proc()

        healthy = iter([urllib.error.URLError("loading"), io.BytesIO(b'{"status":"ok"}')])

        def opener(url, timeout):
            value = next(healthy)
            if isinstance(value, Exception):
                raise value
            return value

        server = LlamaServerProcess(["llama-server"], "http://127.0.0.1:8081/health", log_path=tmp_path / "l.log",
                                    popen=popen, opener=opener, sleep=lambda s: None)
        server.start()
        server.start()  # already running
        server.wait_healthy()
        assert started == [["llama-server"]]
        server.stop()
        assert not server.running


class TestFromEnv:
    def test_shared_is_the_default(self):
        assert type(arbiter_from_env({})) is GpuArbiter

    def test_exclusive_owns_llama_server_with_the_128k_context(self):
        arbiter = arbiter_from_env({"GPU_SHARING": "exclusive", "WEB_API_SECRET": SECRET, "ONYX_ROOT": "/r",
                                    "LLAMA_SERVER_BIN": "/bin/llama-server"})
        assert isinstance(arbiter, ExclusiveGpu)
        command = llama_server_command({"ONYX_ROOT": "/r", "LLM_SERVER_URL": "http://127.0.0.1:9000"})
        assert command[command.index("--ctx-size") + 1] == "131072"
        assert command[command.index("--port") + 1] == "9000"
        assert command[command.index("--model") + 1] == "/r/models/Qwen3.5-9B-Q4_K_M.gguf"
        assert "--kv-unified" in command
