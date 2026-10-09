"""
Who has the GPU while an ingestion batch runs.

- Hosted (GPU_SHARING=shared, 24 GB): the live models keep their half of the card and ingestion uses the other half,
  so nobody has to move. Supervisord owns llama-server.
- Dev hardware (GPU_SHARING=exclusive, 12 GB): llama-server at 128K (~8.1 GB) plus the pinned embedder and reranker
  leave no room for an ingestion model (Qwen2-VL alone is 7.5 GB). So before a batch the worker asks the API to give
  up the GPU (it stops admitting questions, lets those in flight finish and frees its models), stops llama-server,
  and runs the queue; once the queue is empty it starts llama-server again and gives the GPU back. Here the worker
  owns llama-server. Switching costs a minute, so it happens once per queue, not once per batch.
"""

import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from web_deploy.common.signing import sign_service

logger = logging.getLogger(__name__)

HEALTH_TIMEOUT_SECONDS = 600.0
RETRY_SECONDS = 5.0


class GpuArbiter:
    """GPU_SHARING=shared: nothing to hand over."""

    def start(self) -> None:
        pass

    def before_job(self) -> None:
        pass

    def when_idle(self) -> None:
        pass

    def stop(self) -> None:
        pass


class HandoffClient:
    """Asks the box API to give the GPU up (pause) or take it back (resume)."""

    def __init__(self, api_url: str, secret: str, opener=urllib.request.urlopen) -> None:
        self._api_url = api_url.rstrip("/")
        self._secret = secret
        self._open = opener

    def _post(self, action: str, timeout: Optional[float]) -> None:
        path = f"/internal/gpu/{action}"
        request = urllib.request.Request(f"{self._api_url}{path}", data=b"", method="POST",
                                         headers=sign_service(self._secret, "POST", path))
        with self._open(request, timeout=timeout) as response:
            response.read()

    def pause(self) -> None:
        # No timeout: a report in flight is allowed to finish, and that can take tens of minutes
        self._post("pause", None)

    def resume(self) -> None:
        self._post("resume", HEALTH_TIMEOUT_SECONDS)


def llama_server_command(env: Dict[str, str]) -> List[str]:
    """The same launch as web_deploy/provision/llama_server.sh, from the same settings."""
    root = Path(env.get("ONYX_ROOT") or ".")
    port = urlparse(env.get("LLM_SERVER_URL", "http://127.0.0.1:8081")).port or 8081
    return [
        env.get("LLAMA_SERVER_BIN", "/workspace/llama.cpp/build/bin/llama-server"),
        "--model", env.get("LLAMA_SERVER_MODEL", str(root / "models" / "Qwen3.5-9B-Q4_K_M.gguf")),
        "--host", "127.0.0.1", "--port", str(port),
        "--n-gpu-layers", "99",
        "--ctx-size", env.get("LLM_SERVER_KV_TOKENS") or "131072",
        "--parallel", env.get("LLM_SERVER_PARALLEL") or "3",
        "--kv-unified",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--flash-attn", "on",
        "--jinja", "--reasoning-format", "none",
        "--no-webui",
    ]


class LlamaServerProcess:
    def __init__(self, command: List[str], health_url: str, log_path: Optional[Path] = None,
                 popen=subprocess.Popen, opener=urllib.request.urlopen, sleep=time.sleep,
                 clock=time.monotonic) -> None:
        self._command = command
        self._health_url = health_url
        self._log_path = log_path
        self._popen = popen
        self._open = opener
        self._sleep = sleep
        self._clock = clock
        self._process = None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        if self.running:
            return
        log = open(self._log_path, "ab") if self._log_path else subprocess.DEVNULL
        logger.info("Starting llama-server (the chat models' GPU half).")
        self._process = self._popen(self._command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def wait_healthy(self) -> None:
        deadline = self._clock() + HEALTH_TIMEOUT_SECONDS
        while self._clock() < deadline:
            if not self.running:
                raise RuntimeError("llama-server exited while loading; see its log.")
            try:
                with self._open(self._health_url, timeout=5) as response:
                    if b'"ok"' in response.read():
                        return
            except (urllib.error.URLError, OSError):
                pass
            self._sleep(2.0)
        raise TimeoutError("llama-server did not become healthy within 10 minutes.")

    def stop(self) -> None:
        if not self.running:
            self._process = None
            return
        logger.info("Stopping llama-server to free the GPU for ingestion.")
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None


class ExclusiveGpu(GpuArbiter):
    def __init__(self, api: HandoffClient, llama: LlamaServerProcess,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._api = api
        self._llama = llama
        self._sleep = sleep
        self._live = False

    def start(self) -> None:
        # At startup the API loads its pinned models by itself; llama-server only has to be there
        self._llama.start()
        self._live = True

    def before_job(self) -> None:
        if not self._live:
            return
        while True:
            try:
                self._api.pause()
                break
            except (urllib.error.URLError, OSError) as exc:
                logger.warning(f"The API isn't ready to give up the GPU yet ({exc}); retrying.")
                self._sleep(RETRY_SECONDS)
        self._llama.stop()
        self._live = False
        logger.info("The GPU is the ingestion worker's.")

    def when_idle(self) -> None:
        if self._live:
            return
        try:
            self._llama.start()
            self._llama.wait_healthy()
            self._api.resume()
        except (urllib.error.URLError, OSError, RuntimeError, TimeoutError) as exc:
            logger.warning(f"Could not give the GPU back yet ({exc}); will retry.")
            return
        self._live = True
        logger.info("Queue empty: the GPU is back with the live models.")

    def stop(self) -> None:
        self._llama.stop()


def arbiter_from_env(env: Dict[str, str], log_dir: Optional[Path] = None) -> GpuArbiter:
    if env.get("GPU_SHARING", "shared") != "exclusive":
        return GpuArbiter()
    server_url = env.get("LLM_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")
    return ExclusiveGpu(
        HandoffClient(env.get("WEB_API_URL", "http://127.0.0.1:8095"), env["WEB_API_SECRET"]),
        LlamaServerProcess(llama_server_command(env), f"{server_url}/health",
                           log_path=(log_dir / "llama-server.log") if log_dir else None),
    )
