"""
The three stages of the M0 spike (docs/web_deploy.md). Each returns a plain dict for the report.

- sweep:  llama-server's VRAM at each slot count and shared KV size, and what /props says a slot may use.
- stress: at the chosen size, the JSON grammar the client sends, parallel throughput, a ~120K-token request beside
          short ones, and more long requests than the shared cache holds.
- shared: the production layout itself. The box running as provisioned (llama-server, API, ingestion worker),
          a video ingested on the ingestion half while users chat on the live half, VRAM watched throughout.
"""

import concurrent.futures
import json
import logging
import shutil
import time
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence

from web_deploy.spike.clients import BoxApi, Completion, LlamaServer, filler
from web_deploy.spike.gpu import VramSampler

logger = logging.getLogger(__name__)

# What each live request must keep: one report section or long conversation uses a whole 128K context
MIN_KV_TOKENS = 131072
# The plan's live budget on a 24 GB card (docs/web_deploy.md, "VRAM"), and what BGE-M3 + the reranker take under load
LIVE_BUDGET_MB = 13_500
PINNED_MODELS_MB = 3_000
LONG_PROMPT_TOKENS = 120_000

SHORT_PROMPTS = [
    "Explain in about 150 words why a notice period matters in an employment contract.",
    "In about 150 words, describe how a hospital triage nurse prioritises patients.",
    "Summarise in about 150 words what an audit trail is and why regulators ask for one.",
    "In about 150 words, explain the difference between a lease and a licence to occupy.",
]


def _llama_process(env: Dict[str, str], parallel: int, kv_tokens: int, log_path: Path):
    import subprocess
    from web_deploy.ingest_worker.gpu_arbiter import LlamaServerProcess, llama_server_command
    run_env = dict(env, LLM_SERVER_PARALLEL=str(parallel), LLM_SERVER_KV_TOKENS=str(kv_tokens))
    url = run_env.get("LLM_SERVER_URL", "http://127.0.0.1:8081").rstrip("/")
    # The server gets exactly this environment (the patched scheduler reads ONYX_* from it)
    return LlamaServerProcess(llama_server_command(run_env), f"{url}/health", log_path=log_path,
                              popen=lambda command, **kw: subprocess.Popen(command, env=run_env, **kw)), url


def _log_tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


# ---------- stage 1 ----------

def sweep(env: Dict[str, str], sampler: VramSampler, grid: Sequence[tuple], workdir: Path) -> List[dict]:
    rows = []
    for parallel, kv_tokens in grid:
        log = workdir / f"llama-server-p{parallel}-kv{kv_tokens}.log"
        log.unlink(missing_ok=True)
        process, url = _llama_process(env, parallel, kv_tokens, log)
        before = sampler.now()
        row = {"parallel": parallel, "kv_tokens": kv_tokens}
        logger.info(f"sweep: --parallel {parallel}, {kv_tokens} KV tokens")
        try:
            process.start()
            process.wait_healthy()
            server = LlamaServer(url)
            props = server.props()
            settings = props.get("default_generation_settings") or {}
            warm = server.chat(SHORT_PROMPTS[0], max_tokens=32)  # compute buffers are allocated on first use
            row.update(started=True, slots=props.get("total_slots"), slot_ctx=settings.get("n_ctx"),
                       vram_mb=sampler.now() - before, first_request_ok=warm.ok)
        except Exception as exc:
            row.update(started=False, error=f"{type(exc).__name__}: {exc}", log_tail=_log_tail(log))
        finally:
            process.stop()
            time.sleep(3)
        rows.append(row)
        logger.info(f"sweep: {json.dumps(row)}")
    return rows


def recommend(rows: List[dict], budget_mb: int = LIVE_BUDGET_MB, pinned_mb: int = PINNED_MODELS_MB) -> Optional[dict]:
    """
    The settings to put in api.env: the most slots, then the biggest shared cache, among sizes that started, let one
    request use a whole 128K context, and leave the live side (llama-server + the pinned models) within its budget.
    """
    fitting = [r for r in rows if r.get("started") and r.get("first_request_ok")
               and r["kv_tokens"] >= MIN_KV_TOKENS and (r.get("slot_ctx") or 0) >= MIN_KV_TOKENS
               and r["vram_mb"] + pinned_mb <= budget_mb]
    if not fitting:
        return None
    best = max(fitting, key=lambda r: (r["parallel"], r["kv_tokens"]))
    return {"LLM_SERVER_PARALLEL": best["parallel"], "LLM_SERVER_KV_TOKENS": best["kv_tokens"],
            "live_estimate_mb": best["vram_mb"] + pinned_mb}


# ---------- stage 2 ----------

def _summary(results: List[Completion]) -> dict:
    return {"ok": sum(r.ok for r in results), "failed": sum(not r.ok for r in results),
            "seconds": [round(r.seconds, 1) for r in results],
            "errors": sorted({(r.error or str(r.status))[:200] for r in results if not r.ok})}


def _at_once(calls: List[Callable[[], Completion]]) -> List[Completion]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return [f.result() for f in [pool.submit(c) for c in calls]]


def chat_prompt(i: int) -> str:
    """A chat turn as the app sends one: a question over ~4K tokens of retrieved excerpts."""
    return (f"Using only the excerpts below, answer in about 120 words: what did the committee ask in item {i}?\n"
            + filler(70).replace("Section", f"Excerpt-{i}-"))


def long_prompt(server: LlamaServer, tokens: int) -> str:
    """Filler text measured with the server's own tokenizer to land just under `tokens`."""
    sample = filler(50)
    per_paragraph = server.count_tokens(sample) / 50
    paragraphs = int(tokens / per_paragraph * 0.97)
    return "Read the minutes below, then answer in one sentence: which warehouse is mentioned last?\n\n" + \
        filler(paragraphs)


def stress(env: Dict[str, str], sampler: VramSampler, parallel: int, kv_tokens: int, workdir: Path,
           long_tokens: int = LONG_PROMPT_TOKENS) -> dict:
    from app.core.model_manager.data.llama_server_chat_model import LlamaServerChatModel
    from app.core.model_manager.utils.llama_cpp_chat import strip_reasoning

    log = workdir / "llama-server-stress.log"
    process, url = _llama_process(env, parallel, kv_tokens, log)
    report: Dict[str, object] = {"parallel": parallel, "kv_tokens": kv_tokens}
    process.start()
    try:
        process.wait_healthy()
        server = LlamaServer(url)

        # 1. Structured output exactly as the app asks for it: response_format json_object, which the client turns
        # into the raw JSON grammar (this server rejects response_format for Qwen3.5), then strip_reasoning
        client = LlamaServerChatModel(url, n_ctx=kv_tokens, timeout_s=1800, kv_tokens=kv_tokens)
        content, error = "", ""
        try:
            raw = client.chat([{"role": "user", "content": 'Return a JSON object with keys "verdict" (true or '
                                'false) and "reason" (a short string): is Paris in France?'}],
                              enable_thinking=False, max_tokens=64, response_format={"type": "json_object"})
            content = strip_reasoning(raw["choices"][0]["message"].get("content") or "")
            parsed = json.loads(content)
        except Exception as exc:
            parsed, error = None, f"{type(exc).__name__}: {exc}"
        report["json_grammar"] = {"ok": isinstance(parsed, dict), "content": content[:200], "error": error}

        # 2. Throughput: the same four answers one after another, then at once
        started = time.monotonic()
        sequential = [server.chat(p, max_tokens=256) for p in SHORT_PROMPTS]
        sequential_s = time.monotonic() - started
        started = time.monotonic()
        concurrent_ = _at_once([lambda p=p: server.chat(p, max_tokens=256) for p in SHORT_PROMPTS])
        concurrent_s = time.monotonic() - started
        report["throughput"] = {
            "sequential_tokens_per_s": round(sum(r.completion_tokens for r in sequential) / sequential_s, 1),
            "concurrent_tokens_per_s": round(sum(r.completion_tokens for r in concurrent_) / concurrent_s, 1),
            "sequential_s": round(sequential_s, 1), "concurrent_s": round(concurrent_s, 1),
            "failed": sum(not r.ok for r in sequential + concurrent_),
        }

        # 3. One ~120K-token request (a report section, a long conversation) beside three short chats
        prompt = long_prompt(server, long_tokens)
        with sampler:
            results = _at_once([lambda: server.chat(prompt, max_tokens=48)] +
                               [lambda p=p: server.chat(p, max_tokens=200) for p in SHORT_PROMPTS[:3]])
        report["long_beside_short"] = {**_summary(results), "long_prompt_tokens": results[0].prompt_tokens,
                                       "short_median_s": round(median(r.seconds for r in results[1:]), 1),
                                       "peak_vram_mb": sampler.peak}

        # 4. More long requests than the shared cache holds: refused, queued, or broken?
        count = kv_tokens // long_tokens + 1
        # Each a different text: identical prompts share their cached prefix and never fill the cache
        prompts = [f"Copy {i}. " + prompt.replace("Section", f"S{i}-Section") for i in range(count)]
        with sampler:
            results = _at_once([lambda p=p: server.chat(p, max_tokens=24) for p in prompts])
        healthy = True
        try:
            server.props()
        except Exception:
            healthy = False
        report["cache_exhaustion"] = {**_summary(results), "requests": count,
                                      "tokens_requested": count * long_tokens,
                                      "server_alive_after": healthy and process.running,
                                      "peak_vram_mb": sampler.peak}

        # 5. The same burst through the app's own client: KvTokenBudget should make the excess wait, not fail
        def through_client(text: str) -> Completion:
            started = time.monotonic()
            try:
                client.chat([{"role": "user", "content": text}], enable_thinking=False, max_tokens=24)
                return Completion(ok=True, seconds=time.monotonic() - started)
            except Exception as exc:
                return Completion(ok=False, seconds=time.monotonic() - started,
                                  error=f"{type(exc).__name__}: {exc}"[:300])

        results = _at_once([lambda p=p: through_client("Again. " + p) for p in prompts])
        report["cache_exhaustion_via_client"] = {**_summary(results), "requests": count}
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["log_tail"] = _log_tail(log)
    finally:
        process.stop()
    return report


def answers_before_long(env: Dict[str, str], parallel: int, kv_tokens: int, workdir: Path,
                        long_tokens: int = LONG_PROMPT_TOKENS) -> dict:
    """
    Three chats (each ~4K tokens of evidence) asked 2 s after a long request starts, on a fresh server with the
    scheduler patch's rule off and then on: how long the chats take, and what the rule costs the long request.
    """
    settings = {"off": {"ONYX_SHORT_PROMPT_TOKENS": "0"},
                "on": {"ONYX_SHORT_PROMPT_TOKENS": env.get("ONYX_SHORT_PROMPT_TOKENS") or "16384",
                       "ONYX_LONG_PROMPT_TRICKLE": env.get("ONYX_LONG_PROMPT_TRICKLE") or "0"}}
    report: Dict[str, object] = {}
    for name, overrides in settings.items():
        log = workdir / f"llama-server-priority-{name}.log"
        process, url = _llama_process(dict(env, **overrides), parallel, kv_tokens, log)
        process.start()
        try:
            process.wait_healthy()
            server = LlamaServer(url)
            server.chat("warm up", max_tokens=8)
            alone = server.chat(chat_prompt(99), max_tokens=150)
            prompt = long_prompt(server, long_tokens)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                long_future = pool.submit(server.chat, prompt, 32)
                time.sleep(2.0)
                chats = [f.result() for f in [pool.submit(server.chat, chat_prompt(i), 150) for i in range(3)]]
                long_result = long_future.result()
            report[name] = {"settings": overrides, "chat_alone_s": round(alone.seconds, 1),
                            "chats_beside_long_s": [round(r.seconds, 1) for r in chats],
                            "long_s": round(long_result.seconds, 1),
                            "all_ok": all(r.ok for r in chats + [long_result])}
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}", "log_tail": _log_tail(log)}
        finally:
            process.stop()
            time.sleep(3)
    return report


# ---------- stage 3 ----------

FACT_FILE = "spike_facts.txt"
ROUND_SECONDS = 5.0
FACTS = """Izvoria spike test contract between Arden Logistics and Bellweather Freight.

Clause 4. Notice period. Either party may end this agreement by giving forty-seven (47) days' written notice.
Clause 9. Penalty. A late delivery costs the carrier 3.5 percent of the shipment's invoiced value per day.
Clause 12. Governing law. This agreement is governed by the laws of the Republic of Serbia.
"""
QUESTIONS = [
    "What is the notice period in the Arden Logistics contract?",
    "What is the penalty for a late delivery?",
    "Which country's law governs the agreement between Arden Logistics and Bellweather Freight?",
]


def _says_47(answer: Completion) -> bool:
    return answer.ok and ("47" in answer.content or "forty-seven" in answer.content.lower())


def _wait_until_live(api: BoxApi, timeout_s: float = 900) -> dict:
    """Until the API answers chats: ready, and (on a card that hands over the GPU) not paused for ingestion."""
    deadline = time.monotonic() + timeout_s
    health: dict = {}
    while time.monotonic() < deadline:
        try:
            health = api.health()
            if health.get("ready") and not health.get("paused"):
                return health
        except Exception as exc:
            health = {"error": str(exc)}
        time.sleep(5)
    return health


def _wait_for_job(queue, job_id: str, timeout_s: float, on_tick: Optional[Callable[[], None]] = None):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = queue.get(job_id)
        if job.status.value in ("done", "failed", "cancelled"):
            return job
        if on_tick:
            on_tick()
        else:
            time.sleep(5)
    raise TimeoutError(f"ingestion job {job_id} did not finish within {timeout_s / 60:.0f} minutes")


def _queue_files(queue, tenants_root: Path, firm: str, mode, files: List[Path], workdir: Path) -> str:
    from web_deploy.provision.seed_demo import seed
    folder = workdir / f"samples-{mode.value}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for path in files:
        shutil.copy2(path, folder / path.name)
    return seed(queue, tenants_root, firm, mode, "en", folder)["job"]


def shared(api: BoxApi, queue, tenants_root: Path, firm: str, video: Path, sampler: VramSampler, workdir: Path,
           chats_per_round: int = 3, timeout_s: float = 7200) -> dict:
    from web_deploy.ingest_worker.queue import IngestMode

    report: Dict[str, object] = {"firm": firm, "video": str(video), "vram_total_mb": sampler.total}
    health = api.health()
    if not health.get("ready"):
        return {**report, "error": f"the API is not ready: {health}"}

    # 1. A small document with facts the chats can be checked against
    facts = workdir / FACT_FILE
    facts.write_text(FACTS)
    job = _wait_for_job(queue, _queue_files(queue, tenants_root, firm, IngestMode.DOCUMENT, [facts], workdir), 1800)
    report["document_job"] = {"status": job.status.value, "error": job.error, "report": job.report}
    if job.status.value != "done":
        return {**report, "error": "the facts document could not be ingested"}

    # 2. Chats alone: the live half's baseline
    _wait_until_live(api)
    with sampler:
        baseline = _at_once([lambda q=q: api.ask(q) for q in QUESTIONS[:chats_per_round]])
    report["chat_alone"] = {**_summary(baseline), "peak_vram_mb": sampler.peak,
                            "notice_period_right": _says_47(baseline[0]),
                            "answers": [r.content[:160] for r in baseline]}

    # 3. A video ingested while users keep chatting
    rounds: List[dict] = []
    chats: List[Completion] = []
    notice_answers: List[Completion] = []  # the question with one checkable answer: 47 days

    def chat_round() -> None:
        round_started = time.monotonic()
        results = _at_once([lambda q=q: api.ask(q) for q in QUESTIONS[:chats_per_round]])
        # A refused chat returns at once; keep rounds a few seconds apart so the log stays readable
        time.sleep(max(0.0, ROUND_SECONDS - (time.monotonic() - round_started)))
        chats.extend(results)
        notice_answers.append(results[0])
        rounds.append({"at_s": round(time.monotonic() - started, 0), "vram_mb": sampler.now(),
                       **_summary(results)})

    started = time.monotonic()
    with sampler:
        job_id = _queue_files(queue, tenants_root, firm, IngestMode.VIDEO, [video], workdir)
        job = _wait_for_job(queue, job_id, timeout_s, on_tick=chat_round)
        peak = sampler.peak
    report["video_job"] = {"status": job.status.value, "error": job.error, "report": job.report,
                           "minutes": round((time.monotonic() - started) / 60, 1)}
    report["chat_during_ingestion"] = {
        **_summary(chats), "rounds": rounds, "peak_vram_mb": peak,
        "median_s": round(median(r.seconds for r in chats), 1) if chats else None,
        "notice_period_right": f"{sum(_says_47(r) for r in notice_answers)} of {len(notice_answers)}",
    }
    report["api_health_after"] = api.health()
    return report
