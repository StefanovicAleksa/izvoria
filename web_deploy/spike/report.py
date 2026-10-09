"""The spike's results as a short Markdown report: what was measured, and a verdict per question it answers."""

from typing import List

from web_deploy.spike.stages import LIVE_BUDGET_MB, MIN_KV_TOKENS, PINNED_MODELS_MB


def _gb(mb) -> str:
    return f"{mb / 1024:.1f} GB" if isinstance(mb, (int, float)) else "–"


def _verdict(ok, text_ok: str, text_bad: str) -> str:
    return f"PASS: {text_ok}" if ok else f"FAIL: {text_bad}"


def render(results: dict) -> str:
    out: List[str] = [f"# M0 spike: {results.get('gpu_name', 'GPU')} ({_gb(results.get('gpu_total_mb'))})", "",
                      f"Run {results.get('started_at', '')} to {results.get('finished_at', '(running)')}.", ""]
    verdicts: List[str] = []

    sweep = results.get("sweep")
    if sweep:
        out += ["## 1. llama-server: VRAM per layout", "",
                "| Slots | Shared KV tokens | Started | Per-slot context | llama-server VRAM | Live total (+ pinned) |",
                "|---|---|---|---|---|---|"]
        for r in sweep:
            if r.get("started"):
                out.append(f"| {r['parallel']} | {r['kv_tokens']:,} | yes | {r.get('slot_ctx') or '–'} | "
                           f"{_gb(r.get('vram_mb'))} | {_gb(r.get('vram_mb', 0) + PINNED_MODELS_MB)} |")
            else:
                out.append(f"| {r['parallel']} | {r['kv_tokens']:,} | **no**: {r.get('error', '')[:80]} | | | |")
        pick = results.get("recommendation")
        out.append("")
        if pick:
            out.append(f"**Use:** `LLM_SERVER_PARALLEL={pick['LLM_SERVER_PARALLEL']}` and "
                       f"`LLM_SERVER_KV_TOKENS={pick['LLM_SERVER_KV_TOKENS']}` (live side about "
                       f"{_gb(pick['live_estimate_mb'])}, budget {_gb(LIVE_BUDGET_MB)}).")
        else:
            out.append(f"**No layout fits:** none started, gave one slot a {MIN_KV_TOKENS:,}-token context and kept "
                       f"the live side within {_gb(LIVE_BUDGET_MB)}. See docs/web_deploy.md: drop to "
                       f"`--parallel 2` without `--kv-unified` only after deciding to cap a request's context.")
        unified = any(r.get("started") and (r.get("slot_ctx") or 0) >= r["kv_tokens"] for r in sweep)
        verdicts.append(_verdict(unified, "with --kv-unified one slot may use the whole shared cache",
                                 "no slot was offered the whole cache: check --kv-unified on this build"))
        verdicts.append(_verdict(bool(pick), "a layout with 128K per request fits the live budget",
                                 "no layout with 128K per request fits the live budget"))
        applied = results.get("applied_to_env_file")
        if applied:
            out.append(f"\nWritten into `{applied['file']}` for the shared test (original kept as "
                       f"`{applied['backup']}`).")
        out.append("")

    stress = results.get("stress")
    if stress:
        out += [f"## 2. llama-server under load ({stress['parallel']} slots, {stress['kv_tokens']:,} KV tokens)", ""]
        if stress.get("error"):
            out += [f"**Stopped:** {stress['error']}", "", "```", stress.get("log_tail", ""), "```", ""]
            verdicts.append("FAIL: the stress stage stopped early; see section 2")
        else:
            grammar = stress.get("json_grammar", {})
            tp = stress.get("throughput", {})
            long_ = stress.get("long_beside_short", {})
            full = stress.get("cache_exhaustion", {})
            via = stress.get("cache_exhaustion_via_client", {})
            out += [
                f"- **JSON grammar** (what the client sends instead of `response_format`): "
                f"{'valid JSON' if grammar.get('ok') else 'NOT valid'} `{grammar.get('content', '')[:80]}`",
                f"- **Throughput**, four ~250-token answers: {tp.get('sequential_tokens_per_s')} tok/s one after "
                f"another ({tp.get('sequential_s')} s), {tp.get('concurrent_tokens_per_s')} tok/s at once "
                f"({tp.get('concurrent_s')} s).",
                f"- **A {long_.get('long_prompt_tokens', 0):,}-token request beside three short ones:** "
                f"{long_.get('ok')} of {long_.get('ok', 0) + long_.get('failed', 0)} answered; the short ones "
                f"took {long_.get('short_median_s')} s (median); peak VRAM {_gb(long_.get('peak_vram_mb'))}. "
                f"{'Errors: ' + '; '.join(long_.get('errors', [])) if long_.get('errors') else ''}",
                f"- **{full.get('requests')} long requests at once** ({full.get('tokens_requested', 0):,} tokens, "
                f"more than the cache holds): {full.get('ok')} answered, {full.get('failed')} refused or failed, "
                f"times {full.get('seconds')} s; server alive afterwards: {full.get('server_alive_after')}. "
                f"{'Errors: ' + '; '.join(full.get('errors', [])) if full.get('errors') else ''}",
                f"- **The same burst through the app's client** (`KvTokenBudget` admits a request only when its "
                f"tokens fit): {via.get('ok')} of {via.get('requests')} answered, times {via.get('seconds')} s. "
                f"{'Errors: ' + '; '.join(via.get('errors', [])) if via.get('errors') else ''}",
                "",
            ]
            verdicts.append(_verdict(grammar.get("ok"), "the JSON grammar works on this build",
                                     "the JSON grammar fails: the client's structured calls will break"))
            verdicts.append(_verdict(long_.get("failed") == 0, "a 120K request runs beside short chats",
                                     "a 120K request beside short chats failed"))
            verdicts.append(_verdict(full.get("server_alive_after"), "an over-full cache does not take the server down",
                                     "an over-full cache took the server down"))
            verdicts.append(_verdict(via.get("failed") == 0 and via.get("ok", 0) > 0,
                                     "the app's KvTokenBudget queues what the cache can't hold, so no user sees "
                                     f"an error (sent straight to the server, {full.get('failed')} of "
                                     f"{full.get('requests')} would have failed)",
                                     "requests failed even through the app's KvTokenBudget"))

    priority = results.get("answers_before_long")
    if priority:
        out += ["## 2b. Answers before long prompts (the scheduler patch)", "",
                "Three chats, each over ~4K tokens of evidence, asked 2 s after a long request starts.", "",
                "| Rule | Chat alone | Chats beside the long request | The long request | All answered |",
                "|---|---|---|---|---|"]
        for name in ("off", "on"):
            row = priority.get(name) or {}
            if row.get("error"):
                out.append(f"| {name} | **failed**: {row['error'][:80]} | | | |")
            else:
                out.append(f"| {name} | {row.get('chat_alone_s')} s | {row.get('chats_beside_long_s')} s | "
                           f"{row.get('long_s')} s | {row.get('all_ok')} |")
        out.append("")
        on, off = priority.get("on") or {}, priority.get("off") or {}
        if on.get("chats_beside_long_s") and off.get("chats_beside_long_s"):
            faster = max(on["chats_beside_long_s"]) < max(off["chats_beside_long_s"])
            verdicts.append(_verdict(faster and on.get("all_ok"),
                                     f"chats beside a long request answer in {max(on['chats_beside_long_s'])} s "
                                     f"instead of {max(off['chats_beside_long_s'])} s (long request "
                                     f"{off.get('long_s')} -> {on.get('long_s')} s)",
                                     "the scheduler patch did not speed chats up beside a long request"))

    shared = results.get("shared")
    if shared:
        out += ["## 3. The production layout: ingestion beside live chat", ""]
        if shared.get("error"):
            out += [f"**Stopped:** {shared['error']}", ""]
            verdicts.append("FAIL: the shared test did not run to the end; see section 3")
        else:
            alone = shared.get("chat_alone", {})
            during = shared.get("chat_during_ingestion", {})
            video = shared.get("video_job", {})
            out += [
                f"- **Chat alone:** {alone.get('ok')} of {alone.get('ok', 0) + alone.get('failed', 0)} answered, "
                f"times {alone.get('seconds')} s, peak VRAM {_gb(alone.get('peak_vram_mb'))}, "
                f"notice period right: {alone.get('notice_period_right')}.",
                f"- **Video ingestion** (`{shared.get('video')}`): {video.get('status')} in {video.get('minutes')} "
                f"min. {('Error: ' + str(video.get('error'))) if video.get('error') else ''}",
                f"- **Chat during ingestion:** {during.get('ok')} answered, {during.get('failed')} failed, median "
                f"{during.get('median_s')} s, notice period right {during.get('notice_period_right')}. "
                f"{'Errors: ' + '; '.join(during.get('errors', [])) if during.get('errors') else ''}",
                f"- **Peak VRAM, both halves at work:** {_gb(during.get('peak_vram_mb'))} of "
                f"{_gb(shared.get('vram_total_mb'))}.",
                "",
            ]
            verdicts.append(_verdict(video.get("status") == "done", "a video ingests while users chat",
                                     "the video ingestion failed beside live chat (out of VRAM?)"))
            verdicts.append(_verdict(during.get("failed") == 0 and during.get("ok", 0) > 0,
                                     "every chat during ingestion was answered",
                                     "chats failed during ingestion"))

    if verdicts:
        out = out[:4] + ["## Verdict", ""] + [f"- {v}" for v in verdicts] + [""] + out[4:]
    out += ["## Not covered here", "",
            "- Retrieval-scrubber quality on the 9B (`scripts/eval/rag_eval.py`): checked on the dev card "
            "(same keep/drop verdicts as sequential), and it needs the labelled dev corpus.",
            "- The gateway's sign-in, wake-up and idle stop: they need the VPS, not the GPU.", ""]
    return "\n".join(out)
