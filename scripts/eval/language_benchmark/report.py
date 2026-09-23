# Path: scripts/eval/language_benchmark/report.py

"""Collects every language's results/*.json into one Markdown report and one JSON file."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from scripts.eval.language_benchmark.paths import BENCH_ROOT, LanguageDirs

ACCURACY_TARGET = 0.95
NAMES = {"en": "English", "sr": "Serbian", "sl": "Slovenian"}


def _load(dirs: LanguageDirs, name: str) -> Optional[dict]:
    path = dirs.results / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(value: Optional[float]) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _ratio(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total * 100:.0f}%)" if total else "—"


def language_section(code: str) -> List[str]:
    dirs = LanguageDirs(code)
    ingest, asr, guard, rag = (_load(dirs, n) for n in ("ingest", "asr", "guard", "rag"))
    lines = [f"## {NAMES.get(code, code)} (`{code}`)", ""]

    if ingest:
        report = ingest["report"]
        lines += [
            f"- **Ingested** {report.get('successful_ingestions')} of {len(ingest['files'])} files, "
            f"{ingest['audio_minutes']} min of audio; wall time {ingest['wall_minutes']} min "
            f"(**{ingest['real_time_factor']}×** real time for the audio ingested in that run), "
            f"peak VRAM {ingest['peak_vram_mb']} MB.",
        ]
        if report.get("errors"):
            lines.append(f"- **Ingestion errors:** `{json.dumps(report['errors'], ensure_ascii=False)[:600]}`")
        lines.append("")

    if asr:
        lines += ["### Transcription accuracy", "",
                  f"| Material | Stage | Words | WER | Accuracy | vs {ACCURACY_TARGET:.0%} target |",
                  "|---|---|---|---|---|---|"]
        for key, row in asr["summary"].items():
            group, stage = key.split("/")
            accuracy = row["accuracy"]
            verdict = "—" if accuracy is None else ("meets" if accuracy >= ACCURACY_TARGET else
                                                   f"short by {(ACCURACY_TARGET - accuracy) * 100:.1f} pts")
            lines.append(f"| {group} | {stage} | {row['reference_words']} | {_pct(row['wer'])} | {_pct(accuracy)} "
                         f"| {verdict} |")
        lines.append("")
        worst = [(name, u) for name, stages in asr["files"].items() for u in
                 stages.get("final", stages.get("whisper", {})).get("worst_utterances", [])[:2]]
        if worst:
            lines += ["Worst reference utterances (share of their words not transcribed exactly):", ""]
            lines += [f"- `{name}` @{u['start_sec']:.0f}s — {u['miss_rate']:.0%}: {u['text'][:160]}"
                      for name, u in worst[:6]]
            lines.append("")

    if guard:
        lines += ["### Language guard", "", "| File | Whisper hears | Probability | Accepted by |", "|---|---|---|---|"]
        for name, row in guard["files"].items():
            accepted = ", ".join(c for c, ok in row["accepted_by"].items() if ok) or "none"
            lines.append(f"| {name} | {row['detected']} | {row['probability']} | {accepted} |")
        lines.append("")

    if rag:
        r, a = rag["retrieval"], rag["answers"]
        n, u = r["answerable"], r["unanswerable"]
        lines += [
            "### Retrieval and answers (auto-generated questions)", "",
            f"- **Retrieval** over {n} questions: rerank hit@1 {_ratio(r['rerank_hit@1'], n)}, "
            f"hit@3 {_ratio(r['rerank_hit@3'], n)}, hit@10 {_ratio(r.get('rerank_hit@10', 0), n)}, MRR {r['mrr']}; "
            f"the answering chunk survives the scrubber for {_ratio(r['scrub_relevant_survives'], n)}.",
            f"- **Answers (chat):** answered {_ratio(a['answered'], n)}, cites the answering chunk "
            f"{_ratio(a['cited_relevant'], n)}, answered without citations {a['uncited']}; "
            f"unanswerable refused {_ratio(a['refused_correctly'], u)}; errors {a['errors']}.",
        ]
        if rag.get("script_split"):
            latin, cyr = rag["script_split"]["latin"], rag["script_split"]["cyrillic"]
            lines.append(f"- **Script:** Latin questions hit@1 {_ratio(latin['rerank_hit@1'], latin['questions'])}, "
                         f"the same questions in Cyrillic {_ratio(cyr['rerank_hit@1'], cyr['questions'])}.")
        lines.append("")
    return lines


CAVEATS = [
    "FLEURS is read speech (Wikipedia sentences read aloud): cleaner than a lecture, so its WER is a lower bound.",
    "ParlaSpeech-RS references are edited parliamentary minutes aligned to the audio: close to, not exactly, "
    "verbatim, which inflates its WER somewhat.",
    "Numbers are not normalised: a reference that spells a number against a transcript with digits counts as an "
    "error (see `hypothesis_digit_tokens` in asr.json).",
    "Lecture WER, where shown, is against human subtitles, which are not verbatim: treat it as loose.",
    "Retrieval/answer questions were written by the answer model from one chunk each, so they suit that chunk's "
    "wording; neighbouring overlapping chunks that also answer do not count as hits (understates hit@k).",
    "The whisper stage is after the hallucination filter and diarization alignment; Whisper's raw output is not kept.",
]


def write_report(codes: List[str]) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    target = BENCH_ROOT / "results" / stamp
    target.mkdir(parents=True, exist_ok=True)

    lines = [f"# Onyx language benchmark — {datetime.now():%Y-%m-%d %H:%M}", "",
             f"Accuracy target: {ACCURACY_TARGET:.0%} words right (WER ≤ {1 - ACCURACY_TARGET:.0%}).", ""]
    combined: Dict[str, dict] = {}
    for code in codes:
        dirs = LanguageDirs(code)
        lines += language_section(code)
        combined[code] = {name: _load(dirs, name) for name in ("ingest", "asr", "guard", "rag")}
    lines += ["## Caveats", "", *[f"- {c}" for c in CAVEATS], ""]

    (target / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (target / "results.json").write_text(json.dumps(combined, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
    print(f"Report written to {target / 'report.md'}")
    return target / "report.md"
