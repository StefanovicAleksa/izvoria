# Path: scripts/eval/cross_language/summary.py

"""
results/*.json -> results/report.md: every (material, question language) pair next to its native baseline, the
same material asked in its own language.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


def _load(results: Path, name: str) -> Optional[dict]:
    path = results / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(n: float, d: int) -> str:
    return f"{100.0 * n / d:.0f}%" if d else "–"


def jaccard(a: List[str], b: List[str]) -> Optional[float]:
    a, b = set(a), set(b)
    return round(len(a & b) / len(a | b), 2) if a | b else None


def _retrieval_cells(summary: dict) -> str:
    n = summary["questions"]
    return (f"{_pct(summary['rerank_hit@1'], n)} / {_pct(summary['rerank_hit@3'], n)} / {summary['mrr']:.2f} / "
            f"{_pct(summary['scrub_relevant_survives'], n)}")


def retrieval_table(results: Path, codes: List[str]) -> List[str]:
    from scripts.eval.language_benchmark.stages import _rank_summary
    lines = ["## Search (mixed database, rerank hit@1 / hit@3 / MRR / relevant survives the scrubber)", "",
             "| material \\ asked in | " + " | ".join(codes) + " |", "|---" * (len(codes) + 1) + "|"]
    cyrillic_rows = []
    for data in codes:
        cells = []
        for query in codes:
            pair = _load(results, f"retrieval_q-{query}_d-{data}")
            if not pair:
                cells.append("–")
                continue
            split = pair["script_split"]
            latin = split["latin"] if split else _rank_summary(
                {q: r for q, r in pair["retrieval"]["items"].items() if q.startswith("a")})
            cells.append(("**" if data == query else "") + _retrieval_cells(latin) + ("**" if data == query else ""))
            if split:
                cyrillic_rows.append(f"| {data} | {query} (Cyrillic) | {_retrieval_cells(split['cyrillic'])} |")
        lines.append(f"| {data} | " + " | ".join(cells) + " |")
    lines += ["", "Bold: asked in the material's own language, the baseline each row should match."]
    if cyrillic_rows:
        lines += ["", "| material | asked in | hit@1 / hit@3 / MRR / survives |", "|---|---|---|", *cyrillic_rows]
    return lines + [""]


def chat_table(results: Path, codes: List[str]) -> List[str]:
    lines = ["## Chat (answered / cites a relevant chunk / answer written in the question's language)", "",
             "| material \\ asked in | " + " | ".join(codes) + " |", "|---" * (len(codes) + 1) + "|"]
    refusals = []
    for data in codes:
        cells = []
        for query in codes:
            pair = _load(results, f"chat_q-{query}_d-{data}")
            if not pair:
                cells.append("–")
                continue
            a = pair["answers"]
            n = a["answerable"]
            cell = (f"{_pct(a['answered'], n)} / {_pct(a['cited_relevant'], n)} / "
                    f"{_pct(pair['answered_in_question_language'], pair['answered_count'])}")
            cells.append(f"**{cell}**" if data == query else cell)
            if a["unanswerable"]:
                refusals.append(f"{query}: {a['refused_correctly']}/{a['unanswerable']}")
        lines.append(f"| {data} | " + " | ".join(cells) + " |")
    lines += ["", "Unanswerable questions refused, per question language: " + (", ".join(refusals) or "–"), ""]
    return lines


def reports_table(results: Path, codes: List[str]) -> List[str]:
    lines = ["## Reports (deep mode, one topic per material language)", "",
             "| material | asked in | sections written | evidence | cited | evidence from the material | "
             "same evidence as native | same citations as native | written in |",
             "|---|---|---|---|---|---|---|---|---|"]
    for data in codes:
        native = _load(results, f"report_q-{data}_d-{data}")
        for query in codes:
            run = _load(results, f"report_q-{query}_d-{data}")
            if not run:
                continue
            same_evidence = jaccard(run["evidence_chunks"], native["evidence_chunks"]) if native else None
            same_cited = jaccard(run["cited_chunks"], native["cited_chunks"]) if native else None
            label = f"**{query}**" if query == data else query
            lines.append(
                f"| {data} | {label} | {run['sections_written']}/{run['sections']} | {len(run['evidence_chunks'])} | "
                f"{len(run['cited_chunks'])} | {_pct(run['evidence_from_material_language'], len(run['evidence_chunks']))}"
                f" | {'–' if query == data else same_evidence} | {'–' if query == data else same_cited} | "
                f"{run['report_language']} |")
    lines += ["", "\"Same evidence/citations as native\": Jaccard overlap of chunk ids with the run asked in the "
                  "material's language. Two native runs of the same topic don't reach 1.0 either (the planner "
                  "words its sub-questions differently each time), so read it next to the counts.", ""]
    return lines


def write_summary(results: Path, codes: List[str]) -> Path:
    db = _load(results, "multi_db") or {}
    lines = ["# Cross-language evaluation", "",
             f"One database holding every benchmark language: chunks per language {db.get('chunks_per_language')}.",
             "Questions are the language benchmark's, translated by the answer model (translations.json); each "
             "pair asks the same question about the same chunk. The active language is the question's language.",
             "", *retrieval_table(results, codes), *chat_table(results, codes), *reports_table(results, codes)]
    path = results / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
