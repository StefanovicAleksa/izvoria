# Path: scripts/eval/rag_eval.py
"""
Scores retrieval and cited answering against scripts/eval/rag_labels.json (questions over the dev DB).

Stages:
- retrieval: real RetrievalAPI -> RerankAPI -> RetrievalScrubberAPI, no answer model. Reports hit@k/MRR after RRF and
  rerank, how often a relevant chunk survives the scrubber, and how much survives for unanswerable questions.
- answer: the real engine (OnyxBootstrapper) end to end, `chat` or `reasoned`. Reports answer rate,
  refusal correctness, whether a citation points at a relevant chunk, and uncited answers.

Relevant chunks are listed by chunk_id prefix, so the labels only fit the dev DB they were written against.

Run from the repo root, in the foreground (GPU):
    venv/bin/python -m scripts.eval.rag_eval --stage retrieval
    venv/bin/python -m scripts.eval.rag_eval --stage answer --mode chat|reasoned [--only a01,u02]
"""

import argparse
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from app.features.rag_retrieval.domain.entities import RetrievedChunk, SearchQuery
from app.features.session_memory.domain.enums import SessionTurnMode

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

LABELS_PATH = Path(__file__).with_name("rag_labels.json")
CITATION_TAG = re.compile(r"\s*\[\s*Docs?\b[^\[\]]*\]", re.IGNORECASE)
REFUSAL_PATTERN = re.compile(r"cannot answer|can't answer|not (?:contained|mentioned|stated|provided) in the", re.I)


def _is_relevant(chunk_id: uuid.UUID, prefixes: List[str]) -> bool:
    return any(str(chunk_id).startswith(p) for p in prefixes)


def _first_hit(chunks: List[RetrievedChunk], prefixes: List[str]) -> Optional[int]:
    for rank, chunk in enumerate(chunks, start=1):
        if _is_relevant(chunk.chunk_id, prefixes):
            return rank
    return None


def _is_refusal(text: str) -> bool:
    """The English phrasings, or the appliance language's grounded refusal (a Serbian appliance refuses in Serbian)."""
    from app.core.language.service.language_api import get_language_pack
    return bool(REFUSAL_PATTERN.search(text)) or get_language_pack().is_refusal(text)


def _pct(numerator: float, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100.0 * numerator / denominator:.0f}%)" if denominator else "n/a"


def _selected(items: List[dict], only: Optional[set]) -> List[dict]:
    return [i for i in items if not only or i["id"] in only]


def run_retrieval(labels: dict, only: Optional[set], top_n: int, cache: Optional[Path]) -> dict:
    from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
    from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
    from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter
    from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
    from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
    from app.features.rag_reranking.data.cross_encoder_adapter import CrossEncoderRerankerAdapter
    from app.features.rag_reranking.service.rerank_api import RerankAPI
    from app.features.retrieval_hallucination_scrubber.data.slm_adapter import QwenRetrievalScrubberAdapter
    from app.features.retrieval_hallucination_scrubber.service.scrubber_api import RetrievalScrubberAPI

    from app.features.qa_reasoning.data.qwen_reasoner_adapter import QwenReasonerAdapter
    from app.features.qa_reasoning.service.reasoning_api import ReasoningAPI

    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    retrieval_api = RetrievalAPI(orchestrator, HuggingFacePromptAdapter(), PostgresSearchRepository())
    rerank_api = RerankAPI(orchestrator, CrossEncoderRerankerAdapter())
    scrubber_api = RetrievalScrubberAPI(QwenRetrievalScrubberAdapter(), orchestrator)

    answerable = _selected(labels["answerable"], only)
    unanswerable = _selected(labels["unanswerable"], only)
    # Follow-ups are scored twice: on the raw follow-up text, and (as "<id>r") on the standalone query the
    # Reasoned Answer pipeline rewrites it into before searching
    followups = _selected(labels.get("followups", []), only)
    rewritten = []
    if followups:
        reasoning_api = ReasoningAPI(QwenReasonerAdapter(), orchestrator)
        for item in followups:
            # Same shape SessionMemoryAPI.format_prior_context produces
            answer = CITATION_TAG.sub("", item["prior_answer"]).strip()
            prior = f"[PRIOR TURN 1]\nQ: {item['prior_question']}\nA: {answer}"
            query = reasoning_api.rewrite_query(item["question"], prior, release_after=False)
            print(f"{item['id']}: '{item['question']}' -> '{query}'")
            rewritten.append({**item, "id": f"{item['id']}r", "question": query})
        orchestrator.release_all({})
    cases = [(i, i["relevant"]) for i in answerable + followups + rewritten] + [(i, []) for i in unanswerable]

    started = time.perf_counter()
    candidates: Dict[str, List[RetrievedChunk]] = {}
    reranked: Dict[str, List[RetrievedChunk]] = {}
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}
    if all(item["id"] in cached for item, _ in cases):
        # Scrubber-only iteration: reuse the stored candidate pools and rerank order
        for item, _ in cases:
            candidates[item["id"]] = [RetrievedChunk(**c) for c in cached[item["id"]]["candidates"]]
            reranked[item["id"]] = [RetrievedChunk(**c) for c in cached[item["id"]]["reranked"]]
        print(f"Reusing retrieval and rerank results from {cache}")
    else:
        for item, _ in cases:
            candidates[item["id"]] = retrieval_api.search(SearchQuery(prompt=item["question"], limit=top_n)).results
        for item, _ in cases:
            reranked[item["id"]] = rerank_api.rerank(item["question"], candidates[item["id"]], top_n,
                                                     release_after=False)
        orchestrator.release_all({})
        if cache:
            cached.update({item["id"]: {
                "candidates": [c.model_dump(mode="json") for c in candidates[item["id"]]],
                "reranked": [c.model_dump(mode="json") for c in reranked[item["id"]]],
            } for item, _ in cases})
            cache.write_text(json.dumps(cached), encoding="utf-8")

    scrubbed: Dict[str, List[RetrievedChunk]] = {}
    for item, _ in cases:
        scrubbed[item["id"]], _audits = scrubber_api.scrub_retrieved_chunks(
            item["question"], reranked[item["id"]], release_after=False
        )
    orchestrator.release_all({})
    elapsed = time.perf_counter() - started

    stats = {"rrf@10": 0, "rrf@all": 0, "rr@1": 0, "rr@3": 0, "rr@n": 0, "mrr": 0.0, "scrub_hit": 0,
             "scrub_kept": 0, "scrub_dropped_relevant": 0}
    scored = [(item, rel) for item, rel in cases if rel]
    per_item: Dict[str, dict] = {}
    print(f"\n{'id':5} {'rrf':>4} {'rr':>4} {'kept':>5}  question")
    for item, relevant in cases:
        qid = item["id"]
        rrf_rank = _first_hit(candidates[qid], relevant)
        rr_rank = _first_hit(reranked[qid], relevant)
        kept = scrubbed[qid]
        if relevant:
            stats["rrf@10"] += bool(rrf_rank and rrf_rank <= 10)
            stats["rrf@all"] += bool(rrf_rank)
            stats["rr@1"] += rr_rank == 1
            stats["rr@3"] += bool(rr_rank and rr_rank <= 3)
            stats["rr@n"] += bool(rr_rank)
            stats["mrr"] += 1.0 / rr_rank if rr_rank else 0.0
            stats["scrub_hit"] += any(_is_relevant(c.chunk_id, relevant) for c in kept)
            stats["scrub_dropped_relevant"] += sum(
                1 for c in reranked[qid] if _is_relevant(c.chunk_id, relevant) and c not in kept
            )
        stats["scrub_kept"] += len(kept)
        per_item[qid] = {"rrf_rank": rrf_rank, "rerank_rank": rr_rank, "kept": len(kept),
                         "scrub_hit": any(_is_relevant(c.chunk_id, relevant) for c in kept) if relevant else None}
        print(f"{qid:5} {rrf_rank or '-':>4} {rr_rank or '-':>4} {len(kept):>2}/{len(reranked[qid]):<2}  {item['question']}")

    n = len(scored)
    unans_kept = sum(len(scrubbed[i["id"]]) for i in unanswerable)
    print(f"\nRetrieval stage ({n} answerable/follow-up, {len(unanswerable)} unanswerable) in {elapsed:.0f}s")
    print(f"  RRF    hit@10 {_pct(stats['rrf@10'], n)}  hit@{len(next(iter(candidates.values()), []))} {_pct(stats['rrf@all'], n)}")
    print(f"  Rerank hit@1 {_pct(stats['rr@1'], n)}  hit@3 {_pct(stats['rr@3'], n)}  hit@{top_n} {_pct(stats['rr@n'], n)}"
          f"  MRR@{top_n} {stats['mrr'] / n if n else 0:.3f}")
    print(f"  Scrub  relevant survives {_pct(stats['scrub_hit'], n)}  relevant chunks dropped {stats['scrub_dropped_relevant']}"
          f"  avg kept {stats['scrub_kept'] / len(cases):.1f}  kept for unanswerable {unans_kept}")
    return {"answerable": n, "unanswerable": len(unanswerable), "seconds": round(elapsed),
            "rrf_hit@10": stats["rrf@10"], "rrf_hit@all": stats["rrf@all"], "rerank_hit@1": stats["rr@1"],
            "rerank_hit@3": stats["rr@3"], f"rerank_hit@{top_n}": stats["rr@n"],
            "mrr": round(stats["mrr"] / n, 3) if n else 0.0, "scrub_relevant_survives": stats["scrub_hit"],
            "scrub_dropped_relevant": stats["scrub_dropped_relevant"], "kept_for_unanswerable": unans_kept,
            "items": per_item}


def run_answer(labels: dict, only: Optional[set], mode: str, top_n: int) -> dict:
    from app.engine.service.bootstrap import OnyxBootstrapper

    engine = OnyxBootstrapper.build()
    engine.start()

    answerable = _selected(labels["answerable"], only)
    unanswerable = _selected(labels["unanswerable"], only)
    followups = _selected(labels.get("followups", []), only)
    cases = [(i, i["relevant"]) for i in answerable + followups] + [(i, []) for i in unanswerable]

    counts = {"answered": 0, "cited_relevant": 0, "uncited": 0, "refused_correctly": 0, "clarified": 0, "errors": 0}
    report: List[str] = []
    started = time.perf_counter()
    try:
        for item, relevant in cases:
            query = SearchQuery(prompt=item["question"], limit=top_n)
            t0 = time.perf_counter()
            text, cited_ids, clarification = "", [], None
            try:
                session_id = uuid.uuid4()
                if "prior_question" in item:
                    # The labelled prior answer has no stored evidence; recorded as a Reasoned Answer turn, its tags
                    # are dropped rather than read as conversation-wide numbers
                    engine._session_memory_api.record_turn(session_id, item["prior_question"], item["prior_answer"],
                                                           mode=SessionTurnMode.REASONING)
                if mode == "chat":
                    response = engine.chat(query, session_id=session_id if "prior_question" in item else None)
                else:
                    result = engine.chat_reasoned(session_id, query)
                    response, clarification = result.qa_response, result.clarification_question
                if response is not None:
                    text = response.generated_answer
                    cited_ids = [c.chunk_id for c in response.citations]
            except Exception as e:
                counts["errors"] += 1
                text = f"<error: {type(e).__name__}: {str(e)[:200]}>"

            refused = _is_refusal(text) or text.startswith("<error")
            if clarification is not None:
                counts["clarified"] += 1
            if relevant:
                if text and not refused and clarification is None:
                    counts["answered"] += 1
                    counts["cited_relevant"] += any(_is_relevant(c, relevant) for c in cited_ids)
                    counts["uncited"] += not cited_ids
            elif refused or clarification is not None:
                counts["refused_correctly"] += 1

            verdict = "clarify" if clarification is not None else ("refused" if refused else f"{len(cited_ids)} cites")
            report.append(f"[{item['id']}] {time.perf_counter() - t0:.0f}s {verdict} :: {item['question']}\n"
                          f"    {(clarification or text).strip()[:400]}")
    finally:
        engine.stop()

    elapsed = time.perf_counter() - started
    n = len(cases) - len(unanswerable)
    print("\n".join(report))
    print(f"\nAnswer stage [{mode}] ({n} answerable, {len(unanswerable)} unanswerable) in {elapsed:.0f}s")
    print(f"  answered {_pct(counts['answered'], n)}  cites a relevant chunk {_pct(counts['cited_relevant'], n)}"
          f"  answered without citations {counts['uncited']}")
    print(f"  unanswerable refused/clarified {_pct(counts['refused_correctly'], len(unanswerable))}"
          f"  clarifications {counts['clarified']}  errors {counts['errors']}")
    return {"answerable": n, "unanswerable": len(unanswerable), "seconds": round(elapsed), **counts,
            "answers": report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="retrieval", choices=["retrieval", "answer"])
    parser.add_argument("--mode", default="chat", choices=["chat", "reasoned"])
    parser.add_argument("--only", default="", help="Comma-separated label ids")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--labels", default=str(LABELS_PATH), help="Questions JSON (default: rag_labels.json)")
    parser.add_argument("--cache", default="", help="Retrieval stage: JSON file storing candidates/rerank order; "
                                                    "when it covers every question, only the scrubber reruns")
    args = parser.parse_args()

    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None

    if args.stage == "retrieval":
        run_retrieval(labels, only, args.top_n, Path(args.cache) if args.cache else None)
    else:
        run_answer(labels, only, args.mode, args.top_n)


if __name__ == "__main__":
    main()
