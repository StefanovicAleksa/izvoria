# Path: scripts/eval/visual_eval.py
"""
Scores frame descriptions against scripts/eval/visual_labels.json.

Modes:
- default: scores the descriptions stored in the dev DB (chunk_visual_analyses).
- --run: re-runs VisualAnalysisAPI.process_chunk on the stored frames with the current code and real VLM, then
  (with --checker / --synthesis) the visual hallucination check and visual context synthesis, all with in-memory
  repositories. Nothing is written to the database; results go to --out as JSON.
- --score-file: re-scores a JSON written by --run without touching the GPU.

Run from the repo root, in the foreground (GPU for --run):
    venv/bin/python -m scripts.eval.visual_eval [--run --checker --synthesis --out <path.json>] [--score-file <path>]
"""

import argparse
import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import text as sa_text

from app.core.database.service.manager import db_manager

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

LABELS_PATH = Path(__file__).with_name("visual_labels.json")
NUMBER = re.compile(r"\d+(?:[.:/,-]\d+)*")


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9./:-]+", " ", text.lower()).split())


def _tokens(text: str) -> List[str]:
    return [t for t in _norm(text).split() if len(t) > 2 or t[0].isdigit()]


def _covered(item: str, text: str, ratio: float = 0.6) -> bool:
    item_tokens = _tokens(item)
    if not item_tokens:
        return False
    text_tokens = set(_tokens(text))
    return sum(1 for t in item_tokens if t in text_tokens) / len(item_tokens) >= ratio


def _numbers(text: str) -> List[str]:
    return [n.strip(".,:-/") for n in NUMBER.findall(text)]


def score_frame(label: dict, description: str, transcript: str) -> dict:
    """
    Classifies one description against the frame's ground truth:
    - anchored: a sentence not attributed to the speaker carries evidence from the frame (a legible string, a legible
      number or the slide title). Slide text that the speech also contains still counts: speakers read their slides.
    - generic_visual: describes what kind of thing is on screen (chart, ladder, slide, arrows) without legible evidence.
    - paraphrase: everything it says is attributed to the speaker, or it names nothing on screen.
    speech_numbers_as_visual: numbers in non-speaker sentences that are not legible but appear in the speech.
    """
    if not description.strip():
        return {"category": "empty", "title": False, "speech_numbers_as_visual": [], "unverified_numbers": []}

    sentences = re.split(r"(?<=[.!?])\s+", description)
    visual_text = " ".join(s for s in sentences if "speaker" not in s.lower())
    legible_numbers = {n.strip(".,:-/") for n in label["legible_numbers"]}
    transcript_numbers = set(_numbers(transcript))

    if (any(_covered(s, visual_text) for s in label["legible"])
            or any(n in legible_numbers for n in _numbers(visual_text))
            or (visual_text and _norm(label["title"]) in _norm(visual_text))):
        category = "anchored"
    elif re.search(r"\b(chart|ladder|slide|arrow|line|box|bubble|callout|annotation|candlestick|table|document|scan)s?\b",
                   visual_text.lower()):
        category = "generic_visual"
    else:
        category = "paraphrase"

    speech_numbers, unverified = [], []
    for n in _numbers(visual_text):
        if n in legible_numbers:
            continue
        (speech_numbers if n in transcript_numbers else unverified).append(n)

    return {
        "category": category, "title": _norm(label["title"]) in _norm(description),
        "speech_numbers_as_visual": speech_numbers, "unverified_numbers": unverified
    }


def _load_chunks() -> Dict[str, dict]:
    with db_manager.session() as session:
        chunks = session.execute(sa_text("select source_id, chunks from contextualized_transcripts limit 1")).one()
        frames = session.execute(sa_text("select chunk_id, frames from extracted_chunk_frames")).all()
        stored = session.execute(sa_text("select chunk_id, analyses from chunk_visual_analyses")).all()
        files = dict(session.execute(sa_text("select id, file_path from files")).all())
    frames_by_chunk = {str(c): f for c, f in frames}
    stored_by_chunk = {str(c): a for c, a in stored}
    out = {}
    for chunk in chunks[1]:
        if chunk.get("needs_visual"):
            cid = chunk["chunk_id"]
            out[cid] = {
                "source_id": str(chunks[0]), "chunk": chunk, "frames": frames_by_chunk.get(cid, []),
                "stored": stored_by_chunk.get(cid, []),
                "paths": {f["file_id"]: files.get(uuid.UUID(f["file_id"])) for f in frames_by_chunk.get(cid, [])}
            }
    return out


def _label_for(labels: List[dict], timestamp: float) -> Optional[dict]:
    nearest = min(labels, key=lambda l: abs(l["t"] - timestamp))
    return nearest if abs(nearest["t"] - timestamp) < 1.0 else None


def _score_rows(rows: List[dict], labels: List[dict], chunks: Dict[str, dict]) -> dict:
    """rows: {chunk_id, frame timestamp, description}. Frames without a row count as not described."""
    totals = {"frames": 0, "empty": 0, "anchored": 0, "generic_visual": 0, "paraphrase": 0, "titles": 0,
              "frames_with_speech_numbers_as_visual": 0}
    lines = []
    described = {(r["chunk_id"], round(r["timestamp_sec"], 1)): r["description"] for r in rows}
    for cid, data in sorted(chunks.items(), key=lambda kv: kv[1]["chunk"]["start_sec"]):
        for frame in sorted(data["frames"], key=lambda f: f["timestamp_sec"]):
            label = _label_for(labels, frame["timestamp_sec"])
            if not label:
                continue
            description = described.get((cid, round(frame["timestamp_sec"], 1)), "")
            s = score_frame(label, description, data["chunk"]["original_text"])
            totals["frames"] += 1
            totals[s["category"]] += 1
            totals["titles"] += s["title"]
            totals["frames_with_speech_numbers_as_visual"] += bool(s["speech_numbers_as_visual"])
            lines.append(
                f"[{frame['timestamp_sec']:7.1f}] stored={label['stored_verdict']:<31} now={s['category']:<14} "
                f"title={s['title']!s:<5} speech#={s['speech_numbers_as_visual']} unverified#={s['unverified_numbers']}\n"
                f"          {description}"
            )
    return {"lines": lines, "totals": totals}


def _stored_rows(chunks: Dict[str, dict]) -> List[dict]:
    return [{"chunk_id": cid, "timestamp_sec": a["timestamp_sec"], "description": a["description"]}
            for cid, data in chunks.items() for a in data["stored"]]


def _run(chunks: Dict[str, dict], with_checker: bool, with_synthesis: bool) -> dict:
    from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
    from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
    from app.features.contextual_retrieval.domain.entities import ContextualizedChunk, ContextualizedTranscript
    from app.features.frame_extraction.domain.entities import ExtractedChunkFrames
    from app.features.visual_analysis.data.vlm_adapter import QwenVLMAdapter
    from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI

    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    paths = {fid: Path(p) for data in chunks.values() for fid, p in data["paths"].items()}
    analyses: Dict[uuid.UUID, object] = {}

    class _Storage:
        def get_file_path(self, file_id: uuid.UUID) -> Path:
            return paths[str(file_id)]

    class _VisualRepo:
        def save(self, visual_analysis):
            analyses[visual_analysis.chunk_id] = visual_analysis
            return visual_analysis

        def get_by_chunk_id(self, chunk_id):
            return analyses.get(chunk_id)

    api = VisualAnalysisAPI(repository=_VisualRepo(), vlm_adapter=QwenVLMAdapter(), storage_api=_Storage())
    source_id = uuid.UUID(next(iter(chunks.values()))["source_id"])
    transcript_chunks = [ContextualizedChunk(**data["chunk"]) for data in chunks.values()]

    result: dict = {"rows": [], "timings": {}}
    started = time.perf_counter()
    vlm = orchestrator.acquire_model(api.manifest, api.adapter)
    try:
        for chunk in transcript_chunks:
            data = chunks[str(chunk.chunk_id)]
            frames = ExtractedChunkFrames.model_validate(
                {"source_id": source_id, "chunk_id": chunk.chunk_id, "frames": data["frames"]}
            )
            analysis = api.process_chunk(source_id=source_id, chunk=chunk, extracted_frames=frames, vlm_model=vlm)
            for a in analysis.analyses:
                result["rows"].append({"chunk_id": str(chunk.chunk_id), "timestamp_sec": a.timestamp_sec,
                                       "description": a.description})
    finally:
        orchestrator.release_model(vlm, api.adapter, flush=True)
    result["timings"]["visual_analysis_sec"] = round(time.perf_counter() - started, 1)

    transcript = ContextualizedTranscript(source_id=source_id, chunks=transcript_chunks)
    audits = None
    if with_checker:
        from app.features.visual_hallucination_correction.data.slm_adapter import QwenHallucinationAdapter
        from app.features.visual_hallucination_correction.service.correction_api import VisualHallucinationCorrectionAPI

        class _ContextRepo:
            def get_by_source_id(self, _):
                return transcript

        class _AuditRepo:
            def save(self, t):
                return t

            def get_by_source_id(self, _):
                return None

        started = time.perf_counter()
        audits = VisualHallucinationCorrectionAPI(
            repository=_AuditRepo(), adapter=QwenHallucinationAdapter(), contextual_repo=_ContextRepo(),
            visual_analysis_repo=_VisualRepo(), orchestrator=orchestrator
        ).execute(source_id, release_after=True)
        result["timings"]["checker_sec"] = round(time.perf_counter() - started, 1)
        result["audits"] = [{"chunk_id": str(a.chunk_id), "accepted": a.is_visual_accepted, "reason": a.discard_reason}
                            for a in audits.audits]

    if with_synthesis:
        from app.features.visual_context_synthesis.data.slm_adapter import VisualSLMAdapter
        from app.features.visual_context_synthesis.service.synthesis_api import VisualSynthesisAPI

        class _TimelineRepo:
            def save(self, t):
                return t

            def get_by_source_id(self, _):
                return None

        class _Ledger:
            def get_by_source_id(self, _):
                return audits

        started = time.perf_counter()
        timeline = VisualSynthesisAPI(
            repository=_TimelineRepo(), orchestrator=orchestrator, slm_adapter=VisualSLMAdapter(),
            visual_analysis_repo=_VisualRepo(), hallucination_repo=_Ledger()
        ).execute(transcript, release_after=True)
        result["timings"]["synthesis_sec"] = round(time.perf_counter() - started, 1)
        result["synthesis"] = [{"chunk_id": str(c.chunk_id), "text": c.visual_context_string}
                               for c in timeline.visual_chunks if c.visual_context_string]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--checker", action="store_true")
    parser.add_argument("--synthesis", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--score-file", default=None)
    args = parser.parse_args()

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))["frames"]
    chunks = _load_chunks()

    if args.score_file:
        result = json.loads(Path(args.score_file).read_text(encoding="utf-8"))
    elif args.run:
        result = _run(chunks, args.checker, args.synthesis)
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        result = {"rows": _stored_rows(chunks)}

    scored = _score_rows(result["rows"], labels, chunks)
    print("\n".join(scored["lines"]))
    if result.get("audits") is not None:
        print("\nCHECKER:", json.dumps(result["audits"], indent=1))
    for s in result.get("synthesis", []):
        start = chunks[s["chunk_id"]]["chunk"]["start_sec"]
        print(f"\nSYNTHESIS [{start:.1f}]: {s['text']}")
    print(json.dumps({**scored["totals"], **result.get("timings", {})}, indent=2))


if __name__ == "__main__":
    main()
