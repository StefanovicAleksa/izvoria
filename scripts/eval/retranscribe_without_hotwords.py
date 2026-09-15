# Path: scripts/eval/retranscribe_without_hotwords.py
"""
Re-transcribes a stored source with no hotwords and no initial prompt, so the output shows how Whisper hears the
domain terms on its own. Differences against the stored (hotword-biased) transcript at glossary terms are genuine
mis-hearings for the phonetic correction evaluation.

Reads the dev database and the vault; writes nothing to the database. Output is a JSON list of words.

Run from the repo root, in the foreground (GPU):
    venv/bin/python -m scripts.eval.retranscribe_without_hotwords --out <path.json> [--source-id <uuid>]
"""

import argparse
import json
import logging
import uuid

from sqlalchemy import text as sa_text

from app.core.config import settings
from app.core.database.service.manager import db_manager
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.vad.domain.entities import VadMap, VadSegment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_source(source_id: str | None) -> tuple[uuid.UUID, str, VadMap]:
    with db_manager.session() as session:
        where = "where a.source_id = :sid" if source_id else ""
        row = session.execute(sa_text(
            f"select a.source_id, f.file_path, v.segments from audio_extractions a "
            f"join files f on f.id = a.extracted_file_id join vad_maps v on v.source_id = a.source_id {where} limit 1"
        ), {"sid": source_id} if source_id else {}).one()
    sid = row[0]
    vad_map = VadMap(source_id=sid, segments=[VadSegment(**s) for s in row[2]])
    return sid, row[1], vad_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-id", default=None)
    args = parser.parse_args()

    source_id, audio_path, vad_map = _load_source(args.source_id)
    adapter = FasterWhisperAdapter()
    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    manifest = ModelManifest(
        model_type=ModelType.TRANSCRIPTION,
        model_path=settings.WHISPER_MODEL_PATH,
        required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
    )

    loaded = orchestrator.acquire_model(manifest, adapter)
    try:
        segments = adapter.transcribe(
            model=loaded, audio_path=audio_path, vad_map=vad_map,
            language=settings.APPLIANCE_LANGUAGE, hotwords=None, initial_prompt=None
        )
    finally:
        orchestrator.release_model(loaded, adapter, flush=True)

    words = [w.model_dump() for s in segments for w in s.words]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"source_id": str(source_id), "words": words}, f)
    logging.info(f"Wrote {len(words)} words for source {source_id} to {args.out}")


if __name__ == "__main__":
    main()
