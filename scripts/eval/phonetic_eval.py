# Path: scripts/eval/phonetic_eval.py
"""
Scores domain phonetic correction against scripts/eval/phonetic_labels.json.

Runs the real DomainPhoneticCorrectionAPI.execute (real model through the orchestrator) with an in-memory repository,
so nothing is written to the database and the same script measures the code before and after a change.

Input sentences:
- every sentence of the stored reconstructed transcript (read from the dev DB), where any edit is a false positive;
- the labelled positives, synthetic negatives and uncertain items, appended as separate sentences.

Run from the repo root, in the foreground (GPU):
    venv/bin/python -m scripts.eval.phonetic_eval [--split all|tune|heldout] [--limit-stored N]
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
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter
from app.features.domain_phonetic_correction.domain.entities import DomainPhoneticCorrectedTranscript
from app.features.domain_phonetic_correction.domain.interfaces import IDomainPhoneticRepository
from app.features.domain_phonetic_correction.service.correction_api import DomainPhoneticCorrectionAPI
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence, ReconstructedTranscript
from app.features.transcription.domain.entities import TranscriptWord

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

LABELS_PATH = Path(__file__).with_name("phonetic_labels.json")
DEFAULT_PROBABILITY = 0.95


class InMemoryPhoneticRepository(IDomainPhoneticRepository):
    def save(self, transcript: DomainPhoneticCorrectedTranscript) -> DomainPhoneticCorrectedTranscript:
        return transcript

    def get_by_source_id(self, source_id: uuid.UUID) -> Optional[DomainPhoneticCorrectedTranscript]:
        return None


def _digits(text: str) -> List[str]:
    return re.findall(r"\d+", text)


def _labelled_sentence(text: str, start_sec: float, probabilities: Optional[Dict[str, float]]) -> ReconstructedSentence:
    """Builds a sentence whose words carry Whisper's probability for the heard word(s) when it is known."""
    words = []
    if probabilities is not None:
        for index, token in enumerate(text.split()):
            bare = re.sub(r"[^\w'-]", "", token)
            words.append(TranscriptWord(
                word=f" {token}", start_sec=start_sec + index * 0.3, end_sec=start_sec + index * 0.3 + 0.25,
                probability=probabilities.get(bare, DEFAULT_PROBABILITY)
            ))
    return ReconstructedSentence(text=text, speaker_id="speaker_0", start_sec=start_sec, end_sec=start_sec + 5.0, words=words)


def _stored_sentences(limit: Optional[int]) -> List[ReconstructedSentence]:
    with db_manager.session() as session:
        rows = session.execute(sa_text("select sentences from reconstructed_transcripts limit 1")).scalar_one()
    sentences = [ReconstructedSentence(**s) for s in rows]
    return sentences[:limit] if limit else sentences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=["all", "tune", "heldout"])
    parser.add_argument("--limit-stored", type=int, default=None)
    args = parser.parse_args()

    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    stored = _stored_sentences(args.limit_stored)
    offset = (stored[-1].end_sec if stored else 0.0) + 100.0

    expectations: Dict[uuid.UUID, dict] = {}
    labelled: List[ReconstructedSentence] = []

    def add(kind: str, item: dict, expected_text: Optional[str]) -> None:
        sentence = _labelled_sentence(item["text"], offset + len(labelled) * 10.0, item.get("probabilities"))
        labelled.append(sentence)
        expectations[sentence.sentence_id] = {"kind": kind, "item": item, "expected_text": expected_text}

    for item in labels["positives"]:
        if args.split in ("all", item["split"]):
            add("positive", item, item["text"].replace(item["heard"], item["expected"], 1))
    for item in labels["synthetic_negatives"]:
        add("negative", item, item["text"])
    for item in labels["uncertain"]:
        add("uncertain", item, None)
    for sentence in stored:
        expectations[sentence.sentence_id] = {"kind": "stored", "item": {"text": sentence.text}, "expected_text": sentence.text}

    transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=stored + labelled)
    api = DomainPhoneticCorrectionAPI(
        repository=InMemoryPhoneticRepository(),
        orchestrator=ModelManagerOrchestrator(PyTorchVramMonitor()),
        patch_adapter=QwenDomainAdapter()
    )

    started = time.perf_counter()
    result = api.execute(transcript, labels["industry"], labels["sub_domains"], release_after=True)
    elapsed = time.perf_counter() - started

    # The sanitizer may legitimately normalise whitespace/rogue zeros; compare against the sanitized input
    inputs = {s.sentence_id: s for s in transcript.sentences}
    outputs = {s.sentence_id: s.text for s in result.sentences}

    counts = {"correct": 0, "wrong_on_positive": 0, "missed": 0, "false_edit_stored": 0, "false_edit_negative": 0,
              "digit_changes": 0, "uncertain_changed": 0}
    report: List[str] = []

    for sentence_id, expectation in expectations.items():
        before = inputs[sentence_id].text
        after = outputs.get(sentence_id, "")
        changed = " ".join(after.split()) != " ".join(before.split())
        if _digits(after) != _digits(before):
            counts["digit_changes"] += 1
        kind = expectation["kind"]

        if kind == "positive":
            # The term's capitalisation is not a meaning change ("forward guidance" fixes "for word guidance")
            if " ".join(after.split()).casefold() == " ".join(expectation["expected_text"].split()).casefold():
                counts["correct"] += 1
            elif changed:
                counts["wrong_on_positive"] += 1
                report.append(f"WRONG  | {before}\n       -> {after}")
            else:
                counts["missed"] += 1
                report.append(f"MISSED | {expectation['item']['heard']} -> {expectation['item']['expected']} | {before}")
        elif kind == "uncertain":
            if changed:
                counts["uncertain_changed"] += 1
                report.append(f"UNCERTAIN CHANGED | {before}\n       -> {after}")
        elif changed:
            counts["false_edit_stored" if kind == "stored" else "false_edit_negative"] += 1
            report.append(f"FALSE ({kind}) | {before}\n       -> {after}")

    positives = sum(1 for e in expectations.values() if e["kind"] == "positive")
    edits = counts["correct"] + counts["wrong_on_positive"] + counts["false_edit_stored"] + counts["false_edit_negative"]
    precision = counts["correct"] / edits if edits else 1.0
    recall = counts["correct"] / positives if positives else 0.0

    print("\n".join(report))
    print(json.dumps({
        "split": args.split, "stored_sentences": len(stored), "positives": positives, "edits": edits,
        **counts, "precision": round(precision, 3), "recall": round(recall, 3), "seconds": round(elapsed, 1)
    }, indent=2))


if __name__ == "__main__":
    main()
