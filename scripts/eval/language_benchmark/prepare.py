# Path: scripts/eval/language_benchmark/prepare.py

"""
Turns downloaded material into the files Onyx ingests.

Reference utterances are 5-20 s each; ingesting each one alone would never exercise what a real recording does
(VAD over long audio, batched decoding, the hallucination filter, sentence reconstruction across utterances). So
they are stitched into ~10 minute files with a short silence between them, and the reference transcript is the
utterances' text in the same order. Per-utterance offsets are kept so the worst utterances can be listed.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from scripts.eval.language_benchmark.fetch import SAMPLE_RATE, Utterance
from scripts.eval.language_benchmark.paths import LanguageDirs
from scripts.eval.language_benchmark.wer import normalize

GAP_SECONDS = 0.8
FILE_MINUTES = 10


@dataclass(frozen=True)
class StitchPlan:
    name: str
    utterances: List[Utterance]


def plan_stitches(prefix: str, utterances: Sequence[Utterance], file_minutes: float = FILE_MINUTES) -> List[StitchPlan]:
    """Groups utterances, in order, into files of at most `file_minutes` (a single long utterance still fits one)."""
    plans, current, length = [], [], 0.0
    for utterance in utterances:
        added = utterance.seconds + (GAP_SECONDS if current else 0.0)
        if current and length + added > file_minutes * 60:
            plans.append(StitchPlan(f"{prefix}_{len(plans) + 1:02d}", current))
            current, length, added = [], 0.0, utterance.seconds
        current.append(utterance)
        length += added
    if current:
        plans.append(StitchPlan(f"{prefix}_{len(plans) + 1:02d}", current))
    return plans


def reference_record(plan: StitchPlan) -> dict:
    """Reference text plus, per utterance, its start in the stitched audio and its normalised word count."""
    offsets, start = [], 0.0
    for utterance in plan.utterances:
        offsets.append({"start_sec": round(start, 3), "seconds": round(utterance.seconds, 3),
                        "words": len(normalize(utterance.text)), "text": utterance.text})
        start += utterance.seconds + GAP_SECONDS
    return {"name": plan.name, "text": " ".join(u.text for u in plan.utterances), "utterances": offsets}


def write_stitch(plan: StitchPlan, dirs: LanguageDirs) -> Path:
    import numpy as np
    import soundfile as sf

    media = dirs.media / f"{plan.name}.wav"
    record_path = dirs.refs / f"{plan.name}.json"
    if media.exists() and record_path.exists():
        return media

    gap = np.zeros(int(GAP_SECONDS * SAMPLE_RATE), dtype="float32")
    pieces = []
    for index, utterance in enumerate(plan.utterances):
        audio, rate = sf.read(utterance.audio, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
        if index:
            pieces.append(gap)
        pieces.append(audio)
    sf.write(media, np.concatenate(pieces), SAMPLE_RATE)
    record_path.write_text(json.dumps(reference_record(plan), ensure_ascii=False, indent=1), encoding="utf-8")
    return media


def prepare_all(fetched: dict, dirs: LanguageDirs) -> List[Path]:
    media = []
    for prefix in ("fleurs", "parlaspeech"):
        for plan in plan_stitches(prefix, fetched.get(prefix, [])):
            media.append(write_stitch(plan, dirs))
    for lecture_audio in fetched.get("lectures", []):
        target = dirs.media / lecture_audio.name
        if not target.exists():
            shutil.copy2(lecture_audio, target)
        media.append(target)
    return media
