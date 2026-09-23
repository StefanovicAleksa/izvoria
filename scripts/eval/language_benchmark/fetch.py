# Path: scripts/eval/language_benchmark/fetch.py

"""
Downloads benchmark material. Everything is idempotent: a file already on disk is not fetched again.

The Hugging Face datasets are read as plain files (`hf_hub_download`), not through `datasets.load_dataset`:
FLEURS still ships a loading script, which current `datasets` releases refuse to run.
"""

import csv
import json
import logging
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from scripts.eval.language_benchmark.manifest import LanguageSources, Lecture
from scripts.eval.language_benchmark.paths import LanguageDirs

logger = logging.getLogger(__name__)

FLEURS_REPO = "google/fleurs"
PARLASPEECH_REPO = "classla/ParlaSpeech-RS"
SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Utterance:
    """One reference recording: its audio file and the human transcript of it."""
    audio: Path
    text: str
    seconds: float


# ---------------------------------------------------------------- FLEURS

def parse_fleurs_tsv(rows: Iterable[List[str]]) -> List[Dict[str, str]]:
    """
    FLEURS TSV rows: id, file, raw transcription, normalized transcription, characters, samples, gender. Every
    sentence is read by several speakers; one recording per sentence id is kept, so no sentence counts twice.
    """
    seen, kept = set(), []
    for row in rows:
        if len(row) < 6:
            continue
        sentence_id, file_name, raw, _normalized, _chars, samples = row[:6]
        if sentence_id in seen:
            continue
        seen.add(sentence_id)
        kept.append({"id": sentence_id, "file": file_name, "text": raw, "seconds": int(samples) / SAMPLE_RATE})
    return kept


def fetch_fleurs(config: str, minutes: int, dirs: LanguageDirs) -> List[Utterance]:
    from huggingface_hub import hf_hub_download

    target = dirs.raw / "fleurs"
    index_path = target / "utterances.json"
    if index_path.exists():
        return [Utterance(Path(u["audio"]), u["text"], u["seconds"]) for u in json.loads(index_path.read_text())]

    target.mkdir(parents=True, exist_ok=True)
    tsv = hf_hub_download(FLEURS_REPO, f"data/{config}/test.tsv", repo_type="dataset")
    with open(tsv, encoding="utf-8", newline="") as f:
        rows = parse_fleurs_tsv(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))

    budget, chosen = minutes * 60, []
    for row in rows:
        if budget <= 0:
            break
        chosen.append(row)
        budget -= row["seconds"]

    logger.info(f"FLEURS {config}: downloading the test audio for {len(chosen)} sentences ({minutes} min).")
    archive = hf_hub_download(FLEURS_REPO, f"data/{config}/audio/test.tar.gz", repo_type="dataset")
    wanted = {row["file"]: row for row in chosen}
    with tarfile.open(archive) as tar:
        for member in tar:
            name = Path(member.name).name
            if member.isfile() and name in wanted:
                with tar.extractfile(member) as src, open(target / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    utterances = [Utterance(target / row["file"], row["text"], row["seconds"])
                  for row in chosen if (target / row["file"]).exists()]
    index_path.write_text(json.dumps([{"audio": str(u.audio), "text": u.text, "seconds": u.seconds}
                                      for u in utterances], ensure_ascii=False, indent=1))
    return utterances


# ---------------------------------------------------------------- ParlaSpeech-RS

def pick_speaker_window(speakers: List[str], seconds: List[float], minutes: float) -> List[int]:
    """
    Row indexes of the most frequent speaker's consecutive segments, up to `minutes`: one person talking at length
    is the closest a sentence-segmented corpus gets to a real long-form recording.
    """
    if not speakers:
        return []
    counts: Dict[str, int] = {}
    for speaker in speakers:
        counts[speaker] = counts.get(speaker, 0) + 1
    top = max(counts, key=counts.get)
    budget, rows = minutes * 60, []
    for index, speaker in enumerate(speakers):
        if speaker != top:
            continue
        if budget <= 0:
            break
        rows.append(index)
        budget -= seconds[index]
    return rows


def fetch_parlaspeech(shard: str, minutes: int, dirs: LanguageDirs) -> List[Utterance]:
    import io

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    target = dirs.raw / "parlaspeech"
    index_path = target / "utterances.json"
    if index_path.exists():
        return [Utterance(Path(u["audio"]), u["text"], u["seconds"]) for u in json.loads(index_path.read_text())]

    target.mkdir(parents=True, exist_ok=True)
    logger.info(f"ParlaSpeech-RS: downloading {shard} (~400 MB).")
    parquet = pq.ParquetFile(hf_hub_download(PARLASPEECH_REPO, shard, repo_type="dataset"))
    meta = parquet.read(columns=["speaker_name", "audio_length"]).to_pydict()
    rows = set(pick_speaker_window(meta["speaker_name"], meta["audio_length"], minutes))

    utterances, offset = [], 0
    for batch in parquet.iter_batches(columns=["audio", "text_normalised", "audio_length"], batch_size=256):
        for local, record in enumerate(batch.to_pylist()):
            index = offset + local
            if index not in rows:
                continue
            audio, rate = sf.read(io.BytesIO(record["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if rate != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)
            path = target / f"{index:06d}.wav"
            sf.write(path, np.asarray(audio), SAMPLE_RATE)
            utterances.append(Utterance(path, record["text_normalised"], len(audio) / SAMPLE_RATE))
        offset += batch.num_rows

    index_path.write_text(json.dumps([{"audio": str(u.audio), "text": u.text, "seconds": u.seconds}
                                      for u in utterances], ensure_ascii=False, indent=1))
    return utterances


# ---------------------------------------------------------------- YouTube lectures

def fetch_lecture(lecture: Lecture, language: str, dirs: LanguageDirs) -> Path:
    """The first `max_minutes` of a lecture as 16 kHz mono WAV, plus human subtitles in the language if any."""
    target = dirs.raw / "lectures"
    target.mkdir(parents=True, exist_ok=True)
    audio = target / f"{lecture.stem}.wav"
    if audio.exists():
        return audio

    ytdlp = Path(sys.executable).with_name("yt-dlp")
    command = [
        str(ytdlp if ytdlp.exists() else "yt-dlp"),
        f"https://www.youtube.com/watch?v={lecture.youtube_id}",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ExtractAudio:-ar {SAMPLE_RATE} -ac 1",
        "--download-sections", f"*0:00-{lecture.max_minutes}:00",
        # Human subtitles only: auto-generated captions are another ASR system's guesses, not a reference
        "--write-subs", "--no-write-auto-subs", "--sub-langs", language, "--sub-format", "vtt",
        "-o", str(target / f"{lecture.stem}.%(ext)s"),
        "--no-progress", "--quiet", "--no-warnings",
    ]
    logger.info(f"Downloading lecture {lecture.youtube_id} ({lecture.max_minutes} min): {lecture.title}")
    subprocess.run(command, check=True)
    if not audio.exists():
        raise FileNotFoundError(f"yt-dlp finished but {audio} is missing")
    return audio


def lecture_subtitles(lecture: Lecture, language: str, dirs: LanguageDirs) -> Path:
    return dirs.raw / "lectures" / f"{lecture.stem}.{language}.vtt"


def fetch_all(code: str, sources: LanguageSources, dirs: LanguageDirs) -> Dict[str, object]:
    fleurs = fetch_fleurs(sources.fleurs_config, sources.fleurs_minutes, dirs)
    parla = (fetch_parlaspeech(sources.parlaspeech_shard, sources.parlaspeech_minutes, dirs)
             if sources.parlaspeech_shard else [])
    lectures = [fetch_lecture(lecture, code, dirs) for lecture in sources.lectures]
    return {"fleurs": fleurs, "parlaspeech": parla, "lectures": lectures}
