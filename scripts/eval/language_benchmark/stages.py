# Path: scripts/eval/language_benchmark/stages.py

"""
The stages that need the appliance itself: its database, its models, its language.

Each runs inside a process configured as that language's appliance (see `__main__.language_environment`): its own
database, vault and APPLIANCE_LANGUAGE. `app.*` is imported inside the functions so nothing reads `settings` before
that environment is in place.
"""

import json
import logging
import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from scripts.eval.language_benchmark.manifest import LanguageSources
from scripts.eval.language_benchmark.paths import LanguageDirs
from scripts.eval.language_benchmark.wer import align, digit_tokens, normalize, utterance_error_rates

logger = logging.getLogger(__name__)


def _write(dirs: LanguageDirs, name: str, data: dict) -> dict:
    (dirs.results / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str),
                                               encoding="utf-8")
    return data


def _sources_by_file() -> Dict[str, dict]:
    """Every ingested source, keyed by the media file's stem."""
    from sqlalchemy import text
    from app.core.database.service.manager import db_manager

    with db_manager.session() as session:
        rows = session.execute(text(
            "SELECT s.id, s.name, f.file_name FROM sources s JOIN files f ON f.id = s.file_id"
        )).mappings().all()
    return {Path(row["file_name"]).stem: {"source_id": row["id"], "name": row["name"]} for row in rows}


# ---------------------------------------------------------------- ingest

class _VramSampler:
    """Peak GPU memory over a run, sampled from nvidia-smi (the engine's own telemetry is per model)."""

    def __init__(self, interval: float = 1.0):
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(interval,), daemon=True)

    def _run(self, interval: float) -> None:
        while not self._stop.is_set():
            try:
                used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True, timeout=5).stdout.split()
                self.peak_mb = max([self.peak_mb, *map(int, used)])
            except Exception:
                pass
            self._stop.wait(interval)

    def __enter__(self) -> "_VramSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def ingest(dirs: LanguageDirs) -> dict:
    import soundfile as sf
    from app.core.config import settings
    from app.engine.service.bootstrap import OnyxBootstrapper

    media = sorted(dirs.media.glob("*.wav"))
    audio_minutes = sum(sf.info(str(path)).duration for path in media) / 60
    already = set(_sources_by_file()) if _database_has_sources() else set()

    engine = OnyxBootstrapper.build()
    engine.start()
    started = time.perf_counter()
    try:
        with _VramSampler() as vram:
            report = engine.ingest_audios(str(dirs.media))
    finally:
        engine.stop()
    seconds = time.perf_counter() - started

    new_minutes = sum(sf.info(str(p)).duration for p in media if p.stem not in already) / 60
    return _write(dirs, "ingest", {
        "language": settings.APPLIANCE_LANGUAGE,
        "files": [p.name for p in media],
        "audio_minutes": round(audio_minutes, 1),
        "newly_ingested_minutes": round(new_minutes, 1),
        "wall_minutes": round(seconds / 60, 1),
        # Wall time per minute of audio ingested in this run (skipped files cost nothing)
        "real_time_factor": round(seconds / 60 / new_minutes, 2) if new_minutes else None,
        "peak_vram_mb": vram.peak_mb,
        "report": report.model_dump(mode="json"),
        "models": {name: getattr(settings, name) for name in (
            "WHISPER_MODEL_PATH", "DISCOVERY_SCOUT_MODEL_PATH", "CONTEXTUAL_RETRIEVAL_MODEL_PATH",
            "DOMAIN_PHONETIC_CORRECTION_MODEL_PATH", "VISUAL_CONTEXT_SYNTHESIS_MODEL_PATH",
            "QA_GENERATION_MODEL_PATH", "FTS_LANGUAGE_CONFIG")},
    })


def _database_has_sources() -> bool:
    from sqlalchemy import text
    from app.core.database.service.manager import db_manager
    with db_manager.session() as session:
        return bool(session.execute(text("SELECT to_regclass('sources') IS NOT NULL")).scalar())


# ---------------------------------------------------------------- ASR accuracy

def _stage_texts(source_id) -> Dict[str, str]:
    """The transcript after Whisper + hallucination filter + diarization, and the final (phonetic-corrected) text."""
    from app.core.database.service.manager import db_manager
    from app.features.domain_phonetic_correction.data.repositories import DomainPhoneticRepository
    from app.features.sentence_reconstruction.data.repositories import PostgresSentenceRepository
    from app.features.transcription.data.repositories import TranscriptRepository

    texts: Dict[str, str] = {}
    transcript = TranscriptRepository(db_manager).get_by_source_id(source_id)
    if transcript:
        texts["whisper"] = " ".join(s.text for s in sorted(transcript.segments, key=lambda s: s.start_sec))
    final = (DomainPhoneticRepository(db_manager).get_by_source_id(source_id)
             or PostgresSentenceRepository(db_manager).get_by_source_id(source_id))
    if final:
        texts["final"] = " ".join(s.text for s in sorted(final.sentences, key=lambda s: s.start_sec))
    return texts


def _vtt_text(path: Path, max_seconds: float) -> str:
    """Subtitle cues starting before `max_seconds`, tags stripped and rolling duplicates removed."""
    cues, keep, previous = [], False, None
    for line in path.read_text(encoding="utf-8").splitlines():
        timing = re.match(r"(\d+):(\d+):(\d+)\.\d+ -->|(\d+):(\d+)\.\d+ -->", line)
        if timing:
            h, m, s = (timing.group(1), timing.group(2), timing.group(3)) if timing.group(1) else \
                ("0", timing.group(4), timing.group(5))
            keep = int(h) * 3600 + int(m) * 60 + int(s) < max_seconds
            continue
        text = re.sub(r"<[^>]+>", "", line).strip()
        if keep and text and not text.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) and text != previous:
            cues.append(text)
            previous = text
    return " ".join(cues)


def _score(reference: str, hypothesis: str, utterances: Optional[List[dict]] = None) -> dict:
    ref, hyp = normalize(reference), normalize(hypothesis)
    alignment = align(ref, hyp)
    result = {
        "wer": round(alignment.wer, 4), "reference_words": alignment.reference_words,
        "substitutions": alignment.substitutions, "deletions": alignment.deletions,
        "insertions": alignment.insertions, "hypothesis_digit_tokens": digit_tokens(hyp),
        "sample_edits": [{"ref": r, "hyp": h} for r, h in alignment.edits[:40]],
    }
    if utterances:
        rates = utterance_error_rates(alignment, [u["words"] for u in utterances])
        worst = sorted(zip(rates, utterances), key=lambda pair: -pair[0])[:5]
        result["worst_utterances"] = [{"miss_rate": round(rate, 2), "start_sec": u["start_sec"], "text": u["text"]}
                                      for rate, u in worst if rate > 0]
    return result


def measure_asr(dirs: LanguageDirs, code: str, sources: LanguageSources) -> dict:
    ingested = _sources_by_file()
    files: Dict[str, dict] = {}

    for record_path in sorted(dirs.refs.glob("*.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        source = ingested.get(record["name"])
        if source is None:
            files[record["name"]] = {"error": "not ingested (rejected or failed; see ingest.json)"}
            continue
        texts = _stage_texts(source["source_id"])
        files[record["name"]] = {stage: _score(record["text"], text, record["utterances"])
                                 for stage, text in texts.items()}

    for lecture in sources.lectures:
        subtitles = dirs.raw / "lectures" / f"{lecture.stem}.{code}.vtt"
        source = ingested.get(lecture.stem)
        if not subtitles.exists() or source is None:
            continue
        texts = _stage_texts(source["source_id"])
        reference = _vtt_text(subtitles, lecture.max_minutes * 60)
        files[lecture.stem] = {f"{stage}_loose": _score(reference, text) for stage, text in texts.items()}

    summary = {}
    for group in ("fleurs", "parlaspeech", "lecture"):
        for stage in ("whisper", "final", "whisper_loose", "final_loose"):
            scored = [f[stage] for name, f in files.items() if name.startswith(group) and stage in f]
            if scored:
                words = sum(s["reference_words"] for s in scored)
                errors = sum(s["substitutions"] + s["deletions"] + s["insertions"] for s in scored)
                summary[f"{group}/{stage}"] = {"wer": round(errors / words, 4) if words else None,
                                               "accuracy": round(1 - errors / words, 4) if words else None,
                                               "reference_words": words, "files": len(scored)}
    return _write(dirs, "asr", {"summary": summary, "files": files})


# ---------------------------------------------------------------- language guard

def guard(dirs: LanguageDirs) -> dict:
    """Whisper's language ID on every media file, and which packs' guards would accept it."""
    import soundfile as sf
    from app.core.config import settings
    from app.core.database.service.manager import db_manager
    from app.core.language.data.toml_pack_repository import TomlLanguagePackRepository
    from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
    from app.core.model_manager.domain.entities import ModelManifest
    from app.core.model_manager.domain.enums import ModelType
    from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
    from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
    from app.features.vad.data.repositories import VadRepository
    from app.features.vad.domain.entities import VadMap, VadSegment

    repository = TomlLanguagePackRepository(settings.LANGUAGE_PACKS_DIR)
    packs = {code: repository.load(code) for code in repository.available_codes()}
    ingested = _sources_by_file()
    vad_repo = VadRepository(db_manager)

    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    adapter = FasterWhisperAdapter()
    loaded = orchestrator.acquire_model(ModelManifest(
        model_type=ModelType.TRANSCRIPTION, model_path=settings.WHISPER_MODEL_PATH,
        required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB), adapter)
    files = {}
    try:
        for path in sorted(dirs.media.glob("*.wav")):
            source = ingested.get(path.stem)
            # The VAD map the real guard used when there is one, else the whole file
            vad = vad_repo.get_map_by_source_id(source["source_id"]) if source else None
            if not vad or not vad.segments:
                vad = VadMap(source_id=source["source_id"] if source else __import__("uuid").uuid4(),
                             segments=[VadSegment(start_sec=0.0, end_sec=sf.info(str(path)).duration,
                                                  confidence=1.0)])
            detection = adapter.detect_language(loaded, str(path), vad)
            language, probability = detection if detection else (None, 0.0)
            files[path.name] = {
                "detected": language, "probability": round(probability, 3),
                "accepted_by": {code: bool(detection) and (
                    language in pack.transcription.accepted_detections
                    or probability < pack.transcription.min_detection_probability)
                    for code, pack in packs.items()},
            }
    finally:
        orchestrator.release_model(loaded, adapter)
    return _write(dirs, "guard", {"files": files})


# ---------------------------------------------------------------- questions

# Questions no benchmark material can answer: a grounded appliance must refuse every one
UNANSWERABLE = {
    "sr": ["Koliko košta karta za voz od Beograda do Novog Sada?", "Ko je osvojio Svetsko prvenstvo u fudbalu 1998?",
           "Kako se pravi sarma?", "Koja je glavna reka u Parizu?", "Kada je rođen Nikola Tesla?"],
    "sl": ["Koliko stane vozovnica za vlak od Ljubljane do Maribora?", "Kdo je osvojil svetovno prvenstvo v nogometu leta 1998?",
           "Kako se pripravi potica?", "Katera reka teče skozi Pariz?", "Kdaj se je rodil France Prešeren?"],
    "en": ["How much is a train ticket from London to Manchester?", "Who won the 1998 football World Cup?",
           "How do you bake sourdough bread?", "Which river runs through Paris?", "When was Nikola Tesla born?"],
}

_QUESTION_PROMPT = """You write evaluation questions for a search system over transcribed {language} recordings.

Read the transcript excerpt below. Write ONE question, in {language}, that this excerpt answers directly and
specifically. The question must:
- be answerable from this excerpt alone, about its concrete content (a fact, number, name, reason or claim);
- make sense to someone who has not seen the excerpt (never say "the excerpt", "the speaker" or "in this recording");
- be a natural question a user would type, 5 to 20 words.
Reply with the question only.

EXCERPT:
{excerpt}"""

_LATIN_TO_CYRILLIC = [("dž", "џ"), ("lj", "љ"), ("nj", "њ"), ("Dž", "Џ"), ("Lj", "Љ"), ("Nj", "Њ"),
                      ("DŽ", "Џ"), ("LJ", "Љ"), ("NJ", "Њ")]
_LATIN_LETTERS = dict(zip("abcčćdđefghijklmnoprsštuvzž", "абцчћдђефгхијклмнопрсштувзж"))


def to_cyrillic(text: str) -> str:
    """Serbian Latin -> Cyrillic for test queries only: digraph ambiguity ("injekcija") is irrelevant here."""
    for latin, cyrillic in _LATIN_TO_CYRILLIC:
        text = text.replace(latin, cyrillic)
    out = []
    for ch in text:
        cyrillic = _LATIN_LETTERS.get(ch.lower())
        out.append(ch if cyrillic is None else (cyrillic.upper() if ch.isupper() else cyrillic))
    return "".join(out)


def build_labels(code: str, language: str, questions: List[dict], cyrillic_copies: bool) -> dict:
    """Questions in rag_eval's format: `relevant` holds chunk-id prefixes, any one of which answers the question."""
    answerable = [{"id": f"a{i:02d}", "question": q["question"], "relevant": [str(q["chunk_id"])[:8]],
                   "source": q["source"]} for i, q in enumerate(questions, 1)]
    if cyrillic_copies:
        answerable += [{**item, "id": f"{item['id']}c", "question": to_cyrillic(item["question"])}
                       for item in list(answerable)]
    return {
        "_comment": f"Generated {language} questions over the onyx_bench_{code} database: each was written by the "
                    f"answer model from the one chunk listed in `relevant`. Review before trusting. Ids ending in "
                    f"'c' are Cyrillic copies (a Serbian index is Latin).",
        "answerable": answerable,
        "unanswerable": [{"id": f"u{i:02d}", "question": q} for i, q in enumerate(UNANSWERABLE[code], 1)],
    }


def gen_questions(dirs: LanguageDirs, code: str, per_source: int = 10, seed: int = 7) -> dict:
    """Writes labels.json from random chunks of the lecture and ParlaSpeech sources (not the FLEURS sentence mix)."""
    from sqlalchemy import text
    from app.core.config import settings
    from app.core.database.service.manager import db_manager
    from app.core.language.service.language_api import get_language_pack
    from app.core.model_manager.utils.llama_cpp_chat import build_llama, chat_completion, strip_reasoning

    pack = get_language_pack()
    with db_manager.session() as session:
        rows = session.execute(text(
            "SELECT c.chunk_id, c.raw_text, f.file_name FROM vector_chunks c "
            "JOIN sources s ON s.id = c.source_id JOIN files f ON f.id = s.file_id "
            "WHERE f.file_name LIKE 'lecture_%' OR f.file_name LIKE 'parlaspeech_%' ORDER BY c.chunk_id"
        )).mappings().all()

    by_source: Dict[str, list] = {}
    for row in rows:
        if len(row["raw_text"].split()) >= 40:  # too short a chunk rarely holds a specific answer
            by_source.setdefault(row["file_name"], []).append(row)
    rng = random.Random(seed)
    picked = [row for chunks in by_source.values() for row in rng.sample(chunks, min(per_source, len(chunks)))]

    llm = build_llama(settings.QA_GENERATION_MODEL_PATH, n_ctx=8192, kv_cache_type=settings.QA_LLM_KV_CACHE_TYPE)
    questions = []
    try:
        for row in picked:
            reply = chat_completion(llm, [{"role": "user", "content": _QUESTION_PROMPT.format(
                language=pack.name, excerpt=row["raw_text"][:6000])}], enable_thinking=False,
                temperature=0.3, max_tokens=120)
            question = strip_reasoning(reply["choices"][0]["message"]["content"]).strip().strip('"').splitlines()[0]
            if question:
                questions.append({"question": question, "chunk_id": row["chunk_id"], "source": row["file_name"]})
    finally:
        llm.close()

    labels = build_labels(code, pack.name, questions, cyrillic_copies=pack.transcription.script == "latin")
    dirs.labels.write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    return labels


# ---------------------------------------------------------------- retrieval and answers

def _rank_summary(items: Dict[str, dict]) -> dict:
    n = len(items)
    ranks = [item["rerank_rank"] for item in items.values()]
    return {"questions": n, "rerank_hit@1": sum(r == 1 for r in ranks), "rerank_hit@3": sum(bool(r and r <= 3) for r in ranks),
            "rerank_hit@10": sum(bool(r) for r in ranks),
            "mrr": round(sum(1 / r for r in ranks if r) / n, 3) if n else 0.0,
            "scrub_relevant_survives": sum(bool(item["scrub_hit"]) for item in items.values())}


def measure_rag(dirs: LanguageDirs) -> dict:
    from scripts.eval.rag_eval import run_answer, run_retrieval

    labels = json.loads(dirs.labels.read_text(encoding="utf-8"))
    retrieval = run_retrieval(labels, None, 10, None)
    answers = run_answer(labels, None, "chat", 10)

    # Serbian: the Latin questions and their Cyrillic copies, scored separately from the same run
    split = None
    items = retrieval.get("items", {})
    if any(qid.endswith("c") for qid in items):
        split = {script: _rank_summary({qid: r for qid, r in items.items()
                                        if qid.startswith("a") and qid.endswith("c") == (script == "cyrillic")})
                 for script in ("latin", "cyrillic")}
    return _write(dirs, "rag", {"retrieval": retrieval, "answers": answers, "script_split": split})
