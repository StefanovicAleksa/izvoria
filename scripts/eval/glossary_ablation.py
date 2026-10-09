# Path: scripts/eval/glossary_ablation.py
"""
What is a glossary worth? Transcribes already-ingested sources four ways and scores each against a reference:

    A  raw Whisper (no hotwords)
    B  Whisper with the glossary's hotwords (sub-domains routed per source, as ingestion does)
    C  B + domain phonetic correction
    D  A + domain phonetic correction (correction on its own, without hotwords)

For each: word error rate, and the **term error rate** -- glossary-term occurrences in the reference (a term, its
`term_details.forms`, or its plural) not reproduced exactly. The **ceiling** is the share of a configuration's word
errors that fall on glossary-term words: the most phonetic correction could ever recover there. C-vs-B and D-vs-A
count term occurrences correction fixed or broke, and other words it changed.

Comparison is on normalised words (`language_benchmark/wer.py`): case and punctuation don't count, so a term written
"bund" instead of "Bund" is correct here.

Runs the real TranscriptionAPI, DomainAdaptationAPI and DomainPhoneticCorrectionAPI with in-memory repositories
against a database that already holds the sources (VAD maps, extracted audio). Writes nothing to the database.

Reference text per source, matched by the media file's stem:
- `--refs <dir>` of `<stem>.txt` files, or
- by default the language benchmark's references (FLEURS/ParlaSpeech `refs/*.json`, lecture subtitles).

Without references (`--no-reference --sources <stem> ...`) nothing can be scored. The run lists instead every place
where two configurations disagree on a glossary term (A vs B: what hotwords changed; B vs C and A vs D: what
correction changed), with context, so a person can judge which side is right.

Run from the repo root, in the foreground (GPU), e.g. on the benchmark database:
    venv/bin/python -m scripts.eval.glossary_ablation --language en --industry prop_trading_macro \\
        --database onyx_bench_en [--oracle-subdomains] [--glossary-path <root holding <lang>/>] [--refs <dir>]

Results: onyx_data/benchmarks/glossary/<lang>/<industry>/{results.json,report.md}.
"""

import argparse
import json
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from scripts.eval.language_benchmark.paths import BENCH_ROOT, LanguageDirs
from scripts.eval.language_benchmark.wer import align, normalize

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

CONFIGS = ("A_raw", "B_hotwords", "C_hotwords_phonetic", "D_raw_phonetic")


# ---------------------------------------------------------------- scoring (pure)

def term_variants(glossary: dict, plural_suffix: str = "") -> Dict[Tuple[str, ...], str]:
    """Normalised token sequence -> the glossary term it is an occurrence of, over every term of the glossary."""
    terms = list(glossary.get("base_terms", []))
    for entry in glossary.get("sub_domains", {}).values():
        terms += (entry.get("anchor_words", []) + entry.get("terms", [])) if isinstance(entry, dict) else list(entry)
    details = glossary.get("term_details", {})

    variants: Dict[Tuple[str, ...], str] = {}
    for term in terms:
        spellings = [term] + list(details.get(term, {}).get("forms", []))
        for spelling in spellings:
            tokens = tuple(normalize(spelling))
            if not tokens:
                continue
            variants.setdefault(tokens, term)
            if plural_suffix:
                variants.setdefault(tokens[:-1] + (tokens[-1] + plural_suffix,), term)
    return variants


def find_occurrences(tokens: Sequence[str], variants: Dict[Tuple[str, ...], str]) -> List[Tuple[int, int, str]]:
    """Non-overlapping (start, end, term) spans, left to right, longest variant first at each position."""
    lengths = sorted({len(v) for v in variants}, reverse=True)
    found, i = [], 0
    while i < len(tokens):
        for length in (n for n in lengths if i + n <= len(tokens)):
            term = variants.get(tuple(tokens[i:i + length]))
            if term is not None:
                found.append((i, i + length, term))
                i += length
                break
        else:
            i += 1
    return found


@dataclass
class Score:
    wer: float
    reference_words: int
    errors: int
    term_occurrences: int
    term_errors: int
    term_word_errors: int
    # Per reference word: matched exactly. Per occurrence: (start, end, term, reproduced exactly)
    reference_correct: List[bool]
    occurrences: List[Tuple[int, int, str, bool]]

    @property
    def ter(self) -> Optional[float]:
        return self.term_errors / self.term_occurrences if self.term_occurrences else None

    @property
    def ceiling(self) -> Optional[float]:
        return self.term_word_errors / self.errors if self.errors else None

    def summary(self) -> dict:
        return {"wer": round(self.wer, 4), "reference_words": self.reference_words, "errors": self.errors,
                "term_occurrences": self.term_occurrences, "term_errors": self.term_errors,
                "ter": _round(self.ter), "term_word_errors": self.term_word_errors, "ceiling": _round(self.ceiling)}


def score(reference: str, hypothesis: str, variants: Dict[Tuple[str, ...], str]) -> Score:
    ref = normalize(reference)
    alignment = align(ref, normalize(hypothesis))
    occurrences = [(s, e, term, all(alignment.reference_correct[s:e])) for s, e, term in find_occurrences(ref, variants)]
    term_words = {i for s, e, _, _ in occurrences for i in range(s, e)}
    return Score(
        wer=alignment.wer, reference_words=alignment.reference_words, errors=alignment.errors,
        term_occurrences=len(occurrences), term_errors=sum(1 for *_, ok in occurrences if not ok),
        term_word_errors=sum(1 for i in term_words if not alignment.reference_correct[i]),
        reference_correct=alignment.reference_correct, occurrences=occurrences,
    )


def compare(before: Score, after: Score) -> dict:
    """What a correction pass changed, judged against the reference (both scores share the same reference)."""
    fixed = [term for (_, _, term, ok_b), (*_, ok_a) in zip(before.occurrences, after.occurrences) if ok_a and not ok_b]
    broke = [term for (_, _, term, ok_b), (*_, ok_a) in zip(before.occurrences, after.occurrences) if ok_b and not ok_a]
    term_words = {i for s, e, _, _ in before.occurrences for i in range(s, e)}
    pairs = [(b, a) for i, (b, a) in enumerate(zip(before.reference_correct, after.reference_correct))
             if i not in term_words]
    return {"terms_fixed": len(fixed), "terms_broken": len(broke),
            "other_words_fixed": sum(1 for b, a in pairs if a and not b),
            "other_words_broken": sum(1 for b, a in pairs if b and not a),
            "fixed": fixed, "broken": broke}


def term_disagreements(before: str, after: str, variants: Dict[Tuple[str, ...], str],
                       context: int = 8) -> List[dict]:
    """
    Regions where two transcripts of the same audio differ and either side holds a glossary-term occurrence that
    overlaps the region: what one configuration changed about the terms, with `context` words around it.
    """
    from difflib import SequenceMatcher

    a, b = normalize(before), normalize(after)
    occ_a, occ_b = find_occurrences(a, variants), find_occurrences(b, variants)

    def touching(occurrences, start, end):
        return sorted({term for s, e, term in occurrences if s < end and start < e})

    found = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        terms = touching(occ_a, i1, i2) + [t for t in touching(occ_b, j1, j2) if t not in touching(occ_a, i1, i2)]
        if not terms:
            continue
        found.append({
            "terms": terms,
            "before": " ".join(a[i1:i2]), "after": " ".join(b[j1:j2]),
            "context": " ".join(a[max(0, i1 - context):i1]) + " [" + " ".join(a[i1:i2]) + " -> "
                       + " ".join(b[j1:j2]) + "] " + " ".join(a[i2:i2 + context]),
        })
    return found


def aggregate(scores: Sequence[Score]) -> dict:
    words = sum(s.reference_words for s in scores)
    errors = sum(s.errors for s in scores)
    occurrences = sum(s.term_occurrences for s in scores)
    term_errors = sum(s.term_errors for s in scores)
    term_word_errors = sum(s.term_word_errors for s in scores)
    return {"wer": _round(errors / words if words else None), "reference_words": words, "errors": errors,
            "term_occurrences": occurrences, "term_errors": term_errors,
            "ter": _round(term_errors / occurrences if occurrences else None),
            "term_word_errors": term_word_errors, "ceiling": _round(term_word_errors / errors if errors else None)}


def aggregate_comparisons(comparisons: Sequence[dict]) -> dict:
    keys = ("terms_fixed", "terms_broken", "other_words_fixed", "other_words_broken")
    return {key: sum(c[key] for c in comparisons) for key in keys}


def render_report(language: str, industry: str, result: dict) -> str:
    totals = result["totals"]
    lines = [
        f"# Glossary ablation: {industry} ({language})", "",
        f"{len(result['sources'])} source(s), {totals['A_raw']['reference_words']} reference words, "
        f"{totals['A_raw']['term_occurrences']} glossary-term occurrences. "
        f"Sub-domains: {'all (oracle)' if result['oracle_subdomains'] else 'routed per source'}.", "",
        "| Config | WER | Term occurrences | Term errors | TER | Ceiling |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for config in CONFIGS:
        t = totals[config]
        lines.append(f"| {config} | {_pct(t['wer'])} | {t['term_occurrences']} | {t['term_errors']} | "
                     f"{_pct(t['ter'])} | {_pct(t['ceiling'])} |")
    lines += ["", "Ceiling: share of the configuration's word errors on glossary-term words.", "",
              "| Correction | Terms fixed | Terms broken | Other words fixed | Other words broken |",
              "| --- | --- | --- | --- | --- |"]
    for name, c in result["corrections"].items():
        lines.append(f"| {name} | {c['terms_fixed']} | {c['terms_broken']} | {c['other_words_fixed']} | "
                     f"{c['other_words_broken']} |")
    return "\n".join(lines) + "\n"


PAIRS = (("A_raw", "B_hotwords"), ("B_hotwords", "C_hotwords_phonetic"), ("A_raw", "D_raw_phonetic"))


def render_disagreements(language: str, industry: str, result: dict) -> str:
    lines = [f"# Glossary ablation without references: {industry} ({language})", "",
             f"{len(result['sources'])} source(s). Each line: the term(s) involved, then "
             f"`context [first -> second] context`.", ""]
    for first, second in PAIRS:
        key = f"{first} -> {second}"
        lines += [f"## {key}", ""]
        total = 0
        for stem, data in result["sources"].items():
            items = data["disagreements"][key]
            total += len(items)
            if items:
                lines.append(f"### {stem} ({len(items)})")
                lines += [f"- {', '.join(d['terms'])}: {d['context']}" for d in items]
                lines.append("")
        lines += [f"Total: {total}", ""]
    return "\n".join(lines) + "\n"


def _round(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None


def _pct(value: Optional[float]) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


# ---------------------------------------------------------------- references

def load_references(language: str, refs_dir: Optional[Path]) -> Dict[str, str]:
    """Media stem -> reference text."""
    if refs_dir:
        return {path.stem: path.read_text(encoding="utf-8") for path in sorted(refs_dir.glob("*.txt"))}

    from scripts.eval.language_benchmark.manifest import load_manifest
    from scripts.eval.language_benchmark.stages import _vtt_text

    dirs = LanguageDirs(language)
    references = {}
    for path in sorted(dirs.refs.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        references[record["name"]] = record["text"]
    sources = load_manifest().get(language)
    for lecture in (sources.lectures if sources else []):
        subtitles = dirs.raw / "lectures" / f"{lecture.stem}.{language}.vtt"
        if subtitles.exists():
            references[lecture.stem] = _vtt_text(subtitles, lecture.max_minutes * 60)
    return references


# ---------------------------------------------------------------- GPU run

@dataclass
class Source:
    source_id: uuid.UUID
    stem: str
    audio_path: str


def _ingested_sources() -> List[Source]:
    from sqlalchemy import text
    from app.core.database.service.manager import db_manager

    with db_manager.session() as session:
        rows = session.execute(text(
            "SELECT s.id, sf.file_name, af.file_path FROM sources s "
            "JOIN files sf ON sf.id = s.file_id "
            "JOIN audio_extractions a ON a.source_id = s.id "
            "JOIN files af ON af.id = a.extracted_file_id"
        )).all()
    return [Source(source_id=row[0], stem=Path(row[1]).stem, audio_path=row[2]) for row in rows]


class _NoTranscriptCache:
    """TranscriptionAPI's repository: never returns a stored transcript, never writes one."""

    def get_by_source_id(self, source_id):
        return None

    def save(self, transcript):
        return transcript

    def update(self, transcript):
        return transcript


class _NoGlossaryCache:
    def get_glossary_by_source_id(self, source_id):
        return None

    def save_glossary(self, glossary):
        return glossary

    def delete_glossary_by_source_id(self, source_id):
        return False


def _release_if_resident(orchestrator, adapter, model_type, model_path: str, vram_mb: float) -> None:
    """Releases a model an API acquired and kept (release_after=False); acquiring it again hands out that instance."""
    from app.core.model_manager.domain.entities import ModelManifest

    if model_type in orchestrator.resident_types():
        manifest = ModelManifest(model_type=model_type, model_path=model_path, required_vram_mb=vram_mb)
        orchestrator.release_model(orchestrator.acquire_model(manifest, adapter), adapter, flush=True)


def _transcribe_all(api, orchestrator, adapter, sources: List[Source],
                    hotwords: Dict[uuid.UUID, Optional[str]]) -> Dict[uuid.UUID, list]:
    """One Whisper load for the whole list, released at the end (a source with no speech never loads it)."""
    from app.core.config import settings
    from app.core.model_manager.domain.enums import ModelType

    segments = {}
    for source in sources:
        transcript = api.process_audio(source.source_id, source.audio_path, hotwords=hotwords.get(source.source_id))
        segments[source.source_id] = transcript.segments
    _release_if_resident(orchestrator, adapter, ModelType.TRANSCRIPTION, settings.WHISPER_MODEL_PATH,
                         settings.WHISPER_VRAM_REQUIREMENT_MB)
    return segments


def _correct_all(api, orchestrator, adapter, segments: Dict[uuid.UUID, list], industry: str,
                 sub_domains: Dict[uuid.UUID, List[str]]) -> Dict[uuid.UUID, str]:
    """
    The verifier loads only for a source with candidates, so "release after the last source" could leave it
    loaded; it is kept across sources and released once at the end, whichever source loaded it.
    """
    from app.core.config import settings
    from app.core.model_manager.domain.enums import ModelType
    from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence, ReconstructedTranscript

    texts = {}
    for source_id, source_segments in segments.items():
        sentences = [ReconstructedSentence(text=s.text.strip(), speaker_id="speaker_0", start_sec=s.start_sec,
                                           end_sec=s.end_sec, words=s.words) for s in source_segments if s.text.strip()]
        # A fresh id: the in-memory repository never returns a cached result anyway, but nothing should collide
        transcript = ReconstructedTranscript(source_id=uuid.uuid4(), sentences=sentences)
        result = api.execute(transcript, industry, sub_domains[source_id], release_after=False)
        texts[source_id] = " ".join(s.text for s in result.sentences)

    _release_if_resident(orchestrator, adapter, ModelType.DOMAIN_PHONETIC_CORRECTION,
                         settings.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH,
                         settings.DOMAIN_PHONETIC_CORRECTION_VRAM_REQUIREMENT_MB)
    return texts


def transcribe_configs(industry: str, stems: Sequence[str], oracle: bool) -> dict:
    """Runs A-D over the ingested sources whose media stem is in `stems`; returns their texts and routing."""
    from app.core.config import settings
    from app.core.database.service.manager import db_manager
    from app.core.language.service.language_api import get_language_pack
    from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
    from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
    from app.features.domain_adaptation.data.llama_adapter import LlamaAdapter
    from app.features.domain_adaptation.data.scout_adapter import ScoutWhisperAdapter
    from app.features.domain_adaptation.service.adaptation_api import DomainAdaptationAPI
    from app.features.domain_adaptation.utils.glossary_builder import GlossaryBuilder
    from app.features.domain_adaptation.utils.smart_sampler import SmartSampler
    from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter
    from app.features.domain_phonetic_correction.service.correction_api import DomainPhoneticCorrectionAPI
    from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
    from app.features.transcription.service.transcription_api import TranscriptionAPI
    from app.features.vad.data.repositories import VadRepository
    from scripts.eval.phonetic_eval import InMemoryPhoneticRepository

    glossary_path = settings.GLOSSARY_DIR / f"{industry.lower()}.json"
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))
    variants = term_variants(glossary, get_language_pack().phonetic.plural_suffix)

    # The same file ingested twice is one source for this purpose
    by_stem: Dict[str, Source] = {}
    for source in _ingested_sources():
        if source.stem in stems:
            by_stem.setdefault(source.stem, source)
    sources = list(by_stem.values())
    if not sources:
        raise SystemExit(f"None of the {len(stems)} requested source(s) is ingested in {settings.POSTGRES_DB}.")
    print(f"{len(sources)} source(s) in {settings.POSTGRES_DB}; glossary {glossary_path}", flush=True)

    orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
    vad_repo = VadRepository(db_manager)

    # Sub-domains and hotwords per source, as ingestion would choose them
    sub_domains: Dict[uuid.UUID, List[str]] = {}
    hotwords: Dict[uuid.UUID, Optional[str]] = {}
    if oracle:
        labels = list(glossary.get("sub_domains", {}))
        hotword_string, _ = GlossaryBuilder.build_injection_payloads(industry, labels)
        for source in sources:
            sub_domains[source.source_id], hotwords[source.source_id] = labels, hotword_string
    else:
        adaptation = DomainAdaptationAPI(_NoGlossaryCache(), orchestrator, ScoutWhisperAdapter(), LlamaAdapter(),
                                         SmartSampler())
        for source in sources:
            routed = adaptation.process_domain_adaptation(source.source_id, source.audio_path,
                                                          vad_repo.get_map_by_source_id(source.source_id))
            sub_domains[source.source_id], hotwords[source.source_id] = routed.sub_domains, routed.hotword_string
    print("Routing done", flush=True)

    whisper = FasterWhisperAdapter()
    transcription = TranscriptionAPI(_NoTranscriptCache(), vad_repo, orchestrator, whisper)
    raw = _transcribe_all(transcription, orchestrator, whisper, sources, {})
    print("A (raw) transcribed", flush=True)
    biased = _transcribe_all(transcription, orchestrator, whisper, sources, hotwords)
    print("B (hotwords) transcribed", flush=True)

    verifier = QwenDomainAdapter()
    correction = DomainPhoneticCorrectionAPI(InMemoryPhoneticRepository(), orchestrator, verifier)
    texts = {
        "A_raw": {sid: " ".join(s.text for s in segs) for sid, segs in raw.items()},
        "B_hotwords": {sid: " ".join(s.text for s in segs) for sid, segs in biased.items()},
        "C_hotwords_phonetic": _correct_all(correction, orchestrator, verifier, biased, industry, sub_domains),
        "D_raw_phonetic": _correct_all(correction, orchestrator, verifier, raw, industry, sub_domains),
    }
    print("C and D corrected", flush=True)
    return {"glossary_path": str(glossary_path), "variants": variants, "sources": sources, "texts": texts,
            "sub_domains": sub_domains, "hotwords": hotwords}


def run(language: str, industry: str, references: Dict[str, str], oracle: bool) -> dict:
    run_data = transcribe_configs(industry, list(references), oracle)
    variants, texts = run_data["variants"], run_data["texts"]

    per_source, scores = {}, {config: [] for config in CONFIGS}
    comparisons = {"C vs B": [], "D vs A": []}
    for source in run_data["sources"]:
        reference = references[source.stem]
        source_scores = {config: score(reference, texts[config][source.source_id], variants) for config in CONFIGS}
        for config, value in source_scores.items():
            scores[config].append(value)
        c_vs_b = compare(source_scores["B_hotwords"], source_scores["C_hotwords_phonetic"])
        d_vs_a = compare(source_scores["A_raw"], source_scores["D_raw_phonetic"])
        comparisons["C vs B"].append(c_vs_b)
        comparisons["D vs A"].append(d_vs_a)
        per_source[source.stem] = {
            "sub_domains": run_data["sub_domains"][source.source_id],
            "hotwords": run_data["hotwords"][source.source_id],
            "scores": {config: value.summary() for config, value in source_scores.items()},
            "C vs B": c_vs_b, "D vs A": d_vs_a,
            "missed_terms": {config: sorted({t for *_, t, ok in value.occurrences if not ok})
                             for config, value in source_scores.items()},
        }

    return {"language": language, "industry": industry, "glossary": run_data["glossary_path"],
            "oracle_subdomains": oracle,
            "totals": {config: aggregate(values) for config, values in scores.items()},
            "corrections": {name: aggregate_comparisons(values) for name, values in comparisons.items()},
            "sources": per_source}


def run_without_reference(language: str, industry: str, stems: Sequence[str], oracle: bool) -> dict:
    run_data = transcribe_configs(industry, stems, oracle)
    variants, texts = run_data["variants"], run_data["texts"]
    per_source = {}
    for source in run_data["sources"]:
        sid = source.source_id
        per_source[source.stem] = {
            "sub_domains": run_data["sub_domains"][sid], "hotwords": run_data["hotwords"][sid],
            "disagreements": {f"{first} -> {second}": term_disagreements(texts[first][sid], texts[second][sid],
                                                                          variants)
                              for first, second in PAIRS},
            "texts": {config: texts[config][sid] for config in CONFIGS},
        }
    return {"language": language, "industry": industry, "glossary": run_data["glossary_path"],
            "oracle_subdomains": oracle, "sources": per_source}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True)
    parser.add_argument("--industry", required=True)
    parser.add_argument("--database", default=None, help="database holding the ingested sources (POSTGRES_DB)")
    parser.add_argument("--glossary-path", default=None, help="glossary root holding <lang>/ (GLOSSARY_PATH)")
    parser.add_argument("--refs", type=Path, default=None, help="directory of <stem>.txt references")
    parser.add_argument("--oracle-subdomains", action="store_true", help="use every sub-domain instead of routing")
    parser.add_argument("--no-reference", action="store_true", help="list term disagreements instead of scoring")
    parser.add_argument("--sources", nargs="+", default=[], help="with --no-reference: media stems to run")
    args = parser.parse_args()
    if args.no_reference and not args.sources:
        parser.error("--no-reference needs --sources")

    # Settings are read at import, so the environment is set before anything from app/ is imported
    os.environ["APPLIANCE_LANGUAGE"] = args.language
    os.environ["APPLIANCE_INDUSTRY"] = args.industry
    if args.database:
        os.environ["POSTGRES_DB"] = args.database
    if args.glossary_path:
        os.environ["GLOSSARY_PATH"] = args.glossary_path

    out_dir = BENCH_ROOT / "glossary" / args.language / args.industry.lower()
    if args.no_reference:
        result = run_without_reference(args.language, args.industry, args.sources, args.oracle_subdomains)
        report = render_disagreements(args.language, args.industry, result)
        out_dir = out_dir / "no_reference"
    else:
        result = run(args.language, args.industry, load_references(args.language, args.refs),
                     args.oracle_subdomains)
        report = render_report(args.language, args.industry, result)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Results in {out_dir}")


if __name__ == "__main__":
    main()
