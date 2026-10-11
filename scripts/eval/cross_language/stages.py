# Path: scripts/eval/cross_language/stages.py

"""
The stages. Each GPU stage runs in a child process configured as an appliance whose ACTIVE language is the question
language, over the one mixed database: that is how a client asks in Serbian about English material. `app.*` is
imported inside the functions so nothing reads `settings` before the child's environment is set.
"""

import json
import logging
import time
from typing import Dict, List

from scripts.eval.cross_language.labels import pair_labels
from scripts.eval.cross_language.language_id import detect_language, distinctive_stopwords
from scripts.eval.language_benchmark.paths import BENCH_ROOT, LanguageDirs, database_name

logger = logging.getLogger("cross_language")

CROSS_ROOT = BENCH_ROOT / "cross"
TRANSLATIONS = CROSS_ROOT / "translations.json"
RESULTS = CROSS_ROOT / "results"
MULTI_DB = "onyx_bench_multi"
# The language translations between two others go through (see translate)
PIVOT = "en"
# The tables a search, a chat turn and a report read; everything upstream of the index stays in the language DBs
COPIED_TABLES = ("files", "sources", "vector_chunks")


def pair_name(stage: str, query: str, data: str) -> str:
    return f"{stage}_q-{query}_d-{data}"


def _write(name: str, payload: dict) -> dict:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    return payload


# ---------------------------------------------------------------- the mixed database

def build_multi_db(codes: List[str]) -> dict:
    """
    Copies every language benchmark's index into one database, as an appliance holding all of them would. The
    database is brought to head first; the copies keep their chunk ids (the labels name them) and their per-chunk
    `fts_config`. Source and file names get a "<code>/" prefix: every benchmark has a fleurs_01.wav.
    """
    import psycopg2
    from app.core.config import settings
    from app.core.database.service.migrations import upgrade_to_head
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url

    connect = dict(host=settings.POSTGRES_SERVER, port=settings.POSTGRES_PORT, user=settings.POSTGRES_USER,
                   password=settings.POSTGRES_PASSWORD)
    # The child is configured for the database being created, so the admin connection uses the maintenance DB
    admin = psycopg2.connect(dbname="postgres", **connect)
    admin.autocommit = True
    with admin.cursor() as cursor:
        cursor.execute(f'DROP DATABASE IF EXISTS "{MULTI_DB}" WITH (FORCE)')
        cursor.execute(f'CREATE DATABASE "{MULTI_DB}"')
    admin.close()
    target = psycopg2.connect(dbname=MULTI_DB, **connect)
    target.autocommit = True
    with target.cursor() as cursor:
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    engine = create_engine(make_url(settings.DATABASE_URL).set(database=MULTI_DB))
    upgrade_to_head(engine)
    engine.dispose()

    counts = {}
    for code in codes:
        source = psycopg2.connect(dbname=database_name(code), **connect)
        # Each language DB must be at the same schema, or the column lists below would disagree
        upgrade = create_engine(make_url(settings.DATABASE_URL).set(database=database_name(code)))
        upgrade_to_head(upgrade)
        upgrade.dispose()
        for table in COPIED_TABLES:
            with source.cursor() as read, target.cursor() as write:
                read.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s "
                             "AND is_generated = 'NEVER' ORDER BY ordinal_position", (table,))
                columns = ", ".join(f'"{row[0]}"' for row in read.fetchall())
                buffer = _CopyBuffer()
                read.copy_expert(f"COPY (SELECT {columns} FROM {table}) TO STDOUT", buffer)
                write.copy_expert(f"COPY {table} ({columns}) FROM STDIN", buffer.reader())
        source.close()
        with target.cursor() as cursor:
            cursor.execute("UPDATE sources SET name = %s || name WHERE name NOT LIKE '%%/%%'", (f"{code}/",))
            cursor.execute("UPDATE files SET file_name = %s || file_name WHERE file_name NOT LIKE '%%/%%'",
                           (f"{code}/",))
            cursor.execute("SELECT count(*) FROM vector_chunks v JOIN sources s ON s.id = v.source_id "
                           "WHERE s.language = %s", (code,))
            counts[code] = cursor.fetchone()[0]
    with target.cursor() as cursor:
        cursor.execute("SELECT fts_config::text, count(*) FROM vector_chunks GROUP BY 1 ORDER BY 1")
        stemmers = dict(cursor.fetchall())
    target.close()
    logger.info(f"{MULTI_DB}: chunks per language {counts}, per stemmer {stemmers}")
    return _write("multi_db", {"chunks_per_language": counts, "chunks_per_fts_config": stemmers})


class _CopyBuffer:
    """Holds one COPY TO STDOUT so it can be replayed into COPY FROM STDIN (a few hundred rows here)."""

    def __init__(self):
        self._parts: List[bytes] = []

    def write(self, data):
        self._parts.append(data.encode() if isinstance(data, str) else data)

    def reader(self):
        import io
        return io.BytesIO(b"".join(self._parts))


# ---------------------------------------------------------------- translations and topics

_TRANSLATE_PROMPT = """Translate the text below from {source} into {target}. Keep every name, number and technical
term exact, and write it the way a native {target} speaker would. {script_rule}Reply with the translation only.

TEXT:
{text}"""

_TOPIC_PROMPT = """You write evaluation topics for a research-report system over transcribed {language} recordings.

Below are excerpts from one recording. Write ONE research topic, in {language}, asking for a report on the main
subject they discuss: 8 to 20 words, specific to what is actually said. Never mention "the recording", "the
lecture" or "the speaker". Reply with the topic only.

EXCERPTS:
{excerpts}"""


def _chat(llm, prompt: str, max_tokens: int = 200) -> str:
    from app.core.model_manager.utils.llama_cpp_chat import chat_completion, strip_reasoning
    reply = chat_completion(llm, [{"role": "user", "content": prompt}], enable_thinking=False,
                            temperature=0.2, max_tokens=max_tokens)
    return strip_reasoning(reply["choices"][0]["message"]["content"]).strip().strip('"').splitlines()[0].strip()


def translate(codes: List[str]) -> dict:
    """
    Every benchmark question (Latin originals only) in every language, and one report topic per material language,
    written from its lecture in that language and translated into the others. Written by the answer model, as the
    questions themselves were; read translations.json before trusting the numbers.

    Between two languages other than English the text goes through its English translation: asked to translate
    Serbian into Slovenian directly, the model often handed the Serbian back nearly unchanged. An existing
    translations.json keeps its direct translations into and out of English and only redoes those pivots.
    """
    import psycopg2
    from app.core.config import settings
    from app.core.language.service.language_api import get_language_pack
    from app.core.language.utils.script import normalize_script
    from app.core.model_manager.utils.llama_cpp_chat import build_llama

    packs = {code: get_language_pack(code) for code in codes}

    def into(text: str, source: str, target: str, english: str = None) -> str:
        if source != PIVOT and target != PIVOT:
            return into(english or into(text, source, PIVOT), PIVOT, target)
        pack = packs[target]
        rule = "Write it in Latin script. " if pack.transcription.script == "latin" else ""
        out = _chat(llm, _TRANSLATE_PROMPT.format(source=packs[source].name, target=pack.name, script_rule=rule,
                                                  text=text))
        return normalize_script(out, pack.transcription.script)

    connect = dict(host=settings.POSTGRES_SERVER, port=settings.POSTGRES_PORT, user=settings.POSTGRES_USER,
                   password=settings.POSTGRES_PASSWORD)
    previous = json.loads(TRANSLATIONS.read_text(encoding="utf-8")) if TRANSLATIONS.exists() else {}
    llm = build_llama(settings.QA_GENERATION_MODEL_PATH, n_ctx=8192, kv_cache_type=settings.QA_LLM_KV_CACHE_TYPE)
    result = {}
    try:
        for data in codes:
            labels = json.loads(LanguageDirs(data).labels.read_text(encoding="utf-8"))
            kept = {item["id"]: item["text"] for item in previous.get(data, {}).get("answerable", [])}
            answerable = []
            for item in labels["answerable"]:
                if item["id"].endswith("c"):  # the benchmark's Cyrillic copies; made again below from the Latin
                    continue
                old = kept.get(item["id"], {})
                text = {data: item["question"]}
                # English first, so the other languages can be translated from it
                for target in sorted((c for c in codes if c != data), key=lambda c: c != PIVOT):
                    reusable = old.get(target) if PIVOT in (data, target) else None
                    text[target] = reusable or into(item["question"], data, target, english=text.get(PIVOT))
                answerable.append({"id": item["id"], "relevant": item["relevant"], "source": item["source"],
                                   "text": text})
                logger.info(f"[{data}] {item['id']}: {text}")

            db = psycopg2.connect(dbname=database_name(data), **connect)
            with db.cursor() as cursor:
                cursor.execute("SELECT v.raw_text FROM vector_chunks v JOIN sources s ON s.id = v.source_id "
                               "WHERE s.name LIKE 'lecture_%%' ORDER BY v.position")
                chunks = [row[0] for row in cursor.fetchall()]
            db.close()
            # Spread over the whole lecture, so the topic is its subject rather than its opening
            step = max(1, len(chunks) // 6)
            excerpts = "\n---\n".join(c[:900] for c in chunks[::step][:6])
            topic = {data: normalize_script(_chat(llm, _TOPIC_PROMPT.format(language=packs[data].name,
                                                                            excerpts=excerpts)),
                                            packs[data].transcription.script)}
            for target in sorted((c for c in codes if c != data), key=lambda c: c != PIVOT):
                topic[target] = into(topic[data], data, target, english=topic.get(PIVOT))
            logger.info(f"[{data}] topic: {topic}")
            result[data] = {"answerable": answerable, "topic": topic}
    finally:
        llm.close()
    CROSS_ROOT.mkdir(parents=True, exist_ok=True)
    TRANSLATIONS.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# ---------------------------------------------------------------- search, chat, reports

def _translations() -> dict:
    return json.loads(TRANSLATIONS.read_text(encoding="utf-8"))


def _labels(translations: dict, data: str, query: str, with_cyrillic: bool) -> dict:
    from app.core.language.service.language_api import get_language_pack
    from scripts.eval.language_benchmark.stages import UNANSWERABLE, to_cyrillic
    cyrillic = to_cyrillic if with_cyrillic and get_language_pack(query).transcription.script == "latin" else None
    return pair_labels(translations, data, query, UNANSWERABLE, cyrillic=cyrillic)


def retrieval(query: str, data: str) -> dict:
    """Search, rerank and relevance scrubber over `data` material, asked in `query` (the active language)."""
    from scripts.eval.language_benchmark.stages import _rank_summary
    from scripts.eval.rag_eval import run_retrieval

    labels = _labels(_translations(), data, query, with_cyrillic=True)
    result = run_retrieval(labels, None, 10, None)
    items = result["items"]
    split = {script: _rank_summary({qid: r for qid, r in items.items() if qid.startswith("a")
                                    and qid.endswith("c") == (script == "cyrillic")})
             for script in ("latin", "cyrillic")} if any(q.endswith("c") for q in items) else None
    return _write(pair_name("retrieval", query, data), {"retrieval": result, "script_split": split})


def parse_answer_lines(report_lines: List[str]) -> Dict[str, dict]:
    """rag_eval's answer report lines ("[a01] 12s 2 cites :: question\n    answer") back to {id: {verdict, text}}."""
    parsed = {}
    for line in report_lines:
        head, _, body = line.partition("\n")
        qid = head[1:head.index("]")]
        verdict = head[head.index("]") + 1:head.index("::")].split(maxsplit=1)[-1].strip()
        parsed[qid] = {"verdict": verdict, "text": body.strip()}
    return parsed


def chat(query: str, data: str, codes: List[str]) -> dict:
    """Single-pass chat over `data` material, asked and answered in `query`."""
    from app.core.language.service.language_api import get_language_pack
    from scripts.eval.rag_eval import run_answer

    distinctive = distinctive_stopwords({c: get_language_pack(c).text.stopwords for c in codes})
    result = run_answer(_labels(_translations(), data, query, with_cyrillic=False), None, "chat", 10)
    parsed = parse_answer_lines(result["answers"])
    answered = {qid: detect_language(p["text"], distinctive) for qid, p in parsed.items()
                if qid.startswith("a") and p["verdict"].endswith("cites")}
    return _write(pair_name("chat", query, data), {
        "answers": result, "answer_language": answered,
        "answered_in_question_language": sum(lang == query for lang in answered.values()),
        "answered_count": len(answered)})


def reports(query: str, data: str, codes: List[str]) -> dict:
    """
    One deep research report per material language, on its topic asked in `query`. Measures what the run found
    (the evidence bank and what the report cites) rather than the prose: whether asking in another language
    gathers the same evidence as asking in the material's own.
    """
    import psycopg2
    from app.core.config import settings
    from app.core.language.service.language_api import get_language_pack
    from app.engine.service.bootstrap import IzvoriaBootstrapper

    translations = _translations()
    distinctive = distinctive_stopwords({c: get_language_pack(c).text.stopwords for c in codes})
    db = psycopg2.connect(dbname=MULTI_DB, host=settings.POSTGRES_SERVER, port=settings.POSTGRES_PORT,
                          user=settings.POSTGRES_USER, password=settings.POSTGRES_PASSWORD)
    with db.cursor() as cursor:
        cursor.execute("SELECT id::text, language, name FROM sources")
        source_language = {row[0]: row[1] for row in cursor.fetchall()}
    db.close()

    engine = IzvoriaBootstrapper.build()
    engine.start()
    try:
        topic = translations[data]["topic"][query]
        started = time.perf_counter()
        proposal = engine.propose_research_plan(topic)
        report = engine.execute_research_report(proposal.report_id, proposal.plan)
    finally:
        engine.stop()
    cited = set(report.cited_doc_numbers)
    evidence = report.evidence
    cited_entries = [e for e in evidence if e.doc_number in cited]
    prose = " ".join([report.executive_summary] + [s.markdown_body for s in report.written_sections])
    run = {
        "topic": topic,
        "minutes": round((time.perf_counter() - started) / 60, 1),
        "status": report.status.value,
        "sections": len(report.sections),
        "sections_written": len(report.written_sections),
        "sections_skipped": len(report.sections) - len(report.written_sections),
        "evidence_chunks": sorted(str(e.chunk_id) for e in evidence),
        "cited_chunks": sorted(str(e.chunk_id) for e in cited_entries),
        "evidence_from_material_language": sum(source_language.get(str(e.source_id)) == data for e in evidence),
        "cited_from_material_language": sum(source_language.get(str(e.source_id)) == data for e in cited_entries),
        "report_language": detect_language(prose, distinctive, min_hits=5),
        "language_recorded": report.language,
        "markdown_path": report.markdown_path,
    }
    logger.info(f"[{data} / asked in {query}] {run['sections_written']}/{run['sections']} sections, "
                f"{len(cited_entries)} cited of {len(evidence)} evidence")
    return _write(pair_name("report", query, data), run)
