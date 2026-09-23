# Path: scripts/eval/language_benchmark/__main__.py

"""
Language benchmark: pulls real speech from online, ingests it through the real engine, and measures it.

    venv/bin/python -m scripts.eval.language_benchmark all --language sr sl en
    venv/bin/python -m scripts.eval.language_benchmark <stage> --language sr

Stages, in order: fetch, prepare, ingest, measure-asr, guard, gen-questions, measure-rag, report.
fetch/prepare are CPU and network only; the rest run one at a time on the GPU, each in a child process configured
as that language's appliance -- its own database (onyx_bench_<code>), vault and APPLIANCE_LANGUAGE -- so the dev
database (and the chunk ids scripts/eval/rag_labels.json depend on) is never touched.
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import Dict, List

from scripts.eval.language_benchmark.manifest import load_manifest
from scripts.eval.language_benchmark.paths import BENCH_ROOT, LanguageDirs, REPO_ROOT, database_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("language_benchmark")

APPLIANCE_STAGES = ["ingest", "measure-asr", "guard", "gen-questions", "measure-rag"]
STAGES = ["fetch", "prepare", *APPLIANCE_STAGES, "report"]


def language_environment(code: str) -> Dict[str, str]:
    dirs = LanguageDirs(code)
    return {
        **os.environ,
        "ENV": "development",
        "POSTGRES_DB": database_name(code),
        "APPLIANCE_LANGUAGE": code,
        # No glossary exists yet for sr/sl; GENERAL degrades to no hotwords instead of failing domain adaptation
        "APPLIANCE_INDUSTRY": "GENERAL",
        "ONYX_STORAGE_PATH": str(dirs.vault),
        "DATA_DIR": str(dirs.root),
        "REPORTS_PATH": str(dirs.root / "reports"),
        "ONYX_BENCH_LANGUAGE": code,
    }


def ensure_database(code: str) -> None:
    """Creates onyx_bench_<code> with pgvector in the dev Postgres, if it isn't there yet."""
    import psycopg2
    from app.core.config import settings

    connect = dict(host=settings.POSTGRES_SERVER, port=settings.POSTGRES_PORT, user=settings.POSTGRES_USER,
                   password=settings.POSTGRES_PASSWORD)
    name = database_name(code)
    admin = psycopg2.connect(dbname=settings.POSTGRES_DB, **connect)
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,))
            if cursor.fetchone() is None:
                cursor.execute(f'CREATE DATABASE "{name}"')
                logger.info(f"Created database {name}.")
    finally:
        admin.close()
    database = psycopg2.connect(dbname=name, **connect)
    database.autocommit = True
    try:
        with database.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    finally:
        database.close()


def run_in_language_process(stage: str, code: str) -> None:
    """Runs one appliance stage in a child process configured as `code`'s appliance."""
    ensure_database(code)
    logger.info(f"[{code}] {stage}")
    subprocess.run([sys.executable, "-m", "scripts.eval.language_benchmark", stage, "--language", code, "--in-process"],
                   env=language_environment(code), cwd=REPO_ROOT, check=True)


def run_stage(stage: str, code: str, args: argparse.Namespace) -> None:
    from scripts.eval.language_benchmark import fetch, prepare, report

    sources = load_manifest()[code]
    dirs = LanguageDirs(code).make()

    if stage == "fetch":
        fetch.fetch_all(code, sources, dirs)
    elif stage == "prepare":
        prepare.prepare_all(fetch.fetch_all(code, sources, dirs), dirs)
    elif stage in APPLIANCE_STAGES and not args.in_process:
        run_in_language_process(stage, code)
    elif stage == "ingest":
        from scripts.eval.language_benchmark import stages
        stages.ingest(dirs)
    elif stage == "measure-asr":
        from scripts.eval.language_benchmark import stages
        stages.measure_asr(dirs, code, sources)
    elif stage == "guard":
        from scripts.eval.language_benchmark import stages
        stages.guard(dirs)
    elif stage == "gen-questions":
        from scripts.eval.language_benchmark import stages
        if dirs.labels.exists() and not args.regenerate:
            logger.info(f"[{code}] keeping existing {dirs.labels} (pass --regenerate to rewrite it)")
        else:
            stages.gen_questions(dirs, code, per_source=args.questions_per_source)
    elif stage == "measure-rag":
        from scripts.eval.language_benchmark import stages
        stages.measure_rag(dirs)
    elif stage == "report":
        report.write_report(args.language)


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(prog="language_benchmark")
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--language", nargs="+", required=True, choices=sorted(load_manifest()))
    parser.add_argument("--regenerate", action="store_true", help="gen-questions: rewrite an existing labels.json")
    parser.add_argument("--questions-per-source", type=int, default=10)
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.in_process and os.environ.get("ONYX_BENCH_LANGUAGE") != args.language[0]:
        parser.error("--in-process is only for the child process the benchmark starts itself")

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        run_stage("report", args.language[0], args)
        return
    stage_list = [s for s in STAGES if s != "report"] if args.stage == "all" else [args.stage]
    for stage in stage_list:
        for code in args.language:
            run_stage(stage, code, args)
    if args.stage == "all":
        run_stage("report", args.language[0], args)


if __name__ == "__main__":
    main()
