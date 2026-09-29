# Path: scripts/eval/cross_language/__main__.py

"""
Cross-language evaluation: is asking in Serbian about English material as good as asking in English, and so on
for every pair? Retrieval, chat and research reports, over one database holding every language's benchmark.

    venv/bin/python -m scripts.eval.cross_language all
    venv/bin/python -m scripts.eval.cross_language <stage> [--query-language sr]

Stages: build-db, translate, retrieval, chat, reports, summary. Needs the language benchmark's databases and
labels (`scripts.eval.language_benchmark ... gen-questions`). GPU stages run in a child process per question
language, with that language active, over onyx_bench_multi; results land in onyx_data/benchmarks/cross/results.
"""

import argparse
import logging
import os
import subprocess
import sys
from typing import List

from scripts.eval.language_benchmark.paths import REPO_ROOT, served_languages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cross_language")

# Every served language, each as material and as question language
CODES = served_languages()
PER_QUERY_STAGES = ["retrieval", "chat", "reports"]
# The per-pair result file each stage writes (stages.pair_name)
RESULT_NAMES = {"retrieval": "retrieval", "chat": "chat", "reports": "report"}
STAGES = ["build-db", "translate", *PER_QUERY_STAGES, "summary"]


def environment(query: str) -> dict:
    from scripts.eval.cross_language.stages import CROSS_ROOT, MULTI_DB
    return {**os.environ, "ENV": "development", "POSTGRES_DB": MULTI_DB, "APPLIANCE_LANGUAGE": query,
            "APPLIANCE_INDUSTRY": "GENERAL", "REMEMBER_LANGUAGE": "false",
            "ONYX_STORAGE_PATH": str(CROSS_ROOT / "vault"), "DATA_DIR": str(CROSS_ROOT),
            "REPORTS_PATH": str(CROSS_ROOT / "reports"), "ONYX_CROSS_CHILD": "1"}


def run_child(stage: str, query: str, data: str = "") -> None:
    """One process per pair: a process that exits is the only reliable way to hand every byte of VRAM back."""
    logger.info(f"{stage}: {data or '-'} material, asked in {query}")
    subprocess.run([sys.executable, "-m", "scripts.eval.cross_language", stage, "--query-language", query,
                    *(["--data-language", data] if data else []), "--in-process"],
                   env=environment(query), cwd=REPO_ROOT, check=True)


def run(stage: str, queries: List[str], datas: List[str], in_process: bool, skip_done: bool) -> None:
    from scripts.eval.cross_language import stages, summary

    if stage == "summary":
        logger.info(f"Wrote {summary.write_summary(stages.RESULTS, CODES)}")
    elif stage in ("build-db", "translate") and not in_process:
        run_child(stage, "en")
    elif stage == "build-db":
        stages.build_multi_db(CODES)
    elif stage == "translate":
        stages.translate(CODES)
    elif not in_process:
        for query in queries:
            for data in datas:
                name = stages.pair_name(RESULT_NAMES[stage], query, data)
                if skip_done and (stages.RESULTS / f"{name}.json").exists():
                    logger.info(f"{name} already measured; skipping (--redo to measure again)")
                    continue
                run_child(stage, query, data)
    elif stage == "retrieval":
        stages.retrieval(queries[0], datas[0])
    else:
        getattr(stages, stage)(queries[0], datas[0], CODES)


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(prog="cross_language")
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--query-language", nargs="+", default=CODES, choices=CODES)
    parser.add_argument("--data-language", nargs="+", default=CODES, choices=CODES)
    parser.add_argument("--redo", action="store_true", help="measure pairs that already have a result again")
    parser.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.in_process and not os.environ.get("ONYX_CROSS_CHILD"):
        parser.error("--in-process is only for the child process this script starts itself")
    for stage in (STAGES if args.stage == "all" else [args.stage]):
        run(stage, args.query_language, args.data_language, args.in_process, skip_done=not args.redo)


if __name__ == "__main__":
    main()
