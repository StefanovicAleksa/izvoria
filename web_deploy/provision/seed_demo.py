"""
Fills a demo firm with sample material: copies a folder of files into the firm's uploads on the GPU box and puts it
on the ingestion wait list, exactly as an upload from the web app would.

Run on the box (the firm must exist on the gateway too, or nobody can sign in to see it):
  cd /workspace/onyx_v3
  set -a; . /workspace/web/api.env; set +a
  venv/bin/python -m web_deploy.provision.seed_demo legal-demo document en /workspace/samples/contracts
  venv/bin/python -m web_deploy.provision.seed_demo clinic-demo audio sr /workspace/samples/consultations

Use separate runs per kind of file (video, audio, document, image) and per language.
"""

import argparse
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from deploy.instance.agent.uploads import ALLOWED_SUFFIXES, sanitize_filename
from web_deploy.common.tenancy import firm_root, validate_slug
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue

SEED_OWNER = "seed"


def seed(queue: IngestQueue, tenants_root: Path, firm: str, mode: IngestMode, language: str,
         source_dir: Path) -> dict:
    firm = validate_slug(firm)
    files: List[Path] = sorted(p for p in source_dir.iterdir()
                               if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES)
    if not files:
        raise ValueError(f"No ingestible files in {source_dir}.")
    batch = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"
    target = firm_root(tenants_root, firm) / "uploads" / SEED_OWNER / batch
    target.mkdir(parents=True)
    for path in files:
        shutil.copy2(path, target / sanitize_filename(path.name))
    job = queue.enqueue(firm, SEED_OWNER, mode, language, str(target), len(files))
    return {"job": job.id, "files": len(files), "position": queue.queue_positions().get(job.id)}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Queue sample material for a demo firm.")
    parser.add_argument("firm")
    parser.add_argument("mode", choices=[m.value for m in IngestMode])
    parser.add_argument("language", choices=["en", "sr"])
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)

    from web_deploy.api.config import ApiSettings
    settings = ApiSettings.from_env()
    try:
        result = seed(IngestQueue.connect(settings.control_database_url), settings.tenants_root, args.firm,
                      IngestMode(args.mode), args.language, args.folder)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Queued {result['files']} file(s) for {args.firm} as job {result['job']} (position {result['position']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
