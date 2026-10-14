"""
Brings material ingested before the storage settings were turned on in line with them: drops each recording's
per-chunk clips and WAV, swaps its original for a compact copy, and removes upload folders no job will read again.
New ingestions already leave the vault this way (docs/web_deploy.md, "Storage").

For the local version (run_local stopped or idle), or a box from before the data moved to the VPS, with the API's
settings loaded. The hosted box keeps no vault any more and refuses (BOX_GATEWAY_URL is set there):
  cd /workspace/onyx_v3
  set -a; . /workspace/web/api.env; set +a
  venv/bin/python -m web_deploy.provision.compact_storage --all --dry-run    # what would be freed
  venv/bin/python -m web_deploy.provision.compact_storage --all              # free it
  venv/bin/python -m web_deploy.provision.compact_storage legal-demo         # one firm

Locally: set -a; . web_deploy/local/local.env; set +a, plus TENANTS_ROOT=onyx_data/local-web/tenants and
CONTROL_DATABASE_URL=sqlite:///onyx_data/local-web/control.sqlite3 (what run_local derives from local.env).

Each firm runs in its own child process (its database and vault are process-wide settings, as for an ingestion
job). Refused while an ingestion is running: both would rewrite the same vault.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from web_deploy.common.tenancy import validate_slug
from web_deploy.ingest_worker.queue import IngestQueue
from web_deploy.common.uploads import stale_upload_folders
from web_deploy.ingest_worker.supervisor import firm_environment

RESULT_PREFIX = "COMPACTION_RESULT "
# Upload folders no job refers to are removed once untouched this long: anything older is not still arriving
UPLOAD_MIN_AGE_SECONDS = 3600


def firms_under(tenants_root: Path) -> List[str]:
    """Every firm that has a vault on this box."""
    if not tenants_root.is_dir():
        return []
    return sorted(d.name for d in tenants_root.iterdir() if (d / "vault").is_dir())


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def run_child(firm: str, dry_run: bool) -> int:
    """In the firm's environment: compact its indexed recordings and print the report as one line of JSON."""
    from app.engine.service.bootstrap import IzvoriaBootstrapper

    engine = IzvoriaBootstrapper.build()
    engine.start()
    try:
        report = engine.compact_storage(dry_run=dry_run)
    finally:
        engine.stop()
    print(RESULT_PREFIX + json.dumps({**report.model_dump(), "bytes_saved": report.bytes_saved}), flush=True)
    return 0


def compact_firm(firm: str, dry_run: bool, tenants_root: Path, base_env: Dict[str, str],
                 run=subprocess.run) -> dict:
    env = firm_environment(firm, "en", tenants_root, base_env)
    command = [sys.executable, "-m", "web_deploy.provision.compact_storage", "--child", firm]
    if dry_run:
        command.append("--dry-run")
    completed = run(command, env=env, cwd=env.get("ONYX_ROOT") or None, stdout=subprocess.PIPE, text=True)
    lines = [line for line in (completed.stdout or "").splitlines() if line.startswith(RESULT_PREFIX)]
    if completed.returncode != 0 or not lines:
        raise RuntimeError(f"compaction of {firm} failed (exit code {completed.returncode}); see its log above")
    return json.loads(lines[-1][len(RESULT_PREFIX):])


def describe(firm: str, result: dict) -> str:
    dry = result.get("dry_run")
    parts = [f"{firm}: {result['sources_checked']} indexed source(s)",
             f"clips {human(result['clip_bytes'])}", f"WAVs {human(result['audio_bytes'])}"]
    if dry:
        parts.append(f"{result['originals_to_compact']} original(s) to re-encode "
                     f"({human(result['original_bytes_to_compact'])} now; the saving shows after)")
        verb = "would free"
    else:
        parts.append(f"{result['originals_replaced']} original(s) compacted, saving "
                     f"{human(result['original_bytes_saved'])}")
        verb = "freed"
    line = " · ".join(parts) + f" -> {verb} {human(result['bytes_saved'])}"
    for name, error in (result.get("errors") or {}).items():
        line += f"\n    {name}: {error}"
    return line


def main(argv: Optional[list] = None, settings=None, queue: Optional[IngestQueue] = None,
         compact=compact_firm) -> int:
    parser = argparse.ArgumentParser(description="Free what the vault no longer needs, for firms already ingested.")
    parser.add_argument("firms", nargs="*", help="firm sign-in names (default with --all: every firm on the box)")
    parser.add_argument("--all", action="store_true", help="every firm that has a vault")
    parser.add_argument("--dry-run", action="store_true", help="report what would be freed; change nothing")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        return run_child(args.child, args.dry_run)

    if os.environ.get("BOX_GATEWAY_URL"):
        # The hosted GPU box keeps no vault: new ingestions compact as they go, and the VPS has no appliance venv
        print("error: this box holds no files (they live on the VPS); there is nothing here to compact.",
              file=sys.stderr)
        return 1
    if settings is None:
        from web_deploy.api.config import ApiSettings
        settings = ApiSettings.from_env()
    queue = queue or IngestQueue.connect(settings.control_database_url)
    if queue.depth()["running"]:
        print("error: an ingestion is running; try again when the queue is idle.", file=sys.stderr)
        return 1
    firms = firms_under(settings.tenants_root) if args.all else [validate_slug(f) for f in args.firms]
    if not firms:
        print("error: name the firms to compact, or pass --all.", file=sys.stderr)
        return 1

    total = 0
    failed = False
    for firm in firms:
        try:
            result = compact(firm, args.dry_run, settings.tenants_root, dict(os.environ))
        except Exception as exc:
            print(f"{firm}: {exc}", file=sys.stderr)
            failed = True
            continue
        total += result["bytes_saved"]
        print(describe(firm, result))

    stale = stale_upload_folders(queue.active_batch_dirs(), settings.tenants_root, time.time(), UPLOAD_MIN_AGE_SECONDS)
    stale_bytes = sum(folder_size(folder) for folder in stale)
    if not args.dry_run:
        for folder in stale:
            shutil.rmtree(folder, ignore_errors=True)
    total += stale_bytes
    print(f"uploads: {len(stale)} folder(s) no job will read -> {'would free' if args.dry_run else 'freed'} "
          f"{human(stale_bytes)}")
    print(f"total: {'would free' if args.dry_run else 'freed'} {human(total)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
