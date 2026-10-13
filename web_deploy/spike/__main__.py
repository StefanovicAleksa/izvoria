"""
The M0 spike: does the production layout work on one rented 24 GB card? Run ON the Vast box, after sync_to_box.sh
and provision.sh, with the box's settings loaded. No gateway, VPS or accounts are needed: the test signs its own
requests to the box's API with WEB_API_SECRET, as the gateway would.

  cd /workspace/onyx_v3 && set -a && . /workspace/web/api.env && set +a
  venv/bin/python -m web_deploy.spike all --video /workspace/samples/lecture.mp4

Stages (docs/web_deploy.md, "M0 spike"):
  sweep   llama-server's VRAM per slot count and shared KV size; picks LLM_SERVER_PARALLEL / LLM_SERVER_KV_TOKENS
  stress  JSON grammar, parallel throughput, a ~120K-token request beside short ones, an over-full cache
  shared  the box as provisioned: a video ingested while three users chat, VRAM watched throughout
  all     the three in order; the services are stopped for the first two and started again for the third

Results: <out>/report.md and <out>/results.json (default /workspace/spike).

`shared` predates the move of the data to the VPS (docs/web_deploy.md, "Where data lives"): it seeds files into the
box's own tenants and expects the queue's database beside them. On a box provisioned for the new layout it needs the
tunnel up and BOX_GATEWAY_URL unset, and its files then never reach the VPS; adapting it is open. `sweep` and
`stress` touch no data and are unaffected.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from web_deploy.spike import stages
from web_deploy.spike.gpu import VramSampler
from web_deploy.spike.report import render

logger = logging.getLogger("web_deploy.spike")

DEFAULT_GRID = "1:131072,2:131072,4:131072,4:163840,4:196608,4:262144,6:196608,6:262144"
SERVICES = ("api", "ingest", "llama-server")
REPO = Path(__file__).resolve().parents[2]


def parse_grid(text: str) -> List[tuple]:
    grid = []
    for item in text.split(","):
        parallel, kv = item.strip().split(":")
        grid.append((int(parallel), int(kv)))
    return grid


def supervisorctl(*args: str) -> bool:
    if not shutil.which("supervisorctl"):
        return False
    subprocess.run(["supervisorctl", *args], check=False)
    return True


def llama_port_busy(env: Dict[str, str]) -> bool:
    import urllib.request
    url = env.get("LLM_SERVER_URL", "http://127.0.0.1:8081").rstrip("/") + "/health"
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:
        return False


def apply_to_env_file(path: Path, values: Dict[str, object]) -> Path:
    """Writes the sweep's pick into the box's settings (llama-server and the API read them at start); keeps a copy."""
    backup = path.with_name(path.name + ".before-spike")
    if not backup.exists():
        shutil.copy2(path, backup)
    lines = path.read_text().splitlines()
    for key, value in values.items():
        line = f"{key}={value}"
        for i, existing in enumerate(lines):
            if existing.startswith(f"{key}="):
                lines[i] = line
                break
        else:
            lines.append(line)
    path.write_text("\n".join(lines) + "\n")
    return backup


def wait_for_api(api, minutes: float = 15) -> dict:
    deadline = time.monotonic() + minutes * 60
    last: dict = {}
    while time.monotonic() < deadline:
        try:
            last = api.health()
            if last.get("ready"):
                return last
        except Exception as exc:
            last = {"error": str(exc)}
        time.sleep(5)
    return last


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="The M0 spike, on the rented GPU box.")
    parser.add_argument("stage", choices=["sweep", "stress", "shared", "all"])
    parser.add_argument("--grid", default=DEFAULT_GRID, help="slots:kv_tokens pairs for the sweep")
    parser.add_argument("--parallel", type=int, help="stress at this slot count (default: the sweep's pick)")
    parser.add_argument("--kv", type=int, help="stress at this shared KV size (default: the sweep's pick)")
    parser.add_argument("--long-tokens", type=int, default=stages.LONG_PROMPT_TOKENS,
                        help="size of the long request in the stress stage (smaller only for a trial on a small card)")
    parser.add_argument("--video", type=Path, default=REPO / "integration_tests" / "resources" / "test_video.mp4",
                        help="the recording ingested during the shared test (5-20 minutes is a fair test)")
    parser.add_argument("--firm", default="spike-test")
    parser.add_argument("--api-url", help="the box API (default https://127.0.0.1:$BOX_API_PORT)")
    parser.add_argument("--out", type=Path, default=Path("/workspace/spike"))
    parser.add_argument("--env-file", type=Path, default=Path("/workspace/web/api.env"),
                        help="where `all` writes the sweep's pick before the shared test")
    parser.add_argument("--no-manage-services", action="store_true",
                        help="don't stop/start supervisord's programs around the stages")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    env = dict(os.environ)
    if not env.get("WEB_API_SECRET"):
        print("error: load the box's settings first: set -a && . /workspace/web/api.env && set +a", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    results_path = args.out / "results.json"
    results: dict = json.loads(results_path.read_text()) if results_path.exists() else {}
    results["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    sampler = VramSampler()
    results["gpu_total_mb"] = sampler.total
    try:
        results["gpu_name"] = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                                             stdout=subprocess.PIPE, text=True).stdout.strip()
    except OSError:
        pass

    def save() -> None:
        results_path.write_text(json.dumps(results, indent=2, default=str))
        (args.out / "report.md").write_text(render(results))

    manage = not args.no_manage_services
    if args.stage in ("sweep", "stress", "all"):
        if manage:
            supervisorctl("stop", *SERVICES)
            time.sleep(5)
        if llama_port_busy(env):
            print("error: something already serves llama-server's port. Stop it first: supervisorctl stop "
                  "api ingest llama-server", file=sys.stderr)
            return 1

    if args.stage in ("sweep", "all"):
        results["sweep"] = stages.sweep(env, sampler, parse_grid(args.grid), args.out)
        results["recommendation"] = stages.recommend(results["sweep"])
        save()

    if args.stage in ("stress", "all"):
        pick = results.get("recommendation") or {}
        parallel = args.parallel or pick.get("LLM_SERVER_PARALLEL") or int(env.get("LLM_SERVER_PARALLEL", "4"))
        kv = args.kv or pick.get("LLM_SERVER_KV_TOKENS") or int(env.get("LLM_SERVER_KV_TOKENS", "196608"))
        results["stress"] = stages.stress(env, sampler, parallel, kv, args.out, args.long_tokens)
        save()
        results["answers_before_long"] = stages.answers_before_long(env, parallel, kv, args.out, args.long_tokens)
        save()

    if args.stage in ("shared", "all"):
        pick = results.get("recommendation")
        if pick and args.stage == "all" and args.env_file.exists():
            # Test the layout the sweep chose, not the plan's guess: llama-server.sh and the API read api.env
            values = {k: pick[k] for k in ("LLM_SERVER_PARALLEL", "LLM_SERVER_KV_TOKENS")}
            backup = apply_to_env_file(args.env_file, values)
            env.update({k: str(v) for k, v in values.items()})
            results["applied_to_env_file"] = {"file": str(args.env_file), "values": values, "backup": str(backup)}
            logger.info(f"Wrote {values} into {args.env_file} (the original is {backup})")
        if manage:
            supervisorctl("start", *SERVICES)
        from web_deploy.api.config import ApiSettings
        from web_deploy.common.signing import Principal
        from web_deploy.ingest_worker.queue import IngestQueue
        from web_deploy.spike.clients import BoxApi
        settings = ApiSettings.from_env()
        api = BoxApi(args.api_url or f"https://127.0.0.1:{env.get('BOX_API_PORT', '8095')}", settings.secret,
                     Principal(firm=args.firm, user_id="spike", role="admin"))
        health = wait_for_api(api)
        if not health.get("ready"):
            results["shared"] = {"error": f"the API never became ready: {health}"}
        else:
            queue = IngestQueue.connect(settings.control_database_url)
            results["shared"] = stages.shared(api, queue, settings.tenants_root, args.firm, args.video, sampler,
                                              args.out)
        save()

    results["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    print((args.out / "report.md").read_text())
    print(f"\nSaved: {args.out / 'report.md'} and {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
