"""
The web version on dev hardware: one 12 GB GPU, this machine, your local network. For personal testing.

  venv/bin/python -m web_deploy.local.run_local setup                 # once: checks, builds what is missing
  venv/bin/python -m web_deploy.local.run_local admin create-owner me@example.com     # + the console's firm password
  venv/bin/python -m web_deploy.local.run_local admin create-firm legal-demo "Legal demo" --demo   # + its password
  venv/bin/python -m web_deploy.local.run_local admin create-user legal-demo me@example.com --role admin
  venv/bin/python -m web_deploy.local.run_local                       # start; Ctrl+C stops everything

Then open http://<this machine's LAN address>:8000 from any device on the network and sign in: first as the firm
(its sign-in name and shared password; the owner's console is the firm "admin"), then with your email and password.
Fill in the SMTP_* settings in local.env to have sign-up and help requests emailed to you.

It runs the same code as the hosted version, with these differences, all settings in web_deploy/local/local.env:
- BOX_MODE=local: the "GPU box" is this machine and always on; nothing is rented, started or stopped.
- GATEWAY_HTTPS_ONLY=0: plain HTTP on the LAN, so cookies are not marked Secure.
- GPU_SHARING=exclusive: llama-server (8.1 GB at 128K) plus the embedder and reranker fill the 12 GB card, so while
  the ingestion queue has work the live models step aside: chat stops admitting questions, finishes those in flight,
  frees its models and stops llama-server; ingestion runs; when the queue is empty everything comes back.

Three processes, all from the appliance venv: the API (127.0.0.1:8095), the ingestion worker (which owns
llama-server) and the gateway (0.0.0.0:8000, plain HTTP: keep it on a network you trust).
"""

import argparse
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ENV_FILE = HERE / "local.env"
EXAMPLE = HERE / "local.env.example"
API_PORT = 8095


def parse_env_file(path: Path) -> Dict[str, str]:
    """KEY=value lines; quotes around a value are removed, as bash's `.` would (the JSON lists are single-quoted)."""
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def ensure_env_file(env_file: Path = ENV_FILE, example: Path = EXAMPLE) -> bool:
    """Creates local.env from the example with a fresh secret. Returns whether it had to."""
    if env_file.exists():
        return False
    text = example.read_text(encoding="utf-8").replace("WEB_API_SECRET=\n", f"WEB_API_SECRET={secrets.token_hex(32)}\n")
    env_file.write_text(text, encoding="utf-8")
    env_file.chmod(0o600)
    return True


def build_environment(settings: Dict[str, str], base: Dict[str, str], root: Path = ROOT) -> Dict[str, str]:
    """Everything the three processes read, derived from local.env."""
    data = Path(settings.get("LOCAL_DATA_DIR", "onyx_data/local-web"))
    data = data if data.is_absolute() else root / data
    env = dict(base)
    env.update(settings)
    env.update({
        "ENV": base.get("ENV", "development"),
        "ONYX_ROOT": str(root),
        "PYTHONPATH": str(root),
        "TENANTS_ROOT": str(data / "tenants"),
        "SOURCE_CACHE_ROOT": str(data / "source-cache"),
        "CONTROL_DATABASE_URL": f"sqlite:///{data / 'control.sqlite3'}",
        "GATEWAY_DB": str(data / "gateway.sqlite3"),
        "WEB_LOG_DIR": str(data / "logs"),
        "WEB_API_URL": f"http://127.0.0.1:{API_PORT}",
        "BOX_API_PORT": str(API_PORT),
    })
    if not env.get("LLAMA_SERVER_BIN"):
        env["LLAMA_SERVER_BIN"] = str(data / "llama.cpp" / "build" / "bin" / "llama-server")
    env.setdefault("STATIC_DIR", str(root / "web_deploy" / "gateway" / "static"))
    return env


def commands(env: Dict[str, str], python: str = sys.executable) -> Dict[str, List[str]]:
    return {
        "api": [python, "-m", "uvicorn", "web_deploy.api.app:create_production_app", "--factory",
                "--host", "127.0.0.1", "--port", str(API_PORT), "--timeout-keep-alive", "75"],
        "ingest": [python, "-m", "web_deploy.ingest_worker.supervisor"],
        "gateway": [python, "-m", "uvicorn", "web_deploy.gateway.app:create_production_app", "--factory",
                    "--host", "0.0.0.0", "--port", env.get("LOCAL_PORT", "8000")],
    }


def problems(env: Dict[str, str]) -> List[str]:
    """What stops a start, each with how to fix it."""
    found = []
    for module in ("starlette", "uvicorn", "httpx", "yt_dlp"):
        try:
            __import__(module)
        except ImportError:
            found.append(f"'{module}' is not installed in this venv: "
                         f"venv/bin/pip install -r web_deploy/api/requirements.txt httpx")
            break
    if not Path(env["LLAMA_SERVER_BIN"]).exists():
        found.append(f"No llama-server at {env['LLAMA_SERVER_BIN']}: "
                     f"venv/bin/python -m web_deploy.local.run_local build-llama-server")
    if not (Path(env["STATIC_DIR"]) / "index.html").exists():
        found.append("The web app is not built: venv/bin/python -m web_deploy.local.run_local build-web")
    model = ROOT / "models" / "Qwen3.5-9B-Q4_K_M.gguf"
    if not model.exists():
        found.append(f"Missing {model}.")
    return found


def lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))  # no packet is sent; this only picks the outgoing interface
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def start(env: Dict[str, str]) -> int:
    issues = problems(env)
    if issues:
        print("Not ready to start:\n  - " + "\n  - ".join(issues))
        return 1
    logs = Path(env["WEB_LOG_DIR"])
    for directory in (logs, Path(env["TENANTS_ROOT"]), Path(env["SOURCE_CACHE_ROOT"])):
        directory.mkdir(parents=True, exist_ok=True)

    processes: Dict[str, subprocess.Popen] = {}
    for name, command in commands(env).items():
        log = open(logs / f"{name}.log", "ab")
        processes[name] = subprocess.Popen(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)

    port = env.get("LOCAL_PORT", "8000")
    print(f"Izvoria (dev hardware) is starting. Open http://{lan_address()}:{port} on your network "
          f"(http://localhost:{port} here).\nThe models take a minute or two to load. Logs: {logs}\nCtrl+C stops it.")

    stopping = False

    def shutdown(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    exit_code = 0
    while not stopping:
        for name, process in processes.items():
            if process.poll() is not None:
                print(f"'{name}' exited with code {process.returncode}; stopping. See {logs / (name + '.log')}")
                stopping, exit_code = True, 1
                break
        time.sleep(1.0)

    print("Stopping…")
    # The ingestion worker stops llama-server on SIGTERM; an interrupted job is re-queued for the next start
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            process.kill()
    return exit_code


def main(argv: Optional[List[str]] = None) -> int:
    # Status lines must show up at once, also when the output is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(prog="run_local", description=__doc__.split("\n\n")[0])
    parser.add_argument("command", nargs="?", default="start",
                        choices=["start", "setup", "build-llama-server", "build-web", "admin"])
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if ensure_env_file():
        print(f"Created {ENV_FILE} with a fresh secret.")
    env = build_environment(parse_env_file(ENV_FILE), dict(os.environ))

    if args.command == "admin":
        os.environ["GATEWAY_DB"] = env["GATEWAY_DB"]
        Path(env["GATEWAY_DB"]).parent.mkdir(parents=True, exist_ok=True)
        from web_deploy.gateway.admin_cli import main as admin_main
        return admin_main(["--db", env["GATEWAY_DB"], *args.rest])
    if args.command in ("build-llama-server", "setup") and not Path(env["LLAMA_SERVER_BIN"]).exists():
        build_env = dict(env, LLAMA_CPP_DIR=str(Path(env["LLAMA_SERVER_BIN"]).parents[2]))
        code = subprocess.call(["bash", str(ROOT / "web_deploy" / "provision" / "build_llama_server.sh")],
                               env=build_env)
        if code:
            return code
    if args.command in ("build-web", "setup") and not (Path(env["STATIC_DIR"]) / "index.html").exists():
        frontend = ROOT / "web_deploy" / "frontend"
        npm = shutil.which("npm")
        if not npm:
            print("npm is needed to build the web app (Node 18+).")
            return 1
        if subprocess.call([npm, "ci" if (frontend / "package-lock.json").exists() else "install"], cwd=frontend):
            return 1
        if subprocess.call([npm, "run", "build"], cwd=frontend):
            return 1
    if args.command == "setup":
        issues = problems(env)
        print("Ready. Create an owner, a firm and a user with `run_local admin ...`, then start it."
              if not issues else "Still missing:\n  - " + "\n  - ".join(issues))
        return 1 if issues else 0
    if args.command in ("build-llama-server", "build-web"):
        return 0
    return start(env)


if __name__ == "__main__":
    sys.exit(main())
