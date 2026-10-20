"""
Deploys one version of the app to the VPS and every GPU box, so they all run the same code.

Run on the dev box, from a clean checkout of the commit to deploy (system Python, standard library only):

    python3 web_deploy/provision/update_fleet.py                 # the VPS and every box
    python3 web_deploy/provision/update_fleet.py --dry-run       # what it would do
    python3 web_deploy/provision/update_fleet.py --box 55823057  # only this box (and the VPS)

What it does:
1. Takes HEAD as the version (refuses uncommitted changes), builds the web app, and copies exactly that commit plus
   the built web app, with a RELEASE file naming it (web_deploy/common/release.py), to every machine.
2. The VPS: copies it to /opt/onyx_v3, restarts the gateway when gateway code changed, and makes it the current
   version in the gateway's fleet file (fleet.py). From then on the gateway tries up-to-date boxes first.
3. Every box (Vast states read on the VPS, with the gateway's own key, through fleet_cli.py):
   - running (the box in use): copies the code, re-runs provision.sh when dependencies or the box setup changed,
     waits until no question, report or ingestion is running, restarts the services, and checks that /health
     reports the new version;
   - stopped: marks it "under maintenance" so the gateway leaves it alone, starts it, updates it the same way,
     checks the code imports, stops it and lifts the mark. A box whose GPU another Vast customer is using can't be
     started: it is reported as pending, and the gateway uses it only when no up-to-date box can start.
4. Records each updated box's version. Re-running it later updates only the boxes still behind.

Exits 1 when any box is still behind.
"""

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # run as a script: web_deploy.common is standard library only
from web_deploy.common.release import BOX_KEEP, PROVISION_PATHS, provision_digest  # noqa: E402
VPS_ROOT = "/opt/onyx_v3"
BOX_ROOT = "/workspace/onyx_v3"
BOX_ENV = "/workspace/web/api.env"
DIGEST_FILE = "/workspace/web/provision_digest"
GATEWAY_PYTHON = "/opt/onyx-web-gateway/bin/python"
GATEWAY_ENV = "/etc/onyx-web-gateway.env"
# Where the gateway serves each version's package to boxes that update themselves (releases.py); the last few are kept
RELEASES_DIR = "/var/lib/onyx-web/releases"
KEEP_PACKAGES = 3
STATIC = "web_deploy/gateway/static"

# A change here needs the gateway restarted (the code it runs on the VPS)
GATEWAY_PATHS = ("web_deploy/gateway/", "web_deploy/common/", "web_deploy/ingest_worker/", "web_deploy/api/config.py",
                 "web_deploy/backup/", "deploy/gateway/")
# The Vast states of a box that has, or is getting, its GPU
ALLOCATED = ("running", "loading", "created")
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
               "-o", "LogLevel=ERROR"]


class Failed(Exception):
    """One box could not be updated; the run carries on with the others."""


# ---------------------------------------------------------------- decisions (pure, tested)

def touches(changed: Optional[Sequence[str]], paths: Sequence[str]) -> bool:
    """Whether a change set touches any of `paths` (a directory ends in /). Unknown changes touch everything."""
    if changed is None:
        return True
    return any(f == p or (p.endswith("/") and f.startswith(p)) for f in changed for p in paths)


def plan(box: dict, release: str, force: bool) -> str:
    """What to do with one box, as fleet_cli status describes it."""
    if box.get("maintenance"):
        return "skip: under maintenance (another update is running, or one was interrupted)"
    if box.get("release") == release and not force:
        return "up to date"
    if box.get("actual_status") == "running":
        return "update running"
    if box.get("actual_status") in ALLOCATED or box.get("intended_status") == "running":
        return "skip: starting or stopping right now; run again in a minute"
    if box.get("actual_status") == "missing":
        return "skip: Vast doesn't list it (destroyed?)"
    return "update stopped"


# ---------------------------------------------------------------- plumbing

def say(message: str) -> None:
    print(message, flush=True)


def run(command: List[str], check: bool = True, capture: bool = False, cwd: Optional[Path] = None,
        timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, text=True, timeout=timeout,
                            stdout=subprocess.PIPE if capture else None, stderr=subprocess.PIPE if capture else None)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-800:] if capture else ""
        raise Failed(f"{' '.join(command[:4])}… exited {result.returncode}. {detail}")
    return result


def git(*args: str) -> str:
    return run(["git", *args], capture=True, cwd=ROOT).stdout.strip()


def changed_files(old: str, new: str) -> Optional[List[str]]:
    """Files that differ between two commits, or None when the old one is unknown here."""
    if not old or old == "unknown":
        return None
    if run(["git", "cat-file", "-e", f"{old}^{{commit}}"], check=False, capture=True, cwd=ROOT).returncode != 0:
        return None
    return [line for line in git("diff", "--name-only", old, new).splitlines() if line]


class Vps:
    def __init__(self, target: str) -> None:
        self.target = target
        self._public_url: Optional[str] = None

    @property
    def public_url(self) -> str:
        """The gateway's public address (PUBLIC_URL), where boxes fetch new versions at start."""
        if self._public_url is None:
            out = self.ssh(f"sed -n 's/^PUBLIC_URL=//p' {GATEWAY_ENV} | tail -n 1")
            self._public_url = out.strip().strip('"').strip("'").rstrip("/")
        return self._public_url

    def ssh(self, command: str, capture: bool = True, timeout: float = 300) -> str:
        return run(["ssh", *SSH_OPTIONS, self.target, command], capture=capture, timeout=timeout).stdout or ""

    def fleet(self, *args: str, timeout: float = 300) -> dict:
        command = (f"set -a && . {GATEWAY_ENV} && set +a && cd {VPS_ROOT} && "
                   f"{GATEWAY_PYTHON} -m web_deploy.gateway.fleet_cli {' '.join(shlex.quote(a) for a in args)}")
        out = self.ssh(command, timeout=timeout)
        lines = [line for line in out.splitlines() if line.startswith("{")]
        if not lines:
            raise Failed(f"fleet_cli {' '.join(args)} printed no answer: {out[-400:]}")
        return json.loads(lines[-1])


class Box:
    def __init__(self, info: dict) -> None:
        self.id = info["id"]
        self.info = info

    def address(self, vps: Vps) -> List[str]:
        info = self.info
        if not info.get("ssh_host") or not info.get("ssh_port"):
            info = next(b for b in vps.fleet("status")["boxes"] if b["id"] == self.id)
        if not info.get("ssh_host") or not info.get("ssh_port"):
            raise Failed("Vast gives no SSH address for it")
        return ["-p", str(info["ssh_port"]), f"root@{info['ssh_host']}"]

    def ssh(self, vps: Vps, command: str, timeout: float = 600, check: bool = True) -> subprocess.CompletedProcess:
        return run(["ssh", *SSH_OPTIONS, *self.address(vps), command], capture=True, timeout=timeout, check=check)


# ---------------------------------------------------------------- steps

def preflight(allow_unpushed: bool) -> str:
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Uncommitted changes: commit them first, so every machine runs a commit you can find again.")
    release = git("rev-parse", "--short=12", "HEAD")
    upstream = run(["git", "branch", "-r", "--contains", "HEAD"], capture=True, cwd=ROOT, check=False).stdout.strip()
    if not upstream and not allow_unpushed:
        raise SystemExit(f"{release} isn't pushed: push it first (or pass --allow-unpushed).")
    return release


def build_export(release: str, workdir: Path) -> Path:
    say("== Building the web app")
    run(["npm", "run", "build"], cwd=ROOT / "web_deploy" / "frontend", capture=True, timeout=600)
    export = workdir / "release"
    export.mkdir()
    archive = workdir / "release.tar"
    run(["git", "archive", "--format=tar", "-o", str(archive), "HEAD"], cwd=ROOT)
    with tarfile.open(archive) as tar:
        tar.extractall(export, filter="data")
    shutil.copytree(ROOT / STATIC, export / STATIC, dirs_exist_ok=True)
    (export / "RELEASE").write_text(release + "\n")
    return export


def build_package(export: Path, release: str, workdir: Path) -> Dict[str, Path]:
    """The release as boxes fetch it (releases.py): a tarball of the export, and its description."""
    package = workdir / f"{release}.tar.gz"
    with tarfile.open(package, "w:gz") as tar:
        for entry in sorted(export.iterdir()):
            tar.add(entry, arcname=entry.name)
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    meta = workdir / f"{release}.json"
    meta.write_text(json.dumps({"release": release, "sha256": digest, "size": package.stat().st_size,
                                "provision_digest": provision_digest(export)}) + "\n")
    return {"package": package, "meta": meta}


def update_vps(vps: Vps, export: Path, package: Dict[str, Path], release: str, dry_run: bool) -> None:
    say(f"== VPS ({vps.target})")
    previous = vps.ssh(f"cat {VPS_ROOT}/RELEASE 2>/dev/null || echo unknown").strip() or "unknown"
    restart = touches(changed_files(previous, release), GATEWAY_PATHS)
    say(f"   {previous} -> {release}; gateway restart: {'yes' if restart else 'no'}")
    if dry_run:
        return
    # No --delete: the VPS's copy may hold files of its own. The built web app is replaced whole.
    run(["rsync", "-rlptz", "--no-owner", "--no-group", "--chmod=go+rX,go-w", "-e", "ssh " + " ".join(SSH_OPTIONS),
         f"{export}/", f"{vps.target}:{VPS_ROOT}/"])
    run(["rsync", "-rlptz", "--delete", "--no-owner", "--no-group", "--chmod=go+rX,go-w",
         "-e", "ssh " + " ".join(SSH_OPTIONS), f"{export / STATIC}/", f"{vps.target}:{VPS_ROOT}/{STATIC}/"])
    if restart:
        vps.ssh("systemctl restart onyx-web-gateway && sleep 5 && systemctl is-active onyx-web-gateway")
    # The package boxes fetch when they start behind (self_update.py), in place before the version becomes current
    vps.ssh(f"install -d -m 755 {RELEASES_DIR}")
    run(["rsync", "-tz", "--chmod=F644", "-e", "ssh " + " ".join(SSH_OPTIONS), str(package["package"]),
         str(package["meta"]), f"{vps.target}:{RELEASES_DIR}/"])
    vps.ssh(f"cd {RELEASES_DIR} && ls -1t *.tar.gz | tail -n +{KEEP_PACKAGES + 1} | "
            f"while read old; do rm -f \"$old\" \"${{old%.tar.gz}}.json\"; done")
    vps.fleet("set-release", release)
    say("   done; gateway now prefers boxes on this version")


def copy_code(vps: Vps, box: Box, export: Path) -> None:
    excludes = [arg for keep in BOX_KEEP for arg in ("--exclude", keep)]
    port, target = box.address(vps)[1], box.address(vps)[2]
    run(["rsync", "-rlptz", "--delete", *excludes, "-e", f"ssh -p {port} " + " ".join(SSH_OPTIONS),
         f"{export}/", f"{target}:{BOX_ROOT}/"], timeout=1800)
    # What the box's own updates at start need (self_update.py): which Vast instance it is, should Vast's variables be
    # missing, and the gateway's public address
    settings = {"BOX_INSTANCE_ID": str(box.id)}
    if vps.public_url:
        settings["RELEASE_URL"] = vps.public_url
    box.ssh(vps, set_env_lines(BOX_ENV, settings))


def set_env_lines(path: str, values: Dict[str, str]) -> str:
    """A shell command that sets each KEY=value in an env file, replacing the line it already has."""
    commands = []
    for key, value in values.items():
        line = shlex.quote(f"{key}={value}")
        commands.append(f"{{ grep -v '^{key}=' {path} > {path}.new || true; echo {line} >> {path}.new; "
                        f"mv {path}.new {path}; }}")
    return f"chmod 600 {path} && " + " && ".join(commands) + f" && chmod 600 {path}"


def mark_provisioned(vps: Vps, box: Box, export: Path) -> None:
    """Tells the box's self_update.py that its setup matches this version's (so it won't re-run provision.sh)."""
    box.ssh(vps, f"echo {provision_digest(export)} > {DIGEST_FILE}")


def provision(vps: Vps, box: Box) -> None:
    say("   re-running provision.sh (dependencies or box setup changed); minutes, more if llama-server rebuilds")
    result = box.ssh(vps, f"bash {BOX_ROOT}/web_deploy/provision/provision.sh > /workspace/logs/provision.log 2>&1;"
                          f" code=$?; tail -n 15 /workspace/logs/provision.log; exit $code", timeout=3 * 3600,
                     check=False)
    if result.returncode != 0:
        raise Failed(f"provision.sh failed:\n{result.stdout[-1500:]}")


def wait_idle(vps: Vps, minutes: float) -> None:
    deadline = time.monotonic() + minutes * 60
    while True:
        health = vps.fleet("health")
        if health.get("reachable") and not health.get("busy"):
            return
        if time.monotonic() > deadline:
            raise Failed(f"still busy after {minutes:g} min (a question, report or ingestion is running); "
                         f"run again later")
        say("   waiting for running work to finish…")
        time.sleep(20)


def wait_release(vps: Vps, release: str, minutes: float = 12) -> None:
    deadline = time.monotonic() + minutes * 60
    last = {}
    while time.monotonic() < deadline:
        last = vps.fleet("health")
        if last.get("ready") and last.get("release") == release:
            return
        time.sleep(10)
    raise Failed(f"after the restart /health says {last}")


def update_running(vps: Vps, box: Box, export: Path, changed: Optional[List[str]], wait_minutes: float) -> None:
    # Idle first: a running process imports some modules late, and must not pick up new files mid-answer
    wait_idle(vps, wait_minutes)
    say("   copying the code")
    copy_code(vps, box, export)
    reprovision = touches(changed, PROVISION_PATHS)
    if reprovision:
        provision(vps, box)
        wait_idle(vps, wait_minutes)
    mark_provisioned(vps, box, export)
    say("   restarting the services")
    programs = "all" if reprovision else "api ingest"
    # Not checked: a tunnel can take a few respawns to come back. wait_release is the check (ready, this version).
    box.ssh(vps, f"supervisorctl reread >/dev/null; supervisorctl update >/dev/null; supervisorctl restart {programs}",
            check=False)
    wait_release(vps, export.joinpath("RELEASE").read_text().strip())


def update_stopped(vps: Vps, box: Box, export: Path, changed: Optional[List[str]]) -> None:
    vps.fleet("maintenance", "add", str(box.id))
    try:
        say("   starting it (the gateway leaves it alone meanwhile)")
        started = vps.fleet("start", str(box.id), timeout=600)
        if not started.get("started"):
            raise Failed(f"Vast couldn't start it ({started.get('reason')}): another customer has its GPU. "
                         f"The gateway uses it only if no up-to-date box can start; run this again later")
        try:
            wait_ssh(vps, box)
            # Its services start at boot with the old code. The box in use holds the tunnel, so they can't reach the
            # data (ExitOnForwardFailure), but stop them at once rather than rely on that
            box.ssh(vps, "supervisorctl stop all >/dev/null 2>&1; true", check=False)
            say("   copying the code")
            copy_code(vps, box, export)
            if touches(changed, PROVISION_PATHS):
                provision(vps, box)
            release = export.joinpath("RELEASE").read_text().strip()
            # datasets loads pyarrow, which refuses a NumPy it wasn't built for only when imported (engine start)
            check = box.ssh(vps, f"cd {BOX_ROOT} && cat RELEASE && venv/bin/python -c "
                                 f"'import web_deploy.api.app, web_deploy.ingest_worker.supervisor, datasets' 2>&1 | tail -3",
                            check=False)
            if check.returncode != 0 or release not in check.stdout:
                raise Failed(f"the updated code doesn't load:\n{check.stdout[-800:]}")
            mark_provisioned(vps, box, export)
        finally:
            say("   stopping it")
            vps.fleet("stop", str(box.id))
    finally:
        vps.fleet("maintenance", "remove", str(box.id))


def wait_ssh(vps: Vps, box: Box, minutes: float = 6) -> None:
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        try:
            if box.ssh(vps, "true", timeout=30, check=False).returncode == 0:
                return
        except (Failed, subprocess.TimeoutExpired):
            pass
        time.sleep(10)
    raise Failed("its SSH didn't come up")


def update_boxes(vps: Vps, export: Path, release: str, only: Sequence[int], force: bool, dry_run: bool,
                 wait_minutes: float) -> Dict[int, str]:
    results: Dict[int, str] = {}
    for info in vps.fleet("status")["boxes"]:
        box = Box(info)
        if only and box.id not in only:
            continue
        action = plan(info, release, force)
        price = f"${info['price_per_hour']:.3f}/h" if info.get("price_per_hour") else "price unknown"
        say(f"== Box {box.id} ({info['actual_status']}, {price}, holds {info.get('release') or 'unknown'}): {action}")
        if dry_run or not action.startswith("update"):
            results[box.id] = action
            continue
        changed = changed_files(info.get("release", ""), release)
        try:
            if action == "update running":
                update_running(vps, box, export, changed, wait_minutes)
            else:
                update_stopped(vps, box, export, changed)
            vps.fleet("record", str(box.id), release)
            results[box.id] = "updated"
            say(f"   updated to {release}")
        except (Failed, subprocess.TimeoutExpired) as exc:
            results[box.id] = f"NOT updated: {exc}"
            say(f"   NOT updated: {exc}")
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vps", default="root@app.izvoria.com", help="SSH target of the gateway VPS")
    parser.add_argument("--box", type=int, action="append", default=[], help="Only this box (repeatable)")
    parser.add_argument("--skip-vps", action="store_true", help="Leave the VPS as it is")
    parser.add_argument("--force", action="store_true", help="Update boxes that already hold this version")
    parser.add_argument("--dry-run", action="store_true", help="Say what would happen; change nothing")
    parser.add_argument("--wait-idle", type=float, default=20, metavar="MIN",
                        help="How long to wait for running work on the box in use (default 20)")
    parser.add_argument("--allow-unpushed", action="store_true")
    args = parser.parse_args(argv)

    release = preflight(args.allow_unpushed)
    say(f"Deploying {release}: {git('log', '-1', '--format=%s')}")
    vps = Vps(args.vps)
    with tempfile.TemporaryDirectory(prefix="izvoria-release-") as tmp:
        export = build_export(release, Path(tmp)) if not args.dry_run else Path(tmp)
        package = build_package(export, release, Path(tmp)) if not args.dry_run else {}
        if not args.skip_vps:
            update_vps(vps, export, package, release, args.dry_run)
        results = update_boxes(vps, export, release, args.box, args.force, args.dry_run, args.wait_idle)

    say("\n== Summary")
    for instance, result in results.items():
        say(f"   {instance}: {result}")
    behind = [i for i, r in results.items() if not (r in ("updated", "up to date") or args.dry_run)]
    if behind:
        say(f"\n{len(behind)} box(es) still behind. Run this again later; meanwhile the gateway tries them last.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
