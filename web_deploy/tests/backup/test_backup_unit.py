import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from web_deploy.backup.config import BackupSettings
from web_deploy.backup.restic import Restic, ResticError, sftp_target
from web_deploy.backup.restore import Restorer
from web_deploy.backup.runner import (
    RETRY_AFTER_SECONDS, BackupRunner, is_due, is_running, read_status,
)

NOW = 2_000_000_000.0
REPO = "sftp:u123@u123.your-storagebox.de:onyx-backups"


def _settings(tmp_path, **env):
    base = {
        "BACKUP_ENABLED": "true", "RESTIC_REPOSITORY": REPO, "RESTIC_PASSWORD": "pw",
        "WEB_DIR": str(tmp_path / "web"), "TENANTS_ROOT": str(tmp_path / "tenants"),
        "BACKUP_STAGING_DIR": str(tmp_path / "staging"), "POSTGRES_PASSWORD": "dbpw",
    }
    base.update(env)
    return BackupSettings.from_env(base)


class FakeRun:
    """Records every command; answers psql's database list and fails what it is told to."""

    def __init__(self, databases=("onyx_control", "onyx_firm_legal_demo"), fail=()):
        self.calls = []
        self.databases = databases
        self.fail = fail

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        name = command[0] if command[0] != "restic" else "restic " + next(
            a for a in command[1:] if not a.startswith("-") and not a.startswith("sftp.command"))
        if name in self.fail:
            return SimpleNamespace(returncode=1, stdout="", stderr=f"{name} broke")
        if command[0] == "psql":
            return SimpleNamespace(returncode=0, stdout="\n".join(self.databases) + "\n", stderr="")
        if command[0] == "pg_dump":
            Path(command[command.index("--file") + 1]).write_bytes(b"PGDMP")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def names(self):
        out = []
        for command, _ in self.calls:
            if command[0] == "restic":
                rest = [a for a in command[1:] if a not in ("-o",) and not a.startswith("sftp.command")]
                out.append("restic " + rest[0])
            else:
                out.append(command[0])
        return out


class Queue:
    def __init__(self, queued=0, running=0, finished=None):
        self._depth = {"queued": queued, "running": running}
        self._finished = finished

    def depth(self):
        return self._depth

    def last_finished_at(self):
        return self._finished


class TestSettings:
    def test_defaults_are_off_and_point_at_the_box(self):
        settings = BackupSettings.from_env({})
        assert settings.configured is False
        assert (settings.ssh_port, settings.host, settings.interval_hours) == (23, "onyx-box", 24.0)
        assert settings.tenants_root == Path("/workspace/tenants")
        assert settings.extra_paths == (Path("/workspace/web"),)
        assert settings.keep == (("daily", 7), ("weekly", 4), ("monthly", 6))

    def test_configured_needs_the_switch_a_repository_and_a_password(self, tmp_path):
        assert _settings(tmp_path).configured
        assert not _settings(tmp_path, BACKUP_ENABLED="false").configured
        assert not _settings(tmp_path, RESTIC_PASSWORD="").configured


class TestRestic:
    @pytest.mark.parametrize("repository, target", [
        (REPO, "u123@u123.your-storagebox.de"),
        ("sftp://u1@box.de:23//home/backups", "u1@box.de"),
        ("/srv/restic", None),
    ])
    def test_sftp_target(self, repository, target):
        assert sftp_target(repository) == target

    def test_commands_log_in_with_the_boxs_key_on_port_23(self, tmp_path):
        settings = _settings(tmp_path)
        run = FakeRun()

        Restic(settings, run=run).backup([Path("/a")], ["/a/*/uploads"], "box")

        command, kwargs = run.calls[0]
        assert command[:3] == ["restic", "-o", f"sftp.command=ssh -i {tmp_path}/web/backup_ed25519 -p 23 "
                                                f"-o StrictHostKeyChecking=accept-new -o BatchMode=yes "
                                                f"u123@u123.your-storagebox.de -s sftp"]
        assert command[3:] == ["backup", "--host", "onyx-box", "--tag", "box", "--exclude", "/a/*/uploads", "/a"]
        assert (kwargs["env"]["RESTIC_REPOSITORY"], kwargs["env"]["RESTIC_PASSWORD"]) == (REPO, "pw")

    def test_a_known_hosts_file_can_be_given(self, tmp_path):
        run = FakeRun()
        Restic(_settings(tmp_path, BACKUP_KNOWN_HOSTS="/var/lib/onyx-web/known_hosts"), run=run).unlock()

        assert "-o UserKnownHostsFile=/var/lib/onyx-web/known_hosts -o BatchMode=yes" in run.calls[0][0][2]

    def test_forget_keeps_the_policy_for_this_host_only(self, tmp_path):
        run = FakeRun()
        Restic(_settings(tmp_path), run=run).forget("box")

        assert run.calls[0][0][3:] == ["forget", "--host", "onyx-box", "--tag", "box", "--prune",
                                       "--keep-daily", "7", "--keep-weekly", "4", "--keep-monthly", "6"]

    def test_a_new_repository_is_initialised_once(self, tmp_path):
        run = FakeRun(fail=("restic cat",))
        Restic(_settings(tmp_path), run=run).ensure_repository()
        assert run.names() == ["restic cat", "restic init"]

        run = FakeRun()
        Restic(_settings(tmp_path), run=run).ensure_repository()
        assert run.names() == ["restic cat"]

    def test_a_failure_raises_with_restics_message(self, tmp_path):
        with pytest.raises(ResticError, match="restic backup broke"):
            Restic(_settings(tmp_path), run=FakeRun(fail=("restic backup",))).backup([], [], "box")


class TestIsDue:
    @pytest.fixture
    def settings(self, tmp_path):
        return _settings(tmp_path)

    def test_never_backed_up(self, settings):
        assert is_due(settings, {}, None, NOW)

    def test_off_is_never_due(self, tmp_path):
        assert not is_due(_settings(tmp_path, BACKUP_ENABLED="false"), {}, None, NOW)

    def test_after_the_interval(self, settings):
        assert not is_due(settings, {"last_success": NOW - 3600}, None, NOW)
        assert is_due(settings, {"last_success": NOW - 25 * 3600}, None, NOW)

    def test_as_soon_as_new_material_has_landed(self, settings):
        landed = datetime.fromtimestamp(NOW - 60, tz=timezone.utc)
        assert is_due(settings, {"last_success": NOW - 3600}, landed, NOW)
        assert not is_due(settings, {"last_success": NOW - 30}, landed, NOW)

    def test_a_failed_run_waits_an_hour_before_trying_again(self, settings):
        failed = {"last_error": "unreachable", "last_attempt": NOW - 60}
        assert not is_due(settings, failed, None, NOW)
        failed["last_attempt"] = NOW - RETRY_AFTER_SECONDS
        assert is_due(settings, failed, None, NOW)


class TestBackupRunner:
    @pytest.fixture
    def settings(self, tmp_path):
        settings = _settings(tmp_path)
        (settings.tenants_root / "legal-demo" / "vault").mkdir(parents=True)
        (tmp_path / "web").mkdir()
        return settings

    def test_a_backup_dumps_every_firm_database_then_snapshots_and_prunes(self, settings, tmp_path):
        run = FakeRun()
        ok = BackupRunner(settings, Queue(), run=run, clock=lambda: NOW).run_once()

        assert ok
        assert run.names() == ["restic unlock", "restic cat", "psql", "pg_dump", "pg_dump", "restic backup",
                               "restic forget"]
        dump = run.calls[3][0]
        assert dump[:7] == ["pg_dump", "-h", "127.0.0.1", "-p", "5432", "-U", "onyx"]
        assert dump[-1] == "onyx_control" and "--format=custom" in dump
        assert run.calls[3][1]["env"]["PGPASSWORD"] == "dbpw"
        backup = run.calls[5][0]
        assert backup[-3:] == [str(settings.tenants_root), str(settings.staging), str(tmp_path / "web")]
        assert str(settings.tenants_root / "*" / "uploads") in backup
        status = read_status(settings.status_file)
        assert status["running"] is False and status["last_success"] == NOW and status["last_error"] is None
        assert status["databases"] == ["onyx_control", "onyx_firm_legal_demo"]
        assert not (settings.staging / "databases").exists()

    def test_a_failure_is_recorded_and_retried_later(self, settings):
        run = FakeRun(fail=("restic backup",))
        ok = BackupRunner(settings, Queue(), run=run, clock=lambda: NOW).run_once()

        status = read_status(settings.status_file)
        assert not ok
        assert "broke" in status["last_error"] and status["running"] is False
        assert status["last_attempt"] == NOW and "last_success" not in status

    def test_never_while_the_ingestion_queue_has_work(self, settings):
        assert not BackupRunner(settings, Queue(running=1), run=FakeRun(), clock=lambda: NOW).due()
        assert not BackupRunner(settings, Queue(queued=1), run=FakeRun(), clock=lambda: NOW).due()
        assert BackupRunner(settings, Queue(), run=FakeRun(), clock=lambda: NOW).due()


class TestStatus:
    def test_a_run_whose_process_died_is_not_running(self, tmp_path):
        assert is_running({"running": True, "pid": os.getpid()}) is True
        assert is_running({"running": True, "pid": 999_999_999}) is False


class TestRestorer:
    def test_puts_back_databases_vaults_and_the_tls_certificate(self, tmp_path):
        settings = _settings(tmp_path)
        work = tmp_path / "restore"
        run = FakeRun()

        class FakeRestic:
            def restore(self, snapshot, target, tag):
                assert (snapshot, target, tag) == ("latest", work, "box")
                root = target / settings.staging.relative_to("/") / "databases"
                root.mkdir(parents=True)
                (root / "onyx_control.dump").write_bytes(b"PGDMP")
                (root / "onyx_firm_legal_demo.dump").write_bytes(b"PGDMP")
                vault = target / settings.tenants_root.relative_to("/") / "legal-demo" / "vault"
                vault.mkdir(parents=True)
                (vault / "talk.mp4").write_bytes(b"video")
                tls = target / (tmp_path / "web").relative_to("/") / "tls"
                tls.mkdir(parents=True)
                (tls / "cert.pem").write_text("CERT")

        (settings.tenants_root / "legal-demo" / "vault").mkdir(parents=True)
        (settings.tenants_root / "legal-demo" / "vault" / "stale.mp4").write_bytes(b"old")

        names = Restorer(settings, restic=FakeRestic(), run=run, log=lambda _: None).restore("latest", work)

        assert names == ["onyx_control", "onyx_firm_legal_demo"]
        assert run.names() == ["dropdb", "createdb", "pg_restore"] * 2
        assert run.calls[0][0][-2:] == ["--if-exists", "onyx_control"]
        assert "--maintenance-db" in run.calls[1][0]
        vault = settings.tenants_root / "legal-demo" / "vault"
        assert sorted(p.name for p in vault.iterdir()) == ["talk.mp4"]
        assert (settings.tenants_root / "legal-demo" / "uploads").is_dir()
        assert (tmp_path / "web" / "tls" / "cert.pem").read_text() == "CERT"

    def test_a_snapshot_without_dumps_is_refused(self, tmp_path):
        settings = _settings(tmp_path)

        class EmptyRestic:
            def restore(self, snapshot, target, tag):
                target.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="no database dumps"):
            Restorer(settings, restic=EmptyRestic(), run=FakeRun(), log=lambda _: None).restore("latest",
                                                                                                tmp_path / "r")


class TestGatewayBackup:
    def test_copies_the_live_database_and_snapshots_it_as_its_own_series(self, tmp_path):
        import sqlite3
        from web_deploy.backup import gateway
        db = tmp_path / "gateway.sqlite3"
        with sqlite3.connect(db) as conn:
            conn.execute("CREATE TABLE firms(slug TEXT)")
            conn.execute("INSERT INTO firms VALUES ('legal-demo')")
        calls = []

        class FakeRestic:
            def unlock(self): calls.append("unlock")
            def ensure_repository(self): calls.append("init")
            def backup(self, paths, excludes, tag): calls.append(("backup", paths, tag))
            def forget(self, tag): calls.append(("forget", tag))

        code = gateway.run({"BACKUP_ENABLED": "true", "RESTIC_REPOSITORY": REPO, "RESTIC_PASSWORD": "pw",
                            "GATEWAY_DB": str(db)}, restic=FakeRestic())

        copy = tmp_path / "backup" / "gateway.sqlite3"
        assert code == 0
        assert calls == ["unlock", "init", ("backup", [copy.parent], "gateway"), ("forget", "gateway")]
        with sqlite3.connect(copy) as conn:
            assert conn.execute("SELECT slug FROM firms").fetchall() == [("legal-demo",)]

    def test_off_until_configured(self, tmp_path):
        from web_deploy.backup import gateway
        assert gateway.run({"GATEWAY_DB": str(tmp_path / "x.sqlite3")}, restic=None) == 0

    def test_its_series_is_named_for_the_gateway(self, tmp_path, monkeypatch):
        import sqlite3
        from web_deploy.backup import gateway
        seen = {}

        class FakeRestic:
            def __init__(self, settings):
                seen.update(host=settings.host, key=settings.ssh_key)

            def __getattr__(self, name):
                return lambda *args: None

        monkeypatch.setattr(gateway, "Restic", FakeRestic)
        db = tmp_path / "g.sqlite3"
        sqlite3.connect(db).close()

        gateway.run({"BACKUP_ENABLED": "true", "RESTIC_REPOSITORY": REPO, "RESTIC_PASSWORD": "pw",
                     "GATEWAY_DB": str(db)})

        assert seen == {"host": "onyx-gateway", "key": Path("/etc/onyx-web/backup_ed25519")}
