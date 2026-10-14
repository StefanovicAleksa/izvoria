import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from web_deploy.ingest_worker.queue import IngestMode, IngestQueue
from web_deploy.provision import compact_storage as module
from web_deploy.provision.compact_storage import RESULT_PREFIX, compact_firm, describe, firms_under, main

DAY = 24 * 3600


def _result(**overrides):
    result = {"dry_run": False, "sources_checked": 2, "clip_bytes": 6_000_000_000, "audio_bytes": 2_000_000_000,
              "originals_replaced": 1, "original_bytes_saved": 500_000_000, "originals_to_compact": 0,
              "original_bytes_to_compact": 0, "errors": {}, "bytes_saved": 8_500_000_000}
    result.update(overrides)
    return result


def _batch(tenants, firm, name, age_seconds):
    folder = tenants / firm / "uploads" / "u1" / name
    folder.mkdir(parents=True)
    (folder / "talk.mp4").write_bytes(b"x" * 1000)
    when = __import__("time").time() - age_seconds
    for path in (folder, folder / "talk.mp4"):
        os.utime(path, (when, when))
    return folder


class TestFirmsUnder:
    def test_every_firm_with_a_vault(self, tmp_path):
        (tmp_path / "legal-demo" / "vault").mkdir(parents=True)
        (tmp_path / "clinic-demo" / "vault").mkdir(parents=True)
        (tmp_path / "half-made" / "uploads").mkdir(parents=True)

        assert firms_under(tmp_path) == ["clinic-demo", "legal-demo"]

    def test_no_tenants_yet(self, tmp_path):
        assert firms_under(tmp_path / "missing") == []


class TestCompactFirm:
    def test_runs_a_child_in_the_firms_environment_and_reads_its_result(self, tmp_path):
        calls = []

        def run(command, **kwargs):
            calls.append((command, kwargs))
            return SimpleNamespace(returncode=0, stdout="log line\n" + RESULT_PREFIX + json.dumps(_result()) + "\n")

        result = compact_firm("legal-demo", True, tmp_path, {"PATH": "/bin"}, run=run)

        (command, kwargs), = calls
        assert command[1:] == ["-m", "web_deploy.provision.compact_storage", "--child", "legal-demo", "--dry-run"]
        assert kwargs["env"]["POSTGRES_DB"] == "onyx_firm_legal_demo"
        assert kwargs["env"]["ONYX_STORAGE_PATH"] == str(tmp_path / "legal-demo" / "vault")
        assert kwargs["env"]["ENGINE_CONCURRENT_LIVE"] == "false"
        assert kwargs["stdout"] is subprocess.PIPE
        assert result == _result()

    def test_a_child_without_a_result_is_a_failure(self, tmp_path):
        run = lambda command, **kwargs: SimpleNamespace(returncode=1, stdout="Traceback ...")

        with pytest.raises(RuntimeError, match="exit code 1"):
            compact_firm("legal-demo", False, tmp_path, {}, run=run)


class TestDescribe:
    def test_a_real_run(self):
        line = describe("legal-demo", _result(errors={"bad.mkv": "corrupt"}))

        assert line.startswith("legal-demo: 2 indexed source(s) · clips 6.00 GB · WAVs 2.00 GB")
        assert "1 original(s) compacted, saving 500.0 MB -> freed 8.50 GB" in line
        assert line.endswith("\n    bad.mkv: corrupt")

    def test_a_dry_run(self):
        line = describe("legal-demo", _result(dry_run=True, originals_replaced=0, original_bytes_saved=0,
                                              originals_to_compact=3, original_bytes_to_compact=9_000_000_000,
                                              bytes_saved=8_000_000_000))

        assert "3 original(s) to re-encode (9.00 GB now; the saving shows after) -> would free 8.00 GB" in line


class TestMain:
    @pytest.fixture
    def tenants(self, tmp_path):
        root = tmp_path / "tenants"
        (root / "legal-demo" / "vault").mkdir(parents=True)
        return root

    def _settings(self, tenants):
        return SimpleNamespace(tenants_root=tenants, control_database_url="sqlite://")

    def test_compacts_every_firm_and_removes_uploads_no_job_will_read(self, tenants, capsys):
        queue = IngestQueue.connect("sqlite://")
        old = _batch(tenants, "legal-demo", "20261001-000000-aaaaaa", 2 * DAY)
        fresh = _batch(tenants, "legal-demo", "20261009-000000-bbbbbb", 60)
        queued = _batch(tenants, "legal-demo", "20261002-000000-cccccc", 2 * DAY)
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(queued), 1)
        seen = []

        def compact(firm, dry_run, root, env):
            seen.append((firm, dry_run, root))
            return _result()

        code = main(["--all"], settings=self._settings(tenants), queue=queue, compact=compact)

        assert code == 0
        assert seen == [("legal-demo", False, tenants)]
        assert not old.exists()
        assert fresh.exists() and queued.exists()
        out = capsys.readouterr().out
        assert "uploads: 1 folder(s) no job will read -> freed 0.0 MB" in out
        assert "total: freed 8.50 GB" in out

    def test_a_dry_run_removes_nothing(self, tenants, capsys):
        queue = IngestQueue.connect("sqlite://")
        old = _batch(tenants, "legal-demo", "20261001-000000-aaaaaa", 2 * DAY)

        main(["legal-demo", "--dry-run"], settings=self._settings(tenants), queue=queue,
             compact=lambda firm, dry_run, root, env: _result(dry_run=dry_run))

        assert old.exists()
        assert "would free" in capsys.readouterr().out

    def test_refused_while_an_ingestion_runs(self, tenants, capsys):
        queue = IngestQueue.connect("sqlite://")
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", "/b", 1)
        queue.claim_next()

        code = main(["--all"], settings=self._settings(tenants), queue=queue,
                    compact=lambda *a: pytest.fail("must not compact"))

        assert code == 1
        assert "an ingestion is running" in capsys.readouterr().err

    def test_refused_on_a_box_that_keeps_no_files(self, tenants, capsys, monkeypatch):
        monkeypatch.setenv("BOX_GATEWAY_URL", "http://127.0.0.1:18000")

        code = main(["--all"], settings=self._settings(tenants), queue=IngestQueue.connect("sqlite://"),
                    compact=lambda *a: pytest.fail("must not compact"))

        assert code == 1
        assert "holds no files" in capsys.readouterr().err

    def test_a_failed_firm_is_reported_and_the_others_continue(self, tenants, capsys):
        (tenants / "clinic-demo" / "vault").mkdir(parents=True)
        queue = IngestQueue.connect("sqlite://")

        def compact(firm, dry_run, root, env):
            if firm == "clinic-demo":
                raise RuntimeError("compaction of clinic-demo failed (exit code 1)")
            return _result()

        code = main(["--all"], settings=self._settings(tenants), queue=queue, compact=compact)

        captured = capsys.readouterr()
        assert code == 1
        assert "clinic-demo: compaction of clinic-demo failed" in captured.err
        assert "legal-demo: 2 indexed source(s)" in captured.out
