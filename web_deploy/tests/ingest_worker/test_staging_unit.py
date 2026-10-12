import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from web_deploy.ingest_worker.staging import Stager, StagingError, StagingSettings, pending_markers

REMOTE = "izvoria-box@vps.test"


class FakeVps:
    """
    rsync as the stager calls it, against a folder standing in for the VPS's /workspace/tenants. Understands the four
    shapes the stager uses: pull a folder, pull a file list, push a folder, and a --delete push filtered down to the
    files to delete.
    """

    def __init__(self, root: Path, scratch: Path):
        self.root = root
        self.scratch = scratch
        self.commands = []
        self.fail = False

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if self.fail:
            return subprocess.CompletedProcess(command, 255, "", "ssh: connect to host vps.test: Connection refused")
        args = command[command.index("-e") + 2:]
        listing = next((a.split("=", 1)[1] for a in args if a.startswith("--files-from=")), None)
        rules = next((a.split(" ", 1)[1] for a in args if a.startswith("--filter=merge ")), None)
        source, target = args[-2], args[-1]
        if rules:
            assert "--delete" in args and Path(rules).read_text().splitlines()[-1] == "- *"
            for line in Path(rules).read_text().splitlines():
                if line.startswith("+ /") and not line.endswith("/") and not (self.scratch / line[3:]).exists():
                    (self.root / line[3:]).unlink(missing_ok=True)
            return subprocess.CompletedProcess(command, 0, "", "")
        if listing:
            for name in [line for line in Path(listing).read_text().splitlines() if line]:
                if (self.root / name).exists():
                    (self.scratch / name).parent.mkdir(parents=True, exist_ok=True)
                    (self.scratch / name).write_bytes((self.root / name).read_bytes())
            return subprocess.CompletedProcess(command, 0, "", "")
        if source.startswith(REMOTE):
            self._copy(self.root / source.split(":", 1)[1].rstrip("/"), Path(target.rstrip("/")))
        else:
            self._copy(Path(source.rstrip("/")), self.root / target.split(":", 1)[1].rstrip("/"))
        return subprocess.CompletedProcess(command, 0, "", "")

    @staticmethod
    def _copy(src: Path, dst: Path):
        if not src.is_dir():
            raise AssertionError(f"rsync: change_dir {src} failed: No such file or directory")
        for path in src.rglob("*"):
            if path.is_file():
                out = dst / path.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(path.read_bytes())


@pytest.fixture
def scratch(tmp_path):
    return tmp_path / "box" / "tenants"


@pytest.fixture
def vps(tmp_path, scratch):
    root = tmp_path / "vps" / "tenants"
    root.mkdir(parents=True)
    return FakeVps(root, scratch)


@pytest.fixture
def settings(tmp_path, scratch):
    return StagingSettings(remote=REMOTE, tenants_root=scratch, state_dir=tmp_path / "box" / "state",
                           ssh_key=Path("/workspace/web/sync_ed25519"), known_hosts=Path("/workspace/web/known_hosts"))


def _upload(vps, firm="legal-demo", batch="20261012-120000-abcdef", content=b"the recording"):
    folder = vps.root / firm / "uploads" / "u1" / batch
    folder.mkdir(parents=True)
    (folder / "talk.mp4").write_bytes(content)
    return folder


class TestStageIn:
    def test_the_batch_arrives_at_the_same_path_the_job_names(self, settings, vps, scratch):
        _upload(vps)
        batch = scratch / "legal-demo" / "uploads" / "u1" / "20261012-120000-abcdef"

        Stager(settings, lookup=lambda firm, hashes: [], run=vps).stage_in("j1", "legal-demo", str(batch))

        assert (batch / "talk.mp4").read_bytes() == b"the recording"
        assert pending_markers(settings.state_dir) == [settings.state_dir / "j1.json"]

    def test_vault_files_of_a_source_with_the_same_content_come_too(self, settings, vps, scratch):
        _upload(vps)
        original = vps.root / "legal-demo" / "vault" / "2026" / "10" / "01" / "old.mp4"
        original.parent.mkdir(parents=True)
        original.write_bytes(b"compact copy")
        asked = []

        def lookup(firm, hashes):
            asked.append((firm, hashes))
            return [str(scratch / "legal-demo" / "vault" / "2026" / "10" / "01" / "old.mp4"),
                    str(scratch / "legal-demo" / "vault" / "2026" / "10" / "01" / "gone.wav")]  # missing on the VPS

        batch = scratch / "legal-demo" / "uploads" / "u1" / "20261012-120000-abcdef"
        pulled = Stager(settings, lookup=lookup, run=vps).stage_in("j1", "legal-demo", str(batch))

        assert asked == [("legal-demo", [hashlib.sha256(b"the recording").hexdigest()])]
        assert (scratch / "legal-demo" / "vault" / "2026" / "10" / "01" / "old.mp4").read_bytes() == b"compact copy"
        assert [Path(p).name for p in pulled] == ["old.mp4"]  # only what arrived
        assert json.loads((settings.state_dir / "j1.json").read_text())["pulled"] == pulled

    def test_a_youtube_batch_pulls_nothing(self, settings, vps):
        Stager(settings, lookup=lambda f, h: pytest.fail("nothing to hash"), run=vps).stage_in("j1", "legal-demo", None)
        assert vps.commands == []

    def test_nothing_outside_the_tenants_is_ever_fetched(self, settings, vps):
        with pytest.raises(StagingError):
            Stager(settings, lookup=lambda f, h: [], run=vps).stage_in("j1", "legal-demo", "/etc/ssh")

    def test_rsync_runs_over_the_pinned_ssh_key_with_the_vps_host_key_checked(self, settings, vps, scratch):
        _upload(vps)
        batch = scratch / "legal-demo" / "uploads" / "u1" / "20261012-120000-abcdef"
        Stager(settings, lookup=lambda f, h: [], run=vps).stage_in("j1", "legal-demo", str(batch))

        ssh = vps.commands[0][vps.commands[0].index("-e") + 1]
        assert "-i /workspace/web/sync_ed25519" in ssh
        assert "StrictHostKeyChecking=yes" in ssh and "UserKnownHostsFile=/workspace/web/known_hosts" in ssh
        assert vps.commands[0][-2] == f"{REMOTE}:legal-demo/uploads/u1/20261012-120000-abcdef/"


class TestStageOut:
    def _job_that_wrote(self, settings, vps, scratch, pulled=()):
        stager = Stager(settings, lookup=lambda f, h: list(pulled), run=vps)
        batch = scratch / "legal-demo" / "uploads" / "u1" / "20261012-120000-abcdef"
        _upload(vps)
        for path in pulled:
            remote = vps.root / Path(path).relative_to(scratch)
            remote.parent.mkdir(parents=True, exist_ok=True)
            remote.write_bytes(b"old original")
        stager.stage_in("j1", "legal-demo", str(batch))
        new = scratch / "legal-demo" / "vault" / "2026" / "10" / "12" / "new.mp4"
        new.parent.mkdir(parents=True, exist_ok=True)
        new.write_bytes(b"indexed and compacted")
        return stager, new

    def test_the_vault_goes_to_the_vps_and_the_box_keeps_nothing(self, settings, vps, scratch):
        stager, new = self._job_that_wrote(settings, vps, scratch)

        stager.stage_out("j1")

        assert (vps.root / new.relative_to(scratch)).read_bytes() == b"indexed and compacted"
        assert [p for p in scratch.rglob("*") if p.is_file()] == []
        assert pending_markers(settings.state_dir) == []

    def test_a_pulled_original_the_job_replaced_is_deleted_on_the_vps(self, settings, vps, scratch):
        old = str(scratch / "legal-demo" / "vault" / "2026" / "10" / "01" / "old.mov")
        stager, _ = self._job_that_wrote(settings, vps, scratch, pulled=[old])
        Path(old).unlink()  # compaction swapped it for a copy under a new path

        neighbour = vps.root / "legal-demo" / "vault" / "2026" / "10" / "01" / "frame.jpg"
        neighbour.write_bytes(b"another source's frame")

        stager.stage_out("j1")

        assert not (vps.root / Path(old).relative_to(scratch)).exists()
        assert neighbour.exists()

    def test_a_failed_push_keeps_everything_for_the_retry(self, settings, vps, scratch):
        stager, new = self._job_that_wrote(settings, vps, scratch)
        vps.fail = True

        with pytest.raises(StagingError, match="Connection refused"):
            stager.stage_out("j1")

        assert new.exists()
        assert stager.has_pending()

        vps.fail = False
        assert stager.recover() == 0
        assert (vps.root / new.relative_to(scratch)).exists()
        assert not new.exists()
        assert not stager.has_pending()


class TestOff:
    def test_without_a_remote_the_box_and_the_data_share_one_disk(self, tmp_path):
        settings = StagingSettings(remote="", tenants_root=tmp_path, state_dir=tmp_path / "state")
        stager = Stager(settings, lookup=lambda f, h: pytest.fail("no lookup"), run=lambda *a, **k: pytest.fail("no rsync"))
        (tmp_path / "legal-demo" / "vault").mkdir(parents=True)
        (tmp_path / "legal-demo" / "vault" / "keep.mp4").write_bytes(b"x")

        assert stager.stage_in("j1", "legal-demo", str(tmp_path / "legal-demo" / "uploads" / "u1" / "b")) == []
        stager.stage_out("j1")
        assert stager.recover() == 0
        assert (tmp_path / "legal-demo" / "vault" / "keep.mp4").exists()

    def test_settings_come_from_the_environment(self):
        settings = StagingSettings.from_env({"STAGING_REMOTE": REMOTE, "STAGING_SSH_PORT": "2222",
                                             "TENANTS_ROOT": "/workspace/tenants"})
        assert settings.enabled and settings.ssh_port == 2222
        assert not StagingSettings.from_env({}).enabled
