from pathlib import Path

import pytest

from web_deploy.ingest_worker.staging import KEY_MARK, Stager, StagingError, StagingSettings, pending_markers

KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIEd6JRHFS1ewIuFgZ3Ek4IqIecvHOGrDhyM5smgbBgSP onyx-vps staging"


class FakeGateway:
    """The gateway's box routes: records every call, can fail one."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def __call__(self, method, path):
        self.calls.append((method, path))
        if self.fail and self.fail in path:
            raise StagingError(f"{path}: the gateway could not be reached: Connection refused")
        if path.endswith("staging-key"):
            return {"key": KEY}
        return {"batch_files": 1, "vault_files": 0}


@pytest.fixture
def settings(tmp_path):
    return StagingSettings(gateway_url="http://127.0.0.1:18000", tenants_root=tmp_path / "tenants",
                           state_dir=tmp_path / "state", secret="k" * 40)


def _scratch_file(settings, firm="legal-demo"):
    path = settings.tenants_root / firm / "vault" / "2026" / "10" / "13" / "new.mp4"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"indexed")
    return path


class TestStager:
    def test_the_vps_moves_a_jobs_files_in_and_out_and_the_box_keeps_nothing(self, settings):
        gateway = FakeGateway()
        stager = Stager(settings, call=gateway)
        batch = settings.tenants_root / "legal-demo" / "uploads" / "u1" / "b"

        stager.stage_in("j1", "legal-demo", str(batch))
        assert pending_markers(settings.state_dir) and batch.is_dir()
        written = _scratch_file(settings)

        stager.stage_out("j1")

        assert gateway.calls == [("POST", "/internal/box/stage-in/j1"), ("POST", "/internal/box/stage-out/j1")]
        assert not written.exists()
        assert pending_markers(settings.state_dir) == []

    def test_a_failed_transfer_keeps_the_scratch_and_the_next_start_retries(self, settings):
        stager = Stager(settings, call=FakeGateway(fail="stage-out"))
        stager.stage_in("j1", "legal-demo", None)
        written = _scratch_file(settings)

        with pytest.raises(StagingError):
            stager.stage_out("j1")
        assert written.exists() and stager.has_pending()

        again = Stager(settings, call=FakeGateway())
        assert again.recover() == 0
        assert not written.exists() and not again.has_pending()

    def test_the_vps_staging_key_is_let_in_beside_the_existing_keys(self, settings, tmp_path):
        keys = tmp_path / "root" / ".ssh" / "authorized_keys"
        keys.parent.mkdir(parents=True)
        keys.write_text("ssh-ed25519 AAAAyou you@laptop\nrestrict ssh-ed25519 OLD izvoria-staging\n")

        Stager(settings, call=FakeGateway(), authorized_keys=keys).ensure_access()

        lines = keys.read_text().splitlines()
        assert lines[0] == "ssh-ed25519 AAAAyou you@laptop"
        assert lines[1] == f"restrict {' '.join(KEY.split()[:2])} {KEY_MARK}"
        assert len(lines) == 2

    def test_a_gateway_without_a_key_is_an_error(self, settings, tmp_path):
        stager = Stager(settings, call=lambda m, p: {"key": ""}, authorized_keys=tmp_path / "keys")
        with pytest.raises(StagingError):
            stager.ensure_access()


class TestOff:
    def test_without_a_gateway_the_box_and_the_data_share_one_disk(self, tmp_path):
        settings = StagingSettings(gateway_url="", tenants_root=tmp_path, state_dir=tmp_path / "state")
        stager = Stager(settings, call=lambda *a: pytest.fail("no gateway calls"))
        kept = tmp_path / "legal-demo" / "vault" / "keep.mp4"
        kept.parent.mkdir(parents=True)
        kept.write_bytes(b"x")

        stager.ensure_access()
        stager.stage_in("j1", "legal-demo", None)
        stager.stage_out("j1")
        assert stager.recover() == 0
        assert kept.exists()

    def test_settings_come_from_the_environment(self):
        settings = StagingSettings.from_env({"BOX_GATEWAY_URL": "http://127.0.0.1:18000/", "WEB_API_SECRET": "s"})
        assert settings.enabled and settings.gateway_url == "http://127.0.0.1:18000"
        assert not StagingSettings.from_env({}).enabled
