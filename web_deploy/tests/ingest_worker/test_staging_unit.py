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


@pytest.fixture(autouse=True)
def _keys_in_tmp(tmp_path, monkeypatch):
    """Every transfer installs the staging key: never into the real /root/.ssh."""
    from web_deploy.ingest_worker import staging
    monkeypatch.setattr(staging, "AUTHORIZED_KEYS", tmp_path / "default_authorized_keys")


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

        assert gateway.calls == [("GET", "/internal/box/staging-key"), ("POST", "/internal/box/stage-in/j1"),
                                 ("GET", "/internal/box/staging-key"), ("POST", "/internal/box/stage-out/j1")]
        assert not written.exists()
        assert pending_markers(settings.state_dir) == []

    def test_a_firms_glossary_lives_on_the_box_only_while_its_job_runs(self, settings):
        glossary = settings.tenants_root / "legal-demo" / "glossaries" / "en" / "medical.json"
        glossary.parent.mkdir(parents=True)
        glossary.write_text("{}")  # left by an earlier job; the owner may have cleared it since
        stager = Stager(settings, call=FakeGateway())

        stager.stage_in("j1", "legal-demo", None)
        assert not glossary.exists()  # only what the VPS sends now counts
        glossary.parent.mkdir(parents=True)
        glossary.write_text("{}")
        stager.stage_out("j1")

        assert not (settings.tenants_root / "legal-demo" / "glossaries").exists()

    def test_the_youtube_cookies_go_after_the_download_and_at_every_restart(self, tmp_path):
        cookies = tmp_path / "web" / "youtube_cookies.txt"
        cookies.parent.mkdir()
        on = StagingSettings(gateway_url="http://127.0.0.1:18000", tenants_root=tmp_path / "tenants",
                             state_dir=tmp_path / "state", secret="k" * 40, youtube_cookies=cookies)
        cookies.write_text("x")
        Stager(on, call=FakeGateway()).drop_youtube_cookies()
        assert not cookies.exists()
        cookies.write_text("x")
        Stager(on, call=FakeGateway()).recover()
        assert not cookies.exists()

    def test_after_a_download_the_vps_takes_the_cookies_back_or_hears_of_the_refusal(self, tmp_path):
        cookies = tmp_path / "youtube_cookies.txt"
        on = StagingSettings(gateway_url="http://127.0.0.1:18000", tenants_root=tmp_path / "tenants",
                             state_dir=tmp_path / "state", secret="k" * 40, youtube_cookies=cookies)
        gateway = FakeGateway()
        stager = Stager(on, call=gateway)
        stager.youtube_cookies_back("j1", refused=False)  # no cookies were staged: nothing to give back
        cookies.write_text("x")
        stager.youtube_cookies_back("j1", refused=False)
        stager.youtube_cookies_back("j2", refused=True)
        assert gateway.calls == [("POST", "/internal/box/youtube-cookies/j1/refreshed"),
                                 ("POST", "/internal/box/youtube-cookies/j2/refused")]
        Stager(on, call=FakeGateway(fail="youtube")).youtube_cookies_back("j3", refused=True)  # logged, not raised

    def test_on_one_shared_disk_the_gateways_own_file_is_left_alone(self, tmp_path):
        cookies = tmp_path / "youtube_cookies.txt"
        cookies.write_text("x")
        off = StagingSettings(gateway_url="", tenants_root=tmp_path, state_dir=tmp_path / "s", youtube_cookies=cookies)
        Stager(off).drop_youtube_cookies()
        assert cookies.exists()

    def test_a_restart_wipes_every_firms_glossary(self, settings):
        glossary = settings.tenants_root / "other-firm" / "glossaries" / "sr" / "law.json"
        glossary.parent.mkdir(parents=True)
        glossary.write_text("{}")

        assert Stager(settings, call=FakeGateway()).recover() == 0
        assert not glossary.parent.parent.exists()

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

    def test_a_key_dropped_by_a_container_restart_is_back_before_the_next_transfer(self, settings, tmp_path):
        """Vast rewrites authorized_keys on a restart; once, every transfer after it was refused."""
        keys = tmp_path / "root" / ".ssh" / "authorized_keys"
        stager = Stager(settings, call=FakeGateway(), authorized_keys=keys)
        stager.ensure_access()
        keys.write_text("ssh-ed25519 AAAAyou you@laptop\n")  # what Vast leaves

        stager.stage_in("j1", "legal-demo", None)
        assert keys.read_text().splitlines()[-1].endswith(KEY_MARK)

        keys.write_text("ssh-ed25519 AAAAyou you@laptop\n")
        stager.stage_out("j1")
        assert keys.read_text().splitlines() == ["ssh-ed25519 AAAAyou you@laptop",
                                                 f"restrict {' '.join(KEY.split()[:2])} {KEY_MARK}"]

    def test_a_key_already_in_place_is_not_rewritten(self, settings, tmp_path):
        keys = tmp_path / "keys"
        stager = Stager(settings, call=FakeGateway(), authorized_keys=keys)
        stager.ensure_access()
        before = keys.stat().st_mtime_ns
        keys.chmod(0o400)  # a rewrite would fail

        stager.ensure_access()
        assert keys.stat().st_mtime_ns == before

    def test_a_job_whose_files_never_arrived_is_dropped_without_asking_the_vps(self, settings):
        gateway = FakeGateway(fail="stage-in")
        stager = Stager(settings, call=gateway)
        with pytest.raises(StagingError):
            stager.stage_in("j1", "legal-demo", None)
        assert stager.has_pending()
        stray = _scratch_file(settings)

        stager.abandon("j1")

        assert not stager.has_pending() and not stray.exists()
        assert not [c for c in gateway.calls if "stage-out" in c[1]]

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
