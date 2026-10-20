"""A box brings itself to the current version at start, and never starts services on half-updated code."""

import hashlib
import io
import tarfile
import urllib.error

import pytest

from web_deploy.provision import self_update
from web_deploy.provision.self_update import FAILED, MAINTENANCE, Gateway, gateway_url, instance_id, update


def _package(files: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeGateway:
    def __init__(self, meta=None, package=b"", unreachable=False):
        self.meta, self.package, self.unreachable = meta, package, unreachable
        self.applied_releases, self.failures = [], 0

    def current(self):
        if self.unreachable:
            raise urllib.error.URLError("no route")
        return self.meta

    def download(self, target):
        target.write_bytes(self.package)
        return hashlib.sha256(self.package).hexdigest()

    def applied(self, release):
        self.applied_releases.append(release)

    def failed(self):
        self.failures += 1


@pytest.fixture
def box(tmp_path, monkeypatch):
    root = tmp_path / "onyx_v3"
    (root / "venv").mkdir(parents=True)
    (root / "venv/marker").write_text("kept")
    (root / "RELEASE").write_text("old1234\n")
    (root / "stale.py").write_text("gone after the update")
    monkeypatch.setattr(self_update, "SCRATCH", tmp_path / "scratch")
    monkeypatch.setattr(self_update, "DIGEST_FILE", tmp_path / "provision_digest")
    provisioned = []
    monkeypatch.setattr(self_update, "provision", lambda r: provisioned.append(r))
    return root, provisioned


def _meta(package, release="new5678", digest="d1", maintenance=False):
    return {"release": release, "sha256": hashlib.sha256(package).hexdigest(), "size": len(package),
            "provision_digest": digest, "maintenance": maintenance}


NEW = _package({"RELEASE": "new5678\n", "app.py": "print('new')\n"})


class TestUpdate:
    def test_a_box_behind_takes_the_new_version_keeping_what_it_built(self, box, tmp_path):
        root, provisioned = box
        (tmp_path / "provision_digest").write_text("d1\n")
        gateway = FakeGateway(_meta(NEW), package=NEW)

        assert update(gateway, root) == 0

        assert (root / "RELEASE").read_text().strip() == "new5678" and (root / "app.py").exists()
        assert not (root / "stale.py").exists() and (root / "venv/marker").read_text() == "kept"
        assert gateway.applied_releases == ["new5678"] and provisioned == []
        assert not (tmp_path / "scratch/release.tar.gz").exists()

    def test_a_changed_setup_runs_provision_and_remembers_it(self, box, tmp_path):
        root, provisioned = box
        (tmp_path / "provision_digest").write_text("d0\n")

        assert update(FakeGateway(_meta(NEW, digest="d2"), package=NEW), root) == 0

        assert provisioned == [root] and (tmp_path / "provision_digest").read_text().strip() == "d2"

    def test_a_box_on_the_version_only_reports_it(self, box):
        root, provisioned = box
        gateway = FakeGateway(_meta(NEW, release="old1234"))

        assert update(gateway, root) == 0

        assert (root / "stale.py").exists() and gateway.applied_releases == ["old1234"] and provisioned == []

    def test_a_box_started_only_to_be_updated_starts_no_service(self, box):
        root, _ = box
        assert update(FakeGateway(_meta(NEW, maintenance=True)), root) == MAINTENANCE

    def test_an_unreachable_gateway_starts_the_old_code(self, box):
        root, _ = box
        assert update(FakeGateway(unreachable=True), root) == 0
        assert (root / "RELEASE").read_text().strip() == "old1234"

    def test_a_corrupt_download_changes_nothing(self, box):
        root, _ = box
        gateway = FakeGateway({**_meta(NEW), "sha256": "0" * 64}, package=NEW)

        assert update(gateway, root) == 0

        assert (root / "RELEASE").read_text().strip() == "old1234" and gateway.failures == 0

    def test_a_corrupt_download_under_maintenance_is_reported(self, box):
        root, _ = box
        gateway = FakeGateway({**_meta(NEW, maintenance=True), "sha256": "0" * 64}, package=NEW)

        assert update(gateway, root) == MAINTENANCE
        assert gateway.failures == 1

    def test_a_package_of_another_version_is_a_failure(self, box):
        root, _ = box
        wrong = _package({"RELEASE": "other99\n"})
        gateway = FakeGateway(_meta(wrong), package=wrong)

        assert update(gateway, root) == FAILED
        assert gateway.failures == 1 and gateway.applied_releases == []

    def test_a_failed_provision_starts_nothing(self, box, tmp_path, monkeypatch):
        root, _ = box

        def broken(_root):
            raise RuntimeError("pip failed")
        monkeypatch.setattr(self_update, "provision", broken)
        gateway = FakeGateway(_meta(NEW, digest="d9"), package=NEW)

        assert update(gateway, root) == FAILED
        assert gateway.failures == 1 and gateway.applied_releases == []


class TestSetup:
    @pytest.mark.parametrize("env,environ,expected", [
        ({"BOX_INSTANCE_ID": "55823057"}, {}, 55823057),
        ({}, {"CONTAINER_ID": "56923441"}, 56923441),
        ({}, {"VAST_CONTAINERLABEL": "C.55919995"}, 55919995),
        ({}, {}, None),
        ({"BOX_INSTANCE_ID": "nope"}, {}, None),
    ])
    def test_the_instance_id(self, tmp_path, env, environ, expected):
        assert instance_id(env, environ, tmp_path / "missing") == expected

    def test_the_instance_id_from_the_container_environment(self, tmp_path):
        proc = tmp_path / "environ"
        proc.write_bytes(b"PATH=/bin\0VAST_CONTAINERLABEL=C.42\0")
        assert instance_id({}, {}, proc) == 42

    @pytest.mark.parametrize("env,expected", [
        ({"RELEASE_URL": "https://app.izvoria.com/", "VPS_HOST": "2.31.10.175"}, "https://app.izvoria.com"),
        ({"VPS_HOST": "app.izvoria.com"}, "https://app.izvoria.com"),
        ({}, ""),
    ])
    def test_the_gateway_address(self, env, expected):
        assert gateway_url(env) == expected

    def test_requests_are_signed_for_their_own_path(self):
        seen = []

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def opener(request, timeout):
            seen.append(request)
            return Response(b'{"release": "x"}')

        assert Gateway("https://gw/", "s" * 40, 7, opener).current() == {"release": "x"}
        Gateway("https://gw", "s" * 40, 7, opener).applied("abc1234")

        assert [r.full_url for r in seen] == ["https://gw/box-release/7", "https://gw/box-release/7/applied/abc1234"]
        assert [r.get_method() for r in seen] == ["GET", "POST"]
        assert all(any(k.lower().startswith("x-") for k in r.headers) for r in seen)
