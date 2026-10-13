from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from web_deploy.common.signing import sign_service
from web_deploy.gateway.box_internal import BoxInternalSettings, box_internal_routes
from web_deploy.gateway.staging_server import StagingFailed, StagingRefused

SECRET = "k" * 40


class FakeStaging:
    def __init__(self):
        self.done = []

    def stage_in(self, job_id):
        if job_id == "other":
            raise StagingRefused("Job other is not one the box may move files for now.")
        if job_id == "broken":
            raise StagingFailed("rsync failed (255): connection refused")
        self.done.append(("in", job_id))
        return {"batch_files": 2, "vault_files": 0}

    def stage_out(self, job_id):
        self.done.append(("out", job_id))
        return {"removed": 0}


@pytest.fixture
def files(tmp_path):
    access, password, key = tmp_path / "access", tmp_path / "db-password", tmp_path / "staging.pub"
    access.write_text("open\n")
    password.write_text("fresh-random-password\n")
    key.write_text("ssh-ed25519 AAAA onyx-vps staging\n")
    return access, password, key


@pytest.fixture
def staging():
    return FakeStaging()


def client(files, staging, address="127.0.0.1"):
    access, password, key = files
    app = Starlette(routes=box_internal_routes(BoxInternalSettings(SECRET, access, password, key), staging))
    return TestClient(app, client=(address, 50000))


def call(c, method, path, secret=SECRET):
    return c.request(method, path, headers=sign_service(secret, method, path))


class TestCredentials:
    def test_a_started_box_gets_this_starts_password(self, files, staging):
        response = call(client(files, staging), "POST", "/internal/box/credentials")
        assert response.json() == {"password": "fresh-random-password"}
        assert response.headers["cache-control"] == "no-store"

    def test_a_stopped_box_gets_nothing(self, files, staging):
        files[0].write_text("closed\n")
        assert call(client(files, staging), "POST", "/internal/box/credentials").status_code == 403

    def test_unsigned_or_from_outside_gets_nothing(self, files, staging):
        assert client(files, staging).post("/internal/box/credentials").status_code == 401
        assert call(client(files, staging), "POST", "/internal/box/credentials", secret="x" * 40).status_code == 401
        assert call(client(files, staging, address="203.0.113.9"), "POST",
                    "/internal/box/credentials").status_code == 403

    def test_while_the_password_is_being_rotated_the_box_is_told_to_ask_again(self, files, staging):
        files[1].unlink()
        assert call(client(files, staging), "POST", "/internal/box/credentials").status_code == 503


class TestStaging:
    def test_the_staging_key_and_the_transfers(self, files, staging):
        c = client(files, staging)
        assert call(c, "GET", "/internal/box/staging-key").json() == {"key": "ssh-ed25519 AAAA onyx-vps staging"}
        assert call(c, "POST", "/internal/box/stage-in/j1").json() == {"batch_files": 2, "vault_files": 0}
        assert call(c, "POST", "/internal/box/stage-out/j1").json() == {"removed": 0}
        assert staging.done == [("in", "j1"), ("out", "j1")]

    def test_a_job_the_box_may_not_touch_or_a_failed_transfer_is_reported(self, files, staging):
        c = client(files, staging)
        assert call(c, "POST", "/internal/box/stage-in/other").status_code == 409
        assert call(c, "POST", "/internal/box/stage-in/broken").status_code == 502

    def test_a_stopped_box_moves_nothing(self, files, staging):
        files[0].write_text("closed")
        assert call(client(files, staging), "POST", "/internal/box/stage-in/j1").status_code == 403
        assert staging.done == []
