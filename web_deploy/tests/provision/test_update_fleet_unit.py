"""update_fleet.py's decisions: what each box needs, and when a box or the gateway must be set up again."""

import pytest

from web_deploy.provision.update_fleet import GATEWAY_PATHS, PROVISION_PATHS, plan, touches


def box(**overrides):
    info = {"id": 1, "actual_status": "exited", "intended_status": "stopped", "release": "old", "maintenance": False}
    info.update(overrides)
    return info


class TestPlan:
    def test_a_box_at_the_release_is_left_alone(self):
        assert plan(box(release="new"), "new", force=False) == "up to date"

    def test_force_updates_it_anyway(self):
        assert plan(box(release="new"), "new", force=True) == "update stopped"

    def test_the_box_in_use_is_updated_in_place(self):
        assert plan(box(actual_status="running", intended_status="running"), "new", False) == "update running"

    def test_a_stopped_box_is_started_for_its_update(self):
        assert plan(box(), "new", False) == "update stopped"

    @pytest.mark.parametrize("info", [
        box(maintenance=True),
        box(actual_status="loading", intended_status="running"),
        box(actual_status="exited", intended_status="running"),  # a queued start
        box(actual_status="missing", intended_status="missing"),
    ])
    def test_a_box_that_is_busy_changing_or_gone_is_skipped(self, info):
        assert plan(info, "new", False).startswith("skip")


class TestTouches:
    def test_unknown_changes_touch_everything(self):
        assert touches(None, PROVISION_PATHS) and touches(None, GATEWAY_PATHS)

    def test_code_only_changes_need_no_provisioning(self):
        changed = ["app/features/source_profile/service/source_profile_api.py", "web_deploy/api/app.py"]
        assert not touches(changed, PROVISION_PATHS) and not touches(changed, GATEWAY_PATHS)

    @pytest.mark.parametrize("path", ["requirements.txt", "web_deploy/provision/supervisord.conf",
                                      "web_deploy/provision/llama_patches/0001-x.patch"])
    def test_dependencies_or_box_setup_need_provisioning(self, path):
        assert touches([path], PROVISION_PATHS)

    def test_gateway_code_needs_a_gateway_restart(self):
        assert touches(["web_deploy/gateway/vast_pool.py"], GATEWAY_PATHS)
        assert touches(["deploy/gateway/vast_client.py"], GATEWAY_PATHS)
        assert not touches(["web_deploy/gateway_notes.md"], GATEWAY_PATHS)


class TestPackage:
    def test_the_package_holds_the_export_and_describes_itself(self, tmp_path):
        import hashlib
        import json
        import tarfile

        from web_deploy.common.release import provision_digest
        from web_deploy.provision.update_fleet import build_package

        export = tmp_path / "export"
        (export / "web_deploy/provision").mkdir(parents=True)
        (export / "RELEASE").write_text("abc1234\n")
        (export / "requirements.txt").write_text("torch\n")
        (export / "web_deploy/provision/provision.sh").write_text("echo hi\n")
        out = tmp_path / "out"
        out.mkdir()

        built = build_package(export, "abc1234", out)

        meta = json.loads(built["meta"].read_text())
        assert meta == {"release": "abc1234", "sha256": hashlib.sha256(built["package"].read_bytes()).hexdigest(),
                        "size": built["package"].stat().st_size, "provision_digest": provision_digest(export)}
        with tarfile.open(built["package"]) as tar:
            assert {"RELEASE", "requirements.txt", "web_deploy/provision/provision.sh"} <= set(tar.getnames())

    def test_a_setup_file_change_changes_the_digest(self, tmp_path):
        from web_deploy.common.release import provision_digest

        (tmp_path / "requirements.txt").write_text("torch\n")
        before = provision_digest(tmp_path)
        (tmp_path / "web_deploy/provision/llama_patches").mkdir(parents=True)
        (tmp_path / "web_deploy/provision/llama_patches/a.patch").write_text("x")
        assert provision_digest(tmp_path) != before


class TestEnvLines:
    def test_each_value_replaces_its_line(self, tmp_path):
        import subprocess

        from web_deploy.provision.update_fleet import set_env_lines

        env = tmp_path / "api.env"
        env.write_text("WEB_API_SECRET=x\nBOX_INSTANCE_ID=1\n")

        subprocess.run(["bash", "-c", set_env_lines(str(env), {"BOX_INSTANCE_ID": "55823057",
                                                                "RELEASE_URL": "https://app.izvoria.com"})],
                       check=True)

        assert env.read_text().splitlines() == ["WEB_API_SECRET=x", "BOX_INSTANCE_ID=55823057",
                                                "RELEASE_URL=https://app.izvoria.com"]
        assert env.stat().st_mode & 0o777 == 0o600
