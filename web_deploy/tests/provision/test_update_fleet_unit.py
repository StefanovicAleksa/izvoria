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
