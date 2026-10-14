import logging

import pytest

from web_deploy.gateway.config import GatewaySettings

SECRET = "k" * 40


@pytest.fixture
def env(monkeypatch):
    for key in ("PUBLIC_URL", "BASE_DOMAIN", "FIRM_ROUTING", "FAKE_BOX_HOST", "VAST_API_KEY", "BOX_MODE",
                "GATEWAY_HTTPS_ONLY", "SMTP_HOST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WEB_API_SECRET", SECRET)

    def set_env(**values):
        for key, value in values.items():
            monkeypatch.setenv(key, value)
        return GatewaySettings.from_env()
    return set_env


class TestPublicUrl:
    def test_production_needs_the_address_everyone_signs_in_at(self, env):
        with pytest.raises(RuntimeError, match="PUBLIC_URL"):
            env(VAST_API_KEY="key")
        assert env(VAST_API_KEY="key", VAST_INSTANCE_ID="1",
                   PUBLIC_URL="https://onyx.example.com/").public_url == "https://onyx.example.com"

    def test_production_names_its_boxes_primary_first(self, env):
        with pytest.raises(RuntimeError, match="VAST_INSTANCE_ID"):
            env(VAST_API_KEY="key", PUBLIC_URL="https://onyx.example.com")
        settings = env(VAST_API_KEY="key", PUBLIC_URL="https://onyx.example.com", VAST_INSTANCE_ID="55919995,55823057")
        assert settings.vast_instance_ids == (55919995, 55823057)

    def test_an_env_file_from_before_still_works(self, env):
        assert env(VAST_API_KEY="key", VAST_INSTANCE_ID="1", BASE_DOMAIN="Izvoria.Example.com").public_url == "https://izvoria.example.com"

    def test_a_local_end_to_end_run_without_vast_needs_none(self, env):
        # docs/web_deploy.md "Local end-to-end without Vast"
        settings = env(FAKE_BOX_HOST="127.0.0.1", GATEWAY_HTTPS_ONLY="0")
        assert (settings.public_url, settings.https_only) == ("", False)

    def test_dev_hardware_needs_none(self, env):
        assert env(BOX_MODE="local", GATEWAY_HTTPS_ONLY="0").public_url == ""

    def test_a_leftover_firm_routing_is_ignored_with_a_warning(self, env, caplog):
        with caplog.at_level(logging.WARNING):
            settings = env(BOX_MODE="local", FIRM_ROUTING="login")
        assert not hasattr(settings, "firm_routing")
        assert "FIRM_ROUTING is no longer used" in caplog.text

    def test_mail_is_read_too(self, env):
        assert env(BOX_MODE="local", SMTP_HOST="smtp.test").mail.smtp_host == "smtp.test"
