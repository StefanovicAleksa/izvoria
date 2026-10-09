from unittest.mock import MagicMock, patch

import pytest

from web_deploy.gateway.config import mail_settings_from_env
from web_deploy.gateway.mailer import Mailer, MailSettings, build_message, signup_body, smtp_send
from web_deploy.gateway.store import Store
from web_deploy.tests.gateway.support import InlineExecutor, RecordingTransport

SETTINGS = MailSettings(smtp_host="smtp.test", mail_from="onyx@onyx.test", signup_to="sales@onyx.test")


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "g.sqlite3")


def a_signup(store, **profile):
    return store.add_request("signup", "ana@firm.rs", {
        "firm_name": "Firm", "contact_name": "Ana", "phone": "", "contact_role": "",
        "profile": {"industry": "legal", "description": "Law", "proper_nouns": {}, "languages": {}, **profile},
    }, name="Ana", firm="Firm")


class TestMessages:
    def test_the_signup_body_follows_the_firm_profile_template(self, store):
        body = signup_body(a_signup(store, proper_nouns={"people": "Ana Petrović"}))
        headings = [line for line in body.splitlines() if line.startswith("## ")]
        assert headings == ["## 1. Firm name", "## 2. Industry", "## 3. Sub-industries / specialised domains",
                            "## 4. Firm description", "## 5. What the data will be",
                            "## 6. Proper nouns the recordings will contain",
                            "## 7. Languages spoken in the recordings",
                            "## 8. Terms you already know Whisper gets wrong"]
        assert "- **People:** Ana Petrović" in body
        assert "(not given)" in body  # sections left empty say so rather than vanish

    def test_left_out_names_are_flagged_for_onboarding(self, store):
        assert "add these names by hand at onboarding" in signup_body(a_signup(store))

    def test_nothing_typed_can_add_a_header(self, store):
        request = store.add_request("support", "it@firm.com", {"message": "m"},
                                    subject="Help\r\nBcc: victim@example.com")
        message = build_message(SETTINGS, request)
        assert message["Bcc"] is None
        assert message["Subject"] == "Onyx support: Help Bcc: victim@example.com"
        assert message["To"] == "sales@onyx.test"  # no SUPPORT_TO: the sign-up address takes help too


class TestMailer:
    def test_off_without_smtp(self, store):
        transport = RecordingTransport()
        mailer = Mailer(MailSettings(), store, transport=transport, executor=InlineExecutor())
        request = a_signup(store)
        mailer.submit(request)
        assert store.request(request.id).mail_status == "off" and transport.sent == []

    def test_records_success_and_failure(self, store):
        ok = Mailer(SETTINGS, store, transport=RecordingTransport(), executor=InlineExecutor())
        failing = Mailer(SETTINGS, store, transport=RecordingTransport(fail=True), executor=InlineExecutor())
        first, second = a_signup(store), a_signup(store)
        ok.submit(first)
        failing.submit(second)
        assert store.request(first.id).mail_status == "sent"
        assert store.request(second.id).mail_status == "failed"


class TestSmtp:
    def test_starttls_and_login(self):
        settings = MailSettings(smtp_host="smtp.test", smtp_port=587, smtp_user="u", smtp_password="p",
                                mail_from="u@test", signup_to="me@test")
        with patch("web_deploy.gateway.mailer.smtplib") as smtplib:
            client = smtplib.SMTP.return_value.__enter__.return_value
            smtp_send(settings, MagicMock())
        smtplib.SMTP.assert_called_once_with("smtp.test", 587, timeout=20)
        client.starttls.assert_called_once()
        client.login.assert_called_once_with("u", "p")
        client.send_message.assert_called_once()

    def test_implicit_tls(self):
        settings = MailSettings(smtp_host="smtp.test", smtp_port=465, smtp_ssl=True, mail_from="a@test",
                                signup_to="me@test")
        with patch("web_deploy.gateway.mailer.smtplib") as smtplib:
            client = smtplib.SMTP_SSL.return_value.__enter__.return_value
            smtp_send(settings, MagicMock())
        smtplib.SMTP.assert_not_called()
        client.starttls.assert_not_called()
        client.login.assert_not_called()

    def test_settings_from_the_environment(self, monkeypatch):
        for key, value in {"SMTP_HOST": "smtp.gmail.com", "SMTP_USER": "me@gmail.com", "SMTP_PASSWORD": "app pw",
                           "SIGNUP_TO": "me@gmail.com"}.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("SUPPORT_TO", raising=False)
        monkeypatch.delenv("MAIL_FROM", raising=False)
        settings = mail_settings_from_env()
        assert (settings.smtp_port, settings.smtp_starttls, settings.mail_from, settings.support_to) == (
            587, True, "me@gmail.com", "me@gmail.com")
        assert settings.enabled

    def test_a_user_without_a_password_is_off_not_failing(self):
        half = MailSettings(smtp_host="smtp.gmail.com", smtp_user="me@gmail.com", mail_from="me@gmail.com",
                            signup_to="me@gmail.com")
        assert not half.enabled


class TestEncoding:
    def test_names_outside_ascii_survive_the_trip(self, store):
        request = store.add_request("signup", "ana@firm.rs", {
            "firm_name": "Petrović & Partneri — Beograd", "contact_name": "Ana Đorđević",
            "profile": {"proper_nouns": {"people": "Ana Đorđević, Miloš Šćekić"}, "languages": {}}},
            name="Ana Đorđević", firm="Petrović & Partneri — Beograd")
        message = build_message(SETTINGS, request)
        raw = message.as_bytes()  # what smtplib sends: must not raise on č ć đ š ž
        assert b"=?utf-8?" in raw  # an RFC 2047 encoded word, not raw UTF-8 in a header
        from email import message_from_bytes, policy
        parsed = message_from_bytes(raw, policy=policy.default)
        assert parsed["Subject"] == "Onyx access request: Petrović & Partneri — Beograd"
        assert "Miloš Šćekić" in parsed.get_content()
