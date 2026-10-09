"""The two pages anyone can use: a firm asking for Onyx, and a help request. Stored, emailed, and hard to abuse."""

import pytest

from web_deploy.gateway.app import DAILY_MAIL_CAP, PUBLIC_MAX_PER_ADDRESS, SIGNUP_MAX_BYTES
from web_deploy.tests.gateway.support import client_for


def signup(**overrides):
    body = {
        "firm_name": "Petrović & Partners", "contact_name": "Ana Petrović", "contact_role": "Managing partner",
        "email": "Ana@Petrovic.rs", "phone": "+381 11 000", "language": "sr", "website": "", "elapsed_ms": 60000,
        "profile": {
            "industry": "legal", "sub_domains": "M&A — share deals", "description": "Commercial law in Belgrade",
            "data_kinds": "Client meetings, depositions", "mishearings": "",
            "proper_nouns": {"people": "Ana Petrović", "laws": "Zakon o privrednim društvima", "secret": "dropped"},
            "languages": {"share": "sr 80%, en 20%", "english_terms": "due diligence", "latin": "no"},
            "unexpected": "dropped",
        },
    }
    body.update(overrides)
    return body


def support(**overrides):
    body = {"email": "it@firm.com", "name": "IT", "firm": "legal-demo", "subject": "Upload fails",
            "message": "A 2 GB video stops at 40%.", "language": "en", "website": "", "elapsed_ms": 9000}
    body.update(overrides)
    return body


class TestSignupRequest:
    def test_is_stored_and_emailed_to_the_signup_address(self, app, store, transport):
        response = client_for(app).post("/public/signup", json=signup())

        assert response.status_code == 202
        [request] = store.requests()
        assert (request.kind, request.email, request.firm, request.name, request.language, request.mail_status) == (
            "signup", "ana@petrovic.rs", "Petrović & Partners", "Ana Petrović", "sr", "sent")
        assert "unexpected" not in request.payload["profile"]
        assert "secret" not in request.payload["profile"]["proper_nouns"]
        [message] = transport.sent
        assert message["To"] == "sales@onyx.test" and message["Reply-To"] == "ana@petrovic.rs"
        assert "Zakon o privrednim društvima" in message.get_content()

    @pytest.mark.parametrize("missing", ["firm_name", "contact_name", "email"])
    def test_required_fields(self, app, store, missing):
        assert client_for(app).post("/public/signup", json=signup(**{missing: ""})).status_code == 400
        assert store.requests() == []

    def test_a_bad_email_or_an_overlong_field_is_explained(self, app):
        assert client_for(app).post("/public/signup", json=signup(email="not an email")).json()["error"]
        assert client_for(app).post("/public/signup", json=signup(firm_name="x" * 201)).status_code == 400

    def test_too_big_a_body_is_refused_before_it_is_read(self, app, store):
        response = client_for(app).post("/public/signup", content=b"{" + b" " * SIGNUP_MAX_BYTES + b"}",
                                        headers={"content-type": "application/json"})
        assert response.status_code == 413
        assert store.requests() == []


class TestSupportRequest:
    def test_is_stored_and_emailed_to_the_support_address(self, app, store, transport):
        assert client_for(app).post("/public/support", json=support()).status_code == 202
        [request] = store.requests(kind="support")
        assert (request.subject, request.firm, request.payload) == (
            "Upload fails", "legal-demo", {"message": "A 2 GB video stops at 40%."})
        assert transport.sent[0]["To"] == "help@onyx.test"
        assert transport.sent[0]["Subject"] == "Onyx support: Upload fails"

    def test_subject_and_message_are_required(self, app):
        assert client_for(app).post("/public/support", json=support(subject="")).status_code == 400
        assert client_for(app).post("/public/support", json=support(message=" ")).status_code == 400


class TestAbuse:
    @pytest.mark.parametrize("overrides", [{"website": "http://spam"}, {"elapsed_ms": 300}, {"elapsed_ms": None}])
    def test_bots_are_told_it_worked_and_nothing_is_kept(self, app, store, transport, overrides):
        assert client_for(app).post("/public/support", json=support(**overrides)).status_code == 202
        assert store.requests() == [] and transport.sent == []

    def test_one_address_can_only_send_a_few(self, app, store):
        client = client_for(app)
        for _ in range(PUBLIC_MAX_PER_ADDRESS):
            assert client.post("/public/support", json=support()).status_code == 202
        assert client.post("/public/support", json=support()).status_code == 429
        assert client.post("/public/signup", json=signup()).status_code == 202  # counted per form
        assert client_for(app, address="10.9.9.9").post("/public/support", json=support()).status_code == 202

    def test_past_the_daily_cap_requests_are_kept_but_not_emailed(self, app, store, transport, monkeypatch):
        monkeypatch.setattr(store, "mailed_since", lambda since: DAILY_MAIL_CAP)
        assert client_for(app).post("/public/support", json=support()).status_code == 202
        [request] = store.requests()
        assert (request.mail_status, request.mail_error) == ("off", "Daily email limit reached.")
        assert transport.sent == []

    def test_a_mail_failure_is_recorded_not_shown(self, app, store, transport):
        transport.fail = True
        assert client_for(app).post("/public/support", json=support()).status_code == 202
        [request] = store.requests()
        assert request.mail_status == "failed" and "smtp is down" in request.mail_error


class TestOwnerSeesRequests:
    def test_lists_marks_and_resends(self, app, store, transport):
        from web_deploy.gateway.store import OWNER_FIRM
        from web_deploy.tests.gateway.support import CONSOLE_PASSWORD, signed_in
        transport.fail = True
        client_for(app).post("/public/signup", json=signup())
        owner = signed_in(app, OWNER_FIRM, "owner@onyx.test", firm_password=CONSOLE_PASSWORD)

        listed = owner.get("/admin/api/requests?kind=signup").json()
        [request] = listed["requests"]
        assert listed["mail_enabled"] is True and request["mail_status"] == "failed"
        assert "client_ip" not in request

        transport.fail = False
        assert owner.post(f"/admin/api/requests/{request['id']}/resend", json={}).json()["mail_status"] == "sent"
        assert owner.post(f"/admin/api/requests/{request['id']}/status",
                          json={"status": "handled"}).json()["status"] == "handled"
        assert owner.get("/admin/api/requests?status=new").json()["requests"] == []
        assert owner.post(f"/admin/api/requests/{request['id']}/status", json={"status": "x"}).status_code == 400


class TestMalformed:
    @pytest.mark.parametrize("content", [b"not json", b"[1, 2]", b""])
    def test_a_body_that_is_not_an_object_is_refused(self, app, store, content):
        response = client_for(app).post("/public/support", content=content, headers={"content-type": "application/json"})
        assert response.status_code == 400
        assert store.requests() == []

    def test_odd_values_are_dropped_or_defaulted_not_trusted(self, app, store):
        body = support(language="de", name=["not", "text"], firm={"x": 1})
        assert client_for(app).post("/public/support", json=body).status_code == 202
        [request] = store.requests()
        assert (request.language, request.name, request.firm) == ("en", "", "")

    def test_a_profile_that_is_not_an_object_is_an_empty_profile(self, app, store):
        assert client_for(app).post("/public/signup", json=signup(profile="nope")).status_code == 202
        profile = store.requests()[0].payload["profile"]
        assert profile["industry"] == "" and profile["proper_nouns"]["people"] == ""


class TestOwnerRequestActions:
    def owner(self, app):
        from web_deploy.gateway.store import OWNER_FIRM
        from web_deploy.tests.gateway.support import CONSOLE_PASSWORD, signed_in
        return signed_in(app, OWNER_FIRM, "owner@onyx.test", firm_password=CONSOLE_PASSWORD)

    def test_unknown_requests_and_actions_are_404(self, app, store):
        owner = self.owner(app)
        request = store.add_request("support", "a@b.com", {"message": "m"}, subject="s")
        assert owner.post("/admin/api/requests/nope/status", json={"status": "handled"}).status_code == 404
        assert owner.post(f"/admin/api/requests/{request.id}/explode", json={}).status_code == 404

    def test_resending_without_email_set_up_is_explained(self, app, store, monkeypatch):
        owner = self.owner(app)
        request = store.add_request("support", "a@b.com", {"message": "m"}, subject="s", mail_status="off")
        import web_deploy.gateway.mailer as mailer_module
        monkeypatch.setattr(mailer_module.Mailer, "enabled", property(lambda self: False))
        response = owner.post(f"/admin/api/requests/{request.id}/resend", json={})
        assert response.status_code == 400 and "SMTP_HOST" in response.json()["error"]
        assert store.request(request.id).mail_status == "off"
