from unittest.mock import patch

import pytest

from web_deploy.common.signing import verify
from web_deploy.gateway import store as store_module
from web_deploy.gateway.app import (
    COOKIE, FIRM_COOKIE, FIRM_MAX_FAILURES, FIRM_PASS_SECONDS, FIRM_REMEMBER_SECONDS, LOGIN_MAX_FAILURES,
)
from web_deploy.gateway.store import OWNER_FIRM
from web_deploy.tests.gateway.support import (
    CONSOLE_PASSWORD, FIRM_PASSWORD, OTHER_FIRM_PASSWORD, PASSWORD, SECRET, client_for, firm_step, signed_in,
)


def cookie_header(response, name):
    return next(h for h in response.headers.get_list("set-cookie") if h.startswith(f"{name}="))


class TestFirmStep:
    def test_a_new_browser_is_asked_for_its_firm(self, app):
        assert client_for(app).get("/auth/me").json() == {
            "languages": ["en", "sr"], "stage": "firm", "console": None, "firm": None, "user": None}

    def test_the_firm_password_opens_the_account_step(self, app):
        client = client_for(app)
        response = firm_step(client, "Legal-Demo")

        assert response.status_code == 200
        assert response.json()["firm"] == {"slug": "legal-demo", "name": "Legal demo", "default_language": "en"}
        me = client.get("/auth/me").json()
        assert (me["stage"], me["console"], me["firm"]["slug"], me["user"]) == ("account", "firm", "legal-demo", None)

    def test_the_firm_pass_is_a_short_strict_http_only_cookie_unless_remembered(self, app):
        short = cookie_header(firm_step(client_for(app)), FIRM_COOKIE)
        assert "HttpOnly" in short and "SameSite=strict" in short and f"Max-Age={FIRM_PASS_SECONDS}" in short
        remembered = cookie_header(firm_step(client_for(app), remember=True), FIRM_COOKIE)
        assert f"Max-Age={FIRM_REMEMBER_SECONDS}" in remembered

    @pytest.mark.parametrize("firm, password", [
        ("legal-demo", "wrong wrong wrong"),
        ("legal-demo", OTHER_FIRM_PASSWORD),
        ("nobody-here", FIRM_PASSWORD),
        ("", FIRM_PASSWORD),
    ])
    def test_wrong_firm_or_password_gets_no_pass(self, app, firm, password):
        client = client_for(app)
        response = firm_step(client, firm, password)
        assert response.status_code in (400, 401)
        assert FIRM_COOKIE not in client.cookies
        assert client.get("/auth/me").json()["stage"] == "firm"

    def test_an_unknown_firm_and_a_wrong_password_are_the_same_answer_and_the_same_work(self, app):
        with patch.object(store_module, "verify_password", wraps=store_module.verify_password) as checks:
            unknown = firm_step(client_for(app), "nobody-here")
            wrong = firm_step(client_for(app), "legal-demo", "wrong wrong wrong")
        assert unknown.status_code == wrong.status_code == 401
        assert unknown.json() == wrong.json()
        assert checks.call_count == 2

    def test_a_firm_without_a_password_or_disabled_cannot_pass(self, app, store):
        store.create_firm("no-password", "Nobody yet")
        assert firm_step(client_for(app), "no-password", "").status_code == 401
        store.set_firm_disabled("other-firm", True)
        assert firm_step(client_for(app), "other-firm", OTHER_FIRM_PASSWORD).status_code == 401

    def test_repeated_failures_from_one_address_are_throttled(self, app):
        client = client_for(app)
        for _ in range(LOGIN_MAX_FAILURES):
            firm_step(client, "legal-demo", "wrong wrong wrong")
        assert firm_step(client).status_code == 429

    def test_a_firm_is_throttled_too_but_later_than_an_address(self, app):
        for i in range(FIRM_MAX_FAILURES):
            firm_step(client_for(app, address=f"10.0.0.{i}"), "legal-demo", "wrong wrong wrong")
        assert firm_step(client_for(app, address="10.0.1.1")).status_code == 429
        assert firm_step(client_for(app, address="10.0.1.1"), "other-firm", OTHER_FIRM_PASSWORD).status_code == 200

    def test_changing_the_firm_password_closes_every_pass_but_not_sessions(self, app, store):
        signed = signed_in(app)
        waiting = client_for(app)
        firm_step(waiting)

        store.set_firm_password("legal-demo", "a new shared secret")

        assert waiting.get("/auth/me").json()["stage"] == "firm"
        assert signed.get("/auth/me").json()["stage"] == "signed_in"


class TestAccountStep:
    def test_two_steps_sign_in_and_wake_the_box(self, app, vast):
        client = client_for(app)
        firm_step(client)
        response = client.post("/auth/login", json={"email": "alice@firm.com", "password": PASSWORD})

        assert response.status_code == 200
        cookie = cookie_header(response, COOKIE)
        assert "HttpOnly" in cookie and "Domain" not in cookie
        me = client.get("/auth/me").json()
        assert (me["stage"], me["console"], me["user"]["email"]) == ("signed_in", "firm", "alice@firm.com")

    def test_the_account_step_needs_a_firm_pass(self, app):
        response = client_for(app).post("/auth/login", json={"email": "alice@firm.com", "password": PASSWORD})
        assert response.status_code == 401
        assert response.json()["kind"] == "firm_stage"

    def test_the_firm_is_the_pass_s_never_one_in_the_body(self, app):
        client = client_for(app)
        firm_step(client, "other-firm", OTHER_FIRM_PASSWORD)
        response = client.post("/auth/login", json={"firm": "legal-demo", "email": "alice@firm.com",
                                                    "password": PASSWORD})
        assert response.status_code == 401

    def test_a_wrong_password_is_refused_and_throttled(self, app):
        client = client_for(app)
        firm_step(client)
        for _ in range(LOGIN_MAX_FAILURES):
            assert client.post("/auth/login", json={"email": "alice@firm.com",
                                                    "password": "wrong wrong"}).status_code == 401
        assert client.post("/auth/login", json={"email": "alice@firm.com", "password": PASSWORD}).status_code == 429

    def test_signing_out_keeps_the_firm_so_the_next_person_starts_at_their_account(self, app):
        alice = signed_in(app)
        alice.post("/auth/logout")
        me = alice.get("/auth/me").json()
        assert (me["stage"], me["user"]) == ("account", None)

    def test_switching_firm_forgets_both(self, app):
        alice = signed_in(app)
        alice.post("/auth/firm/forget")
        assert alice.get("/auth/me").json()["stage"] == "firm"

    def test_a_disabled_firm_signs_everyone_out(self, app, store):
        alice = signed_in(app)
        store.set_firm_disabled("legal-demo", True)
        assert alice.get("/auth/me").json()["stage"] == "firm"
        assert alice.get("/api/conversations").status_code == 401

    def test_change_password(self, app, store):
        alice = signed_in(app)
        assert alice.post("/auth/password", json={"current": "wrong", "new": "x" * 12}).status_code == 400
        assert alice.post("/auth/password", json={"current": PASSWORD, "new": "a brand new password"}).status_code == 200
        assert alice.get("/auth/me").json()["user"] is not None
        assert store.authenticate("legal-demo", "alice@firm.com", "a brand new password") is not None

    def test_the_owner_signs_in_as_firm_admin_with_the_console_password(self, app):
        client = client_for(app)
        assert firm_step(client, OWNER_FIRM, FIRM_PASSWORD).status_code == 401
        assert firm_step(client, OWNER_FIRM, CONSOLE_PASSWORD).status_code == 200
        assert client.get("/auth/me").json()["console"] == "admin"
        assert client.post("/auth/login", json={"email": "owner@onyx.test", "password": PASSWORD}).status_code == 200
        assert client.get("/admin/api/firms").status_code == 200
        assert client.get("/api/conversations").status_code == 404

    def test_a_firm_user_cannot_sign_in_at_the_console(self, app):
        client = client_for(app)
        firm_step(client, OWNER_FIRM, CONSOLE_PASSWORD)
        assert client.post("/auth/login", json={"email": "alice@firm.com", "password": PASSWORD}).status_code == 401


class TestCrossSite:
    @pytest.mark.parametrize("path", ["/auth/firm", "/auth/login", "/auth/logout", "/auth/firm/forget",
                                      "/public/signup", "/public/support"])
    def test_cross_site_posts_are_refused(self, app, path):
        assert client_for(app, origin="http://evil.example").post(path, json={}).status_code == 403

    def test_cross_site_api_calls_are_refused_when_signed_in(self, app):
        alice = signed_in(app)
        alice.headers["origin"] = "http://evil.example"
        assert alice.post("/api/chat", json={}).status_code == 403
        assert alice.post("/firm/api/users", json={}).status_code == 403

    def test_with_a_public_url_only_that_origin_counts(self, app, tmp_path, store, vast, box_api):
        import httpx
        from web_deploy.gateway.app import create_app
        from web_deploy.gateway.box import BoxController
        from web_deploy.gateway.config import GatewaySettings
        from web_deploy.gateway.idle import IdleMonitor
        settings = GatewaySettings(api_secret=SECRET, database_path=tmp_path / "g.db", static_dir=tmp_path,
                                   fake_box_host="127.0.0.1", https_only=False, public_url="https://onyx.example.com")
        http = httpx.AsyncClient(transport=httpx.MockTransport(box_api))
        box = BoxController(settings, vast, http)
        strict = create_app(settings, store, box, IdleMonitor(box, 1800), http, run_idle_monitor=False)
        assert firm_step(client_for(strict)).status_code == 403
        assert firm_step(client_for(strict, origin="https://onyx.example.com")).status_code == 200


class TestProxy:
    def test_api_calls_are_forwarded_signed_for_the_user_without_cookies(self, app, store, box_api, vast):
        alice = signed_in(app)  # wakes the (fake) box

        response = alice.post("/api/chat?x=1", json={"prompt": "q", "language": "en"})

        assert response.status_code == 202
        assert "set-cookie" not in response.headers and "x-internal" not in response.headers
        forwarded = box_api.forwarded[-1]
        assert "cookie" not in forwarded.headers
        principal = verify(SECRET, "POST", "/api/chat", "x=1", forwarded.headers)
        user = store.authenticate("legal-demo", "alice@firm.com", PASSWORD)
        assert (principal.firm, principal.user_id, principal.role) == ("legal-demo", user.id, "member")

    def test_each_firm_is_signed_as_itself(self, app, box_api):
        eve = signed_in(app, "other-firm", "eve@other.com", firm_password=OTHER_FIRM_PASSWORD)
        eve.get("/api/conversations")
        assert verify(SECRET, "GET", "/api/conversations", "", box_api.forwarded[-1].headers).firm == "other-firm"

    def test_a_sleeping_box_answers_503_while_it_wakes(self, app, vast, box_api, monkeypatch):
        async def slow_start():
            pass  # the GPU stays "scheduling": the box never comes up during this test

        monkeypatch.setattr(vast, "start", slow_start)
        alice = signed_in(app)
        response = alice.get("/api/conversations")
        assert response.status_code == 503
        assert response.json()["kind"] == "box"
        assert response.json()["box"]["state"] in ("starting", "stopped")
        assert box_api.forwarded == []

    def test_signed_out_requests_never_reach_the_box(self, app, box_api):
        client = client_for(app)
        firm_step(client)  # a firm pass alone is not a sign-in
        assert client.get("/api/conversations").status_code == 401
        assert box_api.forwarded == []


class TestAdminConsole:
    def owner(self, app):
        return signed_in(app, OWNER_FIRM, "owner@onyx.test", firm_password=CONSOLE_PASSWORD)

    def test_the_owner_creates_firms_with_a_password_and_users(self, app, store):
        owner = self.owner(app)
        assert owner.post("/admin/api/firms", json={"slug": "clinic-demo", "name": "Clinic"}).status_code == 400
        created = owner.post("/admin/api/firms", json={"slug": "clinic-demo", "name": "Clinic", "is_demo": True,
                                                       "firm_password": "clinic shared secret"})
        assert created.status_code == 201 and created.json()["has_password"] is True
        user = owner.post("/admin/api/firms/clinic-demo/users",
                          json={"email": "doc@clinic.com", "password": PASSWORD, "role": "admin"})
        assert user.status_code == 201
        firms = owner.get("/admin/api/firms").json()
        assert {f["slug"]: f["users"] for f in firms["firms"]} == {"clinic-demo": 1, "legal-demo": 2,
                                                                   "other-firm": 1}
        assert firms["console_has_password"] is True
        assert store.verify_firm("clinic-demo", "clinic shared secret") == "clinic-demo"

        disabled = owner.post(f"/admin/api/users/{user.json()['id']}/disable", json={})
        assert disabled.json()["disabled"] is True

    def test_the_owner_sets_a_firm_s_and_the_console_s_password(self, app, store):
        owner = self.owner(app)
        assert owner.post("/admin/api/firms/legal-demo/password",
                          json={"password": "replaced shared secret"}).status_code == 200
        assert store.verify_firm("legal-demo", "replaced shared secret") == "legal-demo"
        assert owner.post("/admin/api/console-password",
                          json={"account_password": "wrong", "password": "x" * 12}).status_code == 400
        assert owner.post("/admin/api/console-password",
                          json={"account_password": PASSWORD, "password": "a new console door"}).status_code == 200
        assert store.verify_firm(OWNER_FIRM, "a new console door") == OWNER_FIRM

    def test_firm_users_cannot_use_the_console(self, app):
        boss = signed_in(app, email="boss@firm.com")  # a firm admin is still not the owner
        assert boss.get("/admin/api/firms").status_code == 403
        assert boss.get("/admin/api/requests").status_code == 403

    def test_bad_input_is_explained(self, app):
        owner = self.owner(app)
        response = owner.post("/admin/api/firms", json={"slug": "Bad Slug", "name": "x", "firm_password": "x" * 12})
        assert response.status_code == 400
        assert owner.post("/admin/api/firms/legal-demo/users",
                          json={"email": "x@y.com", "password": "short"}).status_code == 400


class TestSpa:
    def test_every_page_is_the_app_at_any_host(self, app):
        assert client_for(app).get("/reports/123").text == "<html>spa</html>"
        assert client_for(app, "192.168.1.50:8000").get("/signin").text == "<html>spa</html>"


class TestEdges:
    def test_signing_in_again_ends_the_session_it_replaces(self, app, store):
        client = signed_in(app)
        old = client.cookies.get(COOKIE)
        assert client.post("/auth/login", json={"email": "boss@firm.com", "password": PASSWORD}).status_code == 200
        assert store.session_user(old) is None
        assert client.get("/auth/me").json()["user"]["email"] == "boss@firm.com"

    def test_a_firm_disabled_after_its_pass_was_given_is_back_at_the_firm_step(self, app, store):
        client = client_for(app)
        firm_step(client)
        store.set_firm_disabled("legal-demo", True)
        assert client.get("/auth/me").json()["stage"] == "firm"
        assert client.post("/auth/login", json={"email": "alice@firm.com", "password": PASSWORD}).json()["kind"] == "firm_stage"

    def test_a_new_firm_pass_ends_the_one_it_replaces(self, app, store):
        client = client_for(app)
        firm_step(client)
        old = client.cookies.get(FIRM_COOKIE)
        firm_step(client, "other-firm", OTHER_FIRM_PASSWORD)
        assert store.firm_pass(old) is None
        assert client.get("/auth/me").json()["firm"]["slug"] == "other-firm"

    def test_the_owner_s_firm_actions_check_their_input(self, app):
        owner = signed_in(app, OWNER_FIRM, "owner@onyx.test", firm_password=CONSOLE_PASSWORD)
        assert owner.post("/admin/api/firms/nobody-here/password", json={"password": "x" * 12}).status_code == 404
        assert owner.post("/admin/api/firms/legal-demo/password", json={"password": "short"}).status_code == 400
        assert owner.post("/admin/api/firms/legal-demo/explode", json={}).status_code == 404

    def test_a_firm_admin_s_unknown_action_is_404(self, app, store):
        boss = signed_in(app, email="boss@firm.com")
        alice = next(u.id for u in store.users("legal-demo") if u.email == "alice@firm.com")
        assert boss.post(f"/firm/api/users/{alice}/explode", json={}).status_code == 404
