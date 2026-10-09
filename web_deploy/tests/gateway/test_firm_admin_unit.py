"""A firm's administrators manage their own firm's accounts and its shared password, and nothing else."""

import pytest

from web_deploy.gateway.store import OWNER_FIRM
from web_deploy.tests.gateway.support import FIRM_PASSWORD, OTHER_FIRM_PASSWORD, PASSWORD, signed_in


@pytest.fixture
def boss(app):
    return signed_in(app, email="boss@firm.com")


def user_id(store, firm, email):
    return next(u.id for u in store.users(firm) if u.email == email)


class TestFirmAdmin:
    def test_lists_only_their_own_firm(self, boss):
        emails = {u["email"] for u in boss.get("/firm/api/users").json()["users"]}
        assert emails == {"alice@firm.com", "boss@firm.com"}

    def test_creates_accounts_in_their_own_firm(self, boss, store):
        created = boss.post("/firm/api/users", json={"email": "new@firm.com", "name": "New", "password": PASSWORD,
                                                     "role": "member", "firm": "other-firm"})
        assert created.status_code == 201
        assert created.json()["firm"] == "legal-demo"
        assert store.authenticate("legal-demo", "new@firm.com", PASSWORD) is not None
        assert boss.post("/firm/api/users", json={"email": "x@firm.com", "password": PASSWORD,
                                                  "role": "owner"}).status_code == 400
        assert boss.post("/firm/api/users", json={"email": "y@firm.com", "password": "short"}).status_code == 400

    def test_resets_disables_and_promotes(self, boss, store):
        alice = user_id(store, "legal-demo", "alice@firm.com")
        assert boss.post(f"/firm/api/users/{alice}/password", json={"password": "alice's new password"}).status_code == 200
        assert store.authenticate("legal-demo", "alice@firm.com", "alice's new password") is not None
        assert boss.post(f"/firm/api/users/{alice}/disable", json={}).json()["disabled"] is True
        assert boss.post(f"/firm/api/users/{alice}/enable", json={}).json()["disabled"] is False
        assert boss.post(f"/firm/api/users/{alice}/role", json={"role": "admin"}).json()["role"] == "admin"
        assert boss.post(f"/firm/api/users/{alice}/role", json={"role": "owner"}).status_code == 400

    def test_another_firm_s_users_and_owners_do_not_exist_for_them(self, boss, store):
        for target in (user_id(store, "other-firm", "eve@other.com"), user_id(store, OWNER_FIRM, "owner@onyx.test")):
            assert boss.post(f"/firm/api/users/{target}/disable", json={}).status_code == 404
            assert boss.post(f"/firm/api/users/{target}/password", json={"password": "x" * 12}).status_code == 404
        assert store.authenticate("other-firm", "eve@other.com", PASSWORD) is not None

    def test_cannot_lock_themselves_or_the_firm_out(self, boss, store):
        me = user_id(store, "legal-demo", "boss@firm.com")
        assert boss.post(f"/firm/api/users/{me}/disable", json={}).status_code == 400
        assert boss.post(f"/firm/api/users/{me}/role", json={"role": "member"}).status_code == 400

    def test_a_firm_keeps_at_least_one_enabled_admin(self, app, boss, store):
        alice = user_id(store, "legal-demo", "alice@firm.com")
        boss.post(f"/firm/api/users/{alice}/role", json={"role": "admin"})
        alice_client = signed_in(app, email="alice@firm.com")
        me = user_id(store, "legal-demo", "boss@firm.com")
        assert alice_client.post(f"/firm/api/users/{me}/role", json={"role": "member"}).status_code == 200
        # Alice is now the only admin, and nobody else is an admin who could demote or disable her
        assert store.enabled_admin_count("legal-demo") == 1

    def test_changes_the_firm_password_after_confirming_their_own(self, boss, store):
        assert boss.post("/firm/api/password", json={"account_password": "wrong", "password": "x" * 12}).status_code == 400
        assert store.verify_firm("legal-demo", FIRM_PASSWORD) == "legal-demo"
        assert boss.post("/firm/api/password", json={"account_password": PASSWORD,
                                                     "password": "the firm's new secret"}).status_code == 200
        assert store.verify_firm("legal-demo", "the firm's new secret") == "legal-demo"
        assert store.verify_firm("legal-demo", FIRM_PASSWORD) is None

    def test_members_and_the_owner_are_not_firm_admins(self, app):
        alice = signed_in(app)
        assert alice.get("/firm/api/users").status_code == 403
        assert alice.post("/firm/api/password", json={}).status_code == 403
        from web_deploy.tests.gateway.support import CONSOLE_PASSWORD
        owner = signed_in(app, OWNER_FIRM, "owner@onyx.test", firm_password=CONSOLE_PASSWORD)
        assert owner.get("/firm/api/users").status_code == 403

    def test_an_admin_of_another_firm_only_sees_theirs(self, app):
        eve = signed_in(app, "other-firm", "eve@other.com", firm_password=OTHER_FIRM_PASSWORD)
        assert [u["email"] for u in eve.get("/firm/api/users").json()["users"]] == ["eve@other.com"]
