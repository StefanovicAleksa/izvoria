import pytest

from web_deploy.gateway.passwords import WeakPassword, hash_password, verify_password
from web_deploy.gateway.store import OWNER_FIRM, SESSION_SECONDS, Store, StoreError

PASSWORD = "correct horse battery"


@pytest.fixture
def clock():
    return [1000.0]


@pytest.fixture
def store(tmp_path, clock):
    s = Store(tmp_path / "g.sqlite3", clock=lambda: clock[0])
    s.create_firm("legal-demo", "Legal demo", "en", is_demo=True)
    return s


class TestPasswords:
    def test_round_trip_and_salting(self):
        first, second = hash_password(PASSWORD), hash_password(PASSWORD)
        assert first != second
        assert verify_password(PASSWORD, first) and not verify_password("wrong password!", first)

    def test_short_passwords_and_garbage_hashes(self):
        with pytest.raises(WeakPassword):
            hash_password("short")
        assert verify_password(PASSWORD, "md5$abc") is False


class TestStore:
    def test_a_user_signs_in_only_at_their_own_firm(self, store):
        store.create_firm("other-firm", "Other", "sr")
        store.create_user("legal-demo", "Alice@Firm.com", PASSWORD, "admin")

        assert store.authenticate("legal-demo", "alice@firm.com", PASSWORD).role == "admin"
        assert store.authenticate("other-firm", "alice@firm.com", PASSWORD) is None
        assert store.authenticate("legal-demo", "alice@firm.com", "wrong password") is None

    def test_the_same_email_can_belong_to_two_firms_but_not_twice_to_one(self, store):
        store.create_firm("other-firm", "Other")
        store.create_user("legal-demo", "a@b.com", PASSWORD)
        store.create_user("other-firm", "a@b.com", PASSWORD)
        with pytest.raises(StoreError, match="already has an account"):
            store.create_user("legal-demo", "a@b.com", PASSWORD)

    @pytest.mark.parametrize("firm, role", [("legal-demo", "owner"), ("nowhere", "member"), (OWNER_FIRM, "member")])
    def test_roles_and_firms_are_checked(self, store, firm, role):
        with pytest.raises(StoreError):
            store.create_user(firm, "x@y.com", PASSWORD, role)

    def test_sessions_expire_and_end(self, store, clock):
        user = store.create_user("legal-demo", "a@b.com", PASSWORD)
        token = store.create_session(user.id)
        assert store.session_user(token).id == user.id
        store.end_session(token)
        assert store.session_user(token) is None

        token = store.create_session(user.id)
        clock[0] += SESSION_SECONDS + 1
        assert store.session_user(token) is None

    def test_disabling_or_resetting_signs_out_everywhere(self, store):
        user = store.create_user("legal-demo", "a@b.com", PASSWORD)
        token = store.create_session(user.id)
        store.set_password(user.id, "another long password")
        assert store.session_user(token) is None
        assert store.authenticate("legal-demo", "a@b.com", "another long password") is not None

        token = store.create_session(user.id)
        store.set_user_disabled(user.id, True)
        assert store.session_user(token) is None
        assert store.authenticate("legal-demo", "a@b.com", "another long password") is None

    def test_a_disabled_firm_cannot_sign_in(self, store):
        user = store.create_user("legal-demo", "a@b.com", PASSWORD)
        token = store.create_session(user.id)
        store.set_firm_disabled("legal-demo", True)
        assert store.session_user(token) is None
        assert store.authenticate("legal-demo", "a@b.com", PASSWORD) is None

    def test_the_session_token_is_not_stored(self, store, tmp_path):
        user = store.create_user("legal-demo", "a@b.com", PASSWORD)
        token = store.create_session(user.id)
        assert token.encode() not in (tmp_path / "g.sqlite3").read_bytes()


FIRM_PASSWORD = "a shared firm secret"

# The firms table of the first release, before firm passwords
_OLD_SCHEMA = """
CREATE TABLE firms (slug TEXT PRIMARY KEY, name TEXT NOT NULL, default_language TEXT NOT NULL DEFAULT 'en',
                    is_demo INTEGER NOT NULL DEFAULT 0, disabled INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);
INSERT INTO firms (slug, name, created_at) VALUES ('old-firm', 'Old firm', 1.0);
"""


class TestMigration:
    def test_an_old_database_gains_firm_passwords_and_keeps_its_firms(self, tmp_path):
        import sqlite3
        path = tmp_path / "old.sqlite3"
        db = sqlite3.connect(str(path))
        db.executescript(_OLD_SCHEMA)
        db.close()

        store = Store(path)
        assert store.firm("old-firm").has_password is False
        assert store.firms_without_password() == ["old-firm"]
        store.set_firm_password("old-firm", FIRM_PASSWORD)
        assert Store(path).verify_firm("old-firm", FIRM_PASSWORD) == "old-firm"  # and a second start is a no-op


class TestFirmPasswords:
    def test_verify_firm(self, store):
        assert store.verify_firm("legal-demo", FIRM_PASSWORD) is None  # made without one: closed
        store.set_firm_password("legal-demo", FIRM_PASSWORD)
        assert store.verify_firm(" Legal-Demo ", FIRM_PASSWORD) == "legal-demo"
        assert store.verify_firm("legal-demo", "wrong password!") is None
        assert store.verify_firm("nobody-here", FIRM_PASSWORD) is None
        store.set_firm_disabled("legal-demo", True)
        assert store.verify_firm("legal-demo", FIRM_PASSWORD) is None

    def test_the_console_password_lives_apart_from_firms(self, store):
        assert not store.has_console_password()
        assert store.verify_firm(OWNER_FIRM, FIRM_PASSWORD) is None
        store.set_firm_password(OWNER_FIRM, FIRM_PASSWORD)
        assert store.has_console_password()
        assert store.verify_firm(OWNER_FIRM, FIRM_PASSWORD) == OWNER_FIRM
        assert store.firm(OWNER_FIRM) is None

    def test_weak_or_unknown(self, store):
        with pytest.raises(WeakPassword):
            store.set_firm_password("legal-demo", "short")
        with pytest.raises(StoreError):
            store.set_firm_password("nobody-here", FIRM_PASSWORD)

    def test_passes_expire_and_close(self, store, clock):
        store.set_firm_password("legal-demo", FIRM_PASSWORD)
        token = store.create_firm_pass("legal-demo", 60)
        assert store.firm_pass(token) == "legal-demo"
        clock[0] += 61
        assert store.firm_pass(token) is None

        token = store.create_firm_pass("legal-demo", 60)
        store.end_firm_pass(token)
        assert store.firm_pass(token) is None

    def test_a_new_password_or_disabling_closes_every_pass(self, store, tmp_path):
        store.set_firm_password("legal-demo", FIRM_PASSWORD)
        token = store.create_firm_pass("legal-demo", 60)
        assert token.encode() not in (tmp_path / "g.sqlite3").read_bytes()
        store.set_firm_password("legal-demo", "another firm secret")
        assert store.firm_pass(token) is None

        token = store.create_firm_pass("legal-demo", 60)
        store.set_firm_disabled("legal-demo", True)
        assert store.firm_pass(token) is None

    def test_admin_count_and_firm_scoped_lookup(self, store):
        store.create_firm("other-firm", "Other")
        admin = store.create_user("legal-demo", "a@b.com", PASSWORD, "admin")
        other = store.create_user("other-firm", "c@d.com", PASSWORD, "admin")
        store.create_user("legal-demo", "m@b.com", PASSWORD, "member")
        assert store.enabled_admin_count("legal-demo") == 1
        assert store.firm_user("legal-demo", admin.id).email == "a@b.com"
        assert store.firm_user("legal-demo", other.id) is None
        store.set_user_disabled(admin.id, True)
        assert store.enabled_admin_count("legal-demo") == 0


class TestInboundRequests:
    def test_add_list_and_mark(self, store, clock):
        first = store.add_request("signup", "Ana@Firm.rs", {"firm_name": "F"}, name="Ana", firm="F")
        clock[0] += 1
        second = store.add_request("support", "it@firm.com", {"message": "m"}, subject="Help")
        assert [r.id for r in store.requests()] == [second.id, first.id]
        assert [r.id for r in store.requests(kind="signup")] == [first.id]
        assert first.email == "ana@firm.rs" and first.payload == {"firm_name": "F"}

        store.set_request_status(first.id, "handled")
        store.set_mail_result(second.id, "failed", "x" * 900)
        assert store.requests(status="new")[0].id == second.id
        assert len(store.request(second.id).mail_error) == 500
        with pytest.raises(StoreError):
            store.set_request_status(first.id, "deleted")

    def test_the_daily_cap_counts_what_was_emailed(self, store, clock):
        store.add_request("support", "a@b.com", {})
        store.add_request("support", "a@b.com", {}, mail_status="off")
        assert store.mailed_since(clock[0] - 10) == 1


class TestPurge:
    def test_expired_sessions_and_firm_passes_go(self, store, clock):
        store.set_firm_password("legal-demo", FIRM_PASSWORD)
        user = store.create_user("legal-demo", "a@b.com", PASSWORD)
        store.create_session(user.id)
        store.create_firm_pass("legal-demo", 60)
        live = store.create_firm_pass("legal-demo", 2 * SESSION_SECONDS)
        clock[0] += SESSION_SECONDS + 1
        assert store.purge_sessions() == 2
        assert store.firm_pass(live) == "legal-demo"
