import pytest

from web_deploy.gateway.admin_cli import main
from web_deploy.gateway.store import OWNER_FIRM, Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_ADMIN_PASSWORD", "a long enough password")
    return Store(tmp_path / "g.sqlite3")


class TestAdminCli:
    def test_owner_firm_and_users(self, store, capsys):
        assert main(["create-owner", "me@x.com"], store=store) == 0
        assert main(["create-firm", "legal-demo", "Legal demo", "--language", "sr", "--demo"], store=store) == 0
        assert main(["create-user", "legal-demo", "a@b.com", "--role", "admin"], store=store) == 0
        assert main(["list"], store=store) == 0

        out = capsys.readouterr().out
        assert "legal-demo (demo): Legal demo, answers in sr" in out
        assert "a@b.com admin" in out
        assert store.users(OWNER_FIRM)[0].role == "owner"

    def test_errors_are_reported_not_raised(self, store, capsys):
        assert main(["create-user", "nowhere", "a@b.com"], store=store) == 1
        assert main(["create-firm", "Bad_Slug", "x"], store=store) == 1
        assert "error:" in capsys.readouterr().err

    def test_reset_and_disable(self, store):
        main(["create-firm", "legal-demo", "Legal"], store=store)
        main(["create-user", "legal-demo", "a@b.com"], store=store)
        assert main(["disable-user", "legal-demo", "a@b.com"], store=store) == 0
        assert store.users("legal-demo")[0].disabled is True
        assert main(["reset-password", "legal-demo", "missing@b.com"], store=store) == 1


class TestFirmPasswords:
    def test_create_owner_also_sets_the_console_password(self, store):
        main(["create-owner", "me@x.com"], store=store)
        assert store.verify_firm(OWNER_FIRM, "a long enough password") == OWNER_FIRM

    def test_firms_get_a_password_or_are_flagged(self, store, capsys):
        main(["create-firm", "legal-demo", "Legal"], store=store)
        main(["create-firm", "closed-firm", "Closed", "--no-password"], store=store)
        assert store.verify_firm("legal-demo", "a long enough password") == "legal-demo"
        main(["list"], store=store)
        out = capsys.readouterr().out
        assert "closed-firm [no firm password]" in out and "legal-demo [no firm password]" not in out

    def test_set_and_generate(self, store, capsys):
        main(["create-firm", "legal-demo", "Legal", "--no-password"], store=store)
        assert main(["set-firm-password", "legal-demo", "--generate"], store=store) == 0
        shown = next(line for line in capsys.readouterr().out.splitlines() if "shown once" in line)
        generated = shown.split(": ")[-1].strip()
        assert store.verify_firm("legal-demo", generated) == "legal-demo"
        assert main(["set-firm-password", "nobody-here"], store=store) == 1

    def test_requests(self, store, capsys):
        request = store.add_request("support", "it@firm.com", {"message": "m"}, subject="Upload fails")
        assert main(["list-requests", "--kind", "support"], store=store) == 0
        assert "Upload fails" in capsys.readouterr().out
        assert main(["mark-request", request.id, "handled"], store=store) == 0
        assert store.request(request.id).status == "handled"


class TestCreateOwnerIsAllOrNothing:
    def test_a_mistyped_console_password_creates_no_owner(self, store, monkeypatch):
        monkeypatch.delenv("ONYX_ADMIN_PASSWORD")
        typed = iter(["owner password 1", "owner password 1", "console door 1", "console door 2"])
        monkeypatch.setattr("web_deploy.gateway.admin_cli.getpass.getpass", lambda prompt="": next(typed))

        assert main(["create-owner", "me@x.com"], store=store) == 1
        assert store.users(OWNER_FIRM) == []
        assert not store.has_console_password()

    def test_a_second_owner_keeps_the_console_password(self, store, monkeypatch):
        main(["create-owner", "me@x.com"], store=store)
        monkeypatch.setenv("ONYX_ADMIN_PASSWORD", "a different password")
        assert main(["create-owner", "two@x.com"], store=store) == 0
        assert store.verify_firm(OWNER_FIRM, "a long enough password") == OWNER_FIRM

    def test_a_too_short_console_password_creates_no_owner(self, store, monkeypatch):
        monkeypatch.setenv("ONYX_FIRM_PASSWORD", "short")
        assert main(["create-owner", "me@x.com"], store=store) == 1
        assert store.users(OWNER_FIRM) == []
