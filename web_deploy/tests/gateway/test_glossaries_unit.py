"""A firm's own glossary per language: checked, stored beside its data, set by the owner only."""

import json

import pytest

from web_deploy.gateway import glossaries
from web_deploy.gateway.admin_cli import main
from web_deploy.gateway.glossaries import GlossaryRejected
from web_deploy.gateway.store import OWNER_FIRM
from web_deploy.tests.gateway.support import CONSOLE_PASSWORD, signed_in

LAW = {
    "industry": "Legal disputes",
    "language": "en",
    "base_terms": ["Petrović"],
    "sub_domains": {"CONTRACTS": {"anchor_words": ["contract"], "terms": ["force majeure"]}},
    "term_details": {"Petrović": {"forms": []}, "contract": {"forms": []}, "force majeure": {"forms": []}},
}


@pytest.fixture
def tenants(tmp_path):
    return tmp_path / "tenants"


def _files(tenants, firm="legal-demo", language="en"):
    return sorted(p.name for p in (tenants / firm / "glossaries" / language).iterdir())


class TestSetting:
    def test_an_upload_is_stored_under_its_industry_with_what_it_holds(self, tenants):
        meta = glossaries.set_uploaded(tenants, "legal-demo", "en", LAW, "law.json")

        assert _files(tenants) == ["legal_disputes.json", "meta.json"]
        assert json.loads((tenants / "legal-demo/glossaries/en/legal_disputes.json").read_text()) == LAW
        assert (meta["industry"], meta["source"], meta["original_name"]) == ("legal_disputes", "upload", "law.json")
        assert (meta["terms"], meta["sub_domains"]) == (3, ["CONTRACTS"])
        assert glossaries.status(tenants, "legal-demo") == {"en": meta, "sr": None}
        assert glossaries.active_file(tenants, "legal-demo", "en").name == "legal_disputes.json"

    def test_a_new_one_replaces_the_old_one_for_that_language_only(self, tenants):
        glossaries.set_uploaded(tenants, "legal-demo", "en", LAW, "law.json")
        glossaries.set_uploaded(tenants, "legal-demo", "sr", {**LAW, "language": "sr"}, "law.json")
        glossaries.set_builtin(tenants, "legal-demo", "en", "medical")

        assert _files(tenants) == ["medical.json", "meta.json"]
        assert _files(tenants, language="sr") == ["legal_disputes.json", "meta.json"]
        assert not list((tenants / "legal-demo" / "glossaries").glob(".*"))  # no staging folder left behind

    def test_the_industry_falls_back_to_the_files_name(self, tenants):
        meta = glossaries.set_uploaded(tenants, "legal-demo", "en", {**LAW, "industry": ""}, "Family Law.json")
        assert meta["industry"] == "family_law"

    @pytest.mark.parametrize("glossary, says", [
        (["not", "an", "object"], "JSON object"),
        ({"industry": "law"}, "no terms"),
        ({**LAW, "sub_domains": {"contracts": {"anchor_words": ["x"], "terms": []}}}, "error"),
        ({**LAW, "industry": "", "base_terms": ["Petrović"]}, "industry name"),
    ])
    def test_refused_uploads_write_nothing(self, tenants, glossary, says):
        with pytest.raises(GlossaryRejected, match=says):
            glossaries.set_uploaded(tenants, "legal-demo", "en", glossary, "")
        assert glossaries.status(tenants, "legal-demo") == {"en": None, "sr": None}

    def test_the_validators_errors_are_handed_back(self, tenants):
        cyrillic = {**LAW, "base_terms": ["Петровић"], "term_details": {**LAW["term_details"], "Петровић": {"forms": []}}}
        with pytest.raises(GlossaryRejected) as refused:
            glossaries.set_uploaded(tenants, "legal-demo", "sr", {**cyrillic, "language": "sr"}, "law.json")
        assert any("Cyrillic" in e for e in refused.value.errors)

    def test_only_served_languages_and_real_built_ins(self, tenants):
        for call in (lambda: glossaries.set_uploaded(tenants, "legal-demo", "sl", LAW, "law.json"),
                     lambda: glossaries.set_uploaded(tenants, "legal-demo", "../x", LAW, "law.json"),
                     lambda: glossaries.set_builtin(tenants, "legal-demo", "en", "../../etc/passwd"),
                     lambda: glossaries.set_builtin(tenants, "legal-demo", "sr", "medical")):
            with pytest.raises(GlossaryRejected):
                call()

    def test_general_and_clear_go_back_to_the_shared_glossary(self, tenants):
        glossaries.set_builtin(tenants, "legal-demo", "en", "finance")
        assert glossaries.set_builtin(tenants, "legal-demo", "en", "general") is None
        assert glossaries.active_file(tenants, "legal-demo", "en") is None
        glossaries.set_builtin(tenants, "legal-demo", "en", "finance")
        glossaries.clear(tenants, "legal-demo", "en")
        assert glossaries.status(tenants, "legal-demo")["en"] is None

    def test_a_claude_answer_sets_every_language_or_none(self, tenants):
        answer = {"en": LAW, "sr": {**LAW, "language": "sr"},
                  "review": [{"language": "sr", "term": "Jovanović", "reason": "unsure"}]}
        metas = glossaries.import_claude_answer(tenants, "legal-demo", answer, "answer.json")
        assert sorted(metas) == ["en", "sr"]
        assert json.loads((tenants / "legal-demo/glossaries/sr/legal_disputes.json").read_text())["review"][0]["term"] \
            == "Jovanović"

        glossaries.clear(tenants, "legal-demo", "en")
        glossaries.clear(tenants, "legal-demo", "sr")
        broken = {"en": LAW, "sr": {**LAW, "sub_domains": {}}}  # labels must match in every language
        with pytest.raises(GlossaryRejected):
            glossaries.import_claude_answer(tenants, "legal-demo", broken, "answer.json")
        assert glossaries.status(tenants, "legal-demo") == {"en": None, "sr": None}


class TestConsole:
    def owner(self, app):
        return signed_in(app, OWNER_FIRM, "owner@onyx.test", firm_password=CONSOLE_PASSWORD)

    def test_the_owner_sets_lists_and_clears(self, app, tmp_path):
        owner = self.owner(app)
        url = "/admin/api/firms/legal-demo/glossaries"

        shown = owner.get(url).json()
        assert shown["glossaries"] == {"en": None, "sr": None}
        assert "medical" in shown["builtins"]["en"]

        put = owner.put(f"{url}/en", json={"glossary": LAW, "file_name": "law.json"})
        assert put.status_code == 200, put.text
        assert put.json()["glossaries"]["en"]["industry"] == "legal_disputes"
        assert (tmp_path / "tenants/legal-demo/glossaries/en/legal_disputes.json").exists()

        assert owner.put(f"{url}/en", json={"builtin": "medical"}).json()["glossaries"]["en"]["builtin"] == "medical"
        assert owner.delete(f"{url}/en").json()["glossaries"]["en"] is None

    def test_a_refused_upload_says_why(self, app):
        bad = {**LAW, "sub_domains": {"contracts": {"anchor_words": ["x"], "terms": []}}}
        response = self.owner(app).put("/admin/api/firms/legal-demo/glossaries/en", json={"glossary": bad})
        assert response.status_code == 400
        assert any("UPPER_SNAKE_CASE" in e for e in response.json()["errors"])

    def test_unknown_firms_languages_and_the_console_itself(self, app):
        owner = self.owner(app)
        assert owner.get("/admin/api/firms/nowhere/glossaries").status_code == 404
        assert owner.get(f"/admin/api/firms/{OWNER_FIRM}/glossaries").status_code == 404
        assert owner.put("/admin/api/firms/legal-demo/glossaries/sl", json={"builtin": "medical"}).status_code == 400

    @pytest.mark.parametrize("email", ["boss@firm.com", "alice@firm.com"])
    def test_a_firms_own_people_cannot(self, app, tmp_path, email):
        client = signed_in(app, email=email)
        assert client.get("/admin/api/firms/legal-demo/glossaries").status_code == 403
        assert client.put("/admin/api/firms/legal-demo/glossaries/en", json={"builtin": "medical"}).status_code == 403
        assert not (tmp_path / "tenants" / "legal-demo" / "glossaries").exists()


class TestCli:
    def test_set_show_and_clear(self, store, tenants, tmp_path, capsys):
        law = tmp_path / "law.json"
        law.write_text(json.dumps(LAW))
        base = ["--tenants", str(tenants)]

        assert main([*base, "set-glossary", "legal-demo", "en", str(law)], store=store) == 0
        assert main([*base, "set-glossary", "legal-demo", "sr", "--builtin", "medical"], store=store) == 1
        assert main([*base, "glossaries", "legal-demo"], store=store) == 0
        out = capsys.readouterr()
        assert "legal-demo [en]: legal_disputes (uploaded law.json): 3 terms, 1 sub-domains" in out.out
        assert "legal-demo [sr]: the shared general glossary" in out.out
        assert "no built-in 'medical' glossary for 'sr'" in out.err

        assert main([*base, "clear-glossary", "legal-demo", "en"], store=store) == 0
        assert glossaries.status(tenants, "legal-demo")["en"] is None
        assert main([*base, "glossaries", "nowhere"], store=store) == 1

    def test_a_claude_answer_in_one_step(self, store, tenants, tmp_path):
        answer = tmp_path / "answer.json"
        answer.write_text(json.dumps({"en": LAW, "sr": {**LAW, "language": "sr"}}))
        assert main(["--tenants", str(tenants), "set-glossary", "legal-demo", "--from-claude", str(answer)],
                    store=store) == 0
        assert all(glossaries.status(tenants, "legal-demo").values())
