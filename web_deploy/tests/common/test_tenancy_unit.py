from pathlib import Path

import pytest

from web_deploy.common.tenancy import InvalidSlug, database_name, firm_root, validate_slug


class TestTenancy:
    @pytest.mark.parametrize("slug", ["legal-demo", "abc", "firm2026", "a1-b2-c3"])
    def test_valid_slugs(self, slug):
        assert validate_slug(slug) == slug

    @pytest.mark.parametrize("slug", ["", "ab", "-abc", "abc-", "a--b", "a_b", "1abc", "a" * 33,
                                      "admin", "www", "a.b"])
    def test_invalid_or_reserved_slugs(self, slug):
        with pytest.raises(InvalidSlug):
            validate_slug(slug)

    def test_a_firm_has_its_own_database_and_directory(self):
        assert database_name("legal-demo") == "onyx_firm_legal_demo"
        assert firm_root(Path("/t"), "legal-demo") == Path("/t/legal-demo")

    def test_a_path_cannot_escape_through_the_slug(self):
        with pytest.raises(InvalidSlug):
            firm_root(Path("/t"), "../etc")

    def test_slugs_are_lowercased(self):
        assert validate_slug(" Legal-Demo ") == "legal-demo"
