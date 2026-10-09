import pytest

from web_deploy.common.signing import (
    H_FIRM, H_ROLE, H_SIG, H_TS, H_USER, MAX_SKEW_SECONDS, Principal, SignatureError, sign, verify,
)

SECRET = "s" * 40
ALICE = Principal(firm="legal-demo", user_id="u1", role="member")


class TestSigning:
    def test_a_signed_request_verifies_to_its_principal(self):
        headers = sign(SECRET, "post", "/api/chat", "a=1", ALICE, now=1000)

        assert verify(SECRET, "POST", "/api/chat", "a=1", headers, now=1000) == ALICE
        assert set(headers) == {H_FIRM, H_USER, H_ROLE, H_TS, H_SIG}

    @pytest.mark.parametrize("change", [
        {"method": "DELETE"}, {"path": "/api/searches"}, {"query": "a=2"},
    ])
    def test_the_signature_covers_method_path_and_query(self, change):
        headers = sign(SECRET, "POST", "/api/chat", "a=1", ALICE, now=1000)
        args = {"method": "POST", "path": "/api/chat", "query": "a=1", **change}

        with pytest.raises(SignatureError, match="does not match"):
            verify(SECRET, args["method"], args["path"], args["query"], headers, now=1000)

    @pytest.mark.parametrize("header, value", [(H_FIRM, "other-firm"), (H_USER, "u2"), (H_ROLE, "admin")])
    def test_the_signature_covers_who_is_asking(self, header, value):
        headers = sign(SECRET, "GET", "/api/x", "", ALICE, now=1000)
        headers[header] = value

        with pytest.raises(SignatureError):
            verify(SECRET, "GET", "/api/x", "", headers, now=1000)

    def test_a_stale_signature_is_refused(self):
        headers = sign(SECRET, "GET", "/api/x", "", ALICE, now=1000)

        with pytest.raises(SignatureError, match="expired"):
            verify(SECRET, "GET", "/api/x", "", headers, now=1000 + MAX_SKEW_SECONDS + 1)

    def test_another_secret_is_refused(self):
        headers = sign("t" * 40, "GET", "/api/x", "", ALICE, now=1000)
        with pytest.raises(SignatureError):
            verify(SECRET, "GET", "/api/x", "", headers, now=1000)

    def test_unsigned_requests_and_missing_secrets_are_refused(self):
        with pytest.raises(SignatureError, match="not signed"):
            verify(SECRET, "GET", "/api/x", "", {}, now=1000)
        with pytest.raises(SignatureError, match="no signing secret"):
            verify("", "GET", "/api/x", "", sign(SECRET, "GET", "/api/x", "", ALICE), now=1000)

    def test_header_names_are_case_insensitive(self):
        headers = {k.upper(): v for k, v in sign(SECRET, "GET", "/api/x", "", ALICE, now=1000).items()}
        assert verify(SECRET, "GET", "/api/x", "", headers, now=1000) == ALICE

    def test_an_unknown_role_or_invalid_firm_is_refused_even_when_signed(self):
        for principal in (Principal("legal-demo", "u1", "root"), Principal("Bad_Firm", "u1", "member")):
            headers = sign(SECRET, "GET", "/api/x", "", principal, now=1000)
            with pytest.raises(SignatureError):
                verify(SECRET, "GET", "/api/x", "", headers, now=1000)


class TestServiceSigning:
    def test_round_trip_and_tampering(self):
        from web_deploy.common.signing import sign_service, verify_service
        headers = sign_service(SECRET, "POST", "/internal/gpu/pause", now=1000)
        verify_service(SECRET, "POST", "/internal/gpu/pause", headers, now=1000)
        for args in (("POST", "/internal/gpu/resume", 1000), ("POST", "/internal/gpu/pause", 1000 + MAX_SKEW_SECONDS + 1)):
            with pytest.raises(SignatureError):
                verify_service(SECRET, args[0], args[1], headers, now=args[2])
        with pytest.raises(SignatureError):
            verify_service(SECRET, "POST", "/internal/gpu/pause", {}, now=1000)

    def test_a_user_signature_is_not_a_service_signature(self):
        from web_deploy.common.signing import verify_service
        with pytest.raises(SignatureError):
            verify_service(SECRET, "POST", "/internal/gpu/pause", sign(SECRET, "POST", "/internal/gpu/pause", "", ALICE))
