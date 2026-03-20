# Canvas: test_glossary_miner_error_mapper_v4
# Path: tests/features/glossary_miner/utils/test_error_mapper.py

import json
import urllib.error
import pytest
from socket import timeout
from email.message import Message
from pydantic import BaseModel, ValidationError

from app.tools.glossary_miner.exceptions import (
    GlossaryMinerError,
    WikidataNetworkError,
    WikidataTimeoutError,
    WikidataConnectionRefusedError,
    WikidataRateLimitError,
    WikidataServiceUnavailableError,
    MalformedSparqlQueryError,
    SparqlServerTimeoutError,
    TermExtractionError,
    ManifestFormattingError,
)
from app.tools.glossary_miner.utils.error_mapper import map_wikidata_error


class DummyModel(BaseModel):
    required_field: int


def create_http_error(code: int, msg: str, headers_dict: dict = None) -> urllib.error.HTTPError:
    """Helper to bypass PyCharm's strict Message[str, str] type hinting for urllib headers."""
    headers = Message()
    if headers_dict:
        for k, v in headers_dict.items():
            headers[k] = v
    return urllib.error.HTTPError("https://query.wikidata.org", code, msg, headers, None)


class TestGlossaryMinerErrorMapper:

    # --- HTTP PROTOCOL ERRORS ---

    def test_map_http_429_rate_limit(self):
        err = create_http_error(429, "Too Many Requests", {"Retry-After": "120"})
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, WikidataRateLimitError)
        assert mapped.retry_after_seconds == 120

    def test_map_http_400_bad_request(self):
        err = create_http_error(400, "Bad Request")
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, MalformedSparqlQueryError)

    def test_map_http_500_timeout(self):
        err = create_http_error(500, "Internal Server Error")
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, SparqlServerTimeoutError)

    def test_map_http_503_service_unavailable(self):
        err = create_http_error(503, "Service Unavailable")
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, WikidataServiceUnavailableError)

    def test_map_http_generic_network_error(self):
        err = create_http_error(418, "I'm a teapot")
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, WikidataNetworkError)

    # --- SOCKET / URL ERRORS ---

    def test_map_url_error_connection_refused(self):
        reason = ConnectionRefusedError("Connection refused")
        err = urllib.error.URLError(reason)
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, WikidataConnectionRefusedError)

    def test_map_url_error_timeout(self):
        reason = timeout("timed out")
        err = urllib.error.URLError(reason)
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, WikidataTimeoutError)

    # --- PROCESSING ERRORS ---

    def test_map_json_decode_error(self):
        err = json.JSONDecodeError("Expecting value", "doc", 0)
        mapped = map_wikidata_error(err)
        assert isinstance(mapped, TermExtractionError)

    def test_map_pydantic_validation_error(self):
        with pytest.raises(ValidationError) as exc_info:
            DummyModel(required_field="not_an_int")  # type: ignore

        mapped = map_wikidata_error(exc_info.value)
        assert isinstance(mapped, ManifestFormattingError)

    # --- FALLBACK ---

    def test_map_unknown_error(self):
        err = Exception("Unknown failure")
        mapped = map_wikidata_error(err)
        assert type(mapped) is GlossaryMinerError