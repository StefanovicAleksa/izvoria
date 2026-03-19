# Canvas: glossary_miner_error_mapper_v4
# Path: app/features/glossary_miner/utils/error_mapper.py

import json
import socket
import urllib.error
from pydantic import ValidationError

from app.features.glossary_miner.exceptions import (
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


def map_wikidata_error(error: Exception) -> GlossaryMinerError:
    """
    Translates raw 3rd-party network, query, and data parsing exceptions
    into the strict Glossary Miner domain exception hierarchy.
    """
    # 1. Handle HTTP Protocol Errors
    if isinstance(error, urllib.error.HTTPError):
        code = error.code

        if code == 429:
            retry_after = error.headers.get("Retry-After", 60)
            return WikidataRateLimitError(
                message="Wikidata API rate limit exceeded. Back off required.",
                retry_after_seconds=int(retry_after)
            )
        if code == 400:
            return MalformedSparqlQueryError(
                message="Wikidata rejected the SPARQL query due to syntax/logic errors.",
                original_exception=error
            )
        if code in (502, 503, 504):
            return WikidataServiceUnavailableError(
                message=f"Wikidata servers are currently unavailable. HTTP {code}.",
                original_exception=error
            )
        if code == 500:
            return SparqlServerTimeoutError(
                message="SPARQL query execution timed out on the Wikidata server.",
                original_exception=error
            )

        return WikidataNetworkError(
            message=f"Unexpected HTTP Error connecting to Wikidata: {code}",
            original_exception=error
        )

    # 2. Handle Base URL / Connection / DNS Errors
    if isinstance(error, urllib.error.URLError):
        if isinstance(error.reason, socket.timeout):
            return WikidataTimeoutError(
                message="The network socket timed out while waiting for Wikidata.",
                original_exception=error
            )
        if isinstance(error.reason, ConnectionRefusedError) or "getaddrinfo" in str(error.reason):
            return WikidataConnectionRefusedError(
                message="Failed to connect to Wikidata (Connection Refused or DNS failure).",
                original_exception=error
            )

        return WikidataNetworkError(
            message=f"Failed to reach the Wikidata network: {str(error.reason)}",
            original_exception=error
        )

    # 3. Handle Raw Socket Timeouts
    if isinstance(error, socket.timeout):
        return WikidataTimeoutError(
            message="The underlying network socket timed out.",
            original_exception=error
        )

    # 4. Handle Processing & Parsing Errors (JSON / Pydantic)
    if isinstance(error, json.JSONDecodeError):
        return TermExtractionError(
            message="Failed to parse the JSON response from Wikidata. The payload may be corrupted.",
            original_exception=error
        )

    if isinstance(error, ValidationError):
        return ManifestFormattingError(
            message="Failed to validate the extracted data against the Onyx appliance Pydantic schema.",
            original_exception=error
        )

    # 5. Fallback for completely unknown errors
    return GlossaryMinerError(
        message=f"An unexpected error occurred during extraction: {str(error)}",
        original_exception=error
    )