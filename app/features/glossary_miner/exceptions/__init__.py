# Canvas: glossary_miner_exceptions_init_v1
# Path: app/features/glossary_miner/exceptions/__init__.py

from app.features.glossary_miner.exceptions.base import GlossaryMinerError
from app.features.glossary_miner.exceptions.network import (
    WikidataNetworkError,
    WikidataTimeoutError,
    WikidataConnectionRefusedError,
    WikidataRateLimitError,
    WikidataServiceUnavailableError,
)
from app.features.glossary_miner.exceptions.query import (
    SparqlExecutionError,
    MalformedSparqlQueryError,
    SparqlServerTimeoutError,
)
from app.features.glossary_miner.exceptions.processing import (
    DataProcessingError,
    TermExtractionError,
    ManifestFormattingError,
)

__all__ = [
    "GlossaryMinerError",

    # Network
    "WikidataNetworkError",
    "WikidataTimeoutError",
    "WikidataConnectionRefusedError",
    "WikidataRateLimitError",
    "WikidataServiceUnavailableError",

    # Query
    "SparqlExecutionError",
    "MalformedSparqlQueryError",
    "SparqlServerTimeoutError",

    # Processing
    "DataProcessingError",
    "TermExtractionError",
    "ManifestFormattingError",
]