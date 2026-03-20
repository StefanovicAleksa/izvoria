# Canvas: glossary_miner_network_exceptions_v2
# Path: app/features/glossary_miner/exceptions/network.py

from app.tools.glossary_miner.exceptions.base import GlossaryMinerError

class WikidataNetworkError(GlossaryMinerError):
    """Base class for all HTTP and TCP level network failures."""
    pass

class WikidataTimeoutError(WikidataNetworkError):
    """Raised when the socket connection times out before receiving data."""
    pass

class WikidataConnectionRefusedError(WikidataNetworkError):
    """Raised when DNS fails or the target server actively refuses the connection."""
    pass

class WikidataRateLimitError(WikidataNetworkError):
    """Raised when Wikidata returns an HTTP 429 (Too Many Requests)."""
    def __init__(self, message: str, retry_after_seconds: int = 60):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds

class WikidataServiceUnavailableError(WikidataNetworkError):
    """Raised when Wikidata returns HTTP 502, 503, or 504 (Server Overload/Maintenance)."""
    pass