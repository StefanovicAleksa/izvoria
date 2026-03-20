# Canvas: glossary_miner_query_exceptions_v1
# Path: app/features/glossary_miner/exceptions/query.py

from app.tools.glossary_miner.exceptions.base import GlossaryMinerError

class SparqlExecutionError(GlossaryMinerError):
    """Base class for errors returned by the SPARQL execution engine."""
    pass

class MalformedSparqlQueryError(SparqlExecutionError):
    """Raised when Wikidata returns HTTP 400 because our SPARQL syntax is invalid."""
    pass

class SparqlServerTimeoutError(SparqlExecutionError):
    """
    Raised when the query is structurally valid, but takes too much compute time
    on Wikidata's Blazegraph database, causing the server to abort it.
    """
    pass