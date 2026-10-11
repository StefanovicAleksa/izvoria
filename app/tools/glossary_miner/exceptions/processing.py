# Canvas: glossary_miner_processing_exceptions_v2
# Path: app/features/glossary_miner/exceptions/processing.py

from app.tools.glossary_miner.exceptions.base import GlossaryMinerError

class DataProcessingError(GlossaryMinerError):
    """Base class for all errors that occur after the payload has been successfully downloaded."""
    pass

class TermExtractionError(DataProcessingError):
    """Raised when the raw Wikidata JSON is malformed or missing the required SPARQL bindings."""
    pass

class ManifestFormattingError(DataProcessingError):
    """Raised when the cleaned Python dictionary fails to validate against the IzvoriaGlossaryManifest Pydantic schema."""
    pass