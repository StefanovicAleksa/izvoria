# Canvas: glossary_miner_base_exceptions_v1
# Path: app/features/glossary_miner/exceptions/base.py

class GlossaryMinerError(Exception):
    """
    Base exception for all errors originating within the Glossary Miner feature.
    """
    def __init__(self, message: str, original_exception: Exception | None = None):
        super().__init__(message)
        self.original_exception = original_exception