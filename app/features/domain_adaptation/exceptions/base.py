# Canvas: domain_adaptation_exceptions_base_v1
# Path: app/features/domain_adaptation/exceptions/base.py

class DomainAdaptationError(Exception):
    """
    The Root Exception for the Domain Adaptation feature.
    All specific exceptions within this feature inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception