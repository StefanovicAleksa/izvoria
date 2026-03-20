# Canvas: nlp_router_exceptions_base_v1
# Path: app/features/nlp_router/exceptions/base.py

class NLPRouterError(Exception):
    """
    The Root Exception for the NLP Router feature.
    All specific NLP routing exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception