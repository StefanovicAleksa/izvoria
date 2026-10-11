# Canvas: engine_exceptions_base_v1
# Path: app/engine/exceptions/base.py

class EngineError(Exception):
    """
    The Root Exception for the unified Izvoria Engine.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception