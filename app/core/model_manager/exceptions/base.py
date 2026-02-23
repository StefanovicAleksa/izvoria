# File: app/core/model_manager/exceptions/base.py
class ModelManagerError(Exception):
    """
    The Root Exception for the Model Manager Core Feature.
    All VRAM and weight-loading exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception