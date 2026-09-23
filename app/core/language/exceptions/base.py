# Path: app/core/language/exceptions/base.py

class LanguageError(Exception):
    """
    The Root Exception for the Language core feature.
    Every language pack failure inherits from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
