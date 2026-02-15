from .base import ScannerError

class ScanPathNotFound(ScannerError):
    """Raised when the target directory does not exist."""
    pass

class ScanPermissionError(ScannerError):
    """Raised when the application lacks read permissions."""
    pass