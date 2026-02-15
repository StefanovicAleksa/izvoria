from app.features.scanner import exceptions as exc

def map_scanner_error(e: Exception, context: str = "") -> exc.ScannerError:
    """
    Translates raw system errors (OSError, PermissionError) into
    Onyx Scanner Domain Exceptions.

    Args:
        e: The original exception caught.
        context: Additional context (e.g., the path being accessed).
    """
    # 1. Access Denied (Common in 'System' folders)
    if isinstance(e, PermissionError):
        return exc.ScanPermissionError(
            f"Permission denied accessing: {context}",
            original_exception=e
        )

    # 2. Path Vanished (Race condition: deleted during scan)
    if isinstance(e, FileNotFoundError):
        return exc.ScanPathNotFound(
            f"Target path not found: {context}",
            original_exception=e
        )

    # 3. Generic System I/O (Disk full, Hardware failure)
    if isinstance(e, OSError):
        return exc.ScannerError(
            f"System I/O error during scan at {context}: {str(e)}",
            original_exception=e
        )

    # 4. Catch-all for unexpected crashes
    return exc.ScannerError(
        f"Unexpected scanner error: {str(e)}",
        original_exception=e
    )