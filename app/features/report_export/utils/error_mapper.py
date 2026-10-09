# Path: app/features/report_export/utils/error_mapper.py

from app.features.report_export import exceptions as exc


def map_export_error(e: Exception, context: str = "") -> exc.ReportExportError:
    """Translates layout, font and file errors into the Report Export feature's typed exceptions."""
    if isinstance(e, exc.ReportExportError):
        return e

    prefix = f"{context}: " if context else ""

    # A font file that went missing or unreadable between the check and the load
    if isinstance(e, (FileNotFoundError, PermissionError)):
        return exc.FontMissingError(f"{prefix}A report font could not be read: {e}", original_exception=e)

    # fpdf2 raises FPDFException for layout and font problems (a glyph table it cannot use, a page error)
    if type(e).__name__ in ("FPDFException", "FPDFUnicodeEncodingException"):
        return exc.DocumentRenderError(f"{prefix}The PDF could not be laid out: {e}", original_exception=e)

    return exc.DocumentRenderError(f"{prefix}An unexpected error occurred while exporting: {e}",
                                   original_exception=e)
