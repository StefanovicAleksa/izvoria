# Path: app/features/report_export/exceptions/base.py

class ReportExportError(Exception):
    """
    The Root Exception for the Report Export feature.
    All export-specific exceptions inherit from this.
    """
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
