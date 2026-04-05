# Canvas: hallucination_correction_exceptions_database_v1
# Path: app/features/hallucination_correction/exceptions/database.py

from app.features.hallucination_correction.exceptions.base import HallucinationCorrectionError

class CorrectionDatabaseError(HallucinationCorrectionError):
    """
    Raised when the Postgres database fails to update the Transcript
    after the hallucination sanitization is complete.
    """
    pass