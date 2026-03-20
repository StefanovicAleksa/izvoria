from sqlalchemy.exc import SQLAlchemyError
from app.core.database import exceptions as core_db_exc
from app.legacy_features.intelligence_bridge import exceptions as exc


def map_bridge_error(e: Exception, context: str = "") -> exc.IntelligenceBridgeError:
    """Translates generic/DB errors into Intelligence Bridge Domain Exceptions."""
    error_msg = str(e).lower()

    # 1. Database and SQLAlchemy core errors
    if isinstance(e, (SQLAlchemyError, core_db_exc.DatabaseError)):
        return exc.BridgeDatabaseError(
            f"Database operation failed in Intelligence Bridge: {error_msg} | Context: {context}",
            original_exception=e
        )

    # 2. Logic/Math errors during map combination
    if isinstance(e, ValueError) and "merge" in error_msg:
        return exc.TimelineMergeError(
            f"Failed to merge scene timelines: {error_msg} | Context: {context}",
            original_exception=e
        )

    # 3. Catch-all
    return exc.IntelligenceBridgeError(
        f"Unexpected Bridge Error: {error_msg} | Context: {context}",
        original_exception=e
    )