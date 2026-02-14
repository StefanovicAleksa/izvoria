from app.core.database import exceptions as db_exc
from app.features.storage import exceptions as storage_exc

def map_storage_error(e: Exception) -> Exception:
    """
    Translates Core Database Exceptions into Feature-Specific Storage Exceptions.
    Usage:
        try:
            repo.add(source)
        except DatabaseError as e:
            raise map_storage_error(e) from e
    """
    # 1. Handle Duplicates (Integrity)
    if isinstance(e, db_exc.DBUniqueError):
        # We assume for now that unique errors in this feature are Source duplicates.
        # Ideally, we would check e.details to see which constraint failed.
        return storage_exc.SourceAlreadyExists(f"Source already exists in the system.", original_exception=e)

    # 2. Handle Missing References (Integrity)
    if isinstance(e, db_exc.DBForeignKeyError):
        return storage_exc.ResourceNotFound("Referenced parent resource (Source) not found.", original_exception=e)

    # 3. Handle Connection Issues (Pass them through or wrap them?)
    # Generally, connection errors are catastrophic, so we might want to let them bubble up
    # OR wrap them in a generic StorageError if we want to hide the DB layer completely.
    if isinstance(e, db_exc.ConnectionError):
        return storage_exc.StorageError("Storage system is currently unavailable.", original_exception=e)

    # 4. Fallback
    return storage_exc.StorageError("An unexpected storage error occurred.", original_exception=e)