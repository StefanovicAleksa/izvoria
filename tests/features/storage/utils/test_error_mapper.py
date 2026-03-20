from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from app.features.storage.utils.error_mapper import map_storage_error
from app.features.storage.exceptions.base import StorageError


class TestErrorMapper:
    def test_map_generic_db_error(self):
        """Should map generic SQLAlchemyError to StorageError."""
        db_err = SQLAlchemyError("Connection failed")
        mapped = map_storage_error(db_err)

        assert isinstance(mapped, StorageError)
        # FIX: Check for generic storage error message or type
        assert "storage error" in str(mapped).lower() or "database error" in str(mapped).lower()

    def test_map_integrity_error(self):
        """Should map IntegrityError to StorageError."""
        mock_orig = Exception("unique constraint")
        db_err = IntegrityError("INSERT...", {}, mock_orig)

        mapped = map_storage_error(db_err)
        assert isinstance(mapped, StorageError)