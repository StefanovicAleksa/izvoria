import os
from pathlib import Path
from typing import Generator

from app.features.scanner.domain.interfaces import IScanner
from app.features.scanner.domain.entities import ScannedFile
from app.features.scanner.utils.filters import is_extension_allowed, is_directory_ignored
from app.features.scanner.utils.error_mapper import map_scanner_error
from app.features.scanner import exceptions as exc


class FileScanner(IScanner):
    def scan(self, root_path: str | Path) -> Generator[ScannedFile, None, None]:
        """
        Recursively scans the directory.
        Delegates error mapping to 'utils.error_mapper'.
        """
        target_path = Path(root_path).resolve()

        # Guard Clause: Fail fast if root doesn't exist
        if not target_path.exists():
            raise exc.ScanPathNotFound(f"Root directory not found: {target_path}")

        # Helper to force os.walk to raise errors immediately
        def _raise_error(e: OSError):
            raise e

        try:
            # FIX: Added 'onerror=_raise_error'
            # Without this, os.walk silently skips locked directories.
            for root, dirs, files in os.walk(target_path, onerror=_raise_error):

                # 1. Filter Directories (In-place modification)
                dirs[:] = [d for d in dirs if not is_directory_ignored(d)]

                for file in files:
                    file_path = Path(root) / file

                    # 2. Filter Files
                    if not is_extension_allowed(file_path):
                        continue

                    # 3. Yield Result
                    try:
                        stat = file_path.stat()

                        yield ScannedFile(
                            path=file_path,
                            file_name=file_path.name,
                            extension=file_path.suffix.lower(),
                            size_bytes=stat.st_size
                        )
                    except OSError:
                        # Strategy: Skip individual unreadable files, don't crash the whole scan.
                        continue

        except Exception as e:
            # Now this block will actually catch the PermissionError raised by _raise_error
            raise map_scanner_error(e, context=str(target_path)) from e