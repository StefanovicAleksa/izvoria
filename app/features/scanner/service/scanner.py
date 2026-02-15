import os
from pathlib import Path
from typing import Generator

from app.features.scanner.domain.interfaces import IScaner
from app.features.scanner.domain.entities import ScannedFile
from app.features.scanner.utils.filters import is_extension_allowed, is_directory_ignored
from app.features.scanner.utils.error_mapper import map_scanner_error
from app.features.scanner import exceptions as exc

class FileScanner(IScaner):
    def scan(self, root_path: str, file_name: str) -> Generator[ScannedFile, None, None]:
        target_path = Path(root_path).resolve()

        if not target_path.exists():
            raise exc.ScanPathNotFound(f"Root directory not found: {target_path}")

        try:
            # os.walk is robust, but we wrap it to catch root-level permission issues
            for root, dirs, files in os.walk(target_path):

                # 1. Filter Directories (In-place modification)
                # We modify 'dirs' to prevent os.walk from entering ignored folders
                dirs[:] = [d for d in dirs if not is_directory_ignored(d)]

                for file in files:
                    file_path = Path(root) / file

                    # 2. Filter Files
                    if not is_extension_allowed(file_path):
                        continue

                    # 3. Yield Result
                    try:
                        # We stat the file to get size.
                        # This can fail for broken symlinks or locked files.
                        stat = file_path.stat()

                        yield ScannedFile(
                            path=file_path,
                            name=file_path.name,
                            extension=file_path.suffix.lower(),
                            size_bytes=stat.st_size
                        )
                    except OSError:
                        # Strategy: Skip individual unreadable files, don't crash the whole scan.
                        continue

        except Exception as e:
            # Catch-all: If the loop crashes (e.g., HDD disconnect), map and re-raise.
            raise map_scanner_error(e, context=str(target_path)) from e