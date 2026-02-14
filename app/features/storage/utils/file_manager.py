import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

class FileManager:
    def __init__(self):
        env_path = os.getenv("ONYX_STORAGE_PATH", "onyx_data")

        # Resolve to absolute path
        self.base_path = Path(env_path).resolve()

    def generate_path(self, file_extension: str) -> Path:
        now = datetime.now(timezone.utc)

        relative_path = Path(f"{now.year}/{now.month:02d}/{now.day:02d}")
        full_dir = self.base_path / relative_path

        full_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{uuid.uuid4()}{file_extension}"

        return full_dir / filename

    def store_file(self, source_path: Union[str, Path], dest_path: Path, move: bool = False) -> int:
        src = Path(source_path)

        if not src.exists():
            raise FileNotFoundError(f"Source path does not exist: {src}")

        try:
            if move:
                shutil.move(str(src), str(dest_path))
            else:
                shutil.copy2(str(src), str(dest_path))

            # Security: Set to Read-Only (644) so the app doesn't accidentally overwrite it later
            os.chmod(dest_path, 0o644)

            # return size of the file
            return dest_path.stat().st_size

        except OSError as e:
            raise OSError(f"Storage operation failed: {str(e)}") from e

    def delete(self, file_path: Union[str, Path]):
        try:
            p = Path(file_path)
            # Security check: Ensure we are only deleting files inside our vault
            # (prevents accidental deletion of system files if path is malformed)
            if self.base_path in p.resolve().parents or p.resolve().parent == self.base_path:
                if p.exists():
                    p.unlink()
        except OSError:
            pass  # Fail silently