import uuid
from pathlib import Path

import pytest

from web_deploy.common.source_removal import remove_sources, unlink_inside

SOURCE = str(uuid.uuid4())


class FakeConnection:
    """A DB-API connection that records statements and answers the file lookup."""

    def __init__(self, files=(), fail_on=None):
        self.statements = []
        self.files = list(files)
        self.fail_on = fail_on
        self.committed = self.rolled_back = False

    def cursor(self):
        connection = self

        class Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params=None):
                statement = " ".join(sql.split())
                connection.statements.append((statement, params))
                if connection.fail_on and connection.fail_on in statement:
                    raise RuntimeError("database refused")

            def fetchall(self):
                return connection.files

        return Cursor()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class TestRemoveSources:
    def test_the_jobs_go_first_then_the_source_then_its_files(self):
        file_a, file_b = str(uuid.uuid4()), str(uuid.uuid4())
        connection = FakeConnection(files=[(file_a, "/v/original.mp4"), (file_b, "/v/audio.wav")])

        paths = remove_sources(connection, [SOURCE])

        assert paths == ["/v/original.mp4", "/v/audio.wav"]
        deletes = [(s.split(" WHERE")[0], p) for s, p in connection.statements if s.startswith("DELETE")]
        assert deletes == [("DELETE FROM jobs", {"source_ids": [SOURCE]}),
                           ("DELETE FROM sources", {"source_ids": [SOURCE]}),
                           ("DELETE FROM files", {"file_ids": [file_a, file_b]})]
        assert connection.committed

    def test_a_failure_rolls_everything_back(self):
        connection = FakeConnection(files=[(str(uuid.uuid4()), "/v/x.pdf")], fail_on="DELETE FROM sources")
        with pytest.raises(RuntimeError):
            remove_sources(connection, [SOURCE])
        assert connection.rolled_back and not connection.committed

    def test_nothing_to_remove_touches_nothing(self):
        connection = FakeConnection()
        assert remove_sources(connection, []) == []
        assert connection.statements == []

    def test_an_id_that_is_not_a_uuid_is_refused_before_any_sql(self):
        connection = FakeConnection()
        with pytest.raises(ValueError):
            remove_sources(connection, ["1; DROP TABLE sources"])
        assert connection.statements == []


class TestUnlinkInside:
    def test_files_inside_go_with_the_folders_they_leave_empty(self, tmp_path):
        vault = tmp_path / "vault"
        kept = vault / "2026" / "10" / "keep.mp4"
        gone = vault / "2026" / "10" / "20" / "gone.mp4"
        frame = vault / "frames" / "abc" / "frame_1.jpg"
        for path in (kept, gone, frame):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")

        assert unlink_inside([str(gone), str(frame), str(vault / "missing.wav")], vault) == 2

        assert kept.exists() and not gone.exists()
        assert not (vault / "2026" / "10" / "20").exists() and (vault / "2026" / "10").is_dir()
        assert not (vault / "frames").exists() and vault.is_dir()

    def test_nothing_outside_the_vault_or_behind_a_symlink_is_touched(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        outside = tmp_path / "other-firm.mp4"
        outside.write_bytes(b"x")
        link = vault / "link.mp4"
        link.symlink_to(outside)

        assert unlink_inside([str(outside), str(link), str(vault / ".." / "other-firm.mp4")], vault) == 0
        assert outside.exists() and link.is_symlink()
