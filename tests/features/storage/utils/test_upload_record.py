# Path: tests/features/storage/utils/test_upload_record.py

import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from app.features.storage.domain.entities import SourceType
from app.features.storage.service.storage_api import StorageAPI
from app.features.storage.utils.upload_record import (
    RECORD_NAME, UploadRecord, read_all, read_record, write_record,
)

SHA = "a" * 64


class TestUploadRecord:
    def test_a_prepared_file_is_known_by_its_uploads_hash_and_name(self, tmp_path):
        write_record(tmp_path, {"talk.mp4": UploadRecord(SHA, "talk.mov")})

        assert read_record(tmp_path / "talk.mp4") == UploadRecord(SHA, "talk.mov")
        assert read_all(tmp_path) == {"talk.mp4": UploadRecord(SHA, "talk.mov")}

    def test_a_file_without_a_record_has_none(self, tmp_path):
        write_record(tmp_path, {"talk.mp4": UploadRecord(SHA, "talk.mov")})
        assert read_record(tmp_path / "other.mp4") is None
        assert read_record(tmp_path / "nowhere" / "x.mp4") is None

    @pytest.mark.parametrize("entry", [{"sha256": "not-a-hash", "original_name": "a.mov"},
                                       {"sha256": SHA, "original_name": ""}, "junk"])
    def test_an_unusable_entry_is_ignored(self, tmp_path, entry):
        import json
        (tmp_path / RECORD_NAME).write_text(json.dumps({"files": {"talk.mp4": entry}}))
        assert read_record(tmp_path / "talk.mp4") is None

    def test_a_broken_record_file_is_ignored(self, tmp_path):
        (tmp_path / RECORD_NAME).write_text("{not json")
        assert read_record(tmp_path / "talk.mp4") is None
        assert read_all(tmp_path) == {}


class TestIngestingAPreparedFile:
    @pytest.fixture
    def api(self):
        with patch("app.features.storage.service.storage_api.FileManager"):
            api = StorageAPI(repository=Mock())
        api.file_manager.generate_path.return_value = Path("/vault/x.mp4")
        api.file_manager.store_file.return_value = 10
        api.repository.get_by_hash.return_value = None
        api.repository.create.side_effect = lambda source: source
        return api

    def test_the_uploads_hash_and_name_are_used_and_the_copy_is_not_hashed(self, api, tmp_path):
        copy = tmp_path / "talk.mp4"
        copy.write_bytes(b"720p copy")
        write_record(tmp_path, {"talk.mp4": UploadRecord(SHA, "talk.mov")})

        with patch("app.features.storage.service.storage_api.calculate_file_hash") as hashing:
            source = api.ingest_local_file(str(copy), SourceType.VIDEO, language="en")

        hashing.assert_not_called()
        api.repository.get_by_hash.assert_called_once_with(SHA)
        assert (source.content_hash, source.name) == (SHA, "talk.mov")

    def test_a_reupload_of_the_same_original_finds_its_source(self, api, tmp_path):
        copy = tmp_path / "talk.mp4"
        copy.write_bytes(b"a different encode")
        write_record(tmp_path, {"talk.mp4": UploadRecord(SHA, "talk.mov")})
        existing = Mock(language="en", id=uuid.uuid4())
        api.repository.get_by_hash.return_value = existing

        assert api.ingest_local_file(str(copy), SourceType.VIDEO, language="en") is existing

    def test_without_a_record_the_file_itself_is_hashed(self, api, tmp_path):
        upload = tmp_path / "talk.mp4"
        upload.write_bytes(b"original")

        with patch("app.features.storage.service.storage_api.calculate_file_hash", return_value="b" * 64):
            source = api.ingest_local_file(str(upload), SourceType.VIDEO, language="en")

        assert (source.content_hash, source.name) == ("b" * 64, "talk.mp4")
