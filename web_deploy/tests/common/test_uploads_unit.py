import asyncio
import os

import pytest

from web_deploy.common.uploads import BatchRejected, UploadJanitor, batch_dir, batch_files, receive
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue

DAY = 24 * 3600
NOW = 2_000_000_000.0


def _batch(tenants, firm="legal-demo", owner="u1", batch="20261009-120000-abcdef"):
    folder = tenants / firm / "uploads" / owner / batch
    folder.mkdir(parents=True)
    (folder / "talk.mp4").write_bytes(b"video")
    return folder


def _age(folder, seconds_ago):
    for path in [folder, *folder.rglob("*")]:
        os.utime(path, (NOW - seconds_ago, NOW - seconds_ago))


async def _chunks(*parts):
    for part in parts:
        yield part


class TestBatches:
    def test_a_batch_lives_under_its_users_uploads(self, tmp_path):
        assert batch_dir(tmp_path, "legal-demo", "u1", "20261009-120000-abcdef") == \
            tmp_path / "legal-demo" / "uploads" / "u1" / "20261009-120000-abcdef"

    @pytest.mark.parametrize("user, batch", [("u1", "../../x"), ("../u1", "20261009-120000-abcdef"), ("u1", "")])
    def test_anything_but_a_batch_id_and_a_user_id_is_refused(self, tmp_path, user, batch):
        with pytest.raises(BatchRejected):
            batch_dir(tmp_path, "legal-demo", user, batch)

    def test_an_upload_arrives_whole_or_not_at_all(self, tmp_path):
        result = asyncio.run(receive(_chunks(b"video-", b"bytes"), tmp_path, "my deposition.mp4", 1024))
        assert result == {"name": "my deposition.mp4", "bytes": 11}

        with pytest.raises(BatchRejected):
            asyncio.run(receive(_chunks(b"x" * 2048), tmp_path, "big.mp4", 1024))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["my deposition.mp4"]

    def test_only_ingestible_visible_files_count(self, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / ".b.mp4.part").write_bytes(b"x")
        (tmp_path / "c.exe").write_bytes(b"x")
        assert [p.name for p in batch_files(tmp_path)] == ["a.pdf"]


class TestUploadJanitor:
    def test_a_finished_or_failed_jobs_folder_goes_at_once(self, tmp_path):
        queue = IngestQueue.connect("sqlite://")
        done, failed = _batch(tmp_path, batch="20261009-120000-aaaaaa"), _batch(tmp_path, batch="20261009-120000-bbbbbb")
        for folder, verdict in ((done, "finish"), (failed, "fail")):
            job = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(folder), 1)
            queue.claim_next()
            getattr(queue, verdict)(job.id, {} if verdict == "finish" else "boom")

        assert UploadJanitor(queue, tmp_path).run_once(NOW) == 2
        assert not done.exists() and not failed.exists()
        assert (tmp_path / "legal-demo" / "uploads" / "u1").is_dir()

    def test_a_queued_or_running_jobs_folder_stays_however_old(self, tmp_path):
        """A job a box stop re-queued reads its folder again on the retry."""
        queue = IngestQueue.connect("sqlite://")
        waiting = _batch(tmp_path)
        _age(waiting, 3 * DAY)
        queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(waiting), 1)

        assert UploadJanitor(queue, tmp_path).run_once(NOW) == 0
        assert waiting.exists()

    def test_a_folder_never_queued_goes_after_a_day_untouched(self, tmp_path):
        queue = IngestQueue.connect("sqlite://")
        abandoned = _batch(tmp_path, batch="20261001-000000-aaaaaa")
        uploading = _batch(tmp_path, batch="20261009-000000-bbbbbb")
        _age(abandoned, 2 * DAY)
        _age(uploading, 2 * DAY)
        (uploading / ".big.mp4.part").write_bytes(b"still arriving")
        os.utime(uploading / ".big.mp4.part", (NOW - 60, NOW - 60))

        assert UploadJanitor(queue, tmp_path).run_once(NOW) == 1
        assert not abandoned.exists() and uploading.exists()

    def test_nothing_outside_the_firms_uploads_is_ever_removed(self, tmp_path):
        queue = IngestQueue.connect("sqlite://")
        elsewhere = tmp_path / "samples"
        elsewhere.mkdir()
        (elsewhere / "keep.mp4").write_bytes(b"x")
        job = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(elsewhere), 1)
        queue.claim_next()
        queue.finish(job.id, {})

        UploadJanitor(queue, tmp_path / "tenants").run_once(NOW)

        assert (elsewhere / "keep.mp4").exists()
