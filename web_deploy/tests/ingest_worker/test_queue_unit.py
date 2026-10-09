import pytest

from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, JobOrigin, JobStatus, QueueError


@pytest.fixture
def queue():
    return IngestQueue.connect("sqlite://")


class TestIngestQueue:
    def test_jobs_run_first_come_first_served_across_firms(self, queue):
        a = queue.enqueue("firm-a", "u1", IngestMode.VIDEO, "en", "/a", 2)
        b = queue.enqueue("firm-b", "u2", IngestMode.DOCUMENT, "sr", "/b", 5)

        assert queue.queue_positions() == {a.id: 1, b.id: 2}
        assert queue.claim_next().id == a.id
        assert queue.queue_positions() == {b.id: 1}
        assert queue.depth() == {"queued": 1, "running": 1}

    def test_a_claimed_job_is_never_claimed_twice(self, queue):
        queue.enqueue("firm-a", "u1", IngestMode.AUDIO, "en", "/a", 1)
        assert queue.claim_next() is not None
        assert queue.claim_next() is None

    def test_finish_and_fail_are_recorded(self, queue):
        a = queue.enqueue("firm-a", "u1", IngestMode.AUDIO, "en", "/a", 1)
        b = queue.enqueue("firm-a", "u1", IngestMode.AUDIO, "en", "/b", 1)
        queue.finish(a.id, {"succeeded": 1})
        queue.fail(b.id, "CUDA OOM")
        assert (queue.get(a.id).status, queue.get(a.id).report) == (JobStatus.DONE, {"succeeded": 1})
        assert (queue.get(b.id).status, queue.get(b.id).error) == (JobStatus.FAILED, "CUDA OOM")

    def test_only_a_queued_job_of_the_firm_can_be_cancelled(self, queue):
        a = queue.enqueue("firm-a", "u1", IngestMode.AUDIO, "en", "/a", 1)
        assert queue.cancel(a.id, "firm-b") is False
        assert queue.cancel(a.id, "firm-a") is True
        assert queue.claim_next() is None
        b = queue.enqueue("firm-a", "u1", IngestMode.AUDIO, "en", "/b", 1)
        queue.claim_next()
        assert queue.cancel(b.id, "firm-a") is False

    def test_jobs_interrupted_by_a_shutdown_go_back_in_line(self, queue):
        a = queue.enqueue("firm-a", "u1", IngestMode.AUDIO, "en", "/a", 1)
        queue.claim_next()
        assert queue.recover_interrupted() == 1
        assert queue.get(a.id).status is JobStatus.QUEUED
        assert queue.claim_next().id == a.id

    def test_the_firm_listing_hides_the_box_path_and_shows_positions(self, queue):
        a = queue.enqueue("firm-a", "u1", IngestMode.IMAGE, "en", "/secret/path", 3)
        listed = queue.list_for_firm("firm-a")
        assert listed[0]["id"] == a.id and listed[0]["position"] == 1
        assert "/secret/path" not in str(listed)
        assert queue.list_for_firm("firm-b") == []

    def test_an_empty_batch_is_refused(self, queue):
        with pytest.raises(QueueError):
            queue.enqueue("firm-a", "u1", IngestMode.IMAGE, "en", "/a", 0)


class TestOrigin:
    def test_an_uploaded_batch_is_the_default_and_a_youtube_one_says_so(self, queue):
        uploaded = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", "/b1", 2)
        fetched = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", "/b2", 3, origin=JobOrigin.YOUTUBE)

        assert queue.get(uploaded.id).origin is JobOrigin.UPLOAD
        assert queue.get(fetched.id).public()["origin"] == "youtube"

    def test_a_queue_from_before_origins_gains_the_column_with_its_jobs_kept(self, tmp_path):
        from sqlalchemy import create_engine, text
        url = f"sqlite:///{tmp_path / 'control.sqlite3'}"
        old = create_engine(url)
        with old.begin() as connection:
            connection.execute(text(
                "CREATE TABLE ingest_jobs (id VARCHAR(36) PRIMARY KEY, firm VARCHAR(64) NOT NULL, owner_id VARCHAR(64) "
                "NOT NULL, mode VARCHAR(16) NOT NULL, language VARCHAR(8) NOT NULL, batch_dir TEXT NOT NULL, "
                "file_count INTEGER NOT NULL, status VARCHAR(16) NOT NULL, created_at DATETIME NOT NULL, "
                "started_at DATETIME, finished_at DATETIME, progress JSON, report JSON, error TEXT)"))
            connection.execute(text(
                "INSERT INTO ingest_jobs (id, firm, owner_id, mode, language, batch_dir, file_count, status, created_at)"
                " VALUES ('j1', 'legal-demo', 'u1', 'video', 'en', '/b', 2, 'done', '2026-10-07 11:47:44')"))
        old.dispose()

        upgraded = IngestQueue.connect(url)

        assert upgraded.get("j1").origin is JobOrigin.UPLOAD
        IngestQueue.connect(url)  # a second start finds the column and leaves it alone
