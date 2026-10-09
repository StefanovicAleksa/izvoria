import pytest

from web_deploy.ingest_worker.queue import IngestMode, IngestQueue
from web_deploy.provision.seed_demo import SEED_OWNER, seed


class TestSeedDemo:
    def test_copies_the_samples_into_the_firms_uploads_and_queues_them(self, tmp_path):
        samples = tmp_path / "samples"
        samples.mkdir()
        (samples / "contract one.pdf").write_bytes(b"%PDF")
        (samples / "notes.txt").write_text("hello")
        (samples / "ignore.exe").write_bytes(b"MZ")
        queue = IngestQueue.connect("sqlite://")

        result = seed(queue, tmp_path / "tenants", "legal-demo", IngestMode.DOCUMENT, "en", samples)

        job = queue.get(result["job"])
        assert (result["files"], result["position"], job.owner_id, job.firm) == (2, 1, SEED_OWNER, "legal-demo")
        copied = sorted(p.name for p in (tmp_path / "tenants" / "legal-demo" / "uploads" / SEED_OWNER).glob("*/*"))
        assert copied == ["contract one.pdf", "notes.txt"]

    def test_an_empty_folder_or_bad_firm_is_refused(self, tmp_path):
        queue = IngestQueue.connect("sqlite://")
        with pytest.raises(ValueError):
            seed(queue, tmp_path, "legal-demo", IngestMode.AUDIO, "en", tmp_path)
        with pytest.raises(ValueError):
            seed(queue, tmp_path, "../x", IngestMode.AUDIO, "en", tmp_path)
