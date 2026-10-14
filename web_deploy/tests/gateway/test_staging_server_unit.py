import hashlib
import subprocess
from pathlib import Path

import pytest

from app.features.storage.utils.upload_record import UploadRecord, write_record
from web_deploy.gateway.staging_server import BoxFiles, StagingRefused, StagingServer
from web_deploy.ingest_worker.queue import IngestMode, IngestQueue, JobOrigin


class FakeBox:
    """rsync to and from the box's sshd, against a folder standing in for the box's /workspace/tenants."""

    def __init__(self, vps_root: Path, box_root: Path):
        self.vps_root = vps_root
        self.box_root = box_root
        self.commands = []

    def _box(self, spec: str) -> Path:
        assert spec.startswith("root@127.0.0.1:")
        return self.box_root / Path(spec.split(":", 1)[1].rstrip("/")).relative_to(self.vps_root)

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        args = command[command.index("-e") + 2:]
        if "--list-only" in command:
            root = self._box(args[-1])
            listing = "\n".join(f"-rw-r--r--   1 2026/10/13 12:00:00 {p.relative_to(root)}"
                                for p in sorted(root.rglob("*")) if p.is_file())
            return subprocess.CompletedProcess(command, 0, listing, "")
        listing = next((a.split("=", 1)[1] for a in args if a.startswith("--files-from=")), None)
        source, target = args[-2], args[-1]
        if listing:
            for name in filter(None, Path(listing).read_text().splitlines()):
                out = self.box_root / name
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes((self.vps_root / name).read_bytes())
        elif target.startswith("root@"):
            self._copy(Path(source.rstrip("/")), self._box(target))
        else:
            self._copy(self._box(source), Path(target.rstrip("/")))
        return subprocess.CompletedProcess(command, 0, "", "")

    @staticmethod
    def _copy(src: Path, dst: Path):
        for path in src.rglob("*"):
            if path.is_file():
                out = dst / path.relative_to(src)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(path.read_bytes())


@pytest.fixture
def roots(tmp_path):
    vps = tmp_path / "workspace" / "tenants"
    vps.mkdir(parents=True)
    return vps, tmp_path / "box"


@pytest.fixture
def queue():
    return IngestQueue.connect("sqlite://")


def _server(roots, queue, lookup=lambda firm, hashes: [], tmp=None):
    vps, box = roots
    files = BoxFiles(tenants_root=vps, state_dir=vps.parent / "state", port=18022, key=Path("/etc/onyx-web/staging"))
    fake = FakeBox(vps, box)
    return StagingServer(files, queue, lookup=lookup, run=fake), fake


def _running_job(queue, vps, files, origin=JobOrigin.UPLOAD, firm="legal-demo"):
    folder = vps / firm / "uploads" / "u1" / "20261013-120000-abcdef"
    folder.mkdir(parents=True)
    for name, content in files.items():
        (folder / name).write_bytes(content)
    job = queue.enqueue(firm, "u1", IngestMode.VIDEO, "en", str(folder), max(1, len(files)), origin=origin)
    queue.claim_next()
    return job, folder


class TestStageIn:
    def test_the_running_jobs_batch_and_matching_vault_files_go_to_the_box(self, roots, queue):
        vps, box = roots
        job, folder = _running_job(queue, vps, {"talk.mp4": b"upload"})
        old = vps / "legal-demo" / "vault" / "2026" / "01" / "01" / "old.mp4"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"earlier copy")
        other_firm = vps / "other-firm" / "vault" / "x.mp4"
        other_firm.parent.mkdir(parents=True)
        other_firm.write_bytes(b"not theirs")
        asked = []

        def lookup(firm, hashes):
            asked.append((firm, hashes))
            return [str(old), str(other_firm), str(vps / "legal-demo" / "vault" / "gone.wav")]

        server, fake = _server(roots, queue, lookup)
        assert server.stage_in(job.id) == {"batch_files": 1, "vault_files": 1}

        assert asked == [("legal-demo", [hashlib.sha256(b"upload").hexdigest()])]
        assert (box / folder.relative_to(vps) / "talk.mp4").read_bytes() == b"upload"
        assert (box / old.relative_to(vps)).read_bytes() == b"earlier copy"
        assert not (box / other_firm.relative_to(vps)).exists()  # another firm's vault never travels

    def test_a_720p_copy_is_looked_up_by_its_uploads_hash(self, roots, queue):
        vps, _ = roots
        job, folder = _running_job(queue, vps, {"talk.mp4": b"copy"})
        write_record(folder, {"talk.mp4": UploadRecord("a" * 64, "talk.mov")})
        asked = []
        server, _ = _server(roots, queue, lambda firm, hashes: asked.append(hashes) or [])

        server.stage_in(job.id)

        assert asked == [["a" * 64]]

    def test_only_the_job_that_is_running(self, roots, queue):
        vps, _ = roots
        queued = queue.enqueue("legal-demo", "u1", IngestMode.VIDEO, "en", str(vps / "legal-demo" / "x"), 1)
        server, fake = _server(roots, queue)
        for job_id in (queued.id, "no-such-job"):
            with pytest.raises(StagingRefused):
                server.stage_in(job_id)
        assert fake.commands == []

    def test_a_youtube_batch_sends_no_batch(self, roots, queue):
        vps, _ = roots
        job, _ = _running_job(queue, vps, {}, origin=JobOrigin.YOUTUBE)
        server, fake = _server(roots, queue)
        assert server.stage_in(job.id) == {"batch_files": 0, "vault_files": 0}
        assert fake.commands == []

    def test_the_firms_glossary_for_the_jobs_language_goes_too(self, roots, queue):
        vps, box = roots
        job, _ = _running_job(queue, vps, {"talk.mp4": b"upload"})
        for firm, language in (("legal-demo", "en"), ("legal-demo", "sr"), ("other-firm", "en")):
            folder = vps / firm / "glossaries" / language
            folder.mkdir(parents=True)
            (folder / "law.json").write_text(f"{firm} {language}")
            (folder / "meta.json").write_text("{}")
        server, _ = _server(roots, queue)

        server.stage_in(job.id)

        assert (box / "legal-demo" / "glossaries" / "en" / "law.json").read_text() == "legal-demo en"
        assert not (box / "legal-demo" / "glossaries" / "sr").exists()
        assert not (box / "other-firm").exists()

    def test_a_youtube_job_takes_the_youtube_cookies_and_an_upload_does_not(self, roots, queue, tmp_path):
        vps, box = roots
        cookies = tmp_path / "youtube_cookies.txt"
        cookies.write_text("cookies")
        files = BoxFiles(tenants_root=vps, state_dir=vps.parent / "state", port=18022,
                         key=Path("/etc/onyx-web/staging"), youtube_cookies=cookies,
                         box_youtube_cookies=Path("/workspace/web/youtube_cookies.txt"))
        sent = []

        def run(command, **kwargs):
            sent.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        upload, _ = _running_job(queue, vps, {"talk.mp4": b"upload"})
        StagingServer(files, queue, lookup=lambda firm, hashes: [], run=run).stage_in(upload.id)
        assert not any(str(cookies) in c for c in sent)
        queue.fail(upload.id, "done")

        tube, _ = _running_job(queue, vps, {}, origin=JobOrigin.YOUTUBE, firm="other-firm")
        sent.clear()
        StagingServer(files, queue, lookup=lambda firm, hashes: [], run=run).stage_in(tube.id)
        command = next(c for c in sent if str(cookies) in c)
        assert "--chmod=F600" in command and command[-1] == "root@127.0.0.1:/workspace/web/youtube_cookies.txt"

    def test_no_glossary_sends_none(self, roots, queue):
        vps, box = roots
        job, _ = _running_job(queue, vps, {"talk.mp4": b"upload"})
        server, fake = _server(roots, queue)

        server.stage_in(job.id)

        assert not (box / "legal-demo" / "glossaries").exists()
        assert not any("glossaries" in " ".join(c) for c in fake.commands)

    def test_the_connection_is_the_boxs_own_forwarded_sshd_with_the_staging_key(self, roots, queue):
        vps, _ = roots
        job, _ = _running_job(queue, vps, {"talk.mp4": b"x"})
        server, fake = _server(roots, queue)
        server.stage_in(job.id)
        ssh = fake.commands[0][fake.commands[0].index("-e") + 1]
        assert "-p 18022" in ssh and "-i /etc/onyx-web/staging" in ssh


class TestStageOut:
    def test_the_firms_vault_comes_back_and_a_replaced_original_goes(self, roots, queue):
        vps, box = roots
        job, _ = _running_job(queue, vps, {"talk.mp4": b"upload"})
        old = vps / "legal-demo" / "vault" / "2026" / "01" / "01" / "old.mov"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"pulled original")
        neighbour = old.with_name("frame.jpg")
        neighbour.write_bytes(b"another source's frame")
        server, fake = _server(roots, queue, lambda firm, hashes: [str(old)])
        server.stage_in(job.id)

        (box / old.relative_to(vps)).unlink()  # compaction swapped it for a copy under a new path
        new = box / "legal-demo" / "vault" / "2026" / "10" / "13" / "new.mp4"
        new.parent.mkdir(parents=True)
        new.write_bytes(b"720p copy")

        assert server.stage_out(job.id) == {"removed": 1}
        assert (vps / new.relative_to(box)).read_bytes() == b"720p copy"
        assert not old.exists() and neighbour.exists()

    def test_a_finished_or_requeued_job_can_still_hand_back_its_results(self, roots, queue):
        vps, _ = roots
        job, _ = _running_job(queue, vps, {"talk.mp4": b"x"})
        queue.fail(job.id, "boom")
        server, _ = _server(roots, queue)
        assert server.stage_out(job.id) == {"removed": 0}


class TestYoutubeCookiesAfterTheDownload:
    def _server(self, roots, queue, tmp_path, box_jar=None, alerts=None):
        vps, _ = roots
        stored = tmp_path / "youtube_cookies.txt"
        signed = ".youtube.com\tTRUE\t/\tTRUE\t4102444800\tSAPISID\t{}\n"
        stored.write_text("# Netscape HTTP Cookie File\n" + signed.format("old"))
        files = BoxFiles(tenants_root=vps, state_dir=tmp_path / "state", port=18022, key=None,
                         youtube_cookies=stored, box_youtube_cookies=Path("/workspace/web/youtube_cookies.txt"))

        def run(command, **kwargs):
            if box_jar is None:
                return subprocess.CompletedProcess(command, 23, "", "No such file")
            Path(command[-1]).write_text(box_jar.format(signed.format("rotated")))
            return subprocess.CompletedProcess(command, 0, "", "")

        server = StagingServer(files, queue, lookup=lambda f, h: [], run=run,
                               on_youtube_refused=(lambda: alerts.append(1)) if alerts is not None else None)
        job, _ = _running_job(queue, vps, {}, origin=JobOrigin.YOUTUBE)
        return server, job, stored

    def test_the_refreshed_cookies_come_back(self, roots, queue, tmp_path):
        server, job, stored = self._server(roots, queue, tmp_path, box_jar="# Netscape HTTP Cookie File\n{}")
        assert server.youtube_cookies_refreshed(job.id) == {"taken_back": True}
        assert "rotated" in stored.read_text()

    def test_nothing_on_the_box_keeps_the_stored_ones(self, roots, queue, tmp_path):
        server, job, stored = self._server(roots, queue, tmp_path)
        assert server.youtube_cookies_refreshed(job.id) == {"taken_back": False}
        assert "old" in stored.read_text()
        assert not list((tmp_path / "state").glob(".youtube_cookies-*"))

    def test_a_refusal_emails_the_owner_once(self, roots, queue, tmp_path):
        alerts = []
        server, job, _ = self._server(roots, queue, tmp_path, alerts=alerts)
        assert server.youtube_refused(job.id) == {"alerted": True}
        assert server.youtube_refused(job.id) == {"alerted": False}
        assert alerts == [1]
