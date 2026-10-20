"""The firm data the gateway holds itself: uploads, the wait list, and cited sources served from its own vault."""

import functools
import uuid
from pathlib import Path

import httpx
import pytest

from deploy.gateway.vast_client import FakeVastClient
from deploy.instance.agent.media import Prepared
from web_deploy.common.signing import verify
from web_deploy.gateway.app import create_app
from web_deploy.gateway.box import BoxController
from web_deploy.gateway.config import GatewaySettings
from web_deploy.gateway.data import DataRoutes, DataSettings, served_languages
from web_deploy.gateway.idle import IdleMonitor
from web_deploy.gateway.prepare import PrepareSettings
from web_deploy.gateway.mailer import Mailer
from web_deploy.ingest_worker.queue import IngestQueue
from web_deploy.tests.gateway.conftest import MAIL
from web_deploy.tests.gateway.support import (
    OTHER_FIRM_PASSWORD, SECRET, InlineExecutor, RecordingTransport, signed_in,
)


class Box:
    """The GPU box API: /health, and where a cited chunk's file is (it never sends the file)."""

    def __init__(self):
        self.locations = {}
        self.asked = []
        self.ready = True

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": self.ready, "busy": False})
        if not request.url.path.endswith("/locate"):
            self.asked.append(("forwarded", request.url.path))
            return httpx.Response(404, json={"error": "Not found.", "kind": "not_found"})
        principal = verify(SECRET, request.method, request.url.path, request.url.query.decode(), request.headers)
        self.asked.append((principal.firm, request.url.path))
        chunk_id = request.url.path.split("/")[3]
        if chunk_id not in self.locations:
            return httpx.Response(404, json={"error": "Not found.", "kind": "not_found"})
        return httpx.Response(200, json=self.locations[chunk_id])


class Preparer:
    def __init__(self):
        self.prepared = []

    async def prepare(self, offer, variant):
        self.prepared.append((offer.path, variant.value))
        return Prepared(offer.path, offer.mime)


@pytest.fixture
def tenants(tmp_path):
    return tmp_path / "tenants"


@pytest.fixture
def queue():
    return IngestQueue.connect("sqlite://")


@pytest.fixture
def box_api():
    return Box()


@pytest.fixture
def preparer():
    return Preparer()


@pytest.fixture
def vast():
    return FakeVastClient("127.0.0.1", (8095,))


@pytest.fixture
def prepare():
    return PrepareSettings()


class FirmDb:
    """Each firm's database, as far as deleting a source goes: which sources exist and the vault files they own."""

    def __init__(self):
        self.sources = {}  # (firm, source id) -> [vault paths]
        self.deleted = []

    def __call__(self, firm):
        db = self

        class Connection:
            def cursor(self):
                class Cursor:
                    rows = []

                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        return False

                    def execute(self, sql, params=None):
                        ids = (params or {}).get("source_ids", [])
                        if sql.startswith("SELECT id FROM sources"):
                            self.rows = [(i,) for i in ids if (firm, i) in db.sources]
                        elif "WITH matched" in sql:
                            self.rows = [(f"file-{n}", path) for i in ids
                                         for n, path in enumerate(db.sources.get((firm, i), []))]
                        elif sql.startswith("DELETE FROM sources"):
                            db.deleted += [(firm, i) for i in ids]

                    def fetchall(self):
                        return self.rows
                return Cursor()

            def commit(self):
                pass

            def rollback(self):
                pass

            def close(self):
                pass
        return Connection()


@pytest.fixture
def firm_db():
    return FirmDb()


@pytest.fixture
def app(tmp_path, store, vast, box_api, tenants, queue, preparer, prepare, firm_db):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>spa</html>")
    settings = GatewaySettings(api_secret=SECRET, database_path=tmp_path / "g.db", static_dir=static,
                               fake_box_host="127.0.0.1", https_only=False, mail=MAIL)
    http = httpx.AsyncClient(transport=httpx.MockTransport(box_api))
    box = BoxController(settings, vast, http)
    data_settings = DataSettings(tenants_root=tenants, source_cache_root=tmp_path / "cache",
                                 control_database_url="sqlite://", max_upload_bytes=1024, languages=("en", "sr"),
                                 prepare=prepare)
    data = functools.partial(DataRoutes, data_settings, queue, http, SECRET, preparer=preparer, connect_firm=firm_db)
    mailer = Mailer(MAIL, store, transport=RecordingTransport(), executor=InlineExecutor())
    return create_app(settings, store, box, IdleMonitor(box, 1800), http, run_idle_monitor=False, mailer=mailer,
                      data=data, run_background=False)


def _upload(client, name="my deposition.mp4", content=b"video-bytes"):
    batch = client.post("/api/uploads").json()["batch"]
    response = client.put(f"/api/uploads/{batch}/{name}", content=content)
    return batch, response


class TestUploads:
    def test_an_upload_is_written_here_and_queued_without_the_box(self, app, tenants, queue, vast, box_api):
        client = signed_in(app)
        batch, response = _upload(client)

        assert response.status_code == 201, response.text
        assert response.json() == {"name": "my deposition.mp4", "bytes": 11}
        user = client.get("/auth/me").json()["user"]["id"]
        folder = tenants / "legal-demo" / "uploads" / user / batch
        assert (folder / "my deposition.mp4").read_bytes() == b"video-bytes"

        job = client.post("/api/ingest", json={"batch": batch, "mode": "video", "language": "sr"}).json()

        assert (job["status"], job["position"], job["file_count"], job["language"]) == ("queued", 1, 1, "sr")
        assert queue.get(job["id"]).batch_dir == str(folder)
        assert box_api.asked == []  # no client byte went to the box

    def test_a_batch_is_queued_while_the_box_is_still_waking(self, app, box_api):
        box_api.ready = False
        client = signed_in(app)
        batch, response = _upload(client)
        assert response.status_code == 201

        job = client.post("/api/ingest", json={"batch": batch, "mode": "video", "language": "en"})

        assert job.status_code == 201

    def test_an_upload_over_the_limit_is_refused_and_leaves_nothing(self, app, tenants):
        client = signed_in(app)
        batch, response = _upload(client, "big.mp4", b"x" * 2048)
        assert response.status_code == 400
        assert [p for p in tenants.rglob("*") if p.is_file()] == []

    @pytest.mark.parametrize("batch, name", [("..%2F..%2Fx", "a.mp4"), ("20261004-120000-abcdef", "evil.exe")])
    def test_bad_batch_or_file_names_are_refused(self, app, batch, name):
        response = signed_in(app).put(f"/api/uploads/{batch}/{name}", content=b"x")
        assert response.status_code in (400, 404)

    def test_an_empty_batch_or_an_unserved_language_cannot_be_queued(self, app):
        client = signed_in(app)
        empty = client.post("/api/uploads").json()["batch"]
        assert client.post("/api/ingest", json={"batch": empty, "mode": "audio", "language": "en"}).status_code == 400
        batch, _ = _upload(client)
        assert client.post("/api/ingest", json={"batch": batch, "mode": "audio", "language": "sl"}).status_code == 400

    def test_signed_out_or_cross_site_uploads_are_refused(self, app):
        from web_deploy.tests.gateway.support import client_for
        assert client_for(app).post("/api/uploads").status_code == 401
        client = signed_in(app)
        client.headers["origin"] = "http://evil.test"
        assert client.post("/api/uploads").status_code == 403


class TestResumableUploads:
    MB = 1024 * 1024

    @pytest.fixture
    def app(self, tmp_path, store, vast, box_api, tenants, queue, preparer, prepare):
        """The module's app, but taking files up to 8 MB (the shared fixture's limit is 1 KB)."""
        static = tmp_path / "static"
        static.mkdir()
        (static / "index.html").write_text("<html>spa</html>")
        settings = GatewaySettings(api_secret=SECRET, database_path=tmp_path / "g.db", static_dir=static,
                                   fake_box_host="127.0.0.1", https_only=False, mail=MAIL)
        http = httpx.AsyncClient(transport=httpx.MockTransport(box_api))
        box = BoxController(settings, vast, http)
        data_settings = DataSettings(tenants_root=tenants, source_cache_root=tmp_path / "cache",
                                     control_database_url="sqlite://", max_upload_bytes=8 * self.MB,
                                     languages=("en", "sr"), prepare=prepare)
        data = functools.partial(DataRoutes, data_settings, queue, http, SECRET, preparer=preparer)
        mailer = Mailer(MAIL, store, transport=RecordingTransport(), executor=InlineExecutor())
        return create_app(settings, store, box, IdleMonitor(box, 1800), http, run_idle_monitor=False,
                          mailer=mailer, data=data, run_background=False)

    def test_a_file_sent_in_pieces_resumes_and_is_queued_whole(self, app, tenants, queue):
        client = signed_in(app)
        content = b"v" * (2 * self.MB + 5)
        batch = client.post("/api/uploads").json()["batch"]
        base = f"/api/uploads/{batch}/talk.mp4/chunks"
        sizes = {"size": len(content), "chunk": self.MB}

        assert client.post(base, json=sizes).json() == {"received": [], "complete": False}
        query = f"?size={len(content)}&chunk={self.MB}"
        assert client.put(f"{base}/2{query}", content=content[2 * self.MB:]).json() == {"complete": False}
        assert client.put(f"{base}/0{query}", content=content[:self.MB]).json() == {"complete": False}

        # the connection broke here; the client asks again and sends only what is missing
        assert client.post(base, json=sizes).json() == {"received": [0, 2], "complete": False}
        assert client.put(f"{base}/1{query}", content=content[self.MB:2 * self.MB]).json() == {"complete": True}

        job = client.post("/api/ingest", json={"batch": batch, "mode": "video", "language": "en"}).json()
        folder = Path(queue.get(job["id"]).batch_dir)
        assert (folder / "talk.mp4").read_bytes() == content
        assert job["file_count"] == 1

    def test_a_half_sent_file_is_not_queued(self, app, queue):
        client = signed_in(app)
        batch = client.post("/api/uploads").json()["batch"]
        client.post(f"/api/uploads/{batch}/talk.mp4/chunks", json={"size": 3 * self.MB, "chunk": self.MB})
        client.put(f"/api/uploads/{batch}/talk.mp4/chunks/0?size={3 * self.MB}&chunk={self.MB}",
                   content=b"x" * self.MB)
        client.put(f"/api/uploads/{batch}/notes.pdf", content=b"%PDF")

        job = client.post("/api/ingest", json={"batch": batch, "mode": "document", "language": "en"}).json()

        folder = Path(queue.get(job["id"]).batch_dir)
        assert sorted(p.name for p in folder.iterdir()) == ["notes.pdf"]

    def test_a_piece_too_long_or_a_file_too_big_is_refused(self, app):
        client = signed_in(app)
        batch = client.post("/api/uploads").json()["batch"]
        base = f"/api/uploads/{batch}/talk.mp4/chunks"
        assert client.post(base, json={"size": 9 * self.MB, "chunk": self.MB}).status_code == 400
        client.post(base, json={"size": 2 * self.MB, "chunk": self.MB})
        assert client.put(f"{base}/0?size={2 * self.MB}&chunk={self.MB}", content=b"x" * (self.MB + 1)).status_code == 400

    def test_the_uploader_records_what_its_copies_were(self, app, queue):
        client = signed_in(app)
        batch = client.post("/api/uploads").json()["batch"]
        sha = "ab" * 32

        response = client.post(f"/api/uploads/{batch}/record",
                               json={"files": {"talk.mp4": {"sha256": sha, "original_name": "talk.mov"}}})
        assert response.json() == {"recorded": 1}
        client.put(f"/api/uploads/{batch}/talk.mp4", content=b"720p")
        job = client.post("/api/ingest", json={"batch": batch, "mode": "audio", "language": "en"}).json()

        from app.features.storage.utils.upload_record import read_record
        record = read_record(Path(queue.get(job["id"]).batch_dir) / "talk.mp4")
        assert (record.sha256, record.original_name) == (sha, "talk.mov")

    def test_a_bad_record_is_refused(self, app):
        client = signed_in(app)
        batch = client.post("/api/uploads").json()["batch"]
        bad = {"files": {"talk.mp4": {"sha256": "nope", "original_name": "talk.mov"}}}
        assert client.post(f"/api/uploads/{batch}/record", json=bad).status_code == 400


class TestWaitList:
    def test_the_list_and_cancelling_are_answered_here_without_the_box(self, app, queue, box_api):
        from web_deploy.ingest_worker.queue import IngestMode
        client = signed_in(app)
        mine = queue.enqueue("legal-demo", "someone-else", IngestMode.AUDIO, "en", "/x", 2)
        queue.enqueue("other-firm", "eve", IngestMode.AUDIO, "en", "/y", 1)

        body = client.get("/api/ingest/jobs").json()
        assert [j["id"] for j in body["jobs"]] == [mine.id]
        assert body["links"] == {"enabled": True, "max_videos": 200}
        assert client.delete(f"/api/ingest/jobs/{mine.id}").status_code == 403  # not hers, not an admin

        boss = signed_in(app, email="boss@firm.com")
        assert boss.delete(f"/api/ingest/jobs/{mine.id}").json() == {"cancelled": True}
        eve = signed_in(app, firm="other-firm", email="eve@other.com", firm_password=OTHER_FIRM_PASSWORD)
        assert eve.delete(f"/api/ingest/jobs/{mine.id}").status_code == 404
        assert [path for _, path in box_api.asked] == []


    def test_a_finished_batch_can_be_removed_from_the_list_by_anyone_in_the_firm(self, app, queue, box_api):
        from web_deploy.ingest_worker.queue import IngestMode
        client = signed_in(app)
        failed = queue.enqueue("legal-demo", "someone-else", IngestMode.DOCUMENT, "sr", "/x", 1)
        waiting = queue.enqueue("legal-demo", "someone-else", IngestMode.AUDIO, "en", "/z", 1)
        queue.claim_next()
        queue.fail(failed.id, "The files could not be moved to the GPU server for ingestion.")

        assert client.post(f"/api/ingest/jobs/{waiting.id}/dismiss").status_code == 409
        assert client.post(f"/api/ingest/jobs/{failed.id}/dismiss").json() == {"dismissed": True}
        assert [j["id"] for j in client.get("/api/ingest/jobs").json()["jobs"]] == [waiting.id]
        eve = signed_in(app, firm="other-firm", email="eve@other.com", firm_password=OTHER_FIRM_PASSWORD)
        assert eve.post(f"/api/ingest/jobs/{waiting.id}/dismiss").status_code == 404
        assert box_api.asked == []


    def test_a_running_batch_is_asked_to_stop_by_whoever_queued_it_or_an_admin(self, app, queue):
        from web_deploy.ingest_worker.queue import IngestMode
        alice = signed_in(app)
        theirs = queue.enqueue("legal-demo", "someone-else", IngestMode.AUDIO, "sr", "/x", 1)
        queue.claim_next()

        assert alice.delete(f"/api/ingest/jobs/{theirs.id}").status_code == 403
        boss = signed_in(app, email="boss@firm.com")
        assert boss.delete(f"/api/ingest/jobs/{theirs.id}").json() == {"stopping": True}
        job = boss.get("/api/ingest/jobs").json()["jobs"][0]
        assert (job["status"], job["stopping"]) == ("running", True)

        queue.finish(theirs.id, {})
        assert boss.delete(f"/api/ingest/jobs/{theirs.id}").status_code == 409  # finished meanwhile


class TestLibrary:
    def _vault_file(self, tenants, firm, name):
        path = tenants / firm / "vault" / "2026" / "10" / "20" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    def test_an_admin_deletes_a_file_and_its_vault_files_here_without_the_box(self, app, tenants, firm_db, box_api):
        source = str(uuid.uuid4())
        original = self._vault_file(tenants, "legal-demo", "talk.mp4")
        audio = self._vault_file(tenants, "legal-demo", "talk.wav")
        firm_db.sources[("legal-demo", source)] = [str(original), str(audio)]
        boss = signed_in(app, email="boss@firm.com")

        assert boss.delete(f"/api/library/{source}").json() == {"deleted": True, "files_removed": 2}

        assert firm_db.deleted == [("legal-demo", source)]
        assert not original.exists() and not audio.exists()
        assert box_api.asked == []

    def test_members_cannot_and_another_firms_file_does_not_exist(self, app, tenants, firm_db):
        source = str(uuid.uuid4())
        firm_db.sources[("legal-demo", source)] = [str(self._vault_file(tenants, "legal-demo", "a.pdf"))]

        assert signed_in(app).delete(f"/api/library/{source}").status_code == 403
        eve = signed_in(app, firm="other-firm", email="eve@other.com", firm_password=OTHER_FIRM_PASSWORD)
        assert eve.delete(f"/api/library/{source}").status_code == 404
        assert signed_in(app, email="boss@firm.com").delete("/api/library/not-a-uuid").status_code == 404
        assert firm_db.deleted == []

    def test_nothing_is_deleted_while_the_firm_has_a_batch_running(self, app, queue, firm_db):
        from web_deploy.ingest_worker.queue import IngestMode
        source = str(uuid.uuid4())
        firm_db.sources[("legal-demo", source)] = []
        queue.enqueue("legal-demo", "u1", IngestMode.AUDIO, "en", "/x", 1)
        queue.claim_next()

        response = signed_in(app, email="boss@firm.com").delete(f"/api/library/{source}")

        assert response.status_code == 409 and firm_db.deleted == []

    def test_a_path_outside_the_firms_vault_is_never_removed(self, app, tenants, firm_db):
        source = str(uuid.uuid4())
        other = self._vault_file(tenants, "other-firm", "theirs.mp4")
        firm_db.sources[("legal-demo", source)] = [str(other)]

        assert signed_in(app, email="boss@firm.com").delete(f"/api/library/{source}").json()["files_removed"] == 0
        assert other.exists()


class TestPreparingVideos:
    @pytest.fixture
    def prepare(self):
        return PrepareSettings(enabled=True)

    def test_a_video_batch_waits_for_its_720p_copies_and_can_be_cancelled_meanwhile(self, app, queue):
        client = signed_in(app)
        batch, _ = _upload(client)

        job = client.post("/api/ingest", json={"batch": batch, "mode": "video", "language": "en"}).json()

        assert (job["status"], job["position"]) == ("preparing", None)
        assert queue.claim_next() is None  # the box can't take it yet
        assert client.delete(f"/api/ingest/jobs/{job['id']}").json() == {"cancelled": True}

    def test_other_batches_go_straight_to_the_line(self, app):
        client = signed_in(app)
        batch, _ = _upload(client, "minutes.pdf", b"%PDF")
        job = client.post("/api/ingest", json={"batch": batch, "mode": "document", "language": "en"}).json()
        assert job["status"] == "queued"


class TestSources:
    def _cite(self, box_api, path, **location):
        chunk_id = str(uuid.uuid4())
        box_api.locations[chunk_id] = {"physical_path": str(path), "anchor_kind": "page", "extracted_file_id": None,
                                       "start_sec": 0.0, "end_sec": 0.0, "page_start": 7, "page_end": 7, **location}
        return chunk_id

    @pytest.fixture
    def ready(self, vast):
        import asyncio
        asyncio.run(vast.start())

    def test_a_cited_file_is_served_from_the_vault_here(self, app, ready, box_api, tenants, preparer):
        pdf = tenants / "legal-demo" / "vault" / "2026" / "contract.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4")
        chunk_id = self._cite(box_api, pdf)
        client = signed_in(app)

        info = client.get(f"/api/sources/{chunk_id}?label=Doc%203").json()
        assert (info["kind"], info["variants"], info["download_name"]) == ("pdf", ["media", "original"],
                                                                           "Izvoria Doc 3.pdf")
        served = client.get(f"/api/sources/{chunk_id}/media")
        assert served.status_code == 200 and served.content == b"%PDF-1.4"
        assert preparer.prepared == [(pdf.resolve(), "media")]
        assert box_api.asked[0] == ("legal-demo", f"/api/sources/{chunk_id}/locate")

    def test_not_even_the_box_can_point_outside_the_firms_vault(self, app, ready, box_api, tenants, tmp_path):
        theirs = tenants / "other-firm" / "vault" / "x.pdf"
        theirs.parent.mkdir(parents=True)
        theirs.write_bytes(b"%PDF")
        secret = tmp_path / "secrets.pdf"
        secret.write_bytes(b"%PDF")
        client = signed_in(app)
        for path in (theirs, secret):
            assert client.get(f"/api/sources/{self._cite(box_api, path)}").status_code == 404

    def test_a_file_still_being_pushed_from_the_box_is_not_found_yet(self, app, ready, box_api, tenants):
        chunk_id = self._cite(box_api, tenants / "legal-demo" / "vault" / "2026" / "new.mp4")
        response = signed_in(app).get(f"/api/sources/{chunk_id}")
        assert response.status_code == 404

    def test_the_locate_route_itself_is_not_offered_to_browsers(self, app, ready, box_api):
        chunk_id = self._cite(box_api, "/x.pdf")
        assert signed_in(app).get(f"/api/sources/{chunk_id}/locate").status_code == 404
        assert box_api.asked == []

    def test_while_the_box_is_waking_the_user_is_told_so(self, app, box_api):
        box_api.ready = False
        response = signed_in(app).get(f"/api/sources/{uuid.uuid4()}")
        assert response.status_code == 503
        assert response.json()["kind"] == "box"


class TestLanguages:
    def test_the_served_packs_are_the_ingestion_languages(self, tmp_path):
        (tmp_path / "data" / "languages" / "legacy").mkdir(parents=True)
        for code in ("en", "sr"):
            (tmp_path / "data" / "languages" / f"{code}.toml").write_text("")
        (tmp_path / "data" / "languages" / "legacy" / "sl.toml").write_text("")
        assert served_languages(tmp_path) == ("en", "sr")


class TestIdleActivity:
    """Only someone working keeps the rented GPU awake: what an open page re-asks on its own never does."""

    def test_status_checks_and_wait_list_refreshes_are_not_activity(self, tmp_path, store, vast, box_api, tenants,
                                                                    queue, preparer):
        static = tmp_path / "static"
        static.mkdir()
        (static / "index.html").write_text("<html>spa</html>")
        settings = GatewaySettings(api_secret=SECRET, database_path=tmp_path / "g.db", static_dir=static,
                                   fake_box_host="127.0.0.1", https_only=False, mail=MAIL)
        http = httpx.AsyncClient(transport=httpx.MockTransport(box_api))
        box = BoxController(settings, vast, http)
        now = [0.0]
        idle = IdleMonitor(box, 1800, clock=lambda: now[0])
        data_settings = DataSettings(tenants_root=tenants, source_cache_root=tmp_path / "cache",
                                     control_database_url="sqlite://", max_upload_bytes=1024, languages=("en",))
        data = functools.partial(DataRoutes, data_settings, queue, http, SECRET, preparer=preparer)
        mailer = Mailer(MAIL, store, transport=RecordingTransport(), executor=InlineExecutor())
        app = create_app(settings, store, box, idle, http, run_idle_monitor=False, mailer=mailer, data=data,
                         run_background=False)
        client = signed_in(app)

        now[0] = 1000.0
        for _ in range(3):
            assert client.get("/box/status").status_code == 200
            assert client.get("/api/ingest/jobs").status_code == 200
        assert idle._last_activity < 1000.0  # still counting from sign-in: an open tab doesn't hold the box

        client.post("/api/uploads")  # someone actually doing something
        assert idle._last_activity == 1000.0
