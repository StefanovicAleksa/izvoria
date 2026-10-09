import json
import time
import uuid
from pathlib import Path

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from web_deploy.api import mapping
from web_deploy.api.app import create_app
from web_deploy.api.config import ApiSettings
from web_deploy.api.runs import RunStore
from web_deploy.common.signing import Principal, sign
from web_deploy.tests.api.fakes import FakeHost, chunk, fake_queue, preparer

SECRET = "k" * 40
ALICE = Principal(firm="legal-demo", user_id="alice", role="member")
BOB = Principal(firm="legal-demo", user_id="bob", role="member")
EVE = Principal(firm="other-firm", user_id="eve", role="admin")


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path / "tenants")


@pytest.fixture
def queue():
    return fake_queue()


@pytest.fixture
def client(tmp_path, host, queue):
    settings = ApiSettings(secret=SECRET, tenants_root=tmp_path / "tenants", media_cache_root=tmp_path / "cache",
                           control_database_url="sqlite://", live_workers=4, report_workers=2,
                           max_upload_bytes=1024)
    app = create_app(host, settings, queue, runs=RunStore(), preparer=preparer(), start_host=False,
                     link_extractor=FAKE_YOUTUBE)
    with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
        yield test_client


YT_LIST = "PLUl4u3cNGP63WbdFxL8giv4yhgdMGaZNA"


def FAKE_YOUTUBE(url):
    """What yt-dlp's flat extraction returns for the links these tests use."""
    if url.endswith(YT_LIST):
        return {"entries": [{"id": "lecture0001", "title": "Lecture 1", "duration": 600},
                            {"id": "lecture0002", "title": "Lecture 2", "duration": 900}]}
    if url.endswith("v=lecture0003"):
        return {"id": "lecture0003", "title": "Lecture 3", "duration": 300}
    raise RuntimeError("ERROR: [youtube] x: Video unavailable")


def call(client, principal, method, path, body=None, query="", **kwargs):
    headers = sign(SECRET, method, path, query, principal)
    url = path + (f"?{query}" if query else "")
    if body is not None:
        kwargs["json"] = body
    return client.request(method, url, headers=headers, **kwargs)


def events(client, principal, run_id, timeout=5.0):
    """Every event of a run, once it has finished."""
    path = f"/api/runs/{run_id}/events"
    deadline = time.time() + timeout
    with client.stream("GET", path, headers=sign(SECRET, "GET", path, "", principal)) as response:
        assert response.status_code == 200
        out, current = [], {}
        for line in response.iter_lines():
            if line.startswith("event: "):
                current["event"] = line[7:]
            elif line.startswith("data: "):
                current["data"] = json.loads(line[6:])
            elif line.startswith("id: "):
                current["id"] = int(line[4:])
            elif line == "" and "event" in current:
                out.append(current)
                if current["event"] in ("result", "error"):
                    return out
                current = {}
            assert time.time() < deadline
    return out


class TestAuth:
    def test_unsigned_requests_are_refused(self, client):
        assert client.get("/api/languages").status_code == 401

    def test_a_signature_for_another_path_is_refused(self, client):
        headers = sign(SECRET, "GET", "/api/languages", "", ALICE)
        assert client.get("/api/library", headers=headers).status_code == 401

    def test_health_is_open_and_reports_counts_only(self, client):
        body = client.get("/health").json()
        assert body == {"ready": True, "starting_error": None, "paused": False, "inflight": 0, "reports_running": 0,
                        "queued": 0, "ingesting": 0, "backup_running": False, "backup_due": False, "busy": False}

    def test_a_due_backup_keeps_the_box_busy_so_the_gateway_does_not_stop_it(self, tmp_path, host, queue):
        from web_deploy.backup.config import BackupSettings
        from web_deploy.backup.runner import write_status
        backup = BackupSettings.from_env({"BACKUP_ENABLED": "true", "RESTIC_REPOSITORY": "sftp:u@h:repo",
                                          "RESTIC_PASSWORD": "pw", "WEB_DIR": str(tmp_path / "web")})
        settings = ApiSettings(secret=SECRET, tenants_root=tmp_path / "tenants", media_cache_root=tmp_path / "c",
                               control_database_url="sqlite://", live_workers=1, report_workers=1)
        app = create_app(host, settings, queue, runs=RunStore(), preparer=preparer(), start_host=False,
                         backup_settings=backup, clock=lambda: 2_000_000_000.0)
        with TestClient(app, client=("127.0.0.1", 50000)) as test_client:
            body = test_client.get("/health").json()
            assert (body["backup_due"], body["busy"]) == (True, True)  # never backed up yet

            write_status(backup.status_file, {"running": False, "last_success": 2_000_000_000.0 - 60})
            body = test_client.get("/health").json()
            assert (body["backup_due"], body["busy"]) == (False, False)

    def test_a_starting_appliance_answers_503(self, client, host):
        host._ready = False
        response = call(client, ALICE, "GET", "/api/languages")
        assert response.status_code == 503
        assert response.json()["kind"] == "starting"


class TestChat:
    def test_a_chat_turn_runs_in_the_chosen_language_for_its_user(self, client, host):
        response = call(client, ALICE, "POST", "/api/chat", {"prompt": "Notice period?", "language": "sr"})
        assert response.status_code == 202

        result = events(client, ALICE, response.json()["run_id"])[-1]

        assert result["event"] == "result"
        message = result["data"]["message"]
        assert message["text"] == "Thirty days [Doc 1]."
        assert message["evidence"][0]["excerpt"] == "The notice period is 30 days."
        assert "physical_path" not in json.dumps(message)
        name, args, kwargs, language = host.engines["legal-demo"].calls[-1]
        assert (name, kwargs["owner_id"], language) == ("chat", "alice", "sr")
        assert str(kwargs["session_id"]) == result["data"]["session_id"]

    def test_a_reasoned_turn_can_ask_for_clarification(self, client):
        run = call(client, ALICE, "POST", "/api/chat",
                   {"prompt": "What about it?", "language": "en", "mode": "reasoned"}).json()["run_id"]
        message = events(client, ALICE, run)[-1]["data"]["message"]
        assert message["outcome"] == "needs_clarification"
        assert message["text"] == "Which contract?"

    @pytest.mark.parametrize("body, error", [
        ({"prompt": "", "language": "en"}, "Ask a question"),
        ({"prompt": "q", "language": "sl"}, "Choose a language"),
        ({"prompt": "q", "language": "en", "mode": "deep"}, "'mode'"),
        ({"prompt": "q" * 5000, "language": "en"}, "under"),
    ])
    def test_bad_requests_are_explained(self, client, body, error):
        response = call(client, ALICE, "POST", "/api/chat", body)
        assert response.status_code == 400
        assert error in response.json()["error"]

    def test_another_user_cannot_follow_my_run(self, client):
        run = call(client, ALICE, "POST", "/api/chat", {"prompt": "q", "language": "en"}).json()["run_id"]
        path = f"/api/runs/{run}/events"
        assert client.get(path, headers=sign(SECRET, "GET", path, "", BOB)).status_code == 404

    def test_a_reconnect_resumes_after_the_last_event(self, client):
        run = call(client, ALICE, "POST", "/api/chat", {"prompt": "q", "language": "en"}).json()["run_id"]
        first = events(client, ALICE, run)
        path = f"/api/runs/{run}/events"
        headers = {**sign(SECRET, "GET", path, "", ALICE), "last-event-id": str(first[-1]["id"])}
        with client.stream("GET", path, headers=headers) as response:
            assert "event:" not in "".join(response.iter_lines())

    def test_an_engine_error_becomes_an_error_event(self, client, host):
        engine = host.engine("legal-demo")

        class EngineBusyError(Exception):
            pass

        def busy(*args, **kwargs):
            raise EngineBusyError("2 research report(s) are already running.")

        engine.chat = busy
        run = call(client, ALICE, "POST", "/api/chat", {"prompt": "q", "language": "en"}).json()["run_id"]
        last = events(client, ALICE, run)[-1]
        assert last == {"event": "error", "id": last["id"],
                        "data": {"kind": "busy", "message": "2 research report(s) are already running."}}


class TestSearchAndHistory:
    def test_search_is_recorded_for_its_user(self, client, host):
        run = call(client, ALICE, "POST", "/api/search", {"prompt": "notice", "mode": "standard"}).json()["run_id"]
        result = events(client, ALICE, run)[-1]["data"]
        assert result["mode"] == "standard"
        assert result["hits"][0]["rank"] == 1
        assert host.engines["legal-demo"].calls[-1][2]["owner_id"] == "alice"

    def test_history_reads_are_scoped_to_the_user(self, client, host):
        assert call(client, ALICE, "GET", "/api/conversations").status_code == 200
        session = uuid.uuid4()
        assert call(client, ALICE, "DELETE", f"/api/conversations/{session}").json() == {"deleted": 3}
        names = [(c[0], c[2].get("owner_id")) for c in host.engines["legal-demo"].calls]
        assert names == [("list_conversations", "alice"), ("delete_conversation", "alice")]

    def test_a_conversation_that_is_not_mine_is_not_found(self, client):
        assert call(client, ALICE, "GET", f"/api/conversations/{uuid.uuid4()}").status_code == 404

    def test_each_firm_reaches_only_its_own_engine(self, client, host):
        call(client, ALICE, "GET", "/api/conversations")
        call(client, EVE, "GET", "/api/conversations")
        assert set(host.engines) == {"legal-demo", "other-firm"}
        assert [c[2]["owner_id"] for c in host.engines["other-firm"].calls] == ["eve"]

    def test_evidence_is_hydrated_in_request_order(self, client, host):
        engine = host.engine("legal-demo")
        first, second = engine.add_chunk(chunk("one")), engine.add_chunk(chunk("two"))
        body = call(client, ALICE, "POST", "/api/evidence",
                    {"chunk_ids": [str(second.chunk_id), str(first.chunk_id), str(uuid.uuid4())]}).json()
        assert [(e["doc_number"], e["excerpt"]) for e in body["evidence"]] == [(1, "two"), (2, "one")]


class TestReports:
    def test_scoping_runs_as_the_user_in_their_language(self, client, host):
        run = call(client, ALICE, "POST", "/api/reports/plan", {"topic": "Termination", "language": "sr"}).json()
        result = events(client, ALICE, run["run_id"])[-1]["data"]
        assert result["sections"][0]["questions"] == ["What happened?"]
        assert result["estimated_minutes"] == 9
        name, _, kwargs, language = host.engines["legal-demo"].calls[-1]
        assert (name, kwargs["owner_id"], language) == ("propose_research_plan", "alice", "sr")

    def test_a_report_runs_in_the_language_it_was_scoped_in_with_the_edited_outline(self, client, host):
        engine = host.engine("legal-demo")
        stored = host.stored_report(engine, "alice", language="sr")
        body = {"mode": "deep", "plan": {"sections": [
            {"key": "s1", "title": "Background, renamed", "brief": "b", "questions": ["When?", " "]},
            {"key": "", "title": "New section", "questions": []},
        ]}}
        run = call(client, ALICE, "POST", f"/api/reports/{stored.id}/run", body).json()["run_id"]
        result = events(client, ALICE, run)[-1]["data"]

        assert result["markdown"].endswith("[Doc 1](onyx://doc/1).")
        name, _, kwargs, language = engine.calls[-1]
        assert (name, language) == ("execute_research_report", "sr")
        plan = kwargs["plan"]
        assert [s.title for s in plan.sections] == ["Background, renamed", "New section"]
        assert [(q.text, q.section_key) for q in plan.sub_questions] == [("When?", "s1")]

    def test_someone_elses_report_is_not_found(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "bob")
        assert call(client, ALICE, "GET", f"/api/reports/{stored.id}").status_code == 404
        assert call(client, ALICE, "POST", f"/api/reports/{stored.id}/cancel").status_code == 404

    def test_a_plan_awaiting_approval_is_returned_as_a_plan(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "alice")
        body = call(client, ALICE, "GET", f"/api/reports/{stored.id}").json()
        assert body["stage"] == "plan"
        assert body["plan"]["sections"][0]["key"] == "s1"

    def test_the_markdown_download_has_plain_citation_tags(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "alice", status="completed")
        response = call(client, ALICE, "GET", f"/api/reports/{stored.id}/markdown")
        assert response.text == "# Termination clauses\n\nFinding [Doc 1]."
        assert "attachment" in response.headers["content-disposition"]

    def test_the_pdf_download_is_an_attachment_named_after_the_title(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "alice", status="completed")
        response = call(client, ALICE, "GET", f"/api/reports/{stored.id}/pdf")
        assert response.status_code == 200
        assert response.content == b"%PDF-1.4 Termination clauses"
        assert response.headers["content-type"] == "application/pdf"
        assert response.headers["content-disposition"] == 'attachment; filename="Termination Rights.pdf"'

    def test_formulas_are_prepared_for_the_page_but_the_download_keeps_them_as_written(self, client, host, monkeypatch):
        stored = host.stored_report(host.engine("legal-demo"), "alice", status="completed")
        monkeypatch.setattr(host, "render_report", lambda report: "# T\n\nElasticity $E$ [Doc 1](onyx://doc/1).")
        engine = MagicMock()
        engine.get_chunks.return_value = []

        shown = mapping.report_document(host, engine, host.finished_report(stored))["markdown"]
        saved = call(client, ALICE, "GET", f"/api/reports/{stored.id}/markdown").text

        assert "$$E$$" in shown
        assert saved == "# T\n\nElasticity $E$ [Doc 1]."

    def test_a_report_from_before_titles_downloads_under_its_request(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "alice", status="completed")
        stored.title = ""
        response = call(client, ALICE, "GET", f"/api/reports/{stored.id}/markdown")
        assert response.headers["content-disposition"] == 'attachment; filename="Termination clauses.md"'

    def test_renaming_a_report_reaches_the_engine_for_its_owner(self, client, host):
        engine = host.engine("legal-demo")
        stored = host.stored_report(engine, "alice", status="completed")

        response = call(client, ALICE, "POST", f"/api/reports/{stored.id}/rename", {"title": "  Notice   Periods "})

        assert response.json() == {"report_id": str(stored.id), "title": "Notice Periods"}
        assert engine.calls[-1][0] == "rename_research_report"
        assert engine.calls[-1][2] == {"title": "Notice Periods", "owner_id": "alice"}

    def test_someone_elses_report_cannot_be_renamed(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "bob", status="completed")
        response = call(client, ALICE, "POST", f"/api/reports/{stored.id}/rename", {"title": "Mine now"})
        assert response.status_code == 404
        assert stored.title == "Termination Rights"

    def test_someone_elses_report_has_no_pdf(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "bob", status="completed")
        assert call(client, ALICE, "GET", f"/api/reports/{stored.id}/pdf").status_code == 404

    def test_an_empty_outline_is_refused(self, client, host):
        stored = host.stored_report(host.engine("legal-demo"), "alice")
        response = call(client, ALICE, "POST", f"/api/reports/{stored.id}/run", {"plan": {"sections": []}})
        assert response.status_code == 400


class TestUploadsAndIngestion:
    def test_upload_then_queue_a_batch(self, client, host, queue, tmp_path):
        batch = call(client, ALICE, "POST", "/api/uploads").json()["batch"]
        put = f"/api/uploads/{batch}/my deposition.mp4"
        response = client.put(put.replace(" ", "%20"), content=b"video-bytes",
                              headers=sign(SECRET, "PUT", put, "", ALICE))
        assert response.status_code == 201, response.text
        assert response.json() == {"name": "my deposition.mp4", "bytes": 11}

        job = call(client, ALICE, "POST", "/api/ingest", {"batch": batch, "mode": "video", "language": "en"}).json()

        assert (job["status"], job["position"], job["file_count"]) == ("queued", 1, 1)
        stored = queue.get(job["id"])
        assert stored.batch_dir == str(tmp_path / "tenants" / "legal-demo" / "uploads" / "alice" / batch)
        assert "batch_dir" not in job

    def test_an_upload_over_the_limit_is_refused_and_leaves_nothing(self, client, tmp_path):
        batch = call(client, ALICE, "POST", "/api/uploads").json()["batch"]
        put = f"/api/uploads/{batch}/big.mp4"
        response = client.put(put, content=b"x" * 2048, headers=sign(SECRET, "PUT", put, "", ALICE))
        assert response.status_code == 400
        folder = tmp_path / "tenants" / "legal-demo" / "uploads" / "alice" / batch
        assert list(folder.iterdir()) == []

    @pytest.mark.parametrize("batch, name", [("../../x", "a.mp4"), ("20261004-120000-abcdef", "evil.exe")])
    def test_bad_batch_or_file_names_are_refused(self, client, batch, name):
        put = f"/api/uploads/{batch}/{name}"
        response = client.put(put, content=b"x", headers=sign(SECRET, "PUT", put, "", ALICE))
        assert response.status_code in (400, 404)

    def test_an_empty_batch_cannot_be_queued(self, client):
        batch = call(client, ALICE, "POST", "/api/uploads").json()["batch"]
        response = call(client, ALICE, "POST", "/api/ingest", {"batch": batch, "mode": "audio", "language": "en"})
        assert response.status_code == 400

    def test_the_queue_shows_only_my_firms_jobs_and_only_owners_or_admins_cancel(self, client, queue):
        from web_deploy.ingest_worker.queue import IngestMode
        mine = queue.enqueue("legal-demo", "bob", IngestMode.AUDIO, "en", "/x", 2)
        queue.enqueue("other-firm", "eve", IngestMode.AUDIO, "en", "/y", 1)

        jobs = call(client, ALICE, "GET", "/api/ingest/jobs").json()["jobs"]
        assert [j["id"] for j in jobs] == [mine.id]
        assert call(client, ALICE, "DELETE", f"/api/ingest/jobs/{mine.id}").status_code == 403
        assert call(client, EVE, "DELETE", f"/api/ingest/jobs/{mine.id}").status_code == 404
        admin = Principal("legal-demo", "carol", "admin")
        assert call(client, admin, "DELETE", f"/api/ingest/jobs/{mine.id}").json() == {"cancelled": True}


class TestSources:
    def test_a_cited_file_inside_the_firms_vault_is_served(self, client, host, tmp_path):
        vault = tmp_path / "tenants" / "legal-demo" / "vault"
        vault.mkdir(parents=True)
        pdf = vault / "contract.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        c = host.engine("legal-demo").add_chunk(chunk(path=str(pdf), anchor="page", page_start=7, page_end=7))

        info = call(client, ALICE, "GET", f"/api/sources/{c.chunk_id}", query="label=Doc%203").json()

        assert info["kind"] == "pdf"
        assert info["variants"] == ["media", "original"]
        assert info["download_name"] == "Onyx Doc 3.pdf"

    def test_a_file_outside_the_vault_is_never_served(self, client, host, tmp_path):
        outside = tmp_path / "secrets.pdf"
        outside.write_bytes(b"%PDF")
        c = host.engine("legal-demo").add_chunk(chunk(path=str(outside)))
        assert call(client, ALICE, "GET", f"/api/sources/{c.chunk_id}").status_code == 404

    def test_another_firms_vault_is_outside_mine(self, client, host, tmp_path):
        theirs = tmp_path / "tenants" / "other-firm" / "vault"
        theirs.mkdir(parents=True)
        pdf = theirs / "x.pdf"
        pdf.write_bytes(b"%PDF")
        c = host.engine("legal-demo").add_chunk(chunk(path=str(pdf)))
        assert call(client, ALICE, "GET", f"/api/sources/{c.chunk_id}").status_code == 404


class TestGpuHandoff:
    """Dev hardware: the ingestion worker takes the whole GPU, and the API gives it up and takes it back."""

    def _service(self, client, action, host="127.0.0.1"):
        from web_deploy.common.signing import sign_service
        path = f"/internal/gpu/{action}"
        return client.post(path, headers=sign_service(SECRET, "POST", path))

    def test_pause_refuses_model_work_but_not_history_and_resume_restores_it(self, client, host):
        assert self._service(client, "pause").json() == {"paused": True}
        assert host.released == 1
        assert client.get("/health").json()["paused"] is True

        response = call(client, ALICE, "POST", "/api/chat", {"prompt": "q", "language": "en"})
        assert response.status_code == 409
        assert response.json()["kind"] == "ingesting"
        assert call(client, ALICE, "POST", "/api/search", {"prompt": "q"}).status_code == 409
        assert call(client, ALICE, "GET", "/api/conversations").status_code == 200

        assert self._service(client, "resume").json() == {"paused": False}
        assert host.warmed == 1
        assert call(client, ALICE, "POST", "/api/chat", {"prompt": "q", "language": "en"}).status_code == 202

    def test_pause_waits_for_a_question_already_being_answered(self, client, host):
        import threading
        engine = host.engine("legal-demo")
        engine.gate = threading.Event()
        run = call(client, ALICE, "POST", "/api/chat", {"prompt": "q", "language": "en"}).json()["run_id"]
        for _ in range(100):
            if engine.calls:
                break
            time.sleep(0.01)

        paused = threading.Thread(target=self._service, args=(client, "pause"))
        paused.start()
        time.sleep(0.1)
        assert host.released == 0  # still answering
        engine.gate.set()
        paused.join(5)

        assert host.released == 1
        assert events(client, ALICE, run)[-1]["event"] == "result"

    def test_only_a_signed_local_service_may_hand_off(self, client, host):
        assert client.post("/internal/gpu/pause").status_code == 401
        remote = TestClient(client.app, client=("203.0.113.9", 4000))
        from web_deploy.common.signing import sign_service
        assert remote.post("/internal/gpu/pause",
                           headers=sign_service(SECRET, "POST", "/internal/gpu/pause")).status_code == 403
        assert host.released == 0


class TestYouTubeLinks:
    PLAYLIST = f"https://www.youtube.com/playlist?list={YT_LIST}"

    def test_the_preview_lists_the_videos_and_what_could_not_be_added(self, client):
        body = call(client, ALICE, "POST", "/api/ingest/links/preview",
                    {"links": [self.PLAYLIST, "https://youtu.be/lecture0004", "https://vimeo.com/1"]}).json()

        assert [(v["id"], v["title"], v["duration"]) for v in body["videos"]] == [
            ("lecture0001", "Lecture 1", 600.0), ("lecture0002", "Lecture 2", 900.0)]
        assert [p["reason"] for p in body["problems"]] == ["Video unavailable", "Only YouTube links can be added."]

    def test_queueing_writes_the_manifest_and_a_youtube_video_job(self, client, queue, tmp_path):
        response = call(client, ALICE, "POST", "/api/ingest/links",
                        {"links": [self.PLAYLIST, "https://youtu.be/lecture0003"], "language": "en",
                         "video_ids": ["lecture0001", "lecture0003"]})

        assert response.status_code == 201, response.text
        job = response.json()
        assert (job["mode"], job["origin"], job["file_count"], job["language"]) == ("video", "youtube", 2, "en")
        stored = queue.get(job["id"])
        manifest = json.loads((Path(stored.batch_dir) / ".youtube.json").read_text())
        assert [v["id"] for v in manifest["videos"]] == ["lecture0001", "lecture0003"]
        assert Path(stored.batch_dir).is_relative_to(tmp_path / "tenants" / "legal-demo" / "uploads" / "alice")

    def test_links_are_video_only(self, client):
        response = call(client, ALICE, "POST", "/api/ingest/links", {"links": [self.PLAYLIST], "mode": "audio"})
        assert response.status_code == 400

    def test_nothing_downloadable_is_refused(self, client):
        response = call(client, ALICE, "POST", "/api/ingest/links", {"links": ["https://youtu.be/lecture0009"]})
        assert response.status_code == 400
        assert "None of these links" in response.json()["error"]

    def test_the_job_list_says_whether_links_can_be_added(self, client):
        assert call(client, ALICE, "GET", "/api/ingest/jobs").json()["links"] == {"enabled": True, "max_videos": 200}

    def test_links_can_be_turned_off(self, tmp_path, host, queue):
        settings = ApiSettings(secret=SECRET, tenants_root=tmp_path / "tenants", media_cache_root=tmp_path / "cache",
                               control_database_url="sqlite://", link_ingest_enabled=False)
        app = create_app(host, settings, queue, runs=RunStore(), preparer=preparer(), start_host=False,
                         link_extractor=FAKE_YOUTUBE)
        with TestClient(app, client=("127.0.0.1", 50000)) as off:
            response = call(off, ALICE, "POST", "/api/ingest/links/preview", {"links": [self.PLAYLIST]})
        assert response.status_code == 400
        assert "turned off" in response.json()["error"]

