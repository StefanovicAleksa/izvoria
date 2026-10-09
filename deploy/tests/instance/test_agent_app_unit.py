from pathlib import Path
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from deploy.instance.agent.app import OFFER_TOKEN_HEADER, SECRET_HEADER, create_app
from deploy.instance.agent.config import AgentSettings
from deploy.instance.agent.media import MediaError, MediaPreparer
from deploy.instance.agent.session import RunningApp, UnknownLanguage

SECRET = "k" * 40
AUTH = {SECRET_HEADER: SECRET}
BATCH = "20260928-153000"
TOKEN = "t" * 43
OFFER_AUTH = {OFFER_TOKEN_HEADER: TOKEN}


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "vault").mkdir()
    return AgentSettings(secret=SECRET, onyx_root=Path("/x"), upload_root=tmp_path / "up",
                         state_file=tmp_path / "app.json", log_dir=tmp_path / "logs",
                         vault_root=tmp_path / "vault", cache_root=tmp_path / "cache")


@pytest.fixture
def session():
    session = MagicMock()
    session.offer_token.return_value = TOKEN
    return session


@pytest.fixture
def sampler():
    return MagicMock(busy=False)


@pytest.fixture
def client(settings, session, sampler):
    # TestClient's requests come from the host "testclient"; it stands in for the box's loopback here.
    app = create_app(settings, session=session, sampler=sampler, postgres_check=lambda: True,
                     display_check=lambda display: display == ":1", sample_in_background=False,
                     loopback_hosts={"testclient"})
    with TestClient(app) as c:
        yield c


def vault_file(settings, name, data=b"0123456789" * 100):
    path = settings.vault / name
    path.write_bytes(data)
    return path


def offer(client, path, **fields):
    response = client.post("/offers", json={"path": str(path), **fields}, headers=OFFER_AUTH)
    assert response.status_code == 201, response.text
    return response.json()["id"]


class TestAuth:
    @pytest.mark.parametrize("headers", [{}, {SECRET_HEADER: "wrong"}])
    @pytest.mark.parametrize("method, path", [
        ("get", "/health"), ("post", "/session"), ("post", "/shutdown"), ("put", f"/upload/sr/{BATCH}/a.mp4"),
        ("get", "/offers"), ("get", "/offers/" + "a" * 22 + "/original"),
    ])
    def test_every_route_needs_the_secret(self, client, session, headers, method, path):
        assert getattr(client, method)(path, headers=headers).status_code == 401
        session.ensure.assert_not_called()
        session.stop.assert_not_called()


class TestHealth:
    def test_reports_every_readiness_signal(self, client, session, sampler):
        session.current.return_value = RunningApp(pid=7, language="en")
        sampler.busy = True
        assert client.get("/health", headers=AUTH).json() == {
            "postgres": True, "display": True, "app": {"pid": 7, "language": "en"}, "busy": True,
        }

    def test_no_app_yet(self, client, session):
        session.current.return_value = None
        assert client.get("/health", headers=AUTH).json()["app"] is None


class TestSession:
    def test_passes_the_starting_language(self, client, session, sampler):
        session.ensure.return_value = RunningApp(pid=7, language="sr")
        response = client.post("/session", json={"language": "sr"}, headers=AUTH)
        assert response.json() == {"pid": 7, "language": "sr"}
        session.ensure.assert_called_once_with("sr")

    def test_an_unknown_language_is_a_bad_request(self, client, session):
        session.ensure.side_effect = UnknownLanguage("de")
        assert client.post("/session", json={"language": "de"}, headers=AUTH).status_code == 400


class TestShutdown:
    def test_stops_the_app(self, client, session):
        assert client.post("/shutdown", headers=AUTH).json() == {"stopped": True}
        session.stop.assert_called_once_with()


class TestUpload:
    def test_streams_the_body_into_the_batch_folder(self, client, settings):
        response = client.put(f"/upload/sr/{BATCH}/Hearing 1.mp4", content=b"abc" * 1000, headers=AUTH)
        assert response.status_code == 201
        folder = (settings.upload_root / "sr" / BATCH).resolve()
        assert response.json() == {"folder": str(folder), "name": "Hearing 1.mp4", "bytes": 3000}
        assert (folder / "Hearing 1.mp4").read_bytes() == b"abc" * 1000
        assert not list(folder.glob("*.part"))

    @pytest.mark.parametrize("path", [
        f"/upload/de/{BATCH}/a.mp4",
        f"/upload/sr/notabatch/a.mp4",
        f"/upload/sr/{BATCH}/a.exe",
        f"/upload/sr/{BATCH}/..%2F..%2Fa.mp4",
    ])
    def test_rejects_bad_targets(self, client, settings, path):
        response = client.put(path, content=b"x", headers=AUTH)
        assert response.status_code in (400, 404)
        escaped = [p for p in settings.upload_root.parent.rglob("a.mp4")
                   if not p.is_relative_to(settings.upload_root)]
        assert escaped == []


class TestOfferSource:
    def test_files_a_vault_file_under_an_opaque_id(self, client, settings):
        path = vault_file(settings, "a.pdf")
        response = client.post("/offers", json={"path": str(path), "media_kind": "document", "page_start": 7},
                               headers=OFFER_AUTH)
        assert response.status_code == 201
        assert response.json()["seq"] == 1 and len(response.json()["id"]) == 22

    @pytest.mark.parametrize("headers", [{}, {OFFER_TOKEN_HEADER: "wrong"}, AUTH])
    def test_needs_the_launch_token_not_the_agent_secret(self, client, settings, headers):
        path = vault_file(settings, "a.pdf")
        assert client.post("/offers", json={"path": str(path)}, headers=headers).status_code == 401

    def test_no_running_app_means_no_offers(self, client, settings, session):
        session.offer_token.return_value = None
        path = vault_file(settings, "a.pdf")
        assert client.post("/offers", json={"path": str(path)}, headers=OFFER_AUTH).status_code == 401

    def test_only_from_inside_the_box(self, settings, session, sampler):
        app = create_app(settings, session=session, sampler=sampler, sample_in_background=False)
        with TestClient(app) as outside:
            path = vault_file(settings, "a.pdf")
            assert outside.post("/offers", json={"path": str(path)}, headers=OFFER_AUTH).status_code == 403

    @pytest.mark.parametrize("path", ["/etc/passwd", "{vault}/../app.json", "{vault}/missing.pdf"])
    def test_refuses_paths_outside_the_vault(self, client, settings, path):
        response = client.post("/offers", json={"path": path.format(vault=settings.vault)}, headers=OFFER_AUTH)
        assert response.status_code == 400
        assert client.get("/offers", headers=AUTH).json()["offers"] == []

    def test_refuses_a_body_that_is_not_json(self, client):
        assert client.post("/offers", content=b"{", headers=OFFER_AUTH).status_code == 400


class TestListOffers:
    def test_lists_offers_after_a_seq_without_their_paths(self, client, settings):
        path = vault_file(settings, "a.pdf")
        first = offer(client, path, doc_label="[Doc 1]")
        second = offer(client, path, doc_label="[Doc 2]")
        body = client.get("/offers?after=1", headers=AUTH).json()
        assert body["latest_seq"] == 2
        assert [o["id"] for o in body["offers"]] == [second]
        assert first != second
        assert str(settings.vault) not in str(body)

    def test_a_long_poll_with_nothing_new_returns_empty(self, client):
        assert client.get("/offers?after=0&wait=0.01", headers=AUTH).json() == {"latest_seq": 0, "offers": []}

    def test_bad_numbers_are_a_bad_request(self, client):
        assert client.get("/offers?after=x", headers=AUTH).status_code == 400


class TestServeOffer:
    def test_range_requests_get_partial_content(self, client, settings):
        offer_id = offer(client, vault_file(settings, "a.pdf"), media_kind="document")
        response = client.get(f"/offers/{offer_id}/media", headers={**AUTH, "Range": "bytes=0-9"})
        assert response.status_code == 206
        assert response.headers["content-range"] == "bytes 0-9/1000"
        assert response.content == b"0123456789"
        assert response.headers["content-type"] == "application/pdf"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "private, no-store"
        assert response.headers["content-disposition"] == "inline"
        # Chrome won't show a PDF under a sandbox
        assert "content-security-policy" not in response.headers

    def test_original_is_a_sandboxed_download_with_a_readable_name(self, client, settings):
        offer_id = offer(client, vault_file(settings, "a.html"), media_kind="document",
                         doc_label="[Doc 4]", location="Uvod · Član 2")
        response = client.get(f"/offers/{offer_id}/original", headers=AUTH)
        assert response.status_code == 200
        assert response.headers["content-security-policy"] == "sandbox"
        assert response.headers["content-disposition"] == (
            "attachment; filename=\"Onyx Doc 4 - Uvod - _lan 2.html\"; "
            "filename*=UTF-8''Onyx%20Doc%204%20-%20Uvod%20-%20%C4%8Clan%202.html")

    def test_html_is_never_served_inline(self, client, settings):
        offer_id = offer(client, vault_file(settings, "a.html"), media_kind="document")
        assert client.get(f"/offers/{offer_id}/media", headers=AUTH).status_code == 404

    @pytest.mark.parametrize("offer_id, variant", [("a" * 22, "media"), ("..", "media"), (None, "nonsense")])
    def test_unknown_offers_and_variants_are_not_found(self, client, settings, offer_id, variant):
        offer_id = offer_id or offer(client, vault_file(settings, "a.pdf"))
        assert client.get(f"/offers/{offer_id}/{variant}", headers=AUTH).status_code == 404

    def test_a_file_gone_since_the_offer_is_gone(self, client, settings):
        path = vault_file(settings, "a.pdf")
        offer_id = offer(client, path)
        path.unlink()
        assert client.get(f"/offers/{offer_id}/original", headers=AUTH).status_code == 410

    def test_a_failed_preparation_says_so(self, settings, session, sampler):
        preparer = MagicMock(spec=MediaPreparer)
        preparer.prepare.side_effect = MediaError("The appliance couldn't prepare this source for the browser.")
        app = create_app(settings, session=session, sampler=sampler, sample_in_background=False,
                         preparer=preparer, loopback_hosts={"testclient"})
        with TestClient(app) as c:
            offer_id = offer(c, vault_file(settings, "a.wma"), media_kind="full_source", end_sec=5)
            response = c.get(f"/offers/{offer_id}/media", headers=AUTH)
        assert response.status_code == 500
        assert response.json() == {"error": "The appliance couldn't prepare this source for the browser."}
