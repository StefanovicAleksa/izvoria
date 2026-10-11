import asyncio
import os

import pytest

from deploy.instance.agent.sources import (
    MAX_OFFERS, MAX_TEXT_BYTES, OFFER_ID, OFFER_TTL_SECONDS, OfferRequest, OfferStore, SourceKind, SourceRejected,
    Variant, classify, download_name, resolve_vault_path,
)


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "onyx_data"
    (root / "2026" / "10" / "03").mkdir(parents=True)
    return root


def stored(vault, name="a.mp4", data=b"x"):
    path = vault / "2026" / "10" / "03" / name
    path.write_bytes(data)
    return path


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class TestResolveVaultPath:
    def test_accepts_a_stored_file(self, vault):
        path = stored(vault)
        assert resolve_vault_path(str(path), vault) == path.resolve()

    def test_resolves_dot_segments_that_stay_inside(self, vault):
        path = stored(vault)
        assert resolve_vault_path(str(vault / "2026" / ".." / "2026" / "10" / "03" / "a.mp4"), vault) == path

    @pytest.mark.parametrize("case", [
        "traversal", "outside", "symlink_out", "directory", "bad_suffix", "missing", "empty", "nul",
    ])
    def test_refuses_anything_else(self, vault, tmp_path, case):
        outside = tmp_path / "secrets.mp4"
        outside.write_bytes(b"x")
        raw = {
            "traversal": str(vault / ".." / "secrets.mp4"),
            "outside": str(outside),
            "directory": str(vault / "2026"),
            "bad_suffix": str(stored(vault, "run.sh")),
            "missing": str(vault / "gone.mp4"),
            "empty": "",
            "nul": str(vault / "a.mp4\x00.txt"),
        }.get(case)
        if case == "symlink_out":
            link = vault / "2026" / "10" / "03" / "link.mp4"
            os.symlink(outside, link)
            raw = str(link)
        with pytest.raises(SourceRejected):
            resolve_vault_path(raw, vault)


class TestClassify:
    @pytest.mark.parametrize("name, kind, mime", [
        ("a.MP4", SourceKind.VIDEO, "video/mp4"),
        ("a.mkv", SourceKind.VIDEO, "video/x-matroska"),
        ("a.wma", SourceKind.AUDIO, "audio/x-ms-wma"),
        ("a.m4a", SourceKind.AUDIO, "audio/mp4"),
        ("a.pdf", SourceKind.PDF, "application/pdf"),
        ("a.md", SourceKind.TEXT, "text/plain"),
        ("a.docx", SourceKind.DOCUMENT,
         "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("a.html", SourceKind.DOCUMENT, "text/html"),
        ("a.tif", SourceKind.IMAGE, "image/tiff"),
    ])
    def test_kind_and_type_come_from_the_suffix(self, vault, name, kind, mime):
        assert classify(stored(vault, name)) == (kind, mime)


class TestDownloadName:
    @pytest.mark.parametrize("label, location, suffix, expected", [
        ("[Doc 3]", "p. 7 · Article 7 Termination", ".PDF", "Izvoria Doc 3 - p. 7 - Article 7 Termination.pdf"),
        ("[Doc 12]", "00:12:31–00:14:02", ".mp4", "Izvoria Doc 12 - 00.12.31-00.14.02.mp4"),
        ("[Doc 1]", "../../etc/passwd", ".txt", "Izvoria Doc 1 - .._.._etc_passwd.txt"),
        ("", "", ".wav", "Izvoria source.wav"),
    ])
    def test_readable_and_safe(self, label, location, suffix, expected):
        assert download_name(label, location, suffix) == expected


class TestOfferRequest:
    def test_reads_every_field(self):
        request = OfferRequest.from_json({
            "path": "/v/a.mp4", "media_kind": "full_source", "anchor_kind": "time", "start_sec": 12.5,
            "end_sec": 20, "page_start": None, "page_end": None, "doc_label": "[Doc 2]",
            "location": "00:00:12–00:00:20", "excerpt": "SPEAKER_00: hi", "speakers": ["SPEAKER_00"],
        })
        assert request == OfferRequest(path="/v/a.mp4", media_kind="full_source", anchor_kind="time",
                                       start_sec=12.5, end_sec=20.0, doc_label="[Doc 2]",
                                       location="00:00:12–00:00:20", excerpt="SPEAKER_00: hi",
                                       speakers=("SPEAKER_00",))

    @pytest.mark.parametrize("body", [[], {"start_sec": "soon"}, {"page_start": "seven"}])
    def test_rejects_malformed_bodies(self, body):
        with pytest.raises(SourceRejected):
            OfferRequest.from_json(body)

    def test_clamps_display_fields(self):
        request = OfferRequest.from_json({"path": "p", "excerpt": "x" * 50_000, "start_sec": -4})
        assert len(request.excerpt) == 20_000
        assert request.start_sec == 0.0


class TestOfferStore:
    @pytest.mark.asyncio
    async def test_an_offer_hides_the_path_behind_a_random_id(self, vault):
        path = stored(vault)
        offer = await OfferStore(vault).add(OfferRequest(path=str(path), media_kind="clip", doc_label="[Doc 1]"))
        assert OFFER_ID.match(offer.id)
        public = offer.public()
        assert str(path) not in str(public) and path.name not in str(public)
        assert public["kind"] == "video" and public["clip"] is True
        assert public["variants"] == ["media", "original"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name, media_kind, end_sec, variants", [
        ("a.mp3", "full_source", 30.0, ["excerpt", "media", "original"]),
        ("a.mp4", "full_source", 30.0, ["excerpt", "media", "original"]),
        ("a.mp4", "clip", 30.0, ["media", "original"]),
        ("a.pdf", "document", 0.0, ["media", "original"]),
        ("a.txt", "document", 0.0, ["text", "original"]),
        ("a.docx", "document", 0.0, ["original"]),
        ("a.html", "document", 0.0, ["original"]),
        ("a.png", "image", 0.0, ["media", "original"]),
    ])
    async def test_variants_per_kind(self, vault, name, media_kind, end_sec, variants):
        offer = await OfferStore(vault).add(
            OfferRequest(path=str(stored(vault, name)), media_kind=media_kind, end_sec=end_sec))
        assert [v.value for v in offer.variants] == variants

    @pytest.mark.asyncio
    async def test_a_huge_text_file_is_download_only(self, vault):
        path = stored(vault, "big.txt", b"a" * (MAX_TEXT_BYTES + 1))
        offer = await OfferStore(vault).add(OfferRequest(path=str(path)))
        assert offer.variants == (Variant.ORIGINAL,)

    @pytest.mark.asyncio
    async def test_a_path_outside_the_vault_is_never_stored(self, vault, tmp_path):
        store = OfferStore(vault)
        with pytest.raises(SourceRejected):
            await store.add(OfferRequest(path=str(tmp_path)))
        assert store.latest_seq == 0 and store.after(0) == []

    @pytest.mark.asyncio
    async def test_seq_orders_offers(self, vault):
        store = OfferStore(vault)
        first = await store.add(OfferRequest(path=str(stored(vault))))
        second = await store.add(OfferRequest(path=str(stored(vault))))
        assert (first.seq, second.seq) == (1, 2)
        assert store.after(1) == [second]
        assert store.get(first.id) is first

    @pytest.mark.asyncio
    @pytest.mark.parametrize("offer_id", ["", "../../etc", "x" * 22 + "/", "short"])
    async def test_get_refuses_malformed_ids(self, vault, offer_id):
        assert OfferStore(vault).get(offer_id) is None

    @pytest.mark.asyncio
    async def test_old_offers_expire(self, vault):
        clock = Clock()
        store = OfferStore(vault, clock=clock)
        offer = await store.add(OfferRequest(path=str(stored(vault))))
        clock.now += OFFER_TTL_SECONDS + 1
        assert store.get(offer.id) is None

    @pytest.mark.asyncio
    async def test_keeps_only_the_newest(self, vault):
        store = OfferStore(vault)
        path = str(stored(vault))
        offers = [await store.add(OfferRequest(path=path)) for _ in range(MAX_OFFERS + 5)]
        assert store.get(offers[0].id) is None
        assert len(store.after(0)) == MAX_OFFERS

    @pytest.mark.asyncio
    async def test_wait_after_wakes_on_a_new_offer(self, vault):
        store = OfferStore(vault)
        waiting = asyncio.create_task(store.wait_after(0, timeout=5))
        await asyncio.sleep(0)
        offer = await store.add(OfferRequest(path=str(stored(vault))))
        assert await asyncio.wait_for(waiting, 1) == [offer]

    @pytest.mark.asyncio
    async def test_wait_after_times_out_empty(self, vault):
        assert await OfferStore(vault).wait_after(0, timeout=0.01) == []

    @pytest.mark.asyncio
    async def test_a_seq_from_before_a_restart_sees_everything_new(self, vault):
        store = OfferStore(vault)
        offer = await store.add(OfferRequest(path=str(stored(vault))))
        assert await store.wait_after(57, timeout=0.01) == [offer]
