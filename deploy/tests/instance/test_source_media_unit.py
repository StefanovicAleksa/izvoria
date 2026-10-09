import json

import pytest

from deploy.instance.agent.media import (
    MediaError, MediaPreparer, Probe, browser_playable, decode_text, parse_probe,
)
from deploy.instance.agent.sources import OfferRequest, OfferStore, SourceKind, Variant

NICE = ["nice", "-n", "10", "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
H264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p"]
AAC = ["-c:a", "aac", "-b:a", "128k"]


def probe_json(*streams):
    return json.dumps({"streams": list(streams)}).encode()


class FakeRunner:
    """Answers ffprobe with `probe`, and 'runs' ffmpeg by writing its last argument (the output file)."""

    def __init__(self, probe=b"{}", ffmpeg_code=0):
        self.calls = []
        self.probe = probe
        self.ffmpeg_code = ffmpeg_code

    async def __call__(self, command, timeout):
        self.calls.append(list(command))
        if command[0] == "ffprobe":
            return 0, self.probe
        if self.ffmpeg_code == 0:
            with open(command[-1], "wb") as handle:
                handle.write(b"made")
        return self.ffmpeg_code, b""

    @property
    def ffmpeg_calls(self):
        return [c for c in self.calls if c[:4] == NICE[:4]]


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "onyx_data"
    root.mkdir()
    return root


async def offer_for(vault, name, data=b"x", **request):
    path = vault / name
    path.write_bytes(data)
    return await OfferStore(vault).add(OfferRequest(path=str(path), **request))


class TestParseProbe:
    def test_first_audio_and_video_codecs(self):
        raw = probe_json({"codec_type": "video", "codec_name": "hevc"}, {"codec_type": "audio", "codec_name": "aac"})
        assert parse_probe(raw) == Probe("hevc", "aac")

    def test_cover_art_is_not_video(self):
        raw = probe_json({"codec_type": "audio", "codec_name": "mp3"},
                         {"codec_type": "video", "codec_name": "mjpeg", "disposition": {"attached_pic": 1}})
        assert parse_probe(raw) == Probe(None, "mp3")

    def test_garbage_is_an_error(self):
        with pytest.raises(MediaError):
            parse_probe(b"not json")


class TestBrowserPlayable:
    @pytest.mark.parametrize("kind, suffix, probe, playable", [
        (SourceKind.VIDEO, ".mp4", Probe("h264", "aac"), True),
        (SourceKind.VIDEO, ".mp4", Probe("h264", None), True),
        (SourceKind.VIDEO, ".mp4", Probe("hevc", "aac"), False),
        (SourceKind.VIDEO, ".mp4", Probe("h264", "pcm_s16le"), False),
        (SourceKind.VIDEO, ".webm", Probe("vp9", "opus"), True),
        (SourceKind.VIDEO, ".mkv", Probe("h264", "aac"), False),
        (SourceKind.VIDEO, ".mp4", Probe(None, "aac"), False),
        (SourceKind.AUDIO, ".mp3", Probe(None, "mp3"), True),
        (SourceKind.AUDIO, ".m4a", Probe(None, "aac"), True),
        (SourceKind.AUDIO, ".m4a", Probe(None, "alac"), False),
        (SourceKind.AUDIO, ".wma", Probe(None, "wmav2"), False),
        (SourceKind.AUDIO, ".amr", Probe(None, "amr_nb"), False),
        (SourceKind.PDF, ".pdf", Probe(None, None), False),
    ])
    def test_table(self, kind, suffix, probe, playable):
        assert browser_playable(kind, suffix, probe) is playable


class TestDecodeText:
    def test_utf8_with_bom(self):
        assert decode_text("Čas\r\ndva".encode("utf-8-sig")) == "Čas\ndva"

    def test_line_breaks_are_counted_as_the_appliance_counts_them(self):
        assert decode_text(b"a\r\nb\rc\n\nd").split("\n") == ["a", "b", "c", "", "d"]

    def test_cp1250_serbian(self):
        text = "Član 7. Raskid ugovora: šuma, žaba, đak, čvor, ćup.\n" * 20
        assert decode_text(text.encode("cp1250")) == text


class TestPrepare:
    @pytest.mark.asyncio
    async def test_original_is_the_vault_file(self, vault, tmp_path):
        offer = await offer_for(vault, "a.wma")
        runner = FakeRunner()
        prepared = await MediaPreparer(tmp_path / "cache", runner).prepare(offer, Variant.ORIGINAL)
        assert (prepared.path, prepared.mime) == (offer.path, "audio/x-ms-wma")
        assert runner.calls == []

    @pytest.mark.asyncio
    async def test_a_playable_clip_is_served_as_is(self, vault, tmp_path):
        offer = await offer_for(vault, "c.mp4", media_kind="clip")
        runner = FakeRunner(probe_json({"codec_type": "video", "codec_name": "h264"},
                                       {"codec_type": "audio", "codec_name": "aac"}))
        prepared = await MediaPreparer(tmp_path / "cache", runner).prepare(offer, Variant.MEDIA)
        assert prepared.path == offer.path and prepared.mime == "video/mp4"
        assert runner.ffmpeg_calls == []

    @pytest.mark.asyncio
    async def test_an_hevc_clip_is_transcoded_once(self, vault, tmp_path):
        offer = await offer_for(vault, "c.mp4", media_kind="clip")
        runner = FakeRunner(probe_json({"codec_type": "video", "codec_name": "hevc"}))
        preparer = MediaPreparer(tmp_path / "cache", runner)
        first = await preparer.prepare(offer, Variant.MEDIA)
        second = await preparer.prepare(offer, Variant.MEDIA)

        assert first.path == second.path and first.path.parent == tmp_path / "cache"
        assert first.mime == "video/mp4" and first.path.read_bytes() == b"made"
        (call,) = runner.ffmpeg_calls
        assert call == NICE + ["-i", str(offer.path), "-map", "0:v:0?", "-map", "0:a:0?", *H264, *AAC,
                               "-movflags", "+faststart", "-f", "mp4", call[-1]]
        assert ".part" in call[-1]
        assert not list((tmp_path / "cache").glob("*.part*"))

    @pytest.mark.asyncio
    async def test_unplayable_audio_becomes_aac(self, vault, tmp_path):
        offer = await offer_for(vault, "r.wma", media_kind="full_source", end_sec=20.0)
        runner = FakeRunner(probe_json({"codec_type": "audio", "codec_name": "wmav2"}))
        prepared = await MediaPreparer(tmp_path / "cache", runner).prepare(offer, Variant.MEDIA)
        assert prepared.mime == "audio/mp4"
        (call,) = runner.ffmpeg_calls
        assert call[len(NICE):-1] == ["-i", str(offer.path), "-vn", *AAC, "-movflags", "+faststart", "-f", "mp4"]

    @pytest.mark.asyncio
    async def test_audio_excerpt_cuts_the_cited_span_with_padding(self, vault, tmp_path):
        offer = await offer_for(vault, "r.mp3", media_kind="full_source", start_sec=62.0, end_sec=80.0)
        runner = FakeRunner()
        prepared = await MediaPreparer(tmp_path / "cache", runner).prepare(offer, Variant.EXCERPT)
        assert prepared.mime == "audio/mp4"
        (call,) = runner.ffmpeg_calls
        assert call[len(NICE):-1] == ["-ss", "57.000", "-i", str(offer.path), "-t", "28.000", "-vn", *AAC,
                                      "-movflags", "+faststart", "-f", "mp4"]
        assert offer.public()["excerpt_start_sec"] == 57.0

    @pytest.mark.asyncio
    async def test_excerpt_never_starts_before_zero(self, vault, tmp_path):
        offer = await offer_for(vault, "r.mp3", media_kind="full_source", start_sec=2.0, end_sec=4.0)
        runner = FakeRunner()
        await MediaPreparer(tmp_path / "cache", runner).prepare(offer, Variant.EXCERPT)
        (call,) = runner.ffmpeg_calls
        assert call[len(NICE):len(NICE) + 2] == ["-ss", "0.000"]
        assert call[len(NICE) + 4:len(NICE) + 6] == ["-t", "9.000"]

    @pytest.mark.asyncio
    async def test_video_excerpt_is_re_encoded_for_an_exact_cut(self, vault, tmp_path):
        offer = await offer_for(vault, "v.mkv", media_kind="full_source", start_sec=10.0, end_sec=20.0)
        runner = FakeRunner()
        prepared = await MediaPreparer(tmp_path / "cache", runner).prepare(offer, Variant.EXCERPT)
        assert prepared.mime == "video/mp4"
        (call,) = runner.ffmpeg_calls
        assert H264[1] in call and "copy" not in call

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name, converted", [("i.png", False), ("i.jpeg", False), ("i.tif", True),
                                                 ("i.bmp", True)])
    async def test_images(self, vault, tmp_path, name, converted):
        offer = await offer_for(vault, name, media_kind="image")
        runner = FakeRunner()
        prepared = await MediaPreparer(tmp_path / "cache", runner).prepare(offer, Variant.MEDIA)
        assert bool(runner.ffmpeg_calls) is converted
        assert (prepared.mime == "image/png") is (converted or name.endswith(".png"))

    @pytest.mark.asyncio
    async def test_pdf_media_is_the_file(self, vault, tmp_path):
        offer = await offer_for(vault, "d.pdf", media_kind="document")
        prepared = await MediaPreparer(tmp_path / "cache", FakeRunner()).prepare(offer, Variant.MEDIA)
        assert (prepared.path, prepared.mime) == (offer.path, "application/pdf")

    @pytest.mark.asyncio
    async def test_text_is_decoded_to_utf8(self, vault, tmp_path):
        line = "Član 7. Raskid ugovora: šuma, žaba, đak, čvor, ćup."
        offer = await offer_for(vault, "t.txt", f"{line}\r\n".encode("cp1250") * 30, media_kind="document")
        prepared = await MediaPreparer(tmp_path / "cache", FakeRunner()).prepare(offer, Variant.TEXT)
        assert prepared.mime == "text/plain; charset=utf-8"
        assert prepared.path.read_text(encoding="utf-8") == f"{line}\n" * 30

    @pytest.mark.asyncio
    async def test_a_failed_ffmpeg_raises_and_leaves_nothing(self, vault, tmp_path):
        offer = await offer_for(vault, "r.mp3", media_kind="full_source", start_sec=2.0, end_sec=4.0)
        with pytest.raises(MediaError):
            await MediaPreparer(tmp_path / "cache", FakeRunner(ffmpeg_code=1)).prepare(offer, Variant.EXCERPT)
        assert list((tmp_path / "cache").iterdir()) == []

    @pytest.mark.asyncio
    async def test_a_variant_the_offer_lacks_is_refused(self, vault, tmp_path):
        offer = await offer_for(vault, "d.docx", media_kind="document")
        with pytest.raises(MediaError):
            await MediaPreparer(tmp_path / "cache", FakeRunner()).prepare(offer, Variant.MEDIA)


class TestCacheBudget:
    @pytest.mark.asyncio
    async def test_the_oldest_files_go_once_the_cache_is_over_budget(self, vault, tmp_path):
        import os

        cache = tmp_path / "cache"
        cache.mkdir()
        for age, name in enumerate(["old.mp4", "mid.mp4"]):
            (cache / name).write_bytes(b"x" * 10)
            os.utime(cache / name, (1000 + age, 1000 + age))
        offer = await offer_for(vault, "c.mp4", media_kind="clip")
        runner = FakeRunner(probe_json({"codec_type": "video", "codec_name": "hevc"}))

        prepared = await MediaPreparer(cache, runner, max_bytes=16).prepare(offer, Variant.MEDIA)

        # 4 bytes made + 10 kept fits 16; 4 + 10 + 10 didn't
        assert sorted(p.name for p in cache.iterdir()) == sorted(["mid.mp4", prepared.path.name])
