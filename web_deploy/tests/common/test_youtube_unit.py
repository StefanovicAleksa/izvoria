import json

import pytest

from web_deploy.common import youtube
from web_deploy.common.youtube import Link, LinkError, Video, parse_link, resolve

VID = "6ybqwZc7XJM"
LIST = "PLUl4u3cNGP63WbdFxL8giv4yhgdMGaZNA"


class TestParseLink:
    @pytest.mark.parametrize("raw, expected", [
        (f"https://www.youtube.com/watch?v={VID}", Link("video", VID)),
        (f"youtube.com/watch?v={VID}&t=42s", Link("video", VID)),
        (f"https://m.youtube.com/watch?v={VID}", Link("video", VID)),
        (f"https://youtu.be/{VID}?si=abc", Link("video", VID)),
        (f"https://www.youtube.com/shorts/{VID}", Link("video", VID)),
        (f"https://www.youtube.com/embed/{VID}", Link("video", VID)),
        (f"https://www.youtube.com/live/{VID}?feature=share", Link("video", VID)),
        (f"https://www.youtube.com/playlist?list={LIST}", Link("playlist", LIST)),
        (f"https://www.youtube.com/watch?v={VID}&list={LIST}&index=3", Link("video", VID, in_playlist=True)),
        (f"  https://www.youtube.com/watch?v={VID}  ", Link("video", VID)),
    ])
    def test_accepted_links(self, raw, expected):
        assert parse_link(raw) == expected

    @pytest.mark.parametrize("raw, message", [
        ("https://vimeo.com/123456", "Only YouTube links"),
        ("https://www.youtube.com/@somechannel", "not a video or a playlist"),
        ("https://www.youtube.com/results?search_query=rates", "not a video or a playlist"),
        ("https://www.youtube.com/watch?v=short", "no video id"),
        ("https://www.youtube.com/playlist", "no playlist id"),
        ("https://youtu.be/", "no video"),
        ("", "empty"),
        ("not a link at all", "Only YouTube links"),
    ])
    def test_refused_links_say_why(self, raw, message):
        with pytest.raises(LinkError, match=message):
            parse_link(raw)

    def test_canonical_urls(self):
        assert Link("video", VID).url == f"https://www.youtube.com/watch?v={VID}"
        assert Link("playlist", LIST).url == f"https://www.youtube.com/playlist?list={LIST}"


def _video(i, **fields):
    return {"id": f"video{i:06d}", "title": f"Lecture {i}", "duration": 60.0 * i, **fields}


class FakeYouTube:
    def __init__(self, pages):
        self.pages = pages
        self.asked = []

    def __call__(self, url):
        self.asked.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page


class TestResolve:
    def test_a_playlist_is_listed_and_unwatchable_entries_are_left_out(self):
        fake = FakeYouTube({Link("playlist", LIST).url: {"entries": [
            _video(1), {"id": "video000002", "title": "[Private video]"}, _video(3, live_status="is_live"),
            None, _video(4),
        ]}})

        result = resolve([f"https://www.youtube.com/playlist?list={LIST}"], 200, fake)

        assert [v.id for v in result.videos] == ["video000001", "video000004"]
        assert result.videos[1] == Video("video000004", "Lecture 4", 240.0)
        assert result.problems == []

    def test_each_video_is_added_once_across_links(self):
        single = Link("video", "video000001").url
        fake = FakeYouTube({single: _video(1), Link("playlist", LIST).url: {"entries": [_video(1), _video(2)]}})

        result = resolve([f"https://youtu.be/video000001", f"https://www.youtube.com/playlist?list={LIST}"], 200, fake)

        assert [v.id for v in result.videos] == ["video000001", "video000002"]

    def test_the_batch_is_capped(self):
        fake = FakeYouTube({Link("playlist", LIST).url: {"entries": [_video(i) for i in range(1, 6)]}})

        result = resolve([f"https://www.youtube.com/playlist?list={LIST}"], 3, fake)

        assert len(result.videos) == 3
        assert result.capped is True

    def test_problems_are_reported_per_link_with_youtubes_reason(self):
        private = Link("video", "video000009").url
        fake = FakeYouTube({private: RuntimeError("ERROR: [youtube] video000009: Private video. Sign in if you've been granted access")})

        result = resolve(["https://vimeo.com/1", "https://youtu.be/video000009", "  "], 200, fake)

        assert result.videos == []
        assert result.problems == [
            {"link": "https://vimeo.com/1", "reason": "Only YouTube links can be added."},
            {"link": "https://youtu.be/video000009",
             "reason": "Private video. Sign in if you've been granted access"},
        ]

    def test_a_live_stream_and_an_empty_playlist_are_explained(self):
        live = Link("video", "video000007").url
        fake = FakeYouTube({live: _video(7, live_status="is_live"), Link("playlist", LIST).url: {"entries": []}})

        result = resolve(["https://youtu.be/video000007", f"https://www.youtube.com/playlist?list={LIST}"], 200, fake)

        assert [p["reason"] for p in result.problems] == [
            "Live streams can't be added until they have ended and been processed.",
            "The playlist has no videos that can be downloaded.",
        ]

    def test_a_video_from_a_playlist_link_adds_only_the_video_and_says_so(self):
        fake = FakeYouTube({Link("video", "video000001").url: _video(1)})
        raw = f"https://www.youtube.com/watch?v=video000001&list={LIST}"

        result = resolve([raw], 200, fake)

        assert [v.id for v in result.videos] == ["video000001"]
        assert result.playlist_hints == [raw]
        assert fake.asked == [Link("video", "video000001").url]


class TestFileName:
    @pytest.mark.parametrize("title, expected", [
        ("UNIT 3 – Part I: Central Bank Preparation", "UNIT 3 – Part I Central Bank Preparation [abcdefghijk].mp4"),
        ("Ponuda i tražnja / predavanje 2", "Ponuda i tražnja predavanje 2 [abcdefghijk].mp4"),
        ('What? "Rates" <now>|', "What Rates now [abcdefghijk].mp4"),
        ("...", "YouTube video [abcdefghijk].mp4"),
    ])
    def test_the_title_names_the_file_and_the_id_finds_it_again(self, title, expected):
        assert Video("abcdefghijk", title).file_name == expected

    def test_a_long_title_is_cut_at_a_word_and_keeps_the_id(self):
        name = Video("abcdefghijk", "word " * 80).file_name

        assert name.endswith(" [abcdefghijk].mp4")
        assert len(name) <= youtube.MAX_TITLE_IN_NAME + len(" [abcdefghijk].mp4")


class TestManifest:
    def test_written_read_and_tracked_until_every_video_is_fetched_or_given_up(self, tmp_path):
        videos = [Video("video000001", "One", 60.0), Video("video000002", "Two"), Video("video000003", "Three")]
        youtube.write_manifest(tmp_path, videos)

        (tmp_path / videos[0].file_name).write_bytes(b"mp4")
        youtube.record_errors(tmp_path, {videos[1].file_name: "Video unavailable"})

        assert youtube.pending(tmp_path) == [videos[2]]
        manifest = json.loads((tmp_path / youtube.MANIFEST).read_text())
        assert manifest["errors"] == {"Two [video000002].mp4": "Video unavailable"}
        assert youtube.manifest_videos(manifest) == videos

    def test_a_batch_without_a_manifest_has_nothing_to_fetch(self, tmp_path):
        assert youtube.read_manifest(tmp_path) is None
        assert youtube.pending(tmp_path) == []
