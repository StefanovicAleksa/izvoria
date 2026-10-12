import threading
from pathlib import Path

import pytest

from app.features.storage.utils.upload_record import UploadRecord
from web_deploy.common.video_copy import CopyFailed
from web_deploy.uploader.client import GatewayError
from web_deploy.uploader.transfer import Plan, State, collect, run, send_file


class FakeGateway:
    """The gateway's upload calls, with the pieces it holds; `fail_on` makes one piece fail once, as a drop does."""

    def __init__(self, fail_on=None):
        self.pieces = {}
        self.sizes = {}
        self.records = None
        self.queued = None
        self.fail_on = set(fail_on or [])
        self.batches = 0
        self.lock = threading.Lock()

    def new_batch(self):
        self.batches += 1
        return f"20261012-120000-00000{self.batches}"

    def status(self, batch, name, size, chunk):
        self.sizes[name] = size
        return {"received": sorted(i for (n, i) in self.pieces if n == name), "complete": False}

    def put_chunk(self, batch, name, index, size, chunk, data):
        with self.lock:
            if (name, index) in self.fail_on:
                self.fail_on.discard((name, index))
                raise GatewayError("The connection failed: reset by peer")
            self.pieces[(name, index)] = data

    def record(self, batch, files):
        self.records = files

    def enqueue(self, batch, mode, language):
        self.queued = (batch, mode, language)
        return {"file_count": len(self.sizes), "status": "queued"}

    def content(self, name):
        return b"".join(self.pieces[(name, i)] for i in sorted(i for (n, i) in self.pieces if n == name))


def fake_copy(source, dest, settings):
    """A 'copy' a tenth of the size; an MP4 already small is left alone (None), as the real rule would."""
    if source.suffix == ".mp4" and source.stat().st_size < 50:
        return None
    dest.write_bytes(source.read_bytes()[: max(1, source.stat().st_size // 10)])
    return UploadRecord("a" * 64, source.name)


def _files(tmp_path, files):
    root = tmp_path / "Depositions"
    for name, content in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(content)
    return root


def _plan(sources, **extra):
    return Plan(mode="video", language="en", sources=sources, chunk=16, parallel=3, **extra)


class TestCollect:
    def test_a_folders_files_of_the_kind_named_as_the_web_app_names_them(self, tmp_path):
        root = _files(tmp_path, {"Day 1/morning.mov": b"x", "Day 2/morning.mov": b"x", "notes.pdf": b"x",
                                 ".DS_Store": b"x"})

        sources, skipped = collect([root], "video")

        assert [s.name for s in sources] == ["Day 1 - morning.mov", "Day 2 - morning.mov"]
        assert skipped == 2

    def test_the_same_name_twice_is_numbered(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        for folder in (a, b):
            folder.mkdir()
            (folder / "talk.mp4").write_bytes(b"x")
        sources, _ = collect([a / "talk.mp4", b / "talk.mp4"], "video")
        assert [s.name for s in sources] == ["talk.mp4", "talk-2.mp4"]


class TestRun:
    def test_videos_are_shrunk_sent_in_pieces_recorded_and_queued(self, tmp_path):
        root = _files(tmp_path, {"talk.mov": bytes(range(200)), "clip.mp4": b"small"})
        sources, _ = collect([root], "video")
        gateway = FakeGateway()
        state = State(tmp_path / "state.json")
        log = []

        job = run(gateway, _plan(sources), state, tmp_path / "scratch", log=log.append, make=fake_copy)

        assert gateway.content("talk.mp4") == bytes(range(20))   # the copy, not the original
        assert gateway.content("clip.mp4") == b"small"            # already compact: sent as it is
        assert gateway.records == {"talk.mp4": {"sha256": "a" * 64, "original_name": "talk.mov"}}
        assert gateway.queued[1:] == ("video", "en") and job["status"] == "queued"
        assert list((tmp_path / "scratch").iterdir()) == []       # copies deleted once sent
        assert not (tmp_path / "state.json").exists()

    def test_a_dropped_piece_is_sent_again(self, tmp_path):
        root = _files(tmp_path, {"talk.mov": bytes(range(200))})
        sources, _ = collect([root], "video")
        gateway = FakeGateway()
        gateway.fail_on = {("talk.mp4", 0)}
        # the gateway client retries a dropped piece itself; here the fake raises once, so wrap a retry around it
        put = gateway.put_chunk

        def put_with_retry(*args):
            try:
                put(*args)
            except GatewayError:
                put(*args)
        gateway.put_chunk = put_with_retry

        run(gateway, _plan(sources), State(tmp_path / "s.json"), tmp_path / "scratch", log=lambda _: None,
            make=fake_copy)
        assert gateway.content("talk.mp4") == bytes(range(20))

    def test_running_it_again_after_an_interruption_continues_the_batch(self, tmp_path):
        root = _files(tmp_path, {"a.mov": bytes(range(100)), "b.mov": bytes(range(100, 200))})
        sources, _ = collect([root], "video")
        gateway = FakeGateway(fail_on=[("b.mp4", 0)])  # the connection is lost for good during the second file
        state_file = tmp_path / "state.json"
        copies = []

        def counting_copy(source, dest, settings):
            copies.append(source.name)
            return fake_copy(source, dest, settings)

        with pytest.raises(GatewayError):
            run(gateway, _plan(sources), State(state_file), tmp_path / "scratch", log=lambda _: None,
                make=counting_copy)
        assert gateway.queued is None and state_file.exists()

        run(gateway, _plan(sources), State(state_file), tmp_path / "scratch", log=lambda _: None, make=counting_copy)

        assert gateway.batches == 1                                  # the same batch, continued
        assert copies == ["a.mov", "b.mov", "b.mov"]                 # a.mov was not shrunk or sent again
        assert set(gateway.records) == {"a.mp4", "b.mp4"}
        assert gateway.content("b.mp4") == bytes(range(100, 200))[:10]

    def test_a_failed_copy_sends_the_original(self, tmp_path):
        root = _files(tmp_path, {"talk.mov": b"raw"})
        sources, _ = collect([root], "video")
        gateway = FakeGateway()

        def broken(source, dest, settings):
            raise CopyFailed("ffmpeg failed: Invalid data")

        run(gateway, _plan(sources), State(tmp_path / "s.json"), tmp_path / "scratch", log=lambda _: None,
            make=broken)

        assert gateway.content("talk.mov") == b"raw"
        assert gateway.records is None

    def test_other_kinds_are_never_shrunk(self, tmp_path):
        root = _files(tmp_path, {"call.wav": b"audio bytes"})
        sources, _ = collect([root], "audio")
        gateway = FakeGateway()
        run(gateway, Plan(mode="audio", language="sr", sources=sources, chunk=16), State(tmp_path / "s.json"),
            tmp_path / "scratch", log=lambda _: None, make=lambda *a: pytest.fail("audio is not shrunk"))
        assert gateway.content("call.wav") == b"audio bytes"


class TestSendFile:
    def test_only_missing_pieces_are_sent(self, tmp_path):
        path = tmp_path / "a.mp4"
        path.write_bytes(bytes(range(40)))
        gateway = FakeGateway()
        gateway.pieces[("a.mp4", 1)] = bytes(range(16, 32))
        sent = []

        send_file(gateway, "b", path, "a.mp4", 16, 4, sent.append)

        assert sent[0] == 16  # what was already there counts as sent
        assert gateway.content("a.mp4") == bytes(range(40))
