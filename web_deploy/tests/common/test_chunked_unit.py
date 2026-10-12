import asyncio

import pytest

from web_deploy.common.chunked import MIN_CHUNK_BYTES, ChunkedUpload, discard_incomplete
from web_deploy.common.uploads import BatchRejected

MB = MIN_CHUNK_BYTES
CONTENT = bytes(range(256)) * (MB * 2 // 256 + 100)  # a little over two pieces: three pieces, the last short


async def _body(*parts):
    for part in parts:
        yield part


def _piece(index):
    return CONTENT[index * MB:(index + 1) * MB]


def _upload(folder, size=len(CONTENT), chunk=MB):
    return ChunkedUpload(folder, "deposition.mp4", size, chunk, max_bytes=10 * MB)


class TestChunkedUpload:
    def test_pieces_in_any_order_make_the_whole_file(self, tmp_path):
        upload = _upload(tmp_path)
        assert upload.begin() == {"received": [], "complete": False}

        for index in (2, 0):
            assert asyncio.run(upload.write(index, _body(_piece(index)))) == {"complete": False}
        assert upload.received() == [0, 2]
        assert asyncio.run(upload.write(1, _body(_piece(1)[:10], _piece(1)[10:]))) == {"complete": True}

        assert (tmp_path / "deposition.mp4").read_bytes() == CONTENT
        assert sorted(p.name for p in tmp_path.iterdir()) == ["deposition.mp4"]

    def test_an_interrupted_upload_resumes_with_what_arrived(self, tmp_path):
        first = _upload(tmp_path)
        first.begin()
        asyncio.run(first.write(0, _body(_piece(0))))

        again = _upload(tmp_path)
        assert again.begin() == {"received": [0], "complete": False}

    def test_a_finished_file_reports_complete(self, tmp_path):
        upload = _upload(tmp_path)
        upload.begin()
        for index in range(3):
            asyncio.run(upload.write(index, _body(_piece(index))))
        assert _upload(tmp_path).begin() == {"received": [0, 1, 2], "complete": True}

    def test_other_sizes_start_the_file_over(self, tmp_path):
        upload = _upload(tmp_path)
        upload.begin()
        asyncio.run(upload.write(0, _body(_piece(0))))

        assert _upload(tmp_path, size=len(CONTENT) - 1).begin() == {"received": [], "complete": False}
        with pytest.raises(BatchRejected, match="restarted"):
            asyncio.run(upload.write(1, _body(_piece(1))))

    @pytest.mark.parametrize("index, data", [(0, b"short"), (2, b"x" * MB), (7, b"x")])
    def test_a_piece_of_the_wrong_length_or_number_is_refused(self, tmp_path, index, data):
        upload = _upload(tmp_path)
        upload.begin()
        with pytest.raises(BatchRejected):
            asyncio.run(upload.write(index, _body(data)))
        assert upload.received() == []

    @pytest.mark.parametrize("size, chunk", [(11 * MB, MB), (100, 1024), (100, 512 * MB)])
    def test_sizes_out_of_bounds_are_refused(self, tmp_path, size, chunk):
        with pytest.raises(BatchRejected):
            ChunkedUpload(tmp_path, "a.mp4", size, chunk, max_bytes=10 * MB)

    def test_an_empty_file_is_one_empty_piece(self, tmp_path):
        upload = ChunkedUpload(tmp_path, "empty.txt", 0, MB, max_bytes=MB)
        upload.begin()
        assert asyncio.run(upload.write(0, _body(b""))) == {"complete": True}
        assert (tmp_path / "empty.txt").read_bytes() == b""

    def test_names_are_sanitised(self, tmp_path):
        assert ChunkedUpload(tmp_path, "../../etc/talk.mp4", 1, MB, max_bytes=MB).target == tmp_path / "talk.mp4"

    def test_queueing_discards_half_sent_files(self, tmp_path):
        upload = _upload(tmp_path)
        upload.begin()
        asyncio.run(upload.write(0, _body(_piece(0))))
        (tmp_path / "kept.pdf").write_bytes(b"%PDF")

        discard_incomplete(tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["kept.pdf"]
