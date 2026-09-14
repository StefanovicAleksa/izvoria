# Path: tests/features/frame_extraction/utils/test_frame_selector.py

from pathlib import Path

from app.features.frame_extraction.utils.frame_selector import phash_distance, select_diverse_frames


def _frame(timestamp_sec: float, quality: float, phash: str):
    return timestamp_sec, Path(f"frame_{timestamp_sec}.jpg"), quality, phash


class TestPhashDistance:
    def test_counts_differing_bits(self):
        assert phash_distance("0000000000000000", "0000000000000000") == 0
        assert phash_distance("0000000000000000", "000000000000000f") == 4
        assert phash_distance("ffffffffffffffff", "0000000000000000") == 64


class TestSelectDiverseFrames:
    def test_returns_every_frame_under_the_cap(self):
        frames = [_frame(1.0, 150.0, "0000000000000000"), _frame(2.0, 120.0, "ffffffffffffffff")]

        assert select_diverse_frames(frames, 4) == (frames, [])

    def test_non_positive_cap_keeps_every_frame(self):
        frames = [_frame(1.0, 150.0, "0000000000000000"), _frame(2.0, 120.0, "ffffffffffffffff")]

        assert select_diverse_frames(frames, 0) == (frames, [])

    def test_keeps_visually_distinct_frames_in_chronological_order(self):
        slide = _frame(1.0, 200.0, "0000000000000000")
        same_slide_noisy = _frame(2.0, 190.0, "0000000000000001")
        chart = _frame(3.0, 150.0, "ffffffff00000000")
        close_up = _frame(4.0, 160.0, "00000000ffffffff")

        selected, dropped = select_diverse_frames([slide, same_slide_noisy, chart, close_up], 3)

        assert selected == [slide, chart, close_up]
        assert dropped == [same_slide_noisy]

    def test_equally_distant_frames_go_to_the_sharper_one(self):
        anchor = _frame(1.0, 300.0, "0000000000000000")
        soft = _frame(2.0, 100.0, "ffffffffffffffff")
        sharp = _frame(3.0, 250.0, "ffffffffffffffff")

        selected, dropped = select_diverse_frames([anchor, soft, sharp], 2)

        assert selected == [anchor, sharp]
        assert dropped == [soft]
