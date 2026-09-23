# Path: tests/scripts/eval/language_benchmark/test_fetch.py

from scripts.eval.language_benchmark.fetch import parse_fleurs_tsv, pick_speaker_window


class TestParseFleursTsv:
    def test_keeps_one_recording_per_sentence(self):
        rows = [
            ["1817", "a.wav", "У том погледу.", "u tom pogledu", "u | t o m", "160000", "MALE"],
            ["1817", "b.wav", "У том погледу.", "u tom pogledu", "u | t o m", "150000", "FEMALE"],
            ["1818", "c.wav", "Друга реченица.", "druga rečenica", "d r u g a", "80000", "MALE"],
            ["bad row"],
        ]

        parsed = parse_fleurs_tsv(rows)

        assert [(r["id"], r["file"], r["seconds"]) for r in parsed] == [("1817", "a.wav", 10.0), ("1818", "c.wav", 5.0)]
        assert parsed[0]["text"] == "У том погледу."


class TestPickSpeakerWindow:
    def test_takes_the_most_frequent_speakers_segments_up_to_the_budget(self):
        speakers = ["Ana", "Marko", "Marko", "Ana", "Marko", "Marko"]
        seconds = [20, 30, 30, 20, 30, 30]

        assert pick_speaker_window(speakers, seconds, minutes=1) == [1, 2]

    def test_empty_shard(self):
        assert pick_speaker_window([], [], minutes=5) == []
