# Path: tests/scripts/eval/language_benchmark/test_prepare.py

from pathlib import Path

from scripts.eval.language_benchmark.fetch import Utterance
from scripts.eval.language_benchmark.prepare import GAP_SECONDS, plan_stitches, reference_record


def _utterance(seconds: float, text: str = "jedna dva tri") -> Utterance:
    return Utterance(Path(f"/tmp/{seconds}.wav"), text, seconds)


class TestPlanStitches:
    def test_groups_in_order_up_to_the_file_length(self):
        plans = plan_stitches("fleurs", [_utterance(200), _utterance(200), _utterance(200)], file_minutes=7)

        assert [p.name for p in plans] == ["fleurs_01", "fleurs_02"]
        assert [len(p.utterances) for p in plans] == [2, 1]

    def test_an_utterance_longer_than_a_file_still_gets_one(self):
        plans = plan_stitches("parlaspeech", [_utterance(900)], file_minutes=10)

        assert [len(p.utterances) for p in plans] == [1]

    def test_nothing_to_stitch(self):
        assert plan_stitches("fleurs", []) == []


class TestReferenceRecord:
    def test_offsets_include_the_silence_between_utterances(self):
        plan = plan_stitches("fleurs", [_utterance(5.0, "Prva rečenica."), _utterance(4.0, "Druga, duža rečenica!")])[0]

        record = reference_record(plan)

        assert record["text"] == "Prva rečenica. Druga, duža rečenica!"
        assert [u["start_sec"] for u in record["utterances"]] == [0.0, 5.0 + GAP_SECONDS]
        assert [u["words"] for u in record["utterances"]] == [2, 3]
