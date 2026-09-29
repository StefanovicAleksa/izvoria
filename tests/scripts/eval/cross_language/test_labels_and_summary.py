# Path: tests/scripts/eval/cross_language/test_labels_and_summary.py

import json

from scripts.eval.cross_language.labels import pair_labels
from scripts.eval.cross_language.stages import parse_answer_lines
from scripts.eval.cross_language.summary import jaccard, write_summary

TRANSLATIONS = {"en": {"answerable": [
    {"id": "a01", "relevant": ["809adb77"], "source": "lecture.wav",
     "text": {"en": "Why exponential weights?", "sr": "Zašto eksponencijalne težine?", "sl": "Zakaj eksponentne uteži?"}},
], "topic": {"en": "t", "sr": "t", "sl": "t"}}}
UNANSWERABLE = {"en": ["Who won 1998?"], "sr": ["Ko je pobedio 1998?"], "sl": ["Kdo je zmagal 1998?"]}


class TestPairLabels:
    def test_the_same_chunk_is_asked_about_in_the_question_language(self):
        labels = pair_labels(TRANSLATIONS, "en", "sr", UNANSWERABLE)

        assert labels["answerable"] == [{"id": "a01", "question": "Zašto eksponencijalne težine?",
                                         "relevant": ["809adb77"], "source": "lecture.wav"}]
        # Unanswerables are asked once per question language, in the native pair
        assert labels["unanswerable"] == []

    def test_the_native_pair_carries_the_unanswerables(self):
        labels = pair_labels(TRANSLATIONS, "en", "en", UNANSWERABLE)

        assert labels["unanswerable"] == [{"id": "u01", "question": "Who won 1998?"}]

    def test_cyrillic_copies_get_their_own_ids(self):
        labels = pair_labels(TRANSLATIONS, "en", "sr", UNANSWERABLE, cyrillic=str.upper)

        assert [(i["id"], i["question"]) for i in labels["answerable"]] == [
            ("a01", "Zašto eksponencijalne težine?"), ("a01c", "ZAŠTO EKSPONENCIJALNE TEŽINE?")]


class TestParseAnswerLines:
    def test_reads_verdict_and_answer_back(self):
        lines = ["[a01] 12s 2 cites :: Why?\n    Because of X [Doc 1].", "[u01] 3s refused :: Who?\n    I cannot."]

        assert parse_answer_lines(lines) == {
            "a01": {"verdict": "2 cites", "text": "Because of X [Doc 1]."},
            "u01": {"verdict": "refused", "text": "I cannot."},
        }


class TestSummary:
    def test_jaccard(self):
        assert jaccard(["a", "b"], ["b", "c"]) == 0.33
        assert jaccard([], []) is None

    def test_writes_every_table_with_the_native_baseline_marked(self, tmp_path):
        summary = {"questions": 2, "rerank_hit@1": 2, "rerank_hit@3": 2, "mrr": 1.0, "scrub_relevant_survives": 1}
        (tmp_path / "retrieval_q-sr_d-en.json").write_text(json.dumps({
            "retrieval": {"items": {}}, "script_split": {"latin": summary, "cyrillic": summary}}))
        (tmp_path / "retrieval_q-en_d-en.json").write_text(json.dumps({
            "retrieval": {"items": {"a01": {"rerank_rank": 1, "scrub_hit": True},
                                    "a02": {"rerank_rank": 2, "scrub_hit": False}}}, "script_split": None}))
        (tmp_path / "chat_q-sr_d-en.json").write_text(json.dumps({
            "answers": {"answerable": 2, "unanswerable": 0, "answered": 2, "cited_relevant": 1, "refused_correctly": 0},
            "answered_in_question_language": 2, "answered_count": 2}))
        run = {"sections": 3, "sections_written": 3, "evidence_chunks": ["a", "b"], "cited_chunks": ["a"],
               "evidence_from_material_language": 2, "report_language": "en"}
        (tmp_path / "report_q-en_d-en.json").write_text(json.dumps(run))
        (tmp_path / "report_q-sr_d-en.json").write_text(json.dumps({**run, "evidence_chunks": ["a"],
                                                                     "evidence_from_material_language": 1,
                                                                     "report_language": "sr"}))

        report = write_summary(tmp_path, ["en", "sr"]).read_text()

        assert "| en | **50% / 100% / 0.75 / 50%** | 100% / 100% / 1.00 / 50% |" in report
        assert "| en | sr (Cyrillic) | 100% / 100% / 1.00 / 50% |" in report
        assert "| en | – | 100% / 50% / 100% |" in report
        assert "| en | sr | 3/3 | 1 | 1 | 100% | 0.5 | 1.0 | sr |" in report
