# Path: tests/scripts/eval/language_benchmark/test_manifest_and_labels.py

import pytest
from pydantic import ValidationError

from scripts.eval.language_benchmark.manifest import Lecture, load_manifest
from scripts.eval.language_benchmark.stages import UNANSWERABLE, _vtt_text, build_labels, to_cyrillic


class TestManifest:
    def test_shipped_manifest_covers_every_language_pack(self):
        manifest = load_manifest()

        assert set(manifest) == {"en", "sr", "sl"} == set(UNANSWERABLE)
        assert all(sources.lectures for sources in manifest.values())
        assert manifest["sr"].parlaspeech_shard

    def test_bad_youtube_id_is_rejected(self):
        with pytest.raises(ValidationError):
            Lecture(youtube_id="not-an-id", title="t", topic="t", max_minutes=30)


class TestLabels:
    def test_labels_use_rag_eval_format_with_cyrillic_copies(self):
        questions = [{"question": "Šta je rekao lekar?", "chunk_id": "24528228-aaaa-bbbb", "source": "lecture_x.wav"}]

        labels = build_labels("sr", "Serbian", questions, cyrillic_copies=True)

        assert labels["answerable"] == [
            {"id": "a01", "question": "Šta je rekao lekar?", "relevant": ["24528228"], "source": "lecture_x.wav"},
            {"id": "a01c", "question": "Шта је рекао лекар?", "relevant": ["24528228"], "source": "lecture_x.wav"},
        ]
        assert [u["id"] for u in labels["unanswerable"]] == ["u01", "u02", "u03", "u04", "u05"]

    def test_no_copies_for_single_script_languages(self):
        labels = build_labels("sl", "Slovenian", [{"question": "Kaj?", "chunk_id": "abcdef12", "source": "x"}], False)

        assert [a["id"] for a in labels["answerable"]] == ["a01"]


def test_to_cyrillic_handles_digraphs_and_case():
    assert to_cyrillic("Ljubiša, NJIHOV džep i EKG") == "Љубиша, ЊИХОВ џеп и ЕКГ"


def test_vtt_text_keeps_cues_inside_the_window_once(tmp_path):
    vtt = tmp_path / "x.en.vtt"
    vtt.write_text("WEBVTT\nKind: captions\n\n00:00:01.000 --> 00:00:03.000\nHello <c>there</c>\n\n"
                   "00:00:03.000 --> 00:00:05.000\nHello there\n\n00:31:00.000 --> 00:31:02.000\nToo late\n",
                   encoding="utf-8")

    assert _vtt_text(vtt, max_seconds=30 * 60) == "Hello there"
