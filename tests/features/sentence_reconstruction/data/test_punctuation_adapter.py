# Canvas: test_punctuation_adapter_v5
# Path: tests/features/sentence_reconstruction/data/test_punctuation_adapter.py

import pytest
import torch
from typing import List, Optional
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord
from app.features.sentence_reconstruction.data.punctuation_adapter import PunctuationAdapter
from app.features.sentence_reconstruction.exceptions.model import PunctuationModelLoadError

ID2LABEL = {0: "0", 1: ".", 2: ",", 3: "?", 4: "-", 5: ":"}
LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}


class FakeEncoding(dict):
    """Minimal stand-in for a fast-tokenizer BatchEncoding."""

    def __init__(self, word_ids: List[Optional[int]]):
        input_ids = torch.arange(len(word_ids)).unsqueeze(0)
        super().__init__(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0) -> List[Optional[int]]:
        return self._word_ids


def _logits(token_labels: List[str]) -> torch.Tensor:
    logits = torch.zeros(1, len(token_labels), len(ID2LABEL))
    for idx, label in enumerate(token_labels):
        logits[0, idx, LABEL2ID[label]] = 10.0
    return logits


def _words(*texts: str) -> List[TranscriptWord]:
    return [TranscriptWord(start_sec=i, end_sec=i + 0.5, word=t, probability=0.99) for i, t in enumerate(texts)]


def _loaded(manifest, tokenizer, hf_model) -> LoadedModel:
    hf_model.config.id2label = ID2LABEL
    return LoadedModel(manifest=manifest, model_instance={"tokenizer": tokenizer, "model": hf_model, "device": "cpu"})


class TestPunctuationAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.SENTENCE_RECONSTRUCTION,
            model_path="/mock/path/fullstop",
            required_vram_mb=500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(PunctuationAdapter, "_load_components")
    def test_load_success(self, mock_load_components, _, manifest):
        components = {"tokenizer": MagicMock(), "model": MagicMock(), "device": "cpu"}
        mock_load_components.return_value = components

        loaded_model = PunctuationAdapter().load(manifest)

        assert loaded_model.manifest == manifest
        assert loaded_model.model_instance == components
        mock_load_components.assert_called_once_with(manifest.model_path)

    @patch("os.path.exists", return_value=False)
    def test_load_missing_file_raises_error(self, _, manifest):
        with pytest.raises(PunctuationModelLoadError):
            PunctuationAdapter().load(manifest)

    def test_predict_punctuation_maps_zero_label_to_none(self, manifest):
        tokenizer = MagicMock(return_value=FakeEncoding([None, 0, 1, None]))
        hf_model = MagicMock()
        hf_model.return_value.logits = _logits(["0", "0", ".", "0"])

        result = PunctuationAdapter().predict_punctuation(_loaded(manifest, tokenizer, hf_model), _words(" Hello", " world"))

        tokenizer.assert_called_once_with(["Hello", "world"], is_split_into_words=True, return_tensors="pt",
                                          truncation=False)
        assert all(isinstance(r, PunctuatedWord) for r in result)
        assert [(r.acoustic_word.word, r.punctuation_tag) for r in result] == [(" Hello", "NONE"), (" world", ".")]

    def test_multi_subtoken_word_takes_label_of_last_subtoken(self, manifest):
        """The model marks punctuation on a word's final piece; the first piece is usually '0'."""
        # "the" -> 1 token, "echocardiogram" -> 3 tokens (first two "0", last ".")
        tokenizer = MagicMock(return_value=FakeEncoding([None, 0, 1, 1, 1, None]))
        hf_model = MagicMock()
        hf_model.return_value.logits = _logits(["0", "0", "0", "0", ".", "0"])

        result = PunctuationAdapter().predict_punctuation(
            _loaded(manifest, tokenizer, hf_model), _words("the", "echocardiogram")
        )

        assert [r.punctuation_tag for r in result] == ["NONE", "."]

    def test_unknown_labels_are_never_emitted(self, manifest):
        tokenizer = MagicMock(return_value=FakeEncoding([None, 0, None]))
        hf_model = MagicMock()
        hf_model.config.id2label = {0: "LABEL_9"}
        hf_model.return_value.logits = torch.zeros(1, 3, 1)
        loaded = LoadedModel(manifest=manifest, model_instance={"tokenizer": tokenizer, "model": hf_model, "device": "cpu"})

        result = PunctuationAdapter().predict_punctuation(loaded, _words("word"))

        assert result[0].punctuation_tag == "NONE"

    @patch.object(PunctuationAdapter, "OVERLAP_WORDS", 1)
    @patch.object(PunctuationAdapter, "WINDOW_WORDS", 3)
    def test_overlapping_windows_let_the_later_window_relabel_overlap_words(self, manifest):
        windows_seen = []

        def tokenize(window, **kwargs):
            windows_seen.append(list(window))
            return FakeEncoding([None] + list(range(len(window))) + [None])

        def forward(**kwargs):
            token_count = kwargs["input_ids"].shape[1]
            labels = ["0"] * token_count
            labels[-2] = "."  # last word of every window
            return MagicMock(logits=_logits(labels))

        tokenizer = MagicMock(side_effect=tokenize)
        hf_model = MagicMock(side_effect=forward)

        result = PunctuationAdapter().predict_punctuation(
            _loaded(manifest, tokenizer, hf_model), _words("w0", "w1", "w2", "w3", "w4")
        )

        assert windows_seen == [["w0", "w1", "w2"], ["w2", "w3", "w4"]]
        # w2 was "." at the end of window 1 but is re-labelled with right-hand context in window 2
        assert [r.punctuation_tag for r in result] == ["NONE", "NONE", "NONE", "NONE", "."]

    @patch.object(PunctuationAdapter, "WINDOW_WORDS", 2)
    def test_window_shrinks_instead_of_truncating_long_token_sequences(self, manifest):
        def tokenize(window, **kwargs):
            if len(window) > 1:
                encoding = FakeEncoding([None] * 600)
                return encoding
            return FakeEncoding([None, 0, None])

        tokenizer = MagicMock(side_effect=tokenize)
        hf_model = MagicMock(return_value=MagicMock(logits=_logits(["0", ",", "0"])))

        result = PunctuationAdapter().predict_punctuation(_loaded(manifest, tokenizer, hf_model), _words("a", "b"))

        assert [r.punctuation_tag for r in result] == [",", ","]
        assert hf_model.call_count == 2

    def test_empty_input(self, manifest):
        assert PunctuationAdapter().predict_punctuation(_loaded(manifest, MagicMock(), MagicMock()), []) == []
