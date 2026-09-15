# Path: tests/features/domain_phonetic_correction/data/test_qwen_adapter.py

import pytest
from unittest.mock import patch, MagicMock
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.domain_phonetic_correction.data.qwen_adapter import QwenDomainAdapter, CONTEXT_CHARS
from app.features.domain_phonetic_correction.domain.entities import PhoneticCandidate
from app.features.domain_phonetic_correction.exceptions import PhoneticCorrectionError

SENTENCE = "What we've seen in the burn is a downtrend."


def _candidate(text: str, heard: str, replacement: str) -> PhoneticCandidate:
    start = text.index(heard)
    return PhoneticCandidate(start=start, end=start + len(heard), heard=heard, term=replacement,
                             replacement=replacement, score=0.8)


def _choices(*letters: str) -> list:
    return [{"choices": [{"message": {"content": letter}}]} for letter in letters]


class TestQwenDomainAdapter:
    @pytest.fixture
    def manifest(self) -> ModelManifest:
        return ModelManifest(
            model_type=ModelType.DOMAIN_PHONETIC_CORRECTION,
            model_path="/fake/qwen.gguf",
            required_vram_mb=2500.0
        )

    @patch("os.path.exists", return_value=True)
    @patch.object(QwenDomainAdapter, "_get_qwen_model")
    def test_load_success(self, mock_get_qwen: MagicMock, mock_exists: MagicMock, manifest: ModelManifest) -> None:
        mock_instance = MagicMock()
        mock_get_qwen.return_value = mock_instance

        loaded = QwenDomainAdapter().load(manifest)

        assert loaded.model_instance == mock_instance
        mock_get_qwen.assert_called_once_with(manifest.model_path)

    @patch.object(QwenDomainAdapter, "_get_grammar")
    def test_confirms_only_when_the_term_wins_in_both_orders(self, mock_grammar: MagicMock,
                                                             manifest: ModelManifest) -> None:
        llm = MagicMock()
        llm.create_chat_completion.side_effect = _choices("A", "B")
        loaded = LoadedModel(manifest=manifest, model_instance=llm)

        verdicts = QwenDomainAdapter().verify_candidates(
            loaded, SENTENCE, "Previous one.", [_candidate(SENTENCE, "burn", "Bund")], "prop trading macro"
        )

        assert verdicts == [True]
        mock_grammar.assert_called_once_with('root ::= "A" | "B"\n')
        first, second = llm.create_chat_completion.call_args_list
        assert first.kwargs["temperature"] == 0.0
        assert first.kwargs["max_tokens"] == 1
        assert first.kwargs["grammar"] is mock_grammar.return_value
        assert "prop trading macro" in first.kwargs["messages"][0]["content"]
        assert first.kwargs["messages"][1]["content"] == (
            "Previous sentence: Previous one.\n\n"
            "A: What we've seen in the [Bund] is a downtrend.\n"
            "B: What we've seen in the [burn] is a downtrend.\n\n"
            "Which version did the speaker say?"
        )
        assert "A: What we've seen in the [burn] is a downtrend." in second.kwargs["messages"][1]["content"]
        assert "B: What we've seen in the [Bund] is a downtrend." in second.kwargs["messages"][1]["content"]

    @pytest.mark.parametrize("letters", [("A", "A"), ("B", "B"), ("B", "A")])
    @patch.object(QwenDomainAdapter, "_get_grammar")
    def test_rejects_position_bias_or_a_heard_version_choice(self, mock_grammar: MagicMock, letters: tuple,
                                                              manifest: ModelManifest) -> None:
        llm = MagicMock()
        llm.create_chat_completion.side_effect = _choices(*letters)
        loaded = LoadedModel(manifest=manifest, model_instance=llm)

        verdicts = QwenDomainAdapter().verify_candidates(loaded, SENTENCE, "", [_candidate(SENTENCE, "burn", "Bund")], "x")

        assert verdicts == [False]

    @patch.object(QwenDomainAdapter, "_get_grammar")
    def test_one_verdict_per_candidate_in_order(self, mock_grammar: MagicMock, manifest: ModelManifest) -> None:
        text = "Mario Draggy spoke about the bond market."
        llm = MagicMock()
        llm.create_chat_completion.side_effect = _choices("A", "B", "B", "A")
        loaded = LoadedModel(manifest=manifest, model_instance=llm)
        candidates = [_candidate(text, "Mario Draggy", "Mario Draghi"), _candidate(text, "bond", "Bund")]

        verdicts = QwenDomainAdapter().verify_candidates(loaded, text, "", candidates, "x")

        assert verdicts == [True, False]
        assert llm.create_chat_completion.call_count == 4
        assert "Previous sentence: (none)" in llm.create_chat_completion.call_args_list[0].kwargs["messages"][1]["content"]

    def test_no_candidates_makes_no_model_call(self, manifest: ModelManifest) -> None:
        llm = MagicMock()
        loaded = LoadedModel(manifest=manifest, model_instance=llm)

        assert QwenDomainAdapter().verify_candidates(loaded, SENTENCE, "", [], "x") == []
        llm.create_chat_completion.assert_not_called()

    def test_long_sentence_is_windowed_around_the_candidate(self) -> None:
        text = "x" * 5000 + " the burn " + "y" * 5000
        candidate = _candidate(text, "burn", "Bund")

        window = QwenDomainAdapter._marked_window(text, candidate, "Bund")

        assert window == "…" + text[candidate.start - CONTEXT_CHARS:candidate.start] + "[Bund]" + \
               text[candidate.end:candidate.end + CONTEXT_CHARS] + "…"

    @patch.object(QwenDomainAdapter, "_get_grammar")
    def test_inference_failure_is_mapped(self, mock_grammar: MagicMock, manifest: ModelManifest) -> None:
        llm = MagicMock()
        llm.create_chat_completion.side_effect = RuntimeError("llama_cpp decode failed")
        loaded = LoadedModel(manifest=manifest, model_instance=llm)

        with pytest.raises(PhoneticCorrectionError):
            QwenDomainAdapter().verify_candidates(loaded, SENTENCE, "", [_candidate(SENTENCE, "burn", "Bund")], "x")

    @patch("app.features.domain_phonetic_correction.data.qwen_adapter.gc.collect")
    def test_unload_explicitly_closes_c_bindings(self, mock_gc: MagicMock, manifest: ModelManifest) -> None:
        mock_llm = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_llm)

        result = QwenDomainAdapter().unload(loaded_model)

        assert result is True
        mock_llm.close.assert_called_once()
        assert loaded_model.model_instance is None
        mock_gc.assert_called_once()
