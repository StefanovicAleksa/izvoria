# Canvas: punctuation_adapter_v5
# Path: app/features/sentence_reconstruction/data/punctuation_adapter.py

import gc
import os
from typing import Any, Dict, List

import torch

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord
from app.features.sentence_reconstruction.domain.interfaces import IPunctuationAdapter
from app.features.sentence_reconstruction.utils.error_mapper import map_sentence_error
from app.features.sentence_reconstruction.exceptions.assembly import TokenMismatchError

# Labels the fullstop model emits; "0" means "no punctuation after this word"
_PUNCTUATION_LABELS = {".", ",", "?", "-", ":"}


class PunctuationAdapter(IPunctuationAdapter):
    """
    Adapter for the 'fullstop-punctuation-multilingual-base' Token Classification model.
    Assigns deterministic punctuation tags to acoustic words, exactly one per word.

    Words are fed pre-split (`is_split_into_words=True`) in overlapping windows and each word takes
    the label of its LAST subtoken — the model puts the punctuation mark on the final piece of a
    word, as its reference implementation (deepmultilingualpunctuation) reads it. The previous
    character-offset mapping over aggregated entity groups took the first overlapping group, which
    dropped punctuation on multi-subtoken words and could leave a literal "0" tag in the text.
    """

    WINDOW_WORDS = 230
    OVERLAP_WORDS = 5
    MAX_TOKENS = 512

    @staticmethod
    def _load_components(model_path: str) -> Dict[str, Any]:
        """FACTORY METHOD & LAZY IMPORT"""
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        hf_model = AutoModelForTokenClassification.from_pretrained(model_path, local_files_only=True)
        hf_model.to(device)
        hf_model.eval()
        return {"tokenizer": tokenizer, "model": hf_model, "device": device}

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Punctuation model directory missing: {manifest.model_path}")

            return LoadedModel(manifest=manifest, model_instance=self._load_components(manifest.model_path))

        except Exception as e:
            raise map_sentence_error(e) from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            raise map_sentence_error(e) from e

    def predict_punctuation(self, model: LoadedModel, words: List[TranscriptWord]) -> List[PunctuatedWord]:
        try:
            if not words:
                return []

            components = model.model_instance
            tokenizer = components["tokenizer"]
            hf_model = components["model"]
            device = components["device"]
            id2label = hf_model.config.id2label

            texts = [w.word.strip() for w in words]
            tags = ["NONE"] * len(texts)

            start = 0
            while start < len(texts):
                window_size = self.WINDOW_WORDS
                while True:
                    window = texts[start:start + window_size]
                    encoding = tokenizer(window, is_split_into_words=True, return_tensors="pt", truncation=False)
                    if encoding["input_ids"].shape[1] <= self.MAX_TOKENS or window_size <= 1:
                        break
                    # Unusually token-heavy words: shrink the window rather than truncating labels away
                    window_size = max(1, int(window_size * 0.75))

                with torch.inference_mode():
                    inputs = {key: value.to(device) for key, value in encoding.items()}
                    logits = hf_model(**inputs).logits[0]
                predicted_ids = logits.argmax(dim=-1).tolist()

                last_label_per_word: Dict[int, int] = {}
                for token_idx, word_idx in enumerate(encoding.word_ids(0)):
                    if word_idx is not None:
                        last_label_per_word[word_idx] = predicted_ids[token_idx]

                # Later windows overwrite the overlap words: there they have right-hand context,
                # which is what decides the punctuation after a word.
                for word_idx, label_id in last_label_per_word.items():
                    tags[start + word_idx] = self._normalize_label(id2label.get(label_id, "0"))

                if start + len(window) >= len(texts):
                    break
                start += max(1, len(window) - self.OVERLAP_WORDS)

            punctuated_words = [
                PunctuatedWord(acoustic_word=word, punctuation_tag=tag) for word, tag in zip(words, tags)
            ]

            if len(punctuated_words) != len(words):
                raise TokenMismatchError(
                    f"Fatal alignment error: Expected {len(words)} tagged words, got {len(punctuated_words)}."
                )

            return punctuated_words

        except Exception as e:
            raise map_sentence_error(e) from e

    @staticmethod
    def _normalize_label(label: str) -> str:
        return label if label in _PUNCTUATION_LABELS else "NONE"
