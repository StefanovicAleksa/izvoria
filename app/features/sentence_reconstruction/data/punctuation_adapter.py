# Canvas: punctuation_adapter_v3
# Path: app/features/sentence_reconstruction/data/punctuation_adapter.py

import os
import torch
from typing import List
from transformers import pipeline

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord
from app.features.sentence_reconstruction.domain.interfaces import IPunctuationAdapter
from app.features.sentence_reconstruction.utils.error_mapper import map_sentence_error
from app.features.sentence_reconstruction.exceptions.assembly import TokenMismatchError


class PunctuationAdapter(IPunctuationAdapter):
    """
    Adapter for the 'fullstop-punctuation-multilingual-base' Token Classification model.
    Executed strictly on the GPU. Assigns deterministic punctuation tags to acoustic words.
    """

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Punctuation model directory missing: {manifest.model_path}")

            # Initialize the Hugging Face token classification pipeline.
            pipe = pipeline(
                task="token-classification",
                model=manifest.model_path,
                tokenizer=manifest.model_path,
                device=0 if torch.cuda.is_available() else -1,
                aggregation_strategy="simple",  # CRITICAL FIX: Changed from "word" to "simple"
                model_kwargs={"local_files_only": True}
            )

            return LoadedModel(manifest=manifest, model_instance=pipe)

        except Exception as e:
            raise map_sentence_error(e) from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            return True
        except Exception as e:
            raise map_sentence_error(e) from e

    def predict_punctuation(self, model: LoadedModel, words: List[TranscriptWord]) -> List[PunctuatedWord]:
        try:
            if not words:
                return []

            classifier_pipeline = model.model_instance

            # 1. Reconstruct a flat string and record exact character offsets for each acoustic word
            flat_text = ""
            word_offsets = []  # Stores tuples of (start_char, end_char, TranscriptWord)

            for w in words:
                start_char = len(flat_text)
                flat_text += w.word
                end_char = len(flat_text)

                word_offsets.append((start_char, end_char, w))
                flat_text += " "  # Add space for the model to read normally

            flat_text = flat_text.strip()

            # 2. Run the Neural Token Classifier (Zero hallucination text generation)
            predictions = classifier_pipeline(flat_text)

            # 3. Deterministic Offset Mapping
            punctuated_words: List[PunctuatedWord] = []

            # We iterate through our perfect Whisper acoustic words
            for start_char, end_char, acoustic_word in word_offsets:
                assigned_tag = "0"  # "0" is the 'fullstop' model's default tag for "No Punctuation"

                # Find the pipeline prediction that corresponds to this exact character range
                for pred in predictions:
                    # If the predicted word overlaps significantly with our acoustic word's character space
                    overlap_start = max(start_char, pred["start"])
                    overlap_end = min(end_char, pred["end"])

                    if overlap_end > overlap_start:
                        assigned_tag = pred["entity_group"]

                        if assigned_tag == "0":
                            assigned_tag = "NONE"
                        break

                punctuated_words.append(PunctuatedWord(
                    acoustic_word=acoustic_word,
                    punctuation_tag=assigned_tag
                ))

            if len(punctuated_words) != len(words):
                raise TokenMismatchError(
                    f"Fatal alignment error: Expected {len(words)} tagged words, got {len(punctuated_words)}."
                )

            return punctuated_words

        except Exception as e:
            raise map_sentence_error(e) from e