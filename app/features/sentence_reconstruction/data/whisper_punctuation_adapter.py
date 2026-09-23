# Path: app/features/sentence_reconstruction/data/whisper_punctuation_adapter.py

from typing import List

from app.core.model_manager.domain.entities import LoadedModel, ModelManifest
from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord
from app.features.sentence_reconstruction.domain.interfaces import IPunctuationAdapter
from app.features.sentence_reconstruction.utils.error_mapper import map_sentence_error

# The marks the deterministic assembler understands; a word ending in anything else gets no tag
_TAGGED_MARKS = {".", ",", "?", "!", ":", ";", "-"}


class WhisperPunctuationAdapter(IPunctuationAdapter):
    """
    Tags every word with the punctuation Whisper itself wrote after it, loading no model.

    For languages the fullstop classifier was never trained on (it knows English, German, French and Italian),
    its tags would be guesses, while Whisper large-v3 punctuates those languages as it transcribes them. The
    words and their text are passed through untouched, exactly one tag per word, like the model-backed adapter.
    """

    @property
    def requires_model(self) -> bool:
        return False

    def load(self, manifest: ModelManifest) -> LoadedModel:
        return LoadedModel(manifest=manifest, model_instance=None)

    def unload(self, model: LoadedModel) -> bool:
        return True

    def predict_punctuation(self, model: LoadedModel, words: List[TranscriptWord]) -> List[PunctuatedWord]:
        try:
            return [PunctuatedWord(acoustic_word=word, punctuation_tag=self._tag(word.word)) for word in words]
        except Exception as e:
            raise map_sentence_error(e) from e

    @staticmethod
    def _tag(word: str) -> str:
        """The last mark Whisper put on the word ("rekao," -> ","; "zar ne?" -> "?"; "..." -> ".")."""
        stripped = word.strip()
        return stripped[-1] if stripped and stripped[-1] in _TAGGED_MARKS else "NONE"
