# Canvas: reconstruction_api_v5
# Path: app/features/sentence_reconstruction/service/reconstruction_api.py

import logging
from typing import List

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.transcription.domain.entities import Transcript, TranscriptWord
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.sentence_reconstruction.domain.interfaces import ISentenceRepository, IPunctuationAdapter
from app.features.sentence_reconstruction.utils.speaker_grouper import group_by_speaker
from app.features.sentence_reconstruction.utils.deterministic_assembler import assemble_sentences
from app.features.sentence_reconstruction.utils.error_mapper import map_sentence_error
from app.features.sentence_reconstruction.exceptions.assembly import MissingDiarizationError

logger = logging.getLogger(__name__)


class SentenceReconstructionAPI:
    """
    Orchestrates Phase 7: Deterministic Sentence Reconstruction.
    Uses Token Classification to inject punctuation into the audio stream without
    hallucinating or altering text, then pure Python to slice sentences cleanly.
    """

    def __init__(
            self,
            repository: ISentenceRepository,
            orchestrator: ModelManagerOrchestrator,
            adapter: IPunctuationAdapter
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._adapter = adapter

    def reconstruct_transcript(self, transcript: Transcript, release_after: bool = True) -> ReconstructedTranscript:
        """
        Executes the pure deterministic assembly flow.
        """
        try:
            # 1. Idempotency Check
            existing_record = self._repo.get_by_source_id(transcript.source_id)
            if existing_record:
                return existing_record

            # Handle edge case where the transcript is empty (e.g., no human speech)
            if not transcript.segments:
                empty_transcript = ReconstructedTranscript(source_id=transcript.source_id, sentences=[])
                return self._repo.save(empty_transcript)

            # 2. Define VRAM Requirements for the Punctuation Model
            manifest = ModelManifest(
                model_type=ModelType.SENTENCE_RECONSTRUCTION,
                model_path=settings.PUNCTUATION_MODEL_PATH,
                required_vram_mb=settings.PUNCTUATION_VRAM_REQUIREMENT_MB
            )

            loaded_model = None
            all_reconstructed_sentences: List[ReconstructedSentence] = []

            try:
                # 3. Safely acquire the GPU resource (an adapter that reuses Whisper's own punctuation needs none)
                if self._adapter.requires_model:
                    loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

                # 4. Enforce the "Speaker Wall"
                # We group the acoustic segments so we never accidentally stitch
                # words from two different speakers into a single sentence.
                speaker_groups = group_by_speaker(transcript.segments)

                # 5. Process each speaker's continuous block
                for group in speaker_groups:
                    speaker_id = group[0].speaker_id

                    if not speaker_id or speaker_id == "Unknown":
                        raise MissingDiarizationError(
                            "Cannot reconstruct sentences without speaker IDs. "
                            "Ensure the Diarization phase completed successfully."
                        )

                    # Extract the flat, chronologically ordered list of words for this speaker's block
                    block_words: List[TranscriptWord] = []
                    for seg in group:
                        block_words.extend(seg.words)

                    if not block_words:
                        continue

                    # Execute Token Classification (AI Phase)
                    # The AI strictly returns 1 tag per word. No text generation.
                    punctuated_words = self._adapter.predict_punctuation(loaded_model, block_words)

                    # Execute Deterministic Slicing (Pure Python Phase)
                    # Converts the tagged stream of words into ReconstructedSentence objects
                    assembled_sentences = assemble_sentences(speaker_id, punctuated_words)

                    all_reconstructed_sentences.extend(assembled_sentences)

                # 6. Assemble the final Aggregate Root
                final_transcript = ReconstructedTranscript(
                    source_id=transcript.source_id,
                    sentences=all_reconstructed_sentences
                )

                # 7. Persist to Postgres JSONB
                return self._repo.save(final_transcript)

            finally:
                # 8. Strict VRAM lifecycle management
                if loaded_model and release_after:
                    self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

        except Exception as e:
            # 9. Explicit Error Mapping
            raise map_sentence_error(e) from e