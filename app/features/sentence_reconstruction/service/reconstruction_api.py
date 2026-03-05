# Canvas: reconstruction_api_v4
# Path: app/features/sentence_reconstruction/service/reconstruction_api.py

import logging
from typing import List

from app.core.config import settings
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType

from app.features.transcription.domain.entities import Transcript
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.sentence_reconstruction.domain.interfaces import ISentenceRepository, ISentenceAdapter
from app.features.sentence_reconstruction.utils.speaker_grouper import group_by_speaker
from app.features.sentence_reconstruction.utils.aligner import align_and_slice_sentences
from app.features.sentence_reconstruction.utils.error_mapper import map_sentence_error

logger = logging.getLogger(__name__)


class SentenceReconstructionAPI:
    """
    Orchestrates Phase 7: Grammatical Sentence Reconstruction.
    Converts fragmented acoustic ASR tokens into clean, RAG-ready sentences.
    """

    def __init__(
            self,
            repository: ISentenceRepository,
            orchestrator: ModelManagerOrchestrator,
            adapter: ISentenceAdapter
    ):
        self._repo = repository
        self._orchestrator = orchestrator
        self._adapter = adapter

    def reconstruct_transcript(self, transcript: Transcript, release_after: bool = True) -> ReconstructedTranscript:
        """
        Executes the SaT neural splitting and Proportional Virtual Slicing.
        """
        try:
            # 1. Idempotency Check
            existing_record = self._repo.get_by_source_id(transcript.source_id)
            if existing_record:
                return existing_record

            # Handle edge case where the transcript is empty
            if not transcript.segments:
                empty_transcript = ReconstructedTranscript(source_id=transcript.source_id, sentences=[])
                return self._repo.save(empty_transcript)

            # 2. Define VRAM Requirements for the SaT Model
            manifest = ModelManifest(
                model_type=ModelType.SENTENCE_RECONSTRUCTION,  # FIXED: Strictly aligned with Core Enums
                model_path=settings.SAT_MODEL_NAME,
                required_vram_mb=settings.SAT_VRAM_REQUIREMENT_MB
            )

            loaded_model = None
            all_reconstructed_sentences: List[ReconstructedSentence] = []

            try:
                # 3. Safely acquire the GPU resource
                loaded_model = self._orchestrator.acquire_model(manifest, self._adapter)

                # 4. Enforce the "Speaker Wall"
                speaker_groups = group_by_speaker(transcript.segments)

                # 5. Process each speaker's continuous block
                for group in speaker_groups:
                    # Concatenate the acoustic fragments into a single string for the model
                    combined_text = " ".join(seg.text.strip() for seg in group)

                    if not combined_text:
                        continue

                    # Execute SaT neural splitting
                    split_sentences = self._adapter.segment_text(loaded_model, combined_text)

                    # Execute Proportional Linear Interpolation (Virtual Slicing)
                    aligned_sentences = align_and_slice_sentences(
                        original_segments=group,
                        split_sentences=split_sentences
                    )

                    all_reconstructed_sentences.extend(aligned_sentences)

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