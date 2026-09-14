# Canvas: hallucination_correction_api_v1
# Path: app/features/hallucination_correction/service/correction_api.py

import uuid
from typing import Tuple, Optional

from app.features.transcription.domain.interfaces import ITranscriptRepository
from app.features.vad.domain.interfaces import IVadRepository

from app.features.hallucination_correction.domain.interfaces import IHallucinationCorrectionAPI
from app.features.hallucination_correction.domain.entities import CorrectionConfig, CorrectionTelemetry
from app.features.hallucination_correction.utils.heuristics import CorrectionHeuristicsEngine
from app.features.hallucination_correction.utils.error_mapper import map_correction_error
from app.features.hallucination_correction.exceptions.validation import MissingPrerequisiteError


class CorrectionAPI(IHallucinationCorrectionAPI):
    """
    The central coordinator for the Zero-VRAM Hallucination Correction feature.
    Orchestrates the retrieval of transcripts, executes the mathematical gauntlet,
    and persists the cleaned results back to the database.
    """

    def __init__(
            self,
            transcript_repository: ITranscriptRepository,
            vad_repository: IVadRepository,
            config: Optional[CorrectionConfig] = None
    ):
        self._transcript_repo = transcript_repository
        self._vad_repo = vad_repository
        self._config = config or CorrectionConfig()

    def process_transcript(self, source_id: uuid.UUID) -> Tuple[any, CorrectionTelemetry]:
        """
        Executes the cleansing pipeline.
        Note: The return type signature uses 'any' for the Transcript to avoid
        circular/complex imports at the Orchestrator level, but it returns a Transcript entity.
        """
        try:
            # 1. Fetch Prerequisites
            transcript = self._transcript_repo.get_by_source_id(source_id)
            if not transcript:
                raise MissingPrerequisiteError(
                    f"Cannot correct hallucinations: Transcript for Source {source_id} missing.")

            vad_map = self._vad_repo.get_map_by_source_id(source_id)
            if not vad_map:
                raise MissingPrerequisiteError(
                    f"Cannot correct hallucinations: VAD Map for Source {source_id} missing.")

            # 2. Initialize Telemetry
            telemetry = CorrectionTelemetry(
                total_original_segments=len(transcript.segments)
            )

            # If the transcript is already empty, just return early
            if telemetry.total_original_segments == 0:
                return transcript, telemetry

            # 3. Execute the 3-Step Gauntlet
            surviving_segments = []

            for segment in transcript.segments:
                is_hallucination, reason = CorrectionHeuristicsEngine.evaluate_segment(
                    segment=segment,
                    vad_map=vad_map,
                    config=self._config
                )

                if is_hallucination:
                    if reason == "vad":
                        telemetry.removed_by_vad += 1
                    elif reason == "metadata":
                        telemetry.removed_by_metadata += 1
                    elif reason == "regex":
                        telemetry.removed_by_regex += 1
                elif CorrectionHeuristicsEngine.is_low_confidence(segment, self._config):
                    telemetry.flagged_low_confidence += 1
                    surviving_segments.append(segment.model_copy(update={"low_confidence": True}))
                else:
                    surviving_segments.append(segment)

            # 4. Mutate and Persist
            transcript.segments = surviving_segments

            # The repository update method will overwrite the existing JSONB array in PostgreSQL
            cleaned_transcript = self._transcript_repo.update(transcript)

            return cleaned_transcript, telemetry

        except Exception as e:
            raise map_correction_error(e) from e