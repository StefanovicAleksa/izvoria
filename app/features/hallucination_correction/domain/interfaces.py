# Canvas: hallucination_correction_interfaces_v1
# Path: app/features/hallucination_correction/domain/interfaces.py

import uuid
from abc import ABC, abstractmethod
from typing import Tuple

from app.features.transcription.domain.entities import Transcript
from app.features.hallucination_correction.domain.entities import CorrectionTelemetry

class IHallucinationCorrectionAPI(ABC):
    """
    Contract for the Hallucination Correction Service.
    Allows the Orchestrator to depend on this abstraction rather than the concrete implementation.
    """

    @abstractmethod
    def process_transcript(self, source_id: uuid.UUID) -> Tuple[Transcript, CorrectionTelemetry]:
        """
        Loads the existing Transcript and VAD Map for a given source,
        scrubs hallucinatory segments, persists the cleaned Transcript,
        and returns the result along with telemetry.
        """
        pass