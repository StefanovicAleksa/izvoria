# Canvas: detection_api_v4
# Path: app/features/visual_cue_detection/service/detection_api.py

from app.core.model_manager.domain.entities import LoadedModel
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.visual_cue_detection.domain.interfaces import IVisualCueDetectorAdapter


class VisualCueDetectionAPI:
    """
    Service layer that orchestrates the NLP gatekeeping logic.
    Iterates through a transcript, detects visual dependencies using the SLM,
    and delegates to the Contextualizer API to persist the updated flags.
    """

    def __init__(
        self,
        detector_adapter: IVisualCueDetectorAdapter,
        contextualizer_api: ContextualizerAPI
    ):
        self._adapter = detector_adapter
        self._contextual_api = contextualizer_api

    def process_transcript(
        self,
        transcript: ContextualizedTranscript,
        model: LoadedModel
    ) -> ContextualizedTranscript:
        """
        Runs the visual cue detection on every chunk in the transcript.
        Passes the updated transcript to the Contextualizer API to safely persist.
        """
        updated_chunks = []

        for chunk in transcript.chunks:
            # Pass the chunk to the Llama 3 Adapter for analysis
            updated_chunk = self._adapter.detect_cues(model=model, chunk=chunk)
            updated_chunks.append(updated_chunk)

        # Create a new transcript entity with the evaluated chunks
        evaluated_transcript = transcript.model_copy(update={"chunks": updated_chunks})

        # Ask the owning feature's API to update the record, maintaining the architectural boundary
        self._contextual_api.update_transcript(evaluated_transcript)

        return evaluated_transcript