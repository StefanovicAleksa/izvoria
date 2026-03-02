# Canvas: nlp_router_api_v1
# Path: app/features/nlp_router/service/router_api.py

from typing import List

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.core.model_manager.domain.entities import ModelManifest
from app.features.transcription.domain.entities import Transcript
from app.features.scene_detector.domain.entities import TimeWindow

from app.features.nlp_router.domain.interfaces import ILLMRouterAdapter
from app.features.nlp_router.utils.chunker import chunk_transcript


class NLPRouterAPI:
    """
    Orchestrates the conversion of a full Whisper Transcript into precise
    FFmpeg TimeWindows using a local LLM.
    """

    def __init__(
            self,
            orchestrator: ModelManagerOrchestrator,
            adapter: ILLMRouterAdapter,
            manifest: ModelManifest
    ):
        self._orchestrator = orchestrator
        self._adapter = adapter
        self._manifest = manifest

    def generate_vision_windows(self, transcript: Transcript) -> List[TimeWindow]:
        """
        Chunks the text, acquires the LLM, processes the chunks, and merges the results.
        """
        chunks = chunk_transcript(transcript, max_duration_sec=180.0)
        if not chunks:
            return []

        merged_windows: List[TimeWindow] = []
        loaded_model = None

        try:
            # 1. Acquire Phi-3.5 Model into VRAM
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter)

            # 2. Process each chronological block
            for chunk in chunks:
                reasoning_result = self._adapter.extract_visual_cues(loaded_model, chunk)
                merged_windows.extend(reasoning_result.windows)

            # 3. Optimize the windows for FFmpeg
            return self._consolidate_windows(merged_windows)

        finally:
            # 4. Guarantee VRAM release even if inference crashes
            if loaded_model:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

    def _consolidate_windows(self, windows: List[TimeWindow], gap_threshold_sec: float = 5.0) -> List[TimeWindow]:
        """
        Merges overlapping or tightly adjacent time windows to optimize FFmpeg extraction.
        """
        if not windows:
            return []

        # Sort windows chronologically by start time
        sorted_windows = sorted(windows, key=lambda w: w.start_sec)
        consolidated = [sorted_windows[0]]

        for current in sorted_windows[1:]:
            last = consolidated[-1]

            # If the current window overlaps or starts within the gap threshold, merge them
            if current.start_sec <= last.end_sec + gap_threshold_sec:
                merged_end = max(last.end_sec, current.end_sec)
                consolidated[-1] = TimeWindow(start_sec=last.start_sec, end_sec=merged_end)
            else:
                consolidated.append(current)

        return consolidated