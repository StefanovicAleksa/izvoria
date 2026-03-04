# Canvas: router_api_v2
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
    Orchestrates the conversion of a full Whisper Transcript into precise FFmpeg TimeWindows using a local LLM.
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
        chunks = chunk_transcript(transcript, max_duration_sec=180.0)
        if not chunks:
            return []

        merged_windows: List[TimeWindow] = []
        loaded_model = None

        try:
            # 1. Acquire Model into VRAM
            loaded_model = self._orchestrator.acquire_model(self._manifest, self._adapter)

            # 2. Process chunks and apply mathematical buffers programmatically
            for chunk in chunks:
                reasoning_result = self._adapter.extract_visual_cues(loaded_model, chunk)

                for w in reasoning_result.windows:
                    # Apply a strict 5-second post-buffer to account for camera delays
                    buffered_window = TimeWindow(
                        start_sec=w.start_sec,
                        end_sec=w.end_sec + 5.0
                    )
                    merged_windows.append(buffered_window)

            # 3. Consolidate overlapping windows for optimal FFmpeg extraction
            return self._consolidate_windows(merged_windows)

        finally:
            # 4. Release VRAM strictly
            if loaded_model:
                self._orchestrator.release_model(loaded_model, self._adapter, flush=True)

    def _consolidate_windows(self, windows: List[TimeWindow], gap_threshold_sec: float = 5.0) -> List[TimeWindow]:
        if not windows:
            return []

        sorted_windows = sorted(windows, key=lambda w: w.start_sec)
        consolidated = [sorted_windows[0]]

        for current in sorted_windows[1:]:
            last = consolidated[-1]

            if current.start_sec <= last.end_sec + gap_threshold_sec:
                merged_end = max(last.end_sec, current.end_sec)
                consolidated[-1] = TimeWindow(start_sec=last.start_sec, end_sec=merged_end)
            else:
                consolidated.append(current)

        return consolidated