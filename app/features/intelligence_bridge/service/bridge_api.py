# Canvas: intelligence_bridge_api_v1
# Path: app/features/intelligence_bridge/service/bridge_api.py

import uuid

from app.features.intelligence_bridge.domain.entities import IntelligentSceneMap, EnrichedScene
from app.features.intelligence_bridge.domain.interfaces import IIntelligentSceneRepository
from app.features.intelligence_bridge.utils.timeline_merger import merge_scene_maps
from app.features.intelligence_bridge.utils.error_mapper import map_bridge_error

from app.features.scene_detector.service.scene_api import SceneDetectorAPI
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.nlp_router.service.router_api import NLPRouterAPI


class IntelligenceBridgeAPI:
    """
    The orchestrator that bridges computer vision (FFmpeg) and semantic NLP (Phi-3.5).
    """

    def __init__(
            self,
            repository: IIntelligentSceneRepository,
            scene_api: SceneDetectorAPI,
            transcription_api: TranscriptionAPI,
            nlp_router_api: NLPRouterAPI
    ):
        self._repo = repository
        self._scene_api = scene_api
        self._transcription_api = transcription_api
        self._nlp_router_api = nlp_router_api

    def process_multimodal_timeline(self, source_id: uuid.UUID, video_path: str) -> IntelligentSceneMap:
        """
        Executes the two-pass pipeline:
        1. Global visual scene detection
        2. Semantic NLP targeting for subtle visual cues
        3. Mathematical deduplication
        """
        # 1. Idempotency Check
        existing_map = self._repo.get_map_by_source_id(source_id)
        if existing_map:
            return existing_map

        try:
            # === PASS 1: GLOBAL VISUAL BRUTE-FORCE ===
            # High threshold (e.g. 0.4) to catch hard camera cuts
            global_map = self._scene_api.process_video(
                source_id=source_id,
                video_path=video_path,
                threshold=0.4
            )

            # === PREREQUISITE: GET TRANSCRIPT ===
            transcript = self._transcription_api.get_transcript(source_id)
            if not transcript or not transcript.segments:
                # Graceful Degradation: If there is no audio, just save the global map and return
                return self._fallback_to_global(source_id, global_map)

            # === PASS 2: SEMANTIC NLP TARGETING ===
            # Ask Phi-3.5 to find specific "Look at this" moments
            nlp_windows = self._nlp_router_api.generate_vision_windows(transcript)

            if not nlp_windows:
                # Graceful Degradation: No semantic cues found
                return self._fallback_to_global(source_id, global_map)

            # === PASS 3: TARGETED VISUAL EXTRACTION ===
            # Low threshold (0.1) explicitly locked to the NLP time windows
            semantic_map = self._scene_api.process_video(
                source_id=source_id,
                video_path=video_path,
                threshold=0.1,
                windows=nlp_windows
            )

            # === PASS 4: MASTER DEDUPLICATION ===
            intelligent_map = merge_scene_maps(
                source_id=source_id,
                global_map=global_map,
                semantic_map=semantic_map,
                phash_threshold=5
            )

            # Save and return the definitive multi-modal timeline
            return self._repo.save_map(intelligent_map)

        except Exception as e:
            raise map_bridge_error(e, context=f"Source ID {source_id}") from e

    def _fallback_to_global(self, source_id: uuid.UUID, global_map) -> IntelligentSceneMap:
        """Helper to convert a standard SceneMap into an IntelligentSceneMap when NLP is skipped."""
        scenes = [
            EnrichedScene(
                timestamp_sec=s.timestamp_sec,
                extracted_file_id=s.extracted_file_id,
                phash_value=s.phash_value,
                extraction_method="global_pass",
                semantic_context=None
            ) for s in global_map.scenes
        ]
        fallback_map = IntelligentSceneMap(source_id=source_id, scenes=scenes)
        return self._repo.save_map(fallback_map)