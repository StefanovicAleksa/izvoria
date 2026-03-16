# Canvas: faster_whisper_adapter_v4
# Path: app/features/transcription/data/faster_whisper_adapter.py

import os
from typing import List, Optional
from faster_whisper import WhisperModel

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.transcription.domain.interfaces import ITranscriptionAdapter
from app.features.transcription.domain.entities import TranscriptSegment
from app.features.vad.domain.entities import VadMap
from app.features.transcription.utils.error_mapper import map_transcription_error


class FasterWhisperAdapter(ITranscriptionAdapter):
    """
    Adapter for the Faster-Whisper CTranslate2 engine, executed on the GPU.
    """

    # noinspection PyMethodMayBeStatic
    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Whisper model directory missing: {manifest.model_path}")

            model = WhisperModel(
                model_size_or_path=manifest.model_path,
                device="cuda",
                compute_type="float16",
                local_files_only=True
            )
            return LoadedModel(manifest=manifest, model_instance=model)
        except Exception as e:
            raise map_transcription_error(e) from e

    # noinspection PyMethodMayBeStatic
    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            return True
        except Exception as e:
            raise map_transcription_error(e) from e

    # noinspection PyMethodMayBeStatic
    def transcribe(
            self,
            model: LoadedModel,
            audio_path: str,
            vad_map: VadMap,
            language: str,
            hotwords: Optional[str] = None
    ) -> List[TranscriptSegment]:
        try:
            whisper_model: WhisperModel = model.model_instance
            segments_out = []

            # Using the VAD map to transcribe only confirmed speech segments
            for vad_seg in vad_map.segments:
                segments, info = whisper_model.transcribe(
                    audio=audio_path,
                    language=language,
                    vad_filter=False,  # We rely on our own NeMo VAD map
                    hotwords=hotwords,  # Domain weights injected here
                    clip_timestamps=[vad_seg.start_sec, vad_seg.end_sec]
                )

                for segment in segments:
                    segments_out.append(
                        TranscriptSegment(
                            start_sec=segment.start,
                            end_sec=segment.end,
                            text=segment.text.strip(),
                            speaker_id="Unknown"
                        )
                    )

            return segments_out

        except Exception as e:
            raise map_transcription_error(e) from e