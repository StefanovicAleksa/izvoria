# Canvas: scout_whisper_adapter_v2
# Path: app/features/domain_adaptation/data/scout_whisper_adapter.py

import os
import uuid
from typing import List
from faster_whisper import WhisperModel

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_adaptation.domain.interfaces import IScoutWhisperAdapter
from app.features.domain_adaptation.domain.entities import AudioSample, ScoutResult
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error

class ScoutWhisperAdapter(IScoutWhisperAdapter):
    """
    Adapter for Faster-Whisper Tiny, executed on the GPU using FP16 for rapid extraction.
    """
    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Scout model directory missing: {manifest.model_path}")

            model = WhisperModel(
                model_size_or_path=manifest.model_path,
                device="cuda",
                compute_type="float16",
                local_files_only=True
            )
            return LoadedModel(manifest=manifest, model_instance=model)
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Load GPU Whisper Scout") from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            return True
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Unload GPU Whisper Scout") from e

    def transcribe_samples(
        self, model: LoadedModel, samples: List[AudioSample], source_id: uuid.UUID
    ) -> ScoutResult:
        try:
            whisper_model: WhisperModel = model.model_instance
            concatenated_text = ""

            for sample in samples:
                if not os.path.exists(sample.file_path):
                    continue

                segments, info = whisper_model.transcribe(
                    audio=sample.file_path,
                    beam_size=1,
                    language="en",
                    vad_filter=False,
                    without_timestamps=True
                )
                for segment in segments:
                    concatenated_text += f"{segment.text.strip()} "

            return ScoutResult(
                source_id=source_id, samples=samples, combined_text=concatenated_text.strip()
            )
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Transcribe Audio Samples") from e