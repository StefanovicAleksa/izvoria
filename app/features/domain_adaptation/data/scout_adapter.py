# Canvas: scout_adapter_v4
# Path: app/features/domain_adaptation/data/scout_adapter.py

import os
import uuid
import torch
import torchaudio
from typing import List
from faster_whisper import WhisperModel

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.domain_adaptation.domain.interfaces import IScoutWhisperAdapter
from app.features.domain_adaptation.domain.entities import TimeSlice, ScoutResult
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error

class ScoutWhisperAdapter(IScoutWhisperAdapter):
    """
    Adapter for Faster-Whisper Small, executed on the GPU using FP16.
    Uses RAM-based audio chunking to prevent O(N) Disk I/O bottlenecks.
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
        self, model: LoadedModel, audio_path: str, slices: List[TimeSlice], source_id: uuid.UUID
    ) -> ScoutResult:
        try:
            whisper_model: WhisperModel = model.model_instance
            concatenated_text = ""

            # 1. Load full audio into memory ONCE
            waveform, sample_rate = torchaudio.load(audio_path)

            # Ensure mono and 16kHz for Whisper
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            waveform_1d = waveform.squeeze().numpy()

            # 2. Transcribe each calculated time slice strictly from RAM
            for ts in slices:
                start_sample = int(ts.start_sec * 16000)
                end_sample = int(ts.end_sec * 16000)
                audio_chunk = waveform_1d[start_sample:end_sample]

                # Skip micro-chunks
                if len(audio_chunk) < 1600:
                    continue

                segments, _ = whisper_model.transcribe(
                    audio=audio_chunk,
                    beam_size=1,  # Beam size 1 for maximum speed on the scout pass
                    language="en",
                    vad_filter=False,
                    without_timestamps=True
                )
                for segment in segments:
                    concatenated_text += f"{segment.text.strip()} "

            return ScoutResult(
                source_id=source_id,
                combined_text=concatenated_text.strip()
            )
        except Exception as e:
            raise map_domain_adaptation_error(e, context="Transcribe RAM Audio Slices") from e