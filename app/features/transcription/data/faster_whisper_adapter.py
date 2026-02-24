import os
import uuid
import numpy as np
import torch
import torchaudio
from faster_whisper import WhisperModel

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.vad.domain.entities import VadMap
from app.features.transcription.domain.interfaces import ITranscriptionAdapter
from app.features.transcription.domain.entities import Transcript, TranscriptSegment
from app.features.transcription.utils.error_mapper import map_transcription_error

class FasterWhisperAdapter(ITranscriptionAdapter):
    """
    Adapter for Faster-Whisper (CTranslate2).
    Optimized for production inference using INT8 quantization and manual VAD slicing.
    Strictly uses local model directories to maintain the air-gapped requirement.
    """

    REQUIRED_SAMPLE_RATE = 16000

    def load(self, manifest: ModelManifest) -> LoadedModel:
        """
        Loads the CTranslate2 Whisper model into VRAM securely.
        """
        try:
            # 1. Strict explicit check for the model directory's existence
            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Model directory missing: {manifest.model_path}")

            # 2. Local restore using INT8 Quantization for massive VRAM savings
            model = WhisperModel(
                model_size_or_path=manifest.model_path,
                device="cuda",
                compute_type="int8",  # Highly recommended for L4/P40/3060 GPUs
                local_files_only=True  # Strictly enforce offline mode
            )

            return LoadedModel(manifest=manifest, model_instance=model)

        except Exception as e:
            raise map_transcription_error(e) from e

    def unload(self, model: LoadedModel) -> bool:
        """Safely removes the model from VRAM and forces garbage collection."""
        try:
            # Faster-Whisper doesn't have an explicit .to("cpu") or .unload() method.
            # Deleting the reference allows Python/PyTorch GC to reclaim the CUDA memory.
            del model.model_instance
            return True
        except Exception as e:
            raise map_transcription_error(e) from e

    def transcribe(
            self,
            model: LoadedModel,
            audio_path: str,
            vad_map: VadMap,
            source_id: uuid.UUID
    ) -> Transcript:
        try:
            whisper_model: WhisperModel = model.model_instance

            # 1. Load and Format Audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != self.REQUIRED_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.REQUIRED_SAMPLE_RATE)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            audio_np: np.ndarray = waveform.squeeze().numpy()
            total_audio_duration = len(audio_np) / self.REQUIRED_SAMPLE_RATE

            transcript_segments = []

            # --- THE FIX: MERGE VAD SEGMENTS ---
            # Merge segments that are less than 2.0 seconds apart to preserve Whisper's context
            # and prevent the 30-second padding penalty.
            MAX_GAP_SEC = 2.0
            BUFFER_SEC = 0.2  # Add 200ms padding to boundaries to prevent cutting off phonemes

            if not vad_map.segments:
                return Transcript(source_id=source_id, segments=[])

            merged_segments = []
            current_merge = {
                "start": max(0.0, vad_map.segments[0].start_sec - BUFFER_SEC),
                "end": vad_map.segments[0].end_sec + BUFFER_SEC
            }

            for seg in vad_map.segments[1:]:
                # If the gap between the current segment and the previous one is small, merge them
                if seg.start_sec - current_merge["end"] <= MAX_GAP_SEC:
                    current_merge["end"] = seg.end_sec + BUFFER_SEC
                else:
                    # Gap is large enough to split, save the current block
                    merged_segments.append(current_merge)
                    current_merge = {
                        "start": max(0.0, seg.start_sec - BUFFER_SEC),
                        "end": seg.end_sec + BUFFER_SEC
                    }
            merged_segments.append(current_merge)

            # 2. Iterate through the MERGED speech blocks
            for block in merged_segments:
                # Ensure we don't exceed the actual audio length
                block["end"] = min(block["end"], total_audio_duration)

                start_idx = int(block["start"] * self.REQUIRED_SAMPLE_RATE)
                end_idx = int(block["end"] * self.REQUIRED_SAMPLE_RATE)

                if start_idx >= len(audio_np) or start_idx >= end_idx:
                    continue

                audio_chunk = audio_np[start_idx:end_idx]

                # 3. Transcribe the continuous, contextual chunk
                segments_generator, info = whisper_model.transcribe(
                    audio=audio_chunk,
                    beam_size=5,
                    language="en",
                    vad_filter=False,
                    without_timestamps=False,
                    condition_on_previous_text=True  # Helps maintain context within the merged chunk
                )

                # 4. Map timestamps back to global timeline
                for segment in segments_generator:
                    global_start = block["start"] + segment.start
                    global_end = block["start"] + segment.end

                    transcript_segments.append(
                        TranscriptSegment(
                            start_sec=round(global_start, 2),
                            end_sec=round(global_end, 2),
                            text=segment.text.strip()
                        )
                    )

            return Transcript(
                source_id=source_id,
                segments=transcript_segments
            )

        except Exception as e:
            raise map_transcription_error(e) from e