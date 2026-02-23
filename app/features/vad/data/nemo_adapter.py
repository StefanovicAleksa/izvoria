# File: app/features/vad/data/nemo_adapter.py
import os
import uuid
import torch
import torchaudio
# noinspection PyPackageRequirements
from nemo.collections.asr.models import EncDecClassificationModel

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.vad.domain.interfaces import IVadModelAdapter
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.vad.utils.error_mapper import map_vad_error


class NeMoVadAdapter(IVadModelAdapter):
    """
    Adapter for NVIDIA NeMo's MarbleNet VAD.
    Optimized for production inference, incorporating Colab-style post-processing
    (Onset/Offset smoothing) to prevent transcript fragmentation.
    Strictly uses local .nemo files.
    """

    REQUIRED_SAMPLE_RATE = 16000
    WINDOW_LENGTH_SEC = 0.63
    SHIFT_LENGTH_SEC = 0.08

    # Hysteresis Thresholds for smooth clustering
    ONSET_THRESHOLD = 0.50  # Confidence needed to START a speech block
    OFFSET_THRESHOLD = 0.40  # Confidence needed to END a speech block
    MIN_SPEECH_DURATION = 0.25  # Ignore random blips shorter than 250ms

    def load(self, manifest: ModelManifest) -> LoadedModel:
        """Restores the NeMo model from a local .nemo file without network calls."""
        try:
            # 1. Strict explicit check for the file's existence
            if not os.path.exists(settings.VAD_MODEL_PATH):
                raise FileNotFoundError(f"Model file missing: {settings.VAD_MODEL_PATH}")

            # 2. Local restore (Bypasses NGC registry download)
            model = EncDecClassificationModel.restore_from(restore_path=settings.VAD_MODEL_PATH)
            model = model.to("cuda")
            model.eval()

            return LoadedModel(manifest=manifest, model_instance=model)

        except Exception as e:
            raise map_vad_error(e) from e

    def unload(self, model: LoadedModel) -> bool:
        """Safely removes the model from VRAM."""
        try:
            del model.model_instance
            return True
        except Exception as e:
            raise map_vad_error(e) from e

    @torch.no_grad()
    def generate_vad_map(self, model: LoadedModel, audio_path: str, source_id: uuid.UUID) -> VadMap:
        """
        Executes In-Memory Virtual Slicing. Loads the audio, formats it for NeMo,
        and maps the human speech segments using hysteresis smoothing.
        """
        try:
            nemo_model: EncDecClassificationModel = model.model_instance

            # 1. Load and Format Audio
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != self.REQUIRED_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.REQUIRED_SAMPLE_RATE)
                waveform = resampler(waveform)

            # NeMo strictly expects mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.to("cuda")

            # 2. Prepare Window Math
            total_samples = waveform.shape[1]
            window_samples = int(self.WINDOW_LENGTH_SEC * self.REQUIRED_SAMPLE_RATE)
            shift_samples = int(self.SHIFT_LENGTH_SEC * self.REQUIRED_SAMPLE_RATE)

            segments = []
            is_speaking = False
            current_speech_start = 0.0
            confidence_accumulator = 0.0
            frame_count = 0

            # 3. Memory-Safe Sequential Inference Loop
            for start_idx in range(0, total_samples - window_samples + 1, shift_samples):
                end_idx = start_idx + window_samples
                audio_chunk = waveform[:, start_idx:end_idx]
                audio_len = torch.tensor([window_samples], device="cuda")

                logits = nemo_model.forward(input_signal=audio_chunk, input_signal_length=audio_len)
                probs = torch.softmax(logits, dim=-1)

                # Class 1 is "Speech" in MarbleNet
                speech_prob = probs[0, 1].item()
                current_time_sec = start_idx / self.REQUIRED_SAMPLE_RATE

                # 4. Hysteresis Smoothing Logic
                if not is_speaking and speech_prob >= self.ONSET_THRESHOLD:
                    is_speaking = True
                    current_speech_start = current_time_sec
                    confidence_accumulator = speech_prob
                    frame_count = 1

                elif is_speaking and speech_prob >= self.OFFSET_THRESHOLD:
                    confidence_accumulator += speech_prob
                    frame_count += 1

                elif is_speaking and speech_prob < self.OFFSET_THRESHOLD:
                    is_speaking = False
                    speech_duration = current_time_sec - current_speech_start

                    if speech_duration >= self.MIN_SPEECH_DURATION:
                        avg_conf = confidence_accumulator / frame_count
                        segments.append(VadSegment(
                            start_sec=current_speech_start,
                            end_sec=current_time_sec,
                            confidence=avg_conf
                        ))

            # 5. Catch trailing speech if audio ends mid-sentence
            if is_speaking:
                speech_duration = (total_samples / self.REQUIRED_SAMPLE_RATE) - current_speech_start
                if speech_duration >= self.MIN_SPEECH_DURATION:
                    avg_conf = confidence_accumulator / frame_count
                    segments.append(VadSegment(
                        start_sec=current_speech_start,
                        end_sec=total_samples / self.REQUIRED_SAMPLE_RATE,
                        confidence=avg_conf
                    ))

            return VadMap(source_id=source_id, segments=segments)

        except Exception as e:
            raise map_vad_error(e) from e