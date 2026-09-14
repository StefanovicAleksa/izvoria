# File: app/features/vad/data/nemo_adapter.py
import os
import uuid
from typing import List, Tuple

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
        Executes In-Memory Virtual Slicing. Loads the audio, scores every 0.63s window (0.08s hop)
        in GPU batches, and maps the human speech segments using hysteresis smoothing.
        """
        try:
            nemo_model: EncDecClassificationModel = model.model_instance
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # 1. Load and Format Audio
            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != self.REQUIRED_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.REQUIRED_SAMPLE_RATE)
                waveform = resampler(waveform)

            # NeMo strictly expects mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            total_samples = waveform.shape[1]
            total_duration = total_samples / self.REQUIRED_SAMPLE_RATE
            window_samples = int(self.WINDOW_LENGTH_SEC * self.REQUIRED_SAMPLE_RATE)
            shift_samples = int(self.SHIFT_LENGTH_SEC * self.REQUIRED_SAMPLE_RATE)

            if total_samples == 0:
                return VadMap(source_id=source_id, segments=[])

            # Audio shorter than one window is still scored (zero-padded), instead of being ignored
            signal = waveform[0].to(device)
            if total_samples < window_samples:
                signal = torch.nn.functional.pad(signal, (0, window_samples - total_samples))

            # 2. Batched window scoring. unfold() is a view, so only one batch is copied at a time.
            speech_probs = self._score_windows(nemo_model, signal, window_samples, shift_samples, device)

            # 3. Hysteresis over window-centre timestamps
            raw_segments = self._hysteresis_segments(speech_probs, total_duration)

            # 4. Pad and merge so segment edges don't clip the first/last words
            segments = self._pad_and_merge(raw_segments, total_duration, settings.VAD_PAD_SEC)

            return VadMap(source_id=source_id, segments=segments)

        except Exception as e:
            raise map_vad_error(e) from e

    def _score_windows(
            self,
            nemo_model: EncDecClassificationModel,
            signal: torch.Tensor,
            window_samples: int,
            shift_samples: int,
            device: str
    ) -> List[float]:
        """
        Speech probability per window. Scoring windows one by one meant ~90k forward calls (each
        with a GPU sync) for a 2-hour recording; batching does the same work in a few hundred.
        """
        windows = signal.unfold(0, window_samples, shift_samples)
        batch_size = max(1, settings.VAD_BATCH_SIZE)

        probs: List[float] = []
        for start in range(0, windows.shape[0], batch_size):
            batch = windows[start:start + batch_size].contiguous()
            lengths = torch.full((batch.shape[0],), window_samples, dtype=torch.long, device=device)

            logits = nemo_model.forward(input_signal=batch, input_signal_length=lengths)
            # Class 1 is "Speech" in MarbleNet
            probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())

        return probs

    def _hysteresis_segments(self, speech_probs: List[float], total_duration: float) -> List[VadSegment]:
        """
        Onset/offset hysteresis. Each window is timestamped at its CENTRE: stamping it at its start
        placed segment ends up to ~0.5s before the speech actually stopped.
        """
        segments: List[VadSegment] = []
        is_speaking = False
        current_speech_start = 0.0
        confidence_accumulator = 0.0
        frame_count = 0
        half_window = self.WINDOW_LENGTH_SEC / 2.0

        for idx, speech_prob in enumerate(speech_probs):
            current_time_sec = min(idx * self.SHIFT_LENGTH_SEC + half_window, total_duration)

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
                if current_time_sec - current_speech_start >= self.MIN_SPEECH_DURATION:
                    segments.append(VadSegment(
                        start_sec=current_speech_start,
                        end_sec=current_time_sec,
                        confidence=confidence_accumulator / frame_count
                    ))

        # Catch trailing speech if audio ends mid-sentence. The audio ran out before an offset window could
        # close the segment, so its length is measured from where the first speech window began.
        if is_speaking and total_duration - max(0.0, current_speech_start - half_window) >= self.MIN_SPEECH_DURATION:
            segments.append(VadSegment(
                start_sec=current_speech_start,
                end_sec=total_duration,
                confidence=confidence_accumulator / frame_count
            ))

        return segments

    @staticmethod
    def _pad_and_merge(segments: List[VadSegment], total_duration: float, pad_sec: float) -> List[VadSegment]:
        """Pads each segment by `pad_sec` on both sides (clamped to the audio) and merges overlaps."""
        padded: List[Tuple[float, float, float]] = [
            (max(0.0, s.start_sec - pad_sec), min(total_duration, s.end_sec + pad_sec), s.confidence)
            for s in segments
        ]

        merged: List[Tuple[float, float, float]] = []
        for start, end, confidence in padded:
            if merged and start <= merged[-1][1]:
                prev_start, prev_end, prev_conf = merged[-1]
                prev_len, cur_len = prev_end - prev_start, end - start
                weighted = (prev_conf * prev_len + confidence * cur_len) / max(prev_len + cur_len, 1e-9)
                merged[-1] = (prev_start, max(prev_end, end), weighted)
            else:
                merged.append((start, end, confidence))

        return [
            VadSegment(start_sec=start, end_sec=end, confidence=min(1.0, max(0.0, confidence)))
            for start, end, confidence in merged
        ]
