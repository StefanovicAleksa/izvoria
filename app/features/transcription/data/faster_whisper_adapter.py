# Canvas: faster_whisper_adapter_v18
# Path: app/features/transcription/data/faster_whisper_adapter.py

import os
import gc
import torch
import torchaudio
from typing import List, Optional, Any

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.transcription.domain.interfaces import ITranscriptionAdapter
from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.transcription.utils.error_mapper import map_transcription_error


def _merge_vad_segments(segments: List[VadSegment], gap_threshold: float = 2.0) -> List[VadSegment]:
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start_sec)
    merged = [sorted_segs[0]]

    for current in sorted_segs[1:]:
        last = merged[-1]
        if current.start_sec - last.end_sec <= gap_threshold:
            merged[-1] = VadSegment(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, current.end_sec),
                confidence=(last.confidence + current.confidence) / 2.0
            )
        else:
            merged.append(current)

    return merged


class FasterWhisperAdapter(ITranscriptionAdapter):
    @staticmethod
    def _get_whisper_model(model_path: str) -> Any:
        from faster_whisper import WhisperModel
        return WhisperModel(
            model_size_or_path=model_path,
            device="cuda",
            compute_type="float16",
            local_files_only=True
        )

    def load(self, manifest: ModelManifest) -> LoadedModel:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not os.path.exists(manifest.model_path):
                raise FileNotFoundError(f"Whisper model directory missing: {manifest.model_path}")

            model = self._get_whisper_model(manifest.model_path)
            return LoadedModel(manifest=manifest, model_instance=model)
        except Exception as e:
            raise map_transcription_error(e) from e

    def unload(self, model: LoadedModel) -> bool:
        try:
            if hasattr(model, "model_instance"):
                del model.model_instance
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        except Exception as e:
            raise map_transcription_error(e) from e

    def transcribe(
            self,
            model: LoadedModel,
            audio_path: str,
            vad_map: VadMap,
            language: str,
            hotwords: Optional[str] = None,
            initial_prompt: Optional[str] = None
    ) -> List[TranscriptSegment]:
        try:
            whisper_model: Any = model.model_instance
            segments_out = []

            # FIX: Removed backend="soundfile" so torchcodec can natively parse .mp4 and .wav via GPU
            waveform, sample_rate = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            waveform_1d = waveform.squeeze().numpy()
            merged_segments = _merge_vad_segments(vad_map.segments, gap_threshold=2.0)

            safe_hotwords = hotwords.strip() if hotwords and hotwords.strip() else None
            safe_prompt = None

            if initial_prompt and initial_prompt.strip():
                words = initial_prompt.strip().split()
                safe_prompt = " ".join(words[:45])

            for m_seg in merged_segments:
                start_sample = int(m_seg.start_sec * 16000)
                end_sample = int(m_seg.end_sec * 16000)
                audio_chunk = waveform_1d[start_sample:end_sample]

                if len(audio_chunk) < 6400:
                    continue

                segments, info = whisper_model.transcribe(
                    audio=audio_chunk,
                    language=language,
                    vad_filter=False,
                    hotwords=safe_hotwords,
                    initial_prompt=safe_prompt,
                    word_timestamps=True,
                    condition_on_previous_text=False
                )

                for segment in segments:
                    domain_words = []
                    if getattr(segment, "words", None):
                        for w in segment.words:
                            domain_words.append(
                                TranscriptWord(
                                    start_sec=m_seg.start_sec + w.start,
                                    end_sec=m_seg.start_sec + w.end,
                                    word=w.word,
                                    probability=w.probability
                                )
                            )

                    segments_out.append(
                        TranscriptSegment(
                            start_sec=m_seg.start_sec + segment.start,
                            end_sec=m_seg.start_sec + segment.end,
                            text=segment.text.strip(),
                            speaker_id="Unknown",
                            words=domain_words
                        )
                    )

            return segments_out

        except Exception as e:
            raise map_transcription_error(e) from e