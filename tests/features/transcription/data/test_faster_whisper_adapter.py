# Canvas: faster_whisper_adapter_v8
# Path: app/features/transcription/data/faster_whisper_adapter.py

import os
import torch
import torchaudio
from typing import List, Optional
from faster_whisper import WhisperModel

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.transcription.domain.interfaces import ITranscriptionAdapter
from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.transcription.utils.error_mapper import map_transcription_error


class FasterWhisperAdapter(ITranscriptionAdapter):
    """
    Adapter for the Faster-Whisper CTranslate2 engine, executed on the GPU.
    Uses RAM-based audio chunking to prevent O(N) Disk I/O bottlenecks.
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
            hotwords: Optional[str] = None,
            initial_prompt: Optional[str] = None
    ) -> List[TranscriptSegment]:
        try:
            whisper_model: WhisperModel = model.model_instance
            segments_out = []

            # 1. Load full audio into memory ONCE (Prevents massive Disk I/O bottlenecks)
            waveform, sample_rate = torchaudio.load(audio_path)

            # Ensure mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Ensure 16kHz for Whisper
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            waveform_1d = waveform.squeeze().numpy()

            # 2. Merge close VAD segments to minimize inference calls
            merged_segments = self._merge_vad_segments(vad_map.segments, gap_threshold=2.0)

            # 3. Transcribe each merged chunk strictly from RAM
            for m_seg in merged_segments:
                start_sample = int(m_seg.start_sec * 16000)
                end_sample = int(m_seg.end_sec * 16000)

                audio_chunk = waveform_1d[start_sample:end_sample]

                # Skip micro-chunks to prevent Whisper from hallucinating
                if len(audio_chunk) < 1600:
                    continue

                # Pass the RAM chunk directly (avoids O(N) re-decoding of the file)
                # Inject both hotwords (beam search bias) and initial_prompt (transformer context)
                segments, info = whisper_model.transcribe(
                    audio=audio_chunk,
                    language=language,
                    vad_filter=False,
                    hotwords=hotwords,
                    initial_prompt=initial_prompt,
                    word_timestamps=True
                )

                for segment in segments:
                    domain_words = []

                    # Map the inner words and offset their timestamps back to the original file's timeline
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

                    # Offset the segment timestamps back to the original file's timeline
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

    # noinspection PyMethodMayBeStatic
    def _merge_vad_segments(self, segments: List[VadSegment], gap_threshold: float = 2.0) -> List[VadSegment]:
        """Merges VAD segments closer than the gap_threshold to reduce total inference calls."""
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