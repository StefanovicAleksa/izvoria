# Canvas: faster_whisper_adapter_v20
# Path: app/features/transcription/data/faster_whisper_adapter.py

import os
import gc
import logging
import numpy as np
import torch
import torchaudio
from typing import List, Optional, Any, Tuple

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.features.transcription.domain.interfaces import ITranscriptionAdapter
from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.transcription.utils.error_mapper import map_transcription_error

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Whisper judges language from one 30 s window, so that much speech is all the language guard needs
LANGUAGE_DETECTION_SEC = 30

# Speech separated by at most this much silence is transcribed as one span
MERGE_GAP_SEC = 2.0

# faster-whisper's batched pipeline only transcribes the first 30s of any clip, so batched clips
# stay safely below that. Longer uninterrupted speech goes through the sequential path instead.
MAX_BATCH_CLIP_SEC = 29.5

# A second of VAD speech with no word on it is a dropped utterance, not a cough. faster-whisper's batched decoder
# sometimes returns only the last sentence of a clip that joins several utterances (2 of 3 lost on FLEURS en
# 13.2-38.0 s); the sequential decoder recovers them. Word timings drift by a little, hence the tolerance.
COVERAGE_MIN_SPEECH_SEC = 1.0
COVERAGE_TOLERANCE_SEC = 0.2


def _merge_vad_segments(segments: List[VadSegment], gap_threshold: float = MERGE_GAP_SEC) -> List[VadSegment]:
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


def _plan_batched_clips(
        segments: List[VadSegment],
        max_clip_sec: float = MAX_BATCH_CLIP_SEC,
        merge_gap_sec: float = MERGE_GAP_SEC
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Packs VAD segments into clips for batched transcription.

    Clips only ever start/end at VAD boundaries (i.e. in silence), so no word is cut: consecutive
    segments join the current clip while the silence between them is short and the clip stays
    under `max_clip_sec`. A single segment longer than that cannot be split safely and is returned
    separately for the sequential path, which handles arbitrary lengths.

    Returns: (batched_clips, sequential_spans) as (start_sec, end_sec) tuples.
    """
    clips: List[Tuple[float, float]] = []
    sequential: List[Tuple[float, float]] = []
    clip_start: Optional[float] = None
    clip_end: Optional[float] = None

    for seg in sorted(segments, key=lambda s: s.start_sec):
        if seg.end_sec - seg.start_sec > max_clip_sec:
            if clip_start is not None:
                clips.append((clip_start, clip_end))
                clip_start = clip_end = None
            sequential.append((seg.start_sec, seg.end_sec))
            continue

        if clip_start is None:
            clip_start, clip_end = seg.start_sec, seg.end_sec
        elif seg.start_sec - clip_end <= merge_gap_sec and max(clip_end, seg.end_sec) - clip_start <= max_clip_sec:
            clip_end = max(clip_end, seg.end_sec)
        else:
            clips.append((clip_start, clip_end))
            clip_start, clip_end = seg.start_sec, seg.end_sec

    if clip_start is not None:
        clips.append((clip_start, clip_end))

    return clips, sequential


def _widen_to_min_duration(start: float, end: float, total_duration: float, min_duration: float) -> Tuple[float, float]:
    """
    Symmetrically widens a span shorter than `min_duration` (within the audio bounds).
    Very short spans used to be skipped outright, silently dropping isolated one-word answers
    such as "Yes." or "Objection."
    """
    if end - start >= min_duration or total_duration <= 0:
        return start, min(end, total_duration)

    missing = min_duration - (end - start)
    new_start = max(0.0, start - missing / 2.0)
    new_end = min(total_duration, end + missing / 2.0)

    if new_end - new_start < min_duration:
        if new_start <= 0.0:
            new_end = min(total_duration, new_start + min_duration)
        else:
            new_start = max(0.0, new_end - min_duration)

    return new_start, new_end


def _is_covered(speech: VadSegment, segments: List[TranscriptSegment], tolerance: float) -> bool:
    lo, hi = speech.start_sec - tolerance, speech.end_sec + tolerance
    for seg in segments:
        spans = [(w.start_sec, w.end_sec) for w in seg.words] or [(seg.start_sec, seg.end_sec)]
        if any(start < hi and end > lo for start, end in spans):
            return True
    return False


def _uncovered_speech(
        speech_segments: List[VadSegment],
        transcribed: List[TranscriptSegment],
        min_speech_sec: float = COVERAGE_MIN_SPEECH_SEC,
        tolerance: float = COVERAGE_TOLERANCE_SEC
) -> List[VadSegment]:
    """VAD speech of at least `min_speech_sec` that no transcribed word overlaps: speech the decoder dropped."""
    return [
        speech for speech in speech_segments
        if speech.end_sec - speech.start_sec >= min_speech_sec and not _is_covered(speech, transcribed, tolerance)
    ]


def _clips_to_redecode(
        clips: List[Tuple[float, float]],
        speech_segments: List[VadSegment],
        transcribed: List[TranscriptSegment]
) -> List[Tuple[float, float]]:
    """The batched clips holding speech that came back without a single word."""
    dropped = _uncovered_speech(speech_segments, transcribed)
    return [(start, end) for start, end in clips
            if any(start <= (s.start_sec + s.end_sec) / 2.0 <= end for s in dropped)]


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

    @staticmethod
    def _get_batched_pipeline(whisper_model: Any) -> Any:
        """FACTORY METHOD & LAZY IMPORT: a thin batching wrapper around the already-loaded model."""
        from faster_whisper import BatchedInferencePipeline
        return BatchedInferencePipeline(model=whisper_model)

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
            waveform_1d = self._load_waveform(audio_path)
            total_duration = len(waveform_1d) / SAMPLE_RATE
            min_slice_sec = settings.WHISPER_MIN_SLICE_SEC

            safe_hotwords = hotwords.strip() if hotwords and hotwords.strip() else None
            safe_prompt = None
            if initial_prompt and initial_prompt.strip():
                safe_prompt = " ".join(initial_prompt.strip().split()[:45])

            batch_size = settings.WHISPER_BATCH_SIZE
            if batch_size > 1:
                batched_clips, sequential_spans = _plan_batched_clips(vad_map.segments)
            else:
                batched_clips = []
                sequential_spans = [(s.start_sec, s.end_sec) for s in _merge_vad_segments(vad_map.segments)]

            segments_out: List[TranscriptSegment] = []
            decode_options = dict(language=language, hotwords=safe_hotwords, initial_prompt=safe_prompt)

            # 1. Batched path: clips are decoded in parallel; returned timestamps are absolute
            if batched_clips:
                widened_clips = []
                for start, end in batched_clips:
                    w_start, w_end = _widen_to_min_duration(start, end, total_duration, min_slice_sec)
                    if w_end > w_start:
                        widened_clips.append((w_start, w_end))

                if widened_clips:
                    pipeline = self._get_batched_pipeline(whisper_model)
                    segments, _ = pipeline.transcribe(
                        audio=waveform_1d,
                        clip_timestamps=[{"start": start, "end": end} for start, end in widened_clips],
                        batch_size=batch_size,
                        vad_filter=False,
                        word_timestamps=True,
                        condition_on_previous_text=False,
                        **decode_options
                    )
                    batched_out = [self._to_domain_segment(segment, offset_sec=0.0) for segment in segments]

                    # A clip whose speech came back wordless is decoded again, sequentially, and its batched
                    # output replaced rather than merged, so nothing it did return is duplicated
                    redecode = _clips_to_redecode(widened_clips, vad_map.segments, batched_out)
                    if redecode:
                        logger.warning(
                            f"Batched Whisper returned no words for speech in {len(redecode)} of "
                            f"{len(widened_clips)} clips of {os.path.basename(audio_path)}; decoding them "
                            f"sequentially."
                        )
                        batched_out = [
                            seg for seg in batched_out
                            if not any(start <= (seg.start_sec + seg.end_sec) / 2.0 <= end for start, end in redecode)
                        ]
                        for start, end in redecode:
                            batched_out.extend(self._decode_span(whisper_model, waveform_1d, start, end, decode_options))
                    segments_out.extend(batched_out)

            # 2. Sequential path: long uninterrupted speech (or batching disabled)
            for start, end in sequential_spans:
                w_start, w_end = _widen_to_min_duration(start, end, total_duration, min_slice_sec)
                segments_out.extend(self._decode_span(whisper_model, waveform_1d, w_start, w_end, decode_options))

            # 3. Last resort: a clip decoded sequentially can still lose its final sentence. Speech that has no
            # word yet is decoded on its own; being wordless, it can't duplicate anything already transcribed.
            for speech in _uncovered_speech(vad_map.segments, segments_out):
                w_start, w_end = _widen_to_min_duration(speech.start_sec, speech.end_sec, total_duration,
                                                        min_slice_sec)
                segments_out.extend(self._decode_span(whisper_model, waveform_1d, w_start, w_end, decode_options))

            still_missing = _uncovered_speech(vad_map.segments, segments_out)
            if still_missing:
                logger.warning(
                    f"{sum(s.end_sec - s.start_sec for s in still_missing):.1f}s of detected speech in "
                    f"{os.path.basename(audio_path)} has no transcribed words ({len(still_missing)} VAD segments)."
                )

            segments_out.sort(key=lambda s: s.start_sec)
            return segments_out

        except Exception as e:
            raise map_transcription_error(e) from e

    def _decode_span(
            self, whisper_model: Any, waveform_1d: Any, start: float, end: float, decode_options: dict
    ) -> List[TranscriptSegment]:
        """Sequential decode of one span of the file, with timestamps made absolute."""
        audio_chunk = waveform_1d[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]
        if len(audio_chunk) == 0:
            return []

        segments, _ = whisper_model.transcribe(
            audio=audio_chunk,
            vad_filter=False,
            word_timestamps=True,
            condition_on_previous_text=False,
            **decode_options
        )
        return [self._to_domain_segment(segment, offset_sec=start) for segment in segments]

    def detect_language(self, model: LoadedModel, audio_path: str, vad_map: VadMap) -> Optional[Tuple[str, float]]:
        try:
            waveform_1d = self._load_waveform(audio_path)
            speech = []
            collected = 0
            for segment in vad_map.segments:
                clip = waveform_1d[int(segment.start_sec * SAMPLE_RATE):int(segment.end_sec * SAMPLE_RATE)]
                speech.append(clip)
                collected += len(clip)
                if collected >= LANGUAGE_DETECTION_SEC * SAMPLE_RATE:
                    break
            if collected == 0:
                return None

            # Speech only: silence and music between VAD segments would dilute Whisper's language ID
            language, probability, _ = model.model_instance.detect_language(
                audio=np.concatenate(speech)[:LANGUAGE_DETECTION_SEC * SAMPLE_RATE],
                language_detection_segments=1
            )
            return language, float(probability)
        except Exception as e:
            raise map_transcription_error(e) from e

    @staticmethod
    def _load_waveform(audio_path: str) -> Any:
        """The file as a 16 kHz mono float array, the only input faster-whisper accepts."""
        # FIX: Removed backend="soundfile" so torchcodec can natively parse .mp4 and .wav via GPU
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        return waveform.squeeze(0).numpy()

    @staticmethod
    def _to_domain_segment(segment: Any, offset_sec: float) -> TranscriptSegment:
        domain_words = []
        if getattr(segment, "words", None):
            for w in segment.words:
                domain_words.append(
                    TranscriptWord(
                        start_sec=offset_sec + w.start,
                        end_sec=offset_sec + w.end,
                        word=w.word,
                        probability=w.probability
                    )
                )

        return TranscriptSegment(
            start_sec=offset_sec + segment.start,
            end_sec=offset_sec + segment.end,
            text=segment.text.strip(),
            speaker_id="Unknown",
            words=domain_words,
            # Extracted metadata for the hallucination filter
            no_speech_prob=getattr(segment, "no_speech_prob", None),
            avg_logprob=getattr(segment, "avg_logprob", None),
            compression_ratio=getattr(segment, "compression_ratio", None)
        )
