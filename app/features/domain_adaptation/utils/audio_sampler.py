# Canvas: audio_sampler_v1
# Path: app/features/domain_adaptation/utils/audio_sampler.py

import os
import tempfile
import subprocess
from pathlib import Path
from typing import List

from app.features.vad.domain.entities import VadMap
from app.features.domain_adaptation.domain.entities import AudioSample
from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error


class AudioSampler:
    """
    Utility to extract intelligent, distributed audio slices for the CPU Scout.
    Leverages the VAD Map to ensure we only sample actual human speech,
    bypassing silent intros/outros or dead air.
    """

    def __init__(self, slice_duration_sec: float = 20.0):
        self._slice_duration = slice_duration_sec
        self._target_percentiles = [0.15, 0.50, 0.85]

    def extract_samples(self, audio_path: str | Path, vad_map: VadMap) -> List[AudioSample]:
        """
        Extracts 3 distinct audio slices based on the VAD timeline.
        """
        src_audio = Path(audio_path).resolve()

        if not src_audio.exists():
            raise map_domain_adaptation_error(FileNotFoundError(f"Source audio missing: {src_audio}"))

        if not vad_map.segments:
            return []

        # Find the absolute end of the speech timeline
        total_duration = vad_map.segments[-1].end_sec
        samples: List[AudioSample] = []

        for percentile in self._target_percentiles:
            target_sec = total_duration * percentile

            # Find the VAD segment closest to this target time
            closest_segment = min(
                vad_map.segments,
                key=lambda seg: abs(seg.start_sec - target_sec)
            )

            start_sec = closest_segment.start_sec
            end_sec = min(closest_segment.end_sec, start_sec + self._slice_duration)

            # Prevent extracting micro-slices that are too short for Whisper to understand
            if end_sec - start_sec < 2.0:
                continue

            try:
                extracted_path = self._slice_audio_ffmpeg(src_audio, start_sec, end_sec)

                samples.append(AudioSample(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    file_path=str(extracted_path)
                ))
            except Exception as e:
                # If one slice fails, we log it but continue so the Scout still gets partial data
                print(f"[WARNING] Failed to extract audio slice at {start_sec}s: {e}")
                continue

        return samples

    def _slice_audio_ffmpeg(self, src_audio: Path, start_sec: float, end_sec: float) -> Path:
        """
        Uses FFmpeg fast-seeking to instantly extract a slice from the master audio file.
        """
        temp_fd, temp_path_str = tempfile.mkstemp(suffix=".wav", prefix="onyx_scout_slice_")
        os.close(temp_fd)
        out_path = Path(temp_path_str)

        duration = end_sec - start_sec

        command: list[str] = [
            "ffmpeg",
            "-ss", str(start_sec),  # Fast-seek before input
            "-i", os.fspath(src_audio),
            "-t", str(duration),  # Duration to extract
            "-vn",  # No video
            "-acodec", "copy",  # Stream copy (instantaneous, no re-encoding)
            "-y",
            os.fspath(out_path)
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=30
            )
            return out_path
        except Exception as e:
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            raise map_domain_adaptation_error(e, context=f"Slicing audio at {start_sec}s") from e

    def cleanup_samples(self, samples: List[AudioSample]) -> None:
        """
        Deletes the temporary OS files after the Scout is finished transcribing them.
        """
        for sample in samples:
            path = Path(sample.file_path)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass