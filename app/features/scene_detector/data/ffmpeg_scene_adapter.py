# Canvas: ffmpeg_scene_adapter_v2
# Path: app/features/scene_detector/data/ffmpeg_scene_adapter.py

import os
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from app.features.scene_detector.domain.interfaces import IFFmpegSceneAdapter
from app.features.scene_detector.domain.entities import TimeWindow
from app.features.scene_detector.utils.error_mapper import map_scene_detector_error


class FFmpegSceneAdapter(IFFmpegSceneAdapter):
    """
    Executes OS-level FFmpeg commands to extract unique scenes.
    Uses Fast-Seeking (-ss) for extreme CPU optimization when TimeWindows are provided.
    """

    def __init__(self, timeout_seconds: int = 3600):
        self._timeout = timeout_seconds
        self._pts_time_pattern = re.compile(r"pts_time:\s*([0-9.]+)")

    def extract_scenes(
            self,
            video_path: str | Path,
            output_dir: str | Path,
            threshold: float = 0.4,
            windows: Optional[List[TimeWindow]] = None
    ) -> List[Tuple[float, Path]]:

        src_path = Path(video_path).resolve()
        out_dir = Path(output_dir).resolve()

        if not src_path.exists():
            raise map_scene_detector_error(FileNotFoundError(f"Video not found: {src_path}"))

        extracted_scenes: List[Tuple[float, Path]] = []

        # If no windows are provided, we scan the entire video (Global Pass)
        if not windows:
            # We wrap the global scan in a fake window starting at 0 to reuse the logic
            # Duration isn't strictly needed for a full scan, but we'll just omit -t
            scenes = self._run_ffmpeg_extraction(src_path, out_dir, threshold, start_sec=0.0, duration=None,
                                                 window_idx=0)
            extracted_scenes.extend(scenes)
            return extracted_scenes

        # If windows ARE provided, we use Fast-Seeking for each window (Targeted Pass)
        for index, window in enumerate(windows):
            duration = window.end_sec - window.start_sec
            if duration <= 0:
                continue

            scenes = self._run_ffmpeg_extraction(
                src_path,
                out_dir,
                threshold,
                start_sec=window.start_sec,
                duration=duration,
                window_idx=index
            )
            extracted_scenes.extend(scenes)

        return extracted_scenes

    def _run_ffmpeg_extraction(
            self,
            src_path: Path,
            out_dir: Path,
            threshold: float,
            start_sec: float,
            duration: Optional[float],
            window_idx: int
    ) -> List[Tuple[float, Path]]:

        vf_string = f"select='gt(scene,{threshold})',showinfo"
        # Prefix the files with the window index to avoid overwriting files between loops
        output_pattern = out_dir / f"w{window_idx}_frame_%04d.jpg"

        command: list[str] = ["ffmpeg"]

        # FAST SEEKING: Must be placed BEFORE -i to skip decoding
        if start_sec > 0:
            command.extend(["-ss", str(start_sec)])
        if duration is not None:
            command.extend(["-t", str(duration)])

        command.extend([
            "-i", os.fspath(src_path),
            "-vf", vf_string,
            "-vsync", "vfr",
            "-q:v", "2",
            "-y",
            os.fspath(output_pattern)
        ])

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=self._timeout
            )

            # Parse Timestamps
            timestamps: List[float] = []
            for line in result.stderr.splitlines():
                if "Parsed_showinfo" in line and "pts_time:" in line:
                    match = self._pts_time_pattern.search(line)
                    if match:
                        # Because we used -ss, FFmpeg resets the internal clock to 0 for this extraction.
                        # We MUST add the start_sec back to map it correctly to the global video timeline.
                        local_pts = float(match.group(1))
                        global_timestamp = start_sec + local_pts
                        timestamps.append(global_timestamp)

            # Match files to timestamps
            scenes: List[Tuple[float, Path]] = []
            for index, timestamp in enumerate(timestamps):
                file_name = f"w{window_idx}_frame_{index + 1:04d}.jpg"
                file_path = out_dir / file_name

                if file_path.exists():
                    scenes.append((timestamp, file_path))

            return scenes

        except Exception as e:
            raise map_scene_detector_error(e, context=f"{src_path} at window {start_sec}s") from e