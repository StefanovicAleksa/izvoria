import subprocess
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import psutil

# A busy appliance isn't always busy on the GPU: RapidOCR and document conversion run on the CPU for minutes at a
# time, and ffmpeg runs as a child process. So activity is GPU utilisation OR CPU across the app's process tree.
GPU_BUSY_PERCENT = 10.0
CPU_BUSY_PERCENT = 20.0
WINDOW_SECONDS = 300.0


@dataclass(frozen=True)
class Sample:
    at: float
    gpu_percent: float
    cpu_percent: float


def read_gpu_utilisation() -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    values = [float(v) for v in out.split() if v.strip().replace(".", "", 1).isdigit()]
    return max(values, default=0.0)


class ProcessTreeCpu:
    """CPU percent of one process and all its children, measured between consecutive calls."""

    def __init__(self) -> None:
        self._tracked: dict = {}

    def percent(self, pid: Optional[int]) -> float:
        if pid is None:
            self._tracked = {}
            return 0.0
        try:
            root = psutil.Process(pid)
            procs = [root] + root.children(recursive=True)
        except psutil.Error:
            self._tracked = {}
            return 0.0
        total = 0.0
        tracked = {}
        for proc in procs:
            known = self._tracked.get(proc.pid, proc)
            try:
                # The first call on a psutil.Process returns 0.0 and starts its measurement interval.
                total += known.cpu_percent(interval=None)
                tracked[proc.pid] = known
            except psutil.Error:
                continue
        self._tracked = tracked
        return total


class ActivitySampler:
    def __init__(
        self,
        pid_source: Callable[[], Optional[int]],
        gpu_reader: Callable[[], float] = read_gpu_utilisation,
        cpu_reader: Optional[ProcessTreeCpu] = None,
        clock: Callable[[], float] = time.monotonic,
        window_seconds: float = WINDOW_SECONDS,
    ) -> None:
        self._pid_source = pid_source
        self._gpu_reader = gpu_reader
        self._cpu = cpu_reader or ProcessTreeCpu()
        self._clock = clock
        self._window = window_seconds
        self._samples: Deque[Sample] = deque()

    def sample(self) -> Sample:
        now = self._clock()
        sample = Sample(now, self._gpu_reader(), self._cpu.percent(self._pid_source()))
        self._samples.append(sample)
        while self._samples and now - self._samples[0].at > self._window:
            self._samples.popleft()
        return sample

    @property
    def busy(self) -> bool:
        return any(
            s.gpu_percent >= GPU_BUSY_PERCENT or s.cpu_percent >= CPU_BUSY_PERCENT for s in self._samples
        )
