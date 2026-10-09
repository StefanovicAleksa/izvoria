"""VRAM as nvidia-smi reports it, sampled on a thread so a run's peak is caught, not just its end."""

import subprocess
import threading
import time
from typing import Callable, Optional, Tuple


def read_vram(run: Callable = subprocess.run) -> Tuple[int, int]:
    """(used MiB, total MiB) of GPU 0."""
    out = run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True).stdout
    used, total = (int(x.strip()) for x in out.strip().splitlines()[0].split(","))
    return used, total


class VramSampler:
    def __init__(self, interval: float = 0.5, reader: Callable[[], Tuple[int, int]] = read_vram) -> None:
        self._interval = interval
        self._read = reader
        self._peak = 0
        self._total = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def now(self) -> int:
        used, self._total = self._read()
        return used

    @property
    def total(self) -> int:
        if not self._total:
            self.now()
        return self._total

    def reset_peak(self) -> None:
        self._peak = self.now()

    @property
    def peak(self) -> int:
        return max(self._peak, self.now())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak = max(self._peak, self.now())
            except Exception:
                pass
            time.sleep(self._interval)

    def __enter__(self) -> "VramSampler":
        self.reset_peak()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
