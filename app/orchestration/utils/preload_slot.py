# Path: app/orchestration/utils/preload_slot.py

import threading
from typing import Any, Optional


class PreloadSlot:
    """
    A Phase attribute whose value is per thread: `_preload_future = PreloadSlot()` on the class.

    A live Phase is one object shared by every request, and it remembers the preload it fired until `execute`
    resolves it. With requests running concurrently (the hosted API) one request's `execute` would otherwise
    resolve, or discard, another request's preload. Pipelines are synchronous, so a request fires and resolves
    its preloads on its own thread; keeping the future per thread gives each request its own slot. On the
    single-threaded appliance it behaves exactly like a plain attribute.
    """

    def __set_name__(self, owner: type, name: str) -> None:
        self._key = f"__preload_slot_{name}"

    def _local(self, instance: Any) -> threading.local:
        local = instance.__dict__.get(self._key)
        if local is None:
            local = instance.__dict__.setdefault(self._key, threading.local())
        return local

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Any:
        if instance is None:
            return self
        return getattr(self._local(instance), "value", None)

    def __set__(self, instance: Any, value: Any) -> None:
        self._local(instance).value = value
