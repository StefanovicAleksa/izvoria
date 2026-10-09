# Path: tests/orchestration/utils/test_preload_slot_unit.py

import threading

from app.orchestration.utils.preload_slot import PreloadSlot


class _Phase:
    _preload_future = PreloadSlot()

    def __init__(self):
        self._preload_future = None


class TestPreloadSlot:
    def test_behaves_like_an_attribute_on_one_thread(self):
        phase = _Phase()
        assert phase._preload_future is None

        phase._preload_future = "future"

        assert phase._preload_future == "future"

    def test_each_thread_sees_only_its_own_value(self):
        phase = _Phase()
        phase._preload_future = "main"
        seen = {}
        barrier = threading.Barrier(2)

        def request(name):
            seen[f"{name}-before"] = phase._preload_future
            phase._preload_future = name
            barrier.wait()
            seen[f"{name}-after"] = phase._preload_future

        threads = [threading.Thread(target=request, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert seen == {"a-before": None, "b-before": None, "a-after": "a", "b-after": "b"}
        assert phase._preload_future == "main"

    def test_instances_do_not_share_a_slot(self):
        first, second = _Phase(), _Phase()
        first._preload_future = "first"

        assert second._preload_future is None

    def test_class_access_returns_the_descriptor(self):
        assert isinstance(_Phase._preload_future, PreloadSlot)
