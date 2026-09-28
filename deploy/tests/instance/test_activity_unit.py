from unittest.mock import MagicMock

import pytest

from deploy.instance.agent.activity import ActivitySampler


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def sampler_with(gpu_values, cpu_values, clock):
    gpu = iter(gpu_values)
    cpu = MagicMock()
    cpu.percent.side_effect = list(cpu_values)
    return ActivitySampler(
        pid_source=lambda: 123, gpu_reader=lambda: next(gpu), cpu_reader=cpu, clock=clock, window_seconds=300,
    )


class TestActivitySampler:
    def test_idle_when_nothing_has_been_sampled(self):
        assert sampler_with([], [], FakeClock()).busy is False

    @pytest.mark.parametrize("gpu, cpu, busy", [
        (0.0, 0.0, False),
        (9.0, 19.0, False),
        (10.0, 0.0, True),
        (0.0, 20.0, True),  # OCR / document conversion: all CPU, no GPU
    ])
    def test_busy_on_either_gpu_or_process_tree_cpu(self, gpu, cpu, busy):
        sampler = sampler_with([gpu], [cpu], FakeClock())
        sampler.sample()
        assert sampler.busy is busy

    def test_one_busy_sample_keeps_it_busy_for_the_whole_window(self):
        clock = FakeClock()
        sampler = sampler_with([80.0, 0.0, 0.0], [0.0, 0.0, 0.0], clock)
        sampler.sample()
        clock.now = 290
        sampler.sample()
        assert sampler.busy is True
        clock.now = 301
        sampler.sample()
        assert sampler.busy is False
