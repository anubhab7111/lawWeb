"""
Thermal guard for long GPU jobs on the dev laptop.

The CPU and GPU share a heat pipe: sustained full-power GPU work drives the CPU
package to its 100°C critical limit, and the machine hard-powers-off (twice so far,
mid-job, leaving no shutdown log). Long jobs call `guard.step()` between batches:
it enforces a duty cycle (idle time proportional to work time) and, whenever the
CPU package reaches `pause_at`, waits until it cools to `resume_at`. The package
sensor spikes by 20°C for a moment on any burst, so readings are a median of three
samples, taken every `check_every` seconds.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Optional


def cpu_package_temp() -> Optional[float]:
    """CPU package temperature in °C (coretemp "Package id 0"), or the hottest core."""
    best = None
    for label in glob.glob("/sys/class/hwmon/hwmon*/temp*_label"):
        try:
            name = open(label).read().strip()
            if not (name.startswith("Package") or name.startswith("Core")):
                continue
            value = int(open(label.replace("_label", "_input")).read()) / 1000
        except (OSError, ValueError):
            continue
        if name.startswith("Package"):
            return value
        best = value if best is None else max(best, value)
    return best


def smoothed_temp(samples: int = 3, gap: float = 0.5) -> Optional[float]:
    values = []
    for i in range(samples):
        t = cpu_package_temp()
        if t is not None:
            values.append(t)
        if i < samples - 1:
            time.sleep(gap)
    return sorted(values)[len(values) // 2] if values else None


class ThermalGuard:
    def __init__(self, duty: Optional[float] = None, pause_at: float = 92.0, resume_at: float = 82.0,
                 threads: Optional[int] = None, check_every: float = 15.0):
        self.duty = duty if duty is not None else float(os.environ.get("GPU_DUTY", "0.75"))
        self.pause_at, self.resume_at = pause_at, resume_at
        self._last = time.time()
        self._checked = time.time()
        self.check_every = check_every
        self.paused_seconds = 0.0
        threads = threads or int(os.environ.get("TORCH_THREADS", "4"))
        try:
            import torch

            torch.set_num_threads(threads)
        except ImportError:
            pass

    def step(self) -> None:
        now = time.time()
        worked = now - self._last
        if 0 < self.duty < 1:
            time.sleep(worked * (1 - self.duty) / self.duty)
        if time.time() - self._checked >= self.check_every:
            self._checked = time.time()
            temp = smoothed_temp()
            if temp is not None and temp >= self.pause_at:
                t0 = time.time()
                print(f"[thermal] CPU package {temp:.0f}°C — pausing until {self.resume_at:.0f}°C", flush=True)
                while (temp := smoothed_temp()) is not None and temp > self.resume_at:
                    time.sleep(5)
                self.paused_seconds += time.time() - t0
        self._last = time.time()
