"""Lightweight latency profiler with rolling statistics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict, Optional


@dataclass
class Stats:
    count: int = 0
    mean_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


class RollingProfiler:
    """
    Records per-stage latencies with a fixed-size rolling window.
    Thread-safe. Negligible overhead (~100 ns per record call).
    """

    def __init__(self, window: int = 1000) -> None:
        self._window = window
        self._lock = Lock()
        self._buffers: Dict[str, Deque[float]] = {}

    def record(self, stage: str, latency_ms: float) -> None:
        with self._lock:
            buf = self._buffers.setdefault(stage, deque(maxlen=self._window))
            buf.append(latency_ms)

    def stats(self, stage: str) -> Optional[Stats]:
        with self._lock:
            buf = self._buffers.get(stage)
            if not buf:
                return None
            data = sorted(buf)

        n = len(data)
        s = Stats(count=n)
        s.mean_ms = sum(data) / n
        s.min_ms = data[0]
        s.max_ms = data[-1]
        s.p50_ms = data[int(n * 0.50)]
        s.p95_ms = data[int(n * 0.95)]
        s.p99_ms = data[min(int(n * 0.99), n - 1)]
        return s

    def all_stats(self) -> Dict[str, Stats]:
        with self._lock:
            stages = list(self._buffers.keys())
        return {s: self.stats(s) for s in stages}

    def reset(self) -> None:
        with self._lock:
            self._buffers.clear()

    def summary(self) -> str:
        lines = []
        for stage, s in self.all_stats().items():
            if s:
                lines.append(
                    f"[{stage}] n={s.count} "
                    f"mean={s.mean_ms:.2f}ms "
                    f"p50={s.p50_ms:.2f}ms "
                    f"p95={s.p95_ms:.2f}ms "
                    f"p99={s.p99_ms:.2f}ms "
                    f"min={s.min_ms:.2f}ms "
                    f"max={s.max_ms:.2f}ms"
                )
        return "\n".join(lines)


class Timer:
    """Context manager for recording a stage into a profiler."""

    def __init__(self, profiler: RollingProfiler, stage: str) -> None:
        self._profiler = profiler
        self._stage = stage
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        ms = (time.perf_counter() - self._t0) * 1000.0
        self._profiler.record(self._stage, ms)
