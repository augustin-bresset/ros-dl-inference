"""Pre-allocated numpy buffer pool to avoid repeated malloc/free."""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np


class BufferPool:
    """
    Thread-safe pool of pre-allocated numpy arrays.

    Avoids GC pressure from repeated allocation during inference.
    Each (shape, dtype) pair gets its own stack of reusable buffers.
    """

    def __init__(self, max_buffers_per_shape: int = 4) -> None:
        self._max = max_buffers_per_shape
        self._lock = threading.Lock()
        self._pool: Dict[Tuple, list] = {}

    def _key(self, shape: Tuple[int, ...], dtype: np.dtype) -> Tuple:
        return (shape, np.dtype(dtype).str)

    def acquire(self, shape: Tuple[int, ...], dtype=np.float32) -> np.ndarray:
        key = self._key(shape, dtype)
        with self._lock:
            stack = self._pool.get(key)
            if stack:
                return stack.pop()
        return np.empty(shape, dtype=dtype)

    def release(self, arr: np.ndarray) -> None:
        key = self._key(arr.shape, arr.dtype)
        with self._lock:
            stack = self._pool.setdefault(key, [])
            if len(stack) < self._max:
                stack.append(arr)

    def preallocate(
        self,
        shape: Tuple[int, ...],
        dtype=np.float32,
        count: int = 2,
    ) -> None:
        key = self._key(shape, dtype)
        with self._lock:
            stack = self._pool.setdefault(key, [])
            needed = max(0, count - len(stack))
            for _ in range(needed):
                stack.append(np.empty(shape, dtype=dtype))

    def clear(self) -> None:
        with self._lock:
            self._pool.clear()


_global_pool: Optional[BufferPool] = None


def get_global_pool() -> BufferPool:
    global _global_pool
    if _global_pool is None:
        _global_pool = BufferPool()
    return _global_pool
