"""Tests for the buffer pool."""

import threading

import numpy as np
import pytest

from ros_rl_inference.utils.memory_pool import BufferPool


def test_acquire_and_release():
    pool = BufferPool()
    arr = pool.acquire((4,), dtype=np.float32)
    assert arr.shape == (4,)
    assert arr.dtype == np.float32
    pool.release(arr)


def test_buffer_is_reused():
    pool = BufferPool()
    pool.preallocate((4,), dtype=np.float32, count=1)
    a = pool.acquire((4,))
    pool.release(a)
    b = pool.acquire((4,))
    assert a is b  # same object returned from pool


def test_different_shapes():
    pool = BufferPool()
    a = pool.acquire((4,))
    b = pool.acquire((8,))
    assert a.shape != b.shape
    pool.release(a)
    pool.release(b)
    a2 = pool.acquire((4,))
    assert a2 is a


def test_max_pool_size():
    pool = BufferPool(max_buffers_per_shape=2)
    bufs = [pool.acquire((4,)) for _ in range(5)]
    for b in bufs:
        pool.release(b)
    # Only 2 should be kept (pool is a LIFO stack, cap=2)
    kept = [pool.acquire((4,)) for _ in range(5)]
    # Exactly 2 of the 5 re-acquired buffers should be objects from bufs
    reused = sum(1 for k in kept if any(k is b for b in bufs))
    assert reused == 2


def test_thread_safety():
    pool = BufferPool()
    errors = []

    def worker():
        try:
            for _ in range(100):
                arr = pool.acquire((16,))
                arr[:] = 1.0
                pool.release(arr)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_clear():
    pool = BufferPool()
    arr = pool.acquire((4,))
    pool.release(arr)
    pool.clear()
    arr2 = pool.acquire((4,))
    assert arr2 is not arr
