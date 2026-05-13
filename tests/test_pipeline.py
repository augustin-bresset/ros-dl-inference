"""Tests for pre/post processing pipeline."""

import numpy as np
import pytest

from ros_dl_inference.core.pipeline import ProcessingPipeline, register_step


def test_normalize():
    pipe = ProcessingPipeline([
        {"type": "normalize", "mean": [0.5], "std": [0.5]}
    ])
    arr = np.array([[0.5]], dtype=np.float32)
    out = pipe(arr)
    np.testing.assert_allclose(out, [[0.0]], atol=1e-5)


def test_clip():
    pipe = ProcessingPipeline([{"type": "clip", "low": -1.0, "high": 1.0}])
    arr = np.array([[-5.0, 0.5, 3.0]], dtype=np.float32)
    out = pipe(arr)
    np.testing.assert_allclose(out, [[-1.0, 0.5, 1.0]])


def test_scale():
    pipe = ProcessingPipeline([{"type": "scale", "factor": 2.0}])
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = pipe(arr)
    np.testing.assert_allclose(out, [[2.0, 4.0, 6.0]])


def test_cast():
    pipe = ProcessingPipeline([{"type": "cast", "dtype": "float16"}])
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    out = pipe(arr)
    assert out.dtype == np.float16


def test_reshape():
    pipe = ProcessingPipeline([{"type": "reshape", "shape": [1, 2, 3]}])
    arr = np.zeros((1, 6), dtype=np.float32)
    out = pipe(arr)
    assert out.shape == (1, 2, 3)


def test_chain():
    pipe = ProcessingPipeline([
        {"type": "scale", "factor": 10.0},
        {"type": "clip", "low": -5.0, "high": 5.0},
    ])
    arr = np.array([[0.3, -0.8]], dtype=np.float32)
    out = pipe(arr)
    np.testing.assert_allclose(out, [[3.0, -5.0]])


def test_custom_step():
    def my_step(arr, offset):
        return arr + offset

    register_step("offset", my_step)
    pipe = ProcessingPipeline([{"type": "offset", "offset": 100.0}])
    arr = np.array([[1.0]], dtype=np.float32)
    out = pipe(arr)
    np.testing.assert_allclose(out, [[101.0]])


def test_unknown_step_raises():
    with pytest.raises(ValueError, match="Unknown processing step"):
        ProcessingPipeline([{"type": "does_not_exist"}])
