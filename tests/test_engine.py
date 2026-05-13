"""Tests for InferenceEngine using a mock backend."""

import threading
import time
from typing import Dict

import numpy as np
import pytest

from ros_dl_inference.core.base_backend import BaseBackend, ModelInfo
from ros_dl_inference.core.config import InferenceConfig
from ros_dl_inference.core.engine import InferenceEngine


class MockBackend(BaseBackend):
    """Multiplies every input by 2 — fast, no ML framework needed."""

    def __init__(self, config, latency_s=0.0):
        super().__init__(config)
        self._latency_s = latency_s
        self.infer_count = 0

    def load(self):
        self._loaded = True

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self._latency_s:
            time.sleep(self._latency_s)
        self.infer_count += 1
        return {k: v * 2.0 for k, v in inputs.items()}

    def warmup(self):
        pass

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            backend="mock",
            model_path="",
            input_names=["obs"],
            output_names=["action"],
            input_shapes={"obs": (1, 4)},
            output_shapes={"action": (1, 4)},
            device="cpu",
            fp16=False,
            int8=False,
        )

    def unload(self):
        self._loaded = False


def make_config(dynamic_batching=False, max_batch=4, timeout_ms=5.0):
    import yaml, tempfile, os
    d = {
        "model": {"path": "/fake/model.pt", "backend": "pytorch",
                  "input_names": ["obs"], "output_names": ["action"]},
        "optimization": {
            "batch_size": 1, "max_batch_size": max_batch,
            "dynamic_batching": dynamic_batching,
            "batch_timeout_ms": timeout_ms,
            "warmup_iterations": 0,
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(d, f)
        return InferenceConfig.from_yaml(f.name)


def test_sync_infer():
    cfg = make_config()
    backend = MockBackend(cfg)
    with InferenceEngine(backend, cfg) as engine:
        obs = np.ones((1, 4), dtype=np.float32)
        out = engine.infer_sync({"obs": obs})
        np.testing.assert_allclose(out["obs"], obs * 2.0)


def test_profiler_records():
    cfg = make_config()
    backend = MockBackend(cfg, latency_s=0.001)
    with InferenceEngine(backend, cfg) as engine:
        obs = np.ones((1, 4), dtype=np.float32)
        for _ in range(10):
            engine.infer_sync({"obs": obs})
        stats = engine.profiler.stats("inference")
    assert stats is not None
    assert stats.count == 10
    assert stats.mean_ms > 0


def test_dynamic_batching():
    cfg = make_config(dynamic_batching=True, max_batch=4, timeout_ms=50.0)
    backend = MockBackend(cfg, latency_s=0.005)
    results = {}
    event = threading.Event()

    with InferenceEngine(backend, cfg) as engine:
        def cb(name):
            def _cb(out):
                results[name] = out
                if len(results) == 3:
                    event.set()
            return _cb

        for i in range(3):
            obs = np.full((1, 4), float(i), dtype=np.float32)
            engine.submit_async({"obs": obs}, cb(f"req_{i}"))

        event.wait(timeout=2.0)

    assert len(results) == 3
    for i in range(3):
        np.testing.assert_allclose(
            results[f"req_{i}"]["obs"],
            np.full((1, 4), float(i) * 2.0, dtype=np.float32),
        )


def test_engine_get_info():
    cfg = make_config()
    backend = MockBackend(cfg)
    with InferenceEngine(backend, cfg) as engine:
        info = engine.get_info()
    assert info.backend == "mock"
    assert "obs" in info.input_names
