"""
Inference engine: ties backend, pipeline, batching, and profiling together.

Architecture:
  - Single-threaded synchronous path (latency-optimal for robotics).
  - Optional dynamic batching via a background thread + queue.
  - Pre/post processing pipelines run on CPU.
  - Profiler tracks each stage independently.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .base_backend import BaseBackend, InferenceResult, ModelInfo
from .config import InferenceConfig
from .pipeline import ProcessingPipeline, build_postprocessing, build_preprocessing
from ..utils.profiler import RollingProfiler, Timer


class InferenceEngine:
    def __init__(self, backend: BaseBackend, config: InferenceConfig) -> None:
        self._backend = backend
        self._config = config
        self._profiler = RollingProfiler(window=500)

        self._preproc: Optional[ProcessingPipeline] = build_preprocessing(config.preprocessing)
        self._postproc: Optional[ProcessingPipeline] = build_postprocessing(config.postprocessing)

        # Dynamic batching state
        self._batch_thread: Optional[threading.Thread] = None
        self._batch_queue: Optional[queue.Queue] = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._backend.load()

        n = self._config.optimization.warmup_iterations
        if n > 0:
            dummy = self._backend._make_dummy_inputs()
            for _ in range(n):
                self._backend.infer(dummy)

        if self._config.optimization.dynamic_batching:
            self._start_batch_thread()

        self._running = True

    def stop(self) -> None:
        self._running = False
        if self._batch_thread is not None:
            self._batch_queue.put(None)  # sentinel
            self._batch_thread.join(timeout=2.0)
        self._backend.unload()

    # ------------------------------------------------------------------
    # Synchronous inference (low-latency path)
    # ------------------------------------------------------------------

    def infer_sync(
        self, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        with Timer(self._profiler, "preprocess"):
            if self._preproc is not None:
                inputs = {k: self._preproc(v) for k, v in inputs.items()}

        with Timer(self._profiler, "inference"):
            outputs = self._backend.infer(inputs)

        with Timer(self._profiler, "postprocess"):
            if self._postproc is not None:
                outputs = {k: self._postproc(v) for k, v in outputs.items()}

        return outputs

    # ------------------------------------------------------------------
    # Dynamic batching path
    # ------------------------------------------------------------------

    def submit_async(
        self,
        inputs: Dict[str, np.ndarray],
        callback: Callable[[Dict[str, np.ndarray]], None],
    ) -> None:
        """Submit inputs for batched async inference. callback is called on the batch thread."""
        if self._batch_queue is None:
            raise RuntimeError("Dynamic batching not enabled. Call infer_sync instead.")
        self._batch_queue.put((inputs, callback))

    def _start_batch_thread(self) -> None:
        self._batch_queue = queue.Queue(maxsize=64)
        self._batch_thread = threading.Thread(
            target=self._batch_worker,
            name="rl_inference_batcher",
            daemon=True,
        )
        self._batch_thread.start()

    def _batch_worker(self) -> None:
        cfg = self._config.optimization
        timeout_s = cfg.batch_timeout_ms / 1000.0
        max_bs = cfg.max_batch_size

        pending: List[Tuple[Dict[str, np.ndarray], Callable]] = []

        def flush():
            if not pending:
                return
            batch_inputs = self._collate([inp for inp, _ in pending])
            with Timer(self._profiler, "inference_batched"):
                batch_outputs = self._backend.infer(batch_inputs)
            split = self._split(batch_outputs, [next(iter(inp.values())).shape[0] for inp, _ in pending])
            for out, (_, cb) in zip(split, pending):
                if self._postproc is not None:
                    out = {k: self._postproc(v) for k, v in out.items()}
                cb(out)
            pending.clear()

        while True:
            try:
                deadline = time.monotonic() + timeout_s
                item = self._batch_queue.get(timeout=timeout_s)
                if item is None:
                    flush()
                    return
                inputs, cb = item
                if self._preproc is not None:
                    inputs = {k: self._preproc(v) for k, v in inputs.items()}
                pending.append((inputs, cb))

                # Drain more items until deadline or max batch
                while len(pending) < max_bs:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        item2 = self._batch_queue.get(timeout=remaining)
                        if item2 is None:
                            flush()
                            return
                        inp2, cb2 = item2
                        if self._preproc is not None:
                            inp2 = {k: self._preproc(v) for k, v in inp2.items()}
                        pending.append((inp2, cb2))
                    except queue.Empty:
                        break

                flush()
            except queue.Empty:
                flush()

    @staticmethod
    def _collate(items: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = items[0].keys()
        return {k: np.concatenate([item[k] for item in items], axis=0) for k in keys}

    @staticmethod
    def _split(
        batch: Dict[str, np.ndarray], sizes: List[int]
    ) -> List[Dict[str, np.ndarray]]:
        result = []
        offsets = [0]
        for s in sizes:
            offsets.append(offsets[-1] + s)
        for i, s in enumerate(sizes):
            result.append({k: v[offsets[i]: offsets[i + 1]] for k, v in batch.items()})
        return result

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def profiler(self) -> RollingProfiler:
        return self._profiler

    def get_info(self) -> ModelInfo:
        return self._backend.get_info()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
