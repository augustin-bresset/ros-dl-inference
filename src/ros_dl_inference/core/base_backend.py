"""Abstract backend interface for model inference."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import InferenceConfig


@dataclass
class ModelInfo:
    backend: str
    model_path: str
    input_names: List[str]
    output_names: List[str]
    input_shapes: Dict[str, Tuple[int, ...]]
    output_shapes: Dict[str, Tuple[int, ...]]
    device: str
    fp16: bool
    int8: bool
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    batch_size: int


class BaseBackend(ABC):
    """
    All backends must implement this interface.

    Thread-safety: infer() must be safe to call from a single thread.
    Backends are NOT required to be thread-safe across multiple callers.
    The engine serializes calls when needed.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._loaded = False
        self._info: Optional[ModelInfo] = None

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory and prepare for inference."""

    @abstractmethod
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on a batch of inputs.

        Args:
            inputs: dict mapping input name -> numpy array (batch first).

        Returns:
            dict mapping output name -> numpy array (batch first).
        """

    @abstractmethod
    def warmup(self) -> None:
        """Run warmup passes to JIT-compile kernels and fill caches."""

    @abstractmethod
    def get_info(self) -> ModelInfo:
        """Return static metadata about the loaded model."""

    @abstractmethod
    def unload(self) -> None:
        """Release all GPU/CPU memory held by the model."""

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_dummy_inputs(self) -> Dict[str, np.ndarray]:
        dummy: Dict[str, np.ndarray] = {}
        bs = self.config.optimization.batch_size
        for name in self.config.model.input_names:
            shape = self.config.model.input_shapes.get(name)
            if shape:
                dummy[name] = np.zeros([bs, *shape], dtype=np.float32)
            else:
                dummy[name] = np.zeros([bs, 1], dtype=np.float32)
        return dummy

    def timed_infer(self, inputs: Dict[str, np.ndarray]) -> InferenceResult:
        t0 = time.perf_counter()
        outputs = self.infer(inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        batch_size = next(iter(inputs.values())).shape[0] if inputs else 1
        return InferenceResult(outputs=outputs, latency_ms=latency_ms, batch_size=batch_size)

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.unload()
