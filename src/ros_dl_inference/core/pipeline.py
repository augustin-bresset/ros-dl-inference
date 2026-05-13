"""Pre/post processing pipeline with composable steps."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import PostprocessConfig, PreprocessConfig


# ------------------------------------------------------------------
# Built-in pre-processing steps
# ------------------------------------------------------------------

def _normalize(arr: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
    mean_ = np.array(mean, dtype=np.float32)
    std_ = np.array(std, dtype=np.float32)
    return (arr - mean_) / (std_ + 1e-8)


def _clip(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip(arr, low, high)


def _cast(arr: np.ndarray, dtype: str) -> np.ndarray:
    return arr.astype(dtype)


def _reshape(arr: np.ndarray, shape: List[int]) -> np.ndarray:
    return arr.reshape(shape)


def _scale(arr: np.ndarray, factor: float) -> np.ndarray:
    return arr * factor


def _add_batch_dim(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr[np.newaxis, :]
    return arr


# ------------------------------------------------------------------
# Step registry
# ------------------------------------------------------------------

_STEP_REGISTRY: Dict[str, Callable] = {
    "normalize": _normalize,
    "clip": _clip,
    "cast": _cast,
    "reshape": _reshape,
    "scale": _scale,
    "add_batch_dim": _add_batch_dim,
}


def register_step(name: str, fn: Callable) -> None:
    """Register a custom pre/post processing step."""
    _STEP_REGISTRY[name] = fn


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class ProcessingPipeline:
    def __init__(self, steps: List[Dict[str, Any]]) -> None:
        self._steps: List[tuple[Callable, Dict[str, Any]]] = []
        for step_cfg in steps:
            step_cfg = dict(step_cfg)
            name = step_cfg.pop("type")
            fn = _STEP_REGISTRY.get(name)
            if fn is None:
                raise ValueError(f"Unknown processing step: '{name}'. "
                                 f"Available: {list(_STEP_REGISTRY)}")
            self._steps.append((fn, step_cfg))

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        for fn, kwargs in self._steps:
            arr = fn(arr, **kwargs)
        return arr

    def __len__(self) -> int:
        return len(self._steps)


def build_preprocessing(cfg: PreprocessConfig) -> Optional[ProcessingPipeline]:
    if not cfg.enabled or not cfg.steps:
        return None
    return ProcessingPipeline(cfg.steps)


def build_postprocessing(cfg: PostprocessConfig) -> Optional[ProcessingPipeline]:
    if not cfg.enabled or not cfg.steps:
        return None
    return ProcessingPipeline(cfg.steps)
