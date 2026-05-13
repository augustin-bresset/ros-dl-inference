"""
PyTorch backends.

Hierarchy:
  PyTorchBackend         — base class: device setup, pinned memory, standard infer()
  ├── PyTorchScriptBackend  — loads TorchScript / torch.jit exported models
  └── PyTorchSourceBackend  — loads from source code (dynamic import) + checkpoint
"""

from __future__ import annotations

import importlib
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.base_backend import BaseBackend, ModelInfo
from ..core.config import InferenceConfig


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class PyTorchBackend(BaseBackend):
    """
    Shared utilities for all PyTorch backends.

    Subclasses must implement load(). They should call _setup_device() from
    load() after the model is ready to move to the target device.
    """

    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self._model = None
        self._device = None
        self._dtype = None
        self._pinned_inputs: Dict[str, "torch.Tensor"] = {}

    @abstractmethod
    def load(self) -> None: ...

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _setup_device(self) -> None:
        """Initialize _device, _dtype, and optionally pinned memory buffers."""
        import torch
        self._device = torch.device(self.config.device.type)
        self._dtype = torch.float16 if self.config.optimization.fp16 else torch.float32
        if self.config.optimization.use_pinned_memory and self._device.type == "cuda":
            self._preallocate_pinned()

    def _preallocate_pinned(self) -> None:
        import torch
        bs = self.config.optimization.batch_size
        for name in self.config.model.input_names:
            shape = self.config.model.input_shapes.get(name)
            if shape:
                t = torch.empty([bs, *shape], dtype=self._dtype, pin_memory=True)
                self._pinned_inputs[name] = t

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import torch

        with torch.inference_mode():
            tensors = {}
            for name, arr in inputs.items():
                if name in self._pinned_inputs:
                    pinned = self._pinned_inputs[name]
                    if pinned.shape == arr.shape:
                        pinned.copy_(torch.from_numpy(arr))
                        tensors[name] = pinned.to(self._device, non_blocking=True)
                        continue
                t = torch.from_numpy(arr)
                if self._dtype == torch.float16:
                    t = t.half()
                tensors[name] = t.to(self._device, non_blocking=True)

            out = (
                self._model(next(iter(tensors.values())))
                if len(tensors) == 1
                else self._model(**tensors)
            )

            if isinstance(out, torch.Tensor):
                return {self.config.model.output_names[0]: out.cpu().numpy()}
            elif isinstance(out, (tuple, list)):
                return {
                    name: t.cpu().numpy()
                    for name, t in zip(self.config.model.output_names, out)
                }
            elif isinstance(out, dict):
                return {k: v.cpu().numpy() for k, v in out.items()}
            else:
                raise TypeError(f"Unexpected model output type: {type(out)}")

    def warmup(self) -> None:
        import torch
        dummy = self._make_dummy_inputs()
        n = self.config.optimization.warmup_iterations
        with torch.inference_mode():
            for _ in range(n):
                self.infer(dummy)
        if self._device and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def get_info(self) -> ModelInfo:
        import torch
        extra: Dict[str, Any] = {}
        try:
            extra["torch_version"] = torch.__version__
            if self._device and self._device.type == "cuda":
                extra["cuda_device"] = torch.cuda.get_device_name(self._device)
        except Exception:
            pass
        return ModelInfo(
            backend=self.config.model.backend,
            model_path=self.config.model.path,
            input_names=self.config.model.input_names,
            output_names=self.config.model.output_names,
            input_shapes={k: tuple(v) for k, v in self.config.model.input_shapes.items()},
            output_shapes={k: tuple(v) for k, v in self.config.model.output_shapes.items()},
            device=str(self._device),
            fp16=self.config.optimization.fp16,
            int8=False,
            extra=extra,
        )

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        self._pinned_inputs.clear()
        self._loaded = False
        try:
            import torch.cuda
            if self._device and self._device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TorchScript mode  (backend: 'pytorch')
# ---------------------------------------------------------------------------

class PyTorchScriptBackend(PyTorchBackend):
    """
    Loads TorchScript-exported models (torch.jit.load).

    Supports fp16 and torch.compile (PyTorch >= 2.0).
    """

    def load(self) -> None:
        import torch

        self._setup_device()

        model_path = Path(self.config.model.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = torch.jit.load(str(model_path), map_location=self._device)
        model.eval()

        if self.config.optimization.fp16 and self._device.type != "cpu":
            model = model.half()

        if hasattr(torch, "compile") and self._device.type == "cuda":
            model = torch.compile(model, mode="reduce-overhead")

        self._model = model
        self._loaded = True


# ---------------------------------------------------------------------------
# Source-code mode  (backend: 'pytorch_source')
# ---------------------------------------------------------------------------

class PyTorchSourceBackend(PyTorchBackend):
    """
    Loads any PyTorch model from source code + a checkpoint file.

    The model class is imported dynamically at runtime, so no code change is
    needed when switching models — only the YAML config changes.

    Config extras (under model.metadata):
      module:             str   dotted path to the model class
                                e.g. "my_pkg.models.net.MyNet"
      project_root:       str   path prepended to sys.path for import resolution
      constructor_kwargs: dict  passed verbatim to the model class __init__
      state_dict_key:     str   optional key to extract from the checkpoint dict
                                e.g. "model", "state_dict", "model_state_dict"
      strict_load:        bool  default True — passed to load_state_dict
    """

    def load(self) -> None:
        import torch
        meta = self.config.model.metadata

        project_root: Optional[str] = meta.get("project_root")
        if project_root and project_root not in sys.path:
            sys.path.insert(0, project_root)

        module_path: Optional[str] = meta.get("module")
        if not module_path:
            raise ValueError(
                "model.metadata.module must be set to the dotted class path, "
                "e.g. 'my_pkg.models.net.MyNet'"
            )

        module_str, class_name = module_path.rsplit(".", 1)
        try:
            mod = importlib.import_module(module_str)
        except ModuleNotFoundError as e:
            raise ImportError(
                f"Could not import '{module_str}'. "
                "Check model.metadata.project_root and model.metadata.module.\n"
                f"sys.path: {sys.path}\n{e}"
            ) from e

        if not hasattr(mod, class_name):
            raise ImportError(f"Module '{module_str}' has no attribute '{class_name}'")

        cls = getattr(mod, class_name)
        constructor_kwargs: Dict[str, Any] = meta.get("constructor_kwargs", {})
        model = cls(**constructor_kwargs)

        model_path = Path(self.config.model.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

        self._setup_device()

        ckpt = torch.load(str(model_path), map_location=self._device)
        state_dict_key: Optional[str] = meta.get("state_dict_key")
        if state_dict_key:
            if state_dict_key not in ckpt:
                raise KeyError(
                    f"state_dict_key='{state_dict_key}' not found in checkpoint. "
                    f"Available keys: {list(ckpt.keys())}"
                )
            ckpt = ckpt[state_dict_key]

        strict: bool = bool(meta.get("strict_load", True))
        model.load_state_dict(ckpt, strict=strict)

        self._move_to_device(model, self._device)
        model.eval()
        self._model = model
        self._loaded = True

    def _move_to_device(
        self, model: "torch.nn.Module", device: "torch.device"
    ) -> None:
        """Override if the model uses a non-standard device placement API."""
        model.to(device)
