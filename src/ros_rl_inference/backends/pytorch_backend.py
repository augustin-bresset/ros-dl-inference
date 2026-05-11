"""PyTorch backend — supports TorchScript, torch.compile, and eager models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.base_backend import BaseBackend, ModelInfo
from ..core.config import InferenceConfig


class PyTorchBackend(BaseBackend):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self._model = None
        self._device = None
        self._dtype = None
        # Pinned memory buffers for fast H2D transfer
        self._pinned_inputs: Dict[str, "torch.Tensor"] = {}

    def load(self) -> None:
        import torch

        device_str = self.config.device.type
        self._device = torch.device(device_str)
        self._dtype = torch.float16 if self.config.optimization.fp16 else torch.float32

        model_path = Path(self.config.model.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Support .pt (TorchScript), .pth (state dict), or pickled module
        model = torch.jit.load(str(model_path), map_location=self._device)
        model.eval()

        if self.config.optimization.fp16 and self._device.type != "cpu":
            model = model.half()

        # torch.compile gives ~20-30% speedup on CUDA (PyTorch >= 2.0)
        if hasattr(torch, "compile") and self._device.type == "cuda":
            model = torch.compile(model, mode="reduce-overhead")

        self._model = model
        self._loaded = True

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

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import torch

        with torch.inference_mode():
            tensors = {}
            for name, arr in inputs.items():
                if name in self._pinned_inputs:
                    pinned = self._pinned_inputs[name]
                    # Reuse pinned buffer if shape matches
                    if pinned.shape == arr.shape:
                        pinned.copy_(torch.from_numpy(arr))
                        tensors[name] = pinned.to(self._device, non_blocking=True)
                        continue
                t = torch.from_numpy(arr)
                if self._dtype == torch.float16:
                    t = t.half()
                tensors[name] = t.to(self._device, non_blocking=True)

            # Models may accept positional or keyword args
            if len(tensors) == 1:
                out = self._model(next(iter(tensors.values())))
            else:
                out = self._model(**tensors)

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
        if self._device.type == "cuda":
            import torch.cuda
            torch.cuda.synchronize(self._device)

    def get_info(self) -> ModelInfo:
        import torch

        input_shapes: Dict[str, Tuple[int, ...]] = {}
        output_shapes: Dict[str, Tuple[int, ...]] = {}
        for name, shape in self.config.model.input_shapes.items():
            input_shapes[name] = tuple(shape)
        for name, shape in self.config.model.output_shapes.items():
            output_shapes[name] = tuple(shape)

        extra: Dict = {}
        try:
            extra["torch_version"] = torch.__version__
            if self._device.type == "cuda":
                extra["cuda_device"] = torch.cuda.get_device_name(self._device)
        except Exception:
            pass

        return ModelInfo(
            backend="pytorch",
            model_path=self.config.model.path,
            input_names=self.config.model.input_names,
            output_names=self.config.model.output_names,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
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
