"""Backend registry — maps config backend name to implementation class."""

from __future__ import annotations

from typing import Dict, Type

from ..core.base_backend import BaseBackend
from ..core.config import InferenceConfig


def get_backend(config: InferenceConfig) -> BaseBackend:
    name = config.model.backend
    if name == "pytorch":
        from .pytorch_backend import PyTorchBackend
        return PyTorchBackend(config)
    elif name == "tensorrt":
        from .tensorrt_backend import TensorRTBackend
        return TensorRTBackend(config)
    elif name == "onnx":
        from .onnx_backend import ONNXBackend
        return ONNXBackend(config)
    elif name == "jax":
        from .jax_backend import JAXBackend
        return JAXBackend(config)
    elif name == "torchsparse":
        from .torchsparse_backend import TorchSparseBackend
        return TorchSparseBackend(config)
    else:
        raise ValueError(
            f"Unknown backend: '{name}'. "
            f"Supported: pytorch, tensorrt, onnx, jax, torchsparse"
        )
