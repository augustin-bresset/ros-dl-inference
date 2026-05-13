"""Backend registry — maps config backend name to implementation class."""

from __future__ import annotations

from ..core.base_backend import BaseBackend
from ..core.config import InferenceConfig


def get_backend(config: InferenceConfig) -> BaseBackend:
    name = config.model.backend
    if name == "pytorch":
        from .pytorch_backend import PyTorchScriptBackend
        return PyTorchScriptBackend(config)
    elif name == "pytorch_source":
        from .pytorch_backend import PyTorchSourceBackend
        return PyTorchSourceBackend(config)
    elif name == "torchsparse":
        from .torchsparse_backend import TorchSparseBackend
        return TorchSparseBackend(config)
    elif name == "tensorrt":
        from .tensorrt_backend import TensorRTBackend
        return TensorRTBackend(config)
    elif name == "onnx":
        from .onnx_backend import ONNXBackend
        return ONNXBackend(config)
    elif name == "jax":
        from .jax_backend import JAXBackend
        return JAXBackend(config)
    else:
        raise ValueError(
            f"Unknown backend: '{name}'. "
            f"Supported: pytorch, pytorch_source, torchsparse, tensorrt, onnx, jax"
        )
