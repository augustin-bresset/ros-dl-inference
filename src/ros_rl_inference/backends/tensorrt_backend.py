"""
TensorRT backend — lowest latency on NVIDIA hardware.

Supports:
  - Pre-built .trt / .engine files
  - ONNX-to-TRT conversion on the fly
  - FP16 and INT8 precision
  - CUDA stream-based async execution
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.base_backend import BaseBackend, ModelInfo
from ..core.config import InferenceConfig


class TensorRTBackend(BaseBackend):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self._engine = None
        self._context = None
        self._stream = None
        self._bindings: list = []
        self._host_inputs: Dict[str, np.ndarray] = {}
        self._host_outputs: Dict[str, np.ndarray] = {}
        self._device_buffers: list = []

    def load(self) -> None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "TensorRT backend requires tensorrt and pycuda. "
                f"Install via: pip install tensorrt pycuda\n{e}"
            )

        model_path = Path(self.config.model.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)

        # Build or load engine
        suffix = model_path.suffix.lower()
        if suffix in (".trt", ".engine"):
            self._engine = self._load_engine(model_path, trt_logger)
        elif suffix == ".onnx":
            self._engine = self._build_from_onnx(model_path, trt_logger, trt)
        else:
            raise ValueError(f"TensorRT backend expects .trt/.engine or .onnx, got: {suffix}")

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()
        self._setup_bindings(cuda, trt)
        self._loaded = True

    def _load_engine(self, path: Path, logger) -> object:
        import tensorrt as trt

        runtime = trt.Runtime(logger)
        with open(path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())

    def _build_from_onnx(self, path: Path, logger, trt) -> object:
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        with open(path, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"ONNX parse failed:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

        if self.config.optimization.fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        if self.config.optimization.int8 and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)

        bs = self.config.optimization.max_batch_size
        if bs > 1:
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                shape = list(inp.shape[1:])  # drop batch
                profile.set_shape(inp.name, [1, *shape], [bs, *shape], [bs, *shape])
            config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        runtime = trt.Runtime(logger)
        return runtime.deserialize_cuda_engine(serialized)

    def _setup_bindings(self, cuda, trt) -> None:
        self._bindings = []
        self._host_inputs = {}
        self._host_outputs = {}
        self._device_buffers = []

        bs = self.config.optimization.batch_size
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = self._engine.get_tensor_shape(name)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            # Replace dynamic dims with batch_size
            full_shape = tuple(bs if d == -1 else d for d in shape)
            host_buf = cuda.pagelocked_empty(full_shape, dtype=dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)
            self._device_buffers.append(dev_buf)
            self._bindings.append(int(dev_buf))

            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._host_inputs[name] = host_buf
            else:
                self._host_outputs[name] = host_buf

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        import pycuda.driver as cuda

        # Copy inputs to pinned host memory then to device
        for name, arr in inputs.items():
            if name not in self._host_inputs:
                continue
            np.copyto(self._host_inputs[name], arr)

        for name, host_buf in self._host_inputs.items():
            idx = list(self._host_inputs.keys()).index(name)
            cuda.memcpy_htod_async(self._device_buffers[idx], host_buf, self._stream)

        self._context.execute_async_v3(stream_handle=self._stream.handle)

        n_inputs = len(self._host_inputs)
        for i, (name, host_buf) in enumerate(self._host_outputs.items()):
            cuda.memcpy_dtoh_async(host_buf, self._device_buffers[n_inputs + i], self._stream)

        self._stream.synchronize()
        return {name: np.copy(buf) for name, buf in self._host_outputs.items()}

    def warmup(self) -> None:
        dummy = self._make_dummy_inputs()
        for _ in range(self.config.optimization.warmup_iterations):
            self.infer(dummy)

    def get_info(self) -> ModelInfo:
        try:
            import tensorrt as trt
            trt_ver = trt.__version__
        except Exception:
            trt_ver = "unknown"

        return ModelInfo(
            backend="tensorrt",
            model_path=self.config.model.path,
            input_names=list(self._host_inputs.keys()),
            output_names=list(self._host_outputs.keys()),
            input_shapes={k: v.shape for k, v in self._host_inputs.items()},
            output_shapes={k: v.shape for k, v in self._host_outputs.items()},
            device=self.config.device.type,
            fp16=self.config.optimization.fp16,
            int8=self.config.optimization.int8,
            extra={"tensorrt_version": trt_ver},
        )

    def unload(self) -> None:
        self._host_inputs.clear()
        self._host_outputs.clear()
        for buf in self._device_buffers:
            try:
                buf.free()
            except Exception:
                pass
        self._device_buffers.clear()
        self._context = None
        self._engine = None
        self._stream = None
        self._loaded = False
