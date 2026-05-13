"""
ONNX Runtime backend — CPU/GPU portable, works without PyTorch/TF.

Supports all ONNX Runtime execution providers:
  - CPUExecutionProvider
  - CUDAExecutionProvider
  - TensorrtExecutionProvider
  - OpenVINOExecutionProvider
  - CoreMLExecutionProvider (Apple Silicon)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.base_backend import BaseBackend, ModelInfo
from ..core.config import InferenceConfig


_DEVICE_TO_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
}


class ONNXBackend(BaseBackend):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__(config)
        self._session = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("ONNX backend requires: pip install onnxruntime-gpu")

        model_path = Path(self.config.model.path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        device = self.config.device.type.split(":")[0]  # strip device index
        providers = _DEVICE_TO_PROVIDERS.get(device, ["CPUExecutionProvider"])

        # Provider options for CUDA
        provider_options = []
        for p in providers:
            if p == "CUDAExecutionProvider":
                dev_id = 0
                if ":" in self.config.device.type:
                    dev_id = int(self.config.device.type.split(":")[1])
                provider_options.append({
                    "device_id": dev_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "gpu_mem_limit": int(
                        8 * 1024**3 * self.config.device.memory_fraction
                    ),
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                })
            elif p == "TensorrtExecutionProvider":
                provider_options.append({
                    "trt_fp16_enable": self.config.optimization.fp16,
                    "trt_int8_enable": self.config.optimization.int8,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "/tmp/trt_cache",
                })
            else:
                provider_options.append({})

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = self.config.optimization.num_threads
        sess_opts.inter_op_num_threads = 1

        if self.config.optimization.use_pinned_memory:
            sess_opts.enable_mem_pattern = True
            sess_opts.enable_cpu_mem_arena = True

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=providers,
            provider_options=provider_options,
        )

        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._loaded = True

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Filter to only the inputs the model needs
        feed = {k: v for k, v in inputs.items() if k in self._input_names}
        outputs = self._session.run(self._output_names, feed)
        return dict(zip(self._output_names, outputs))

    def warmup(self) -> None:
        dummy = self._make_dummy_inputs()
        for _ in range(self.config.optimization.warmup_iterations):
            self.infer(dummy)

    def get_info(self) -> ModelInfo:
        import onnxruntime as ort

        input_shapes: Dict[str, Tuple] = {}
        output_shapes: Dict[str, Tuple] = {}
        for inp in self._session.get_inputs():
            input_shapes[inp.name] = tuple(
                d if isinstance(d, int) else -1 for d in inp.shape
            )
        for out in self._session.get_outputs():
            output_shapes[out.name] = tuple(
                d if isinstance(d, int) else -1 for d in out.shape
            )

        return ModelInfo(
            backend="onnx",
            model_path=self.config.model.path,
            input_names=self._input_names,
            output_names=self._output_names,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            device=self.config.device.type,
            fp16=self.config.optimization.fp16,
            int8=self.config.optimization.int8,
            extra={
                "ort_version": ort.__version__,
                "providers": self._session.get_providers(),
            },
        )

    def unload(self) -> None:
        self._session = None
        self._loaded = False
