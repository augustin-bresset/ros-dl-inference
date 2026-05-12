"""Configuration loading and validation for ros_rl_inference."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class DeviceConfig:
    type: str = "cpu"          # cpu, cuda, cuda:0, cuda:1, ...
    memory_fraction: float = 1.0
    allow_growth: bool = True


@dataclass
class OptimizationConfig:
    fp16: bool = False
    int8: bool = False
    batch_size: int = 1
    max_batch_size: int = 1
    dynamic_batching: bool = False
    batch_timeout_ms: float = 5.0
    warmup_iterations: int = 3
    use_cuda_graphs: bool = False
    use_pinned_memory: bool = True
    num_threads: int = 1


@dataclass
class PreprocessConfig:
    enabled: bool = False
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PostprocessConfig:
    enabled: bool = False
    steps: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ROSConfig:
    input_topic: str = "/observation"
    output_topic: str = "/action"
    info_service: str = "~/get_model_info"
    queue_size: int = 1
    use_shared_memory: bool = False
    shared_memory_key: str = "ros_rl_inference"
    latched: bool = False
    publish_latency: bool = False
    latency_topic: str = "~/latency"


@dataclass
class ModelConfig:
    path: str = ""
    backend: str = "pytorch"   # pytorch, jax, tensorrt, onnx
    input_names: List[str] = field(default_factory=lambda: ["obs"])
    output_names: List[str] = field(default_factory=lambda: ["action"])
    input_shapes: Dict[str, List[int]] = field(default_factory=dict)
    output_shapes: Dict[str, List[int]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    preprocessing: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocessing: PostprocessConfig = field(default_factory=PostprocessConfig)
    ros: ROSConfig = field(default_factory=ROSConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "InferenceConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        return cls._from_dict(raw or {})

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "InferenceConfig":
        def _load(cls_, data):
            if data is None:
                return cls_()
            fields = {f.name for f in cls_.__dataclass_fields__.values()}
            return cls_(**{k: v for k, v in data.items() if k in fields})

        model_raw = d.get("model", {})
        model = ModelConfig(
            path=model_raw.get("path", ""),
            backend=model_raw.get("backend", "pytorch"),
            input_names=model_raw.get("input_names", ["obs"]),
            output_names=model_raw.get("output_names", ["action"]),
            input_shapes=model_raw.get("input_shapes", {}),
            output_shapes=model_raw.get("output_shapes", {}),
            metadata=model_raw.get("metadata", {}),
        )

        device_raw = d.get("device", {})
        device = DeviceConfig(
            type=device_raw.get("type", "cpu"),
            memory_fraction=device_raw.get("memory_fraction", 1.0),
            allow_growth=device_raw.get("allow_growth", True),
        )

        opt_raw = d.get("optimization", {})
        optimization = OptimizationConfig(
            fp16=opt_raw.get("fp16", False),
            int8=opt_raw.get("int8", False),
            batch_size=opt_raw.get("batch_size", 1),
            max_batch_size=opt_raw.get("max_batch_size", 1),
            dynamic_batching=opt_raw.get("dynamic_batching", False),
            batch_timeout_ms=opt_raw.get("batch_timeout_ms", 5.0),
            warmup_iterations=opt_raw.get("warmup_iterations", 3),
            use_cuda_graphs=opt_raw.get("use_cuda_graphs", False),
            use_pinned_memory=opt_raw.get("use_pinned_memory", True),
            num_threads=opt_raw.get("num_threads", 1),
        )

        pre_raw = d.get("preprocessing", {})
        preprocessing = PreprocessConfig(
            enabled=pre_raw.get("enabled", False),
            steps=pre_raw.get("steps", []),
        )

        post_raw = d.get("postprocessing", {})
        postprocessing = PostprocessConfig(
            enabled=post_raw.get("enabled", False),
            steps=post_raw.get("steps", []),
        )

        ros_raw = d.get("ros", {})
        ros = ROSConfig(
            input_topic=ros_raw.get("input_topic", "/observation"),
            output_topic=ros_raw.get("output_topic", "/action"),
            info_service=ros_raw.get("info_service", "~/get_model_info"),
            queue_size=ros_raw.get("queue_size", 1),
            use_shared_memory=ros_raw.get("use_shared_memory", False),
            shared_memory_key=ros_raw.get("shared_memory_key", "ros_rl_inference"),
            latched=ros_raw.get("latched", False),
            publish_latency=ros_raw.get("publish_latency", False),
            latency_topic=ros_raw.get("latency_topic", "~/latency"),
        )

        return cls(
            model=model,
            device=device,
            optimization=optimization,
            preprocessing=preprocessing,
            postprocessing=postprocessing,
            ros=ros,
        )

    def validate(self) -> None:
        if not self.model.path:
            raise ValueError("model.path must be set")
        if self.model.backend not in ("pytorch", "pytorch_source", "jax", "tensorrt", "onnx", "torchsparse"):
            raise ValueError(f"Unknown backend: {self.model.backend}")
        if self.optimization.fp16 and self.optimization.int8:
            raise ValueError("fp16 and int8 cannot both be enabled")
        if not (0.0 < self.device.memory_fraction <= 1.0):
            raise ValueError("device.memory_fraction must be in (0, 1]")
        if self.optimization.batch_size > self.optimization.max_batch_size:
            raise ValueError("batch_size cannot exceed max_batch_size")
