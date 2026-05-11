"""Tests for config loading and validation."""

import textwrap
from pathlib import Path

import pytest
import yaml

from ros_rl_inference.core.config import InferenceConfig


def write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


def test_load_minimal(tmp_path):
    cfg_data = {"model": {"path": "/tmp/model.pt", "backend": "pytorch"}}
    cfg = InferenceConfig.from_yaml(write_yaml(tmp_path, cfg_data))
    assert cfg.model.backend == "pytorch"
    assert cfg.model.path == "/tmp/model.pt"
    assert cfg.device.type == "cpu"
    assert cfg.optimization.batch_size == 1


def test_load_full(tmp_path):
    cfg_data = {
        "model": {
            "path": "/tmp/model.onnx",
            "backend": "onnx",
            "input_names": ["obs"],
            "output_names": ["action"],
            "input_shapes": {"obs": [48]},
            "output_shapes": {"action": [12]},
        },
        "device": {"type": "cuda:0", "memory_fraction": 0.8},
        "optimization": {
            "fp16": True,
            "batch_size": 4,
            "max_batch_size": 8,
            "dynamic_batching": True,
            "batch_timeout_ms": 10.0,
            "warmup_iterations": 5,
            "num_threads": 2,
        },
        "preprocessing": {
            "enabled": True,
            "steps": [{"type": "clip", "low": -10.0, "high": 10.0}],
        },
        "ros": {
            "input_topic": "/obs",
            "output_topic": "/action",
            "queue_size": 1,
            "publish_latency": True,
        },
    }
    cfg = InferenceConfig.from_yaml(write_yaml(tmp_path, cfg_data))
    assert cfg.model.backend == "onnx"
    assert cfg.device.type == "cuda:0"
    assert cfg.optimization.fp16 is True
    assert cfg.optimization.max_batch_size == 8
    assert cfg.preprocessing.enabled is True
    assert len(cfg.preprocessing.steps) == 1
    assert cfg.ros.publish_latency is True


def test_validate_missing_path(tmp_path):
    cfg_data = {"model": {"backend": "pytorch"}}
    cfg = InferenceConfig.from_yaml(write_yaml(tmp_path, cfg_data))
    with pytest.raises(ValueError, match="model.path"):
        cfg.validate()


def test_validate_unknown_backend(tmp_path):
    cfg_data = {"model": {"path": "/x", "backend": "unknown_backend"}}
    cfg = InferenceConfig.from_yaml(write_yaml(tmp_path, cfg_data))
    with pytest.raises(ValueError, match="Unknown backend"):
        cfg.validate()


def test_validate_fp16_and_int8(tmp_path):
    cfg_data = {
        "model": {"path": "/x", "backend": "tensorrt"},
        "optimization": {"fp16": True, "int8": True},
    }
    cfg = InferenceConfig.from_yaml(write_yaml(tmp_path, cfg_data))
    with pytest.raises(ValueError, match="fp16 and int8"):
        cfg.validate()


def test_validate_batch_size_exceeds_max(tmp_path):
    cfg_data = {
        "model": {"path": "/x", "backend": "onnx"},
        "optimization": {"batch_size": 8, "max_batch_size": 4},
    }
    cfg = InferenceConfig.from_yaml(write_yaml(tmp_path, cfg_data))
    with pytest.raises(ValueError, match="batch_size cannot exceed"):
        cfg.validate()


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        InferenceConfig.from_yaml("/nonexistent/path/config.yaml")
