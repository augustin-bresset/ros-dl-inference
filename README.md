# ros_dl_inference

Optimized model inference package for **ROS 1** and **ROS 2**, designed for real-time robot control.

## Features

- **Pluggable backends** — PyTorch, TensorRT, ONNX Runtime, JAX. Switch with one config line.
- **Ultra-low latency** — pinned memory, CUDA streams, `torch.compile`, TensorRT FP16/INT8.
- **Dynamic batching** — optional background batcher with configurable timeout.
- **Pre/post processing pipelines** — composable YAML-defined steps (normalize, clip, scale, …).
- **Rolling profiler** — per-stage latency stats (mean, p50, p95, p99) with zero malloc overhead.
- **Buffer pool** — reusable numpy arrays to eliminate GC pressure during inference.
- **ROS 1 & 2** — single codebase, thin node wrappers for each.

## Architecture

```
ros_dl_inference/
├── src/ros_dl_inference/
│   ├── core/
│   │   ├── config.py          # YAML config loading + validation
│   │   ├── base_backend.py    # Abstract backend interface
│   │   ├── engine.py          # Inference engine (sync + async batching)
│   │   └── pipeline.py        # Pre/post processing pipeline
│   ├── backends/
│   │   ├── pytorch_backend.py # TorchScript + torch.compile + pinned memory
│   │   ├── tensorrt_backend.py# TRT engine or ONNX-to-TRT, CUDA streams
│   │   ├── onnx_backend.py    # ONNX Runtime (CPU/CUDA/TRT/OpenVINO)
│   │   └── jax_backend.py     # JAX jit, orbax checkpoints
│   ├── utils/
│   │   ├── profiler.py        # Rolling latency profiler
│   │   └── memory_pool.py     # Pre-allocated buffer pool
│   ├── ros1/node.py           # ROS 1 subscriber/publisher
│   └── ros2/node.py           # ROS 2 subscriber/publisher
├── nodes/
│   ├── inference_node_ros1    # ROS 1 entrypoint
│   └── inference_node_ros2    # ROS 2 entrypoint
├── config/                    # Example configs for each backend
├── launch/ros1/               # .launch file
├── launch/ros2/               # .launch.py file
├── msg/                       # InferenceInput.msg, InferenceOutput.msg
└── srv/                       # GetModelInfo.srv
```

## Quick Start

### 1. Install

```bash
# Core only (no ROS)
pip install -e ".[pytorch]"           # PyTorch backend
pip install -e ".[onnx]"              # ONNX Runtime
pip install -e ".[jax]"               # JAX
pip install -e ".[tensorrt]"          # TensorRT + pycuda
```

### 2. Build with ROS

**ROS 1 (catkin):**
```bash
cd ~/catkin_ws/src
ln -s /path/to/ros_dl_inference .
cd ~/catkin_ws && catkin_make
```

**ROS 2 (colcon):**
```bash
cd ~/ros2_ws/src
ln -s /path/to/ros_dl_inference .
cd ~/ros2_ws && colcon build --packages-select ros_dl_inference
```

### 3. Run

**ROS 1:**
```bash
rosrun ros_dl_inference inference_node_ros1 --config /path/to/config.yaml
# or
roslaunch ros_dl_inference inference.launch config:=/path/to/config.yaml
```

**ROS 2:**
```bash
ros2 run ros_dl_inference inference_node_ros2 --config /path/to/config.yaml
# or
ros2 launch ros_dl_inference inference.launch.py config:=/path/to/config.yaml
```

## Configuration

All behaviour is controlled by a single YAML file.

```yaml
model:
  path: /path/to/policy.pt     # model file
  backend: pytorch              # pytorch | tensorrt | onnx | jax
  input_names: [obs]
  output_names: [action]
  input_shapes:
    obs: [48]
  output_shapes:
    action: [12]

device:
  type: cuda:0                  # cpu | cuda | cuda:0 | cuda:1
  memory_fraction: 0.5

optimization:
  fp16: true                    # half precision (2x faster on modern GPUs)
  batch_size: 1
  max_batch_size: 4
  dynamic_batching: false       # enable background batching thread
  batch_timeout_ms: 5.0
  warmup_iterations: 10
  use_pinned_memory: true
  num_threads: 1

preprocessing:
  enabled: true
  steps:
    - type: clip
      low: -10.0
      high: 10.0

postprocessing:
  enabled: true
  steps:
    - type: clip
      low: -1.0
      high: 1.0

ros:
  input_topic: /robot/observation
  output_topic: /robot/action
  queue_size: 1
  publish_latency: true
  latency_topic: ~/latency_ms
```

See `config/` for full examples per backend.

## Custom Backend

Subclass `BaseBackend` and call `get_backend` with your registered name:

```python
from ros_dl_inference.core.base_backend import BaseBackend, ModelInfo
from ros_dl_inference.backends import get_backend

class MyBackend(BaseBackend):
    def load(self): ...
    def infer(self, inputs): ...
    def warmup(self): ...
    def get_info(self) -> ModelInfo: ...
    def unload(self): ...
```

## Custom Processing Step

```python
from ros_dl_inference.core.pipeline import register_step
import numpy as np

def my_transform(arr: np.ndarray, alpha: float) -> np.ndarray:
    return arr * alpha

register_step("my_transform", my_transform)
```

Then in your config:
```yaml
preprocessing:
  enabled: true
  steps:
    - type: my_transform
      alpha: 0.5
```

## Performance Tips

| Scenario | Recommendation |
|---|---|
| NVIDIA GPU | TensorRT backend + fp16: true |
| Jetson Nano/Orin | TensorRT backend, consider int8 |
| CPU-only | ONNX backend, num_threads: N |
| Research / TPU | JAX backend |
| Minimal latency | disable dynamic_batching, warmup_iterations: 20 |

## Tests

```bash
python3 -m pytest tests/ -v
```
