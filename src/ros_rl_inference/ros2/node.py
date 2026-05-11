"""ROS 2 inference node implementation."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import Float32, Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from ...backends import get_backend
from ...core.config import InferenceConfig
from ...core.engine import InferenceEngine


def _best_effort_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _reliable_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        reliability=QoSReliabilityPolicy.RELIABLE,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=depth,
    )


def _make_float32_msg(arr: np.ndarray) -> Float32MultiArray:
    msg = Float32MultiArray()
    msg.layout = MultiArrayLayout()
    msg.layout.data_offset = 0
    for size in arr.shape:
        dim = MultiArrayDimension()
        dim.size = size
        dim.stride = 1
        msg.layout.dim.append(dim)
    msg.data = arr.flatten().tolist()
    return msg


def _parse_float32_msg(msg: Float32MultiArray) -> np.ndarray:
    data = np.array(msg.data, dtype=np.float32)
    if msg.layout.dim:
        shape = [d.size for d in msg.layout.dim]
        data = data.reshape(shape)
    return data


class InferenceNodeROS2(Node):
    def __init__(self, config: InferenceConfig) -> None:
        super().__init__("rl_inference_node")
        self._config = config
        self._engine: Optional[InferenceEngine] = None
        self._setup()

    def _setup(self) -> None:
        cfg = self._config

        backend = get_backend(cfg)
        self._engine = InferenceEngine(backend, cfg)
        self._engine.start()

        self.get_logger().info(
            f"Backend: {cfg.model.backend} | Model: {cfg.model.path} | Device: {cfg.device.type}"
        )
        info = self._engine.get_info()
        self.get_logger().info(
            f"Inputs: {info.input_names} | Outputs: {info.output_names}"
        )

        # Use BEST_EFFORT for sensor data (matches most sensor drivers)
        sub_qos = _best_effort_qos(cfg.ros.queue_size)
        pub_qos = _best_effort_qos(cfg.ros.queue_size)

        self._pub = self.create_publisher(
            Float32MultiArray, cfg.ros.output_topic, pub_qos
        )

        self._latency_pub: Optional[object] = None
        if cfg.ros.publish_latency:
            self._latency_pub = self.create_publisher(
                Float32, cfg.ros.latency_topic, 10
            )

        self._sub = self.create_subscription(
            Float32MultiArray,
            cfg.ros.input_topic,
            self._callback,
            sub_qos,
        )

        self.get_logger().info("Node ready.")

    def _callback(self, msg: Float32MultiArray) -> None:
        cfg = self._config
        obs = _parse_float32_msg(msg)
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]

        t0 = time.perf_counter()
        inputs = {cfg.model.input_names[0]: obs}
        outputs = self._engine.infer_sync(inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        action = next(iter(outputs.values())).squeeze()
        self._pub.publish(_make_float32_msg(action))

        if self._latency_pub is not None:
            self._latency_pub.publish(Float32(data=float(latency_ms)))

    def destroy_node(self) -> None:
        if self._engine is not None:
            self.get_logger().info(
                "Profiler summary:\n" + self._engine.profiler.summary()
            )
            self._engine.stop()
        super().destroy_node()


def main(config: InferenceConfig) -> None:
    rclpy.init()
    node = InferenceNodeROS2(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
