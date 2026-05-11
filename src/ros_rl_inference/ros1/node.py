"""ROS 1 inference node implementation."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from ...backends import get_backend
from ...core.config import InferenceConfig
from ...core.engine import InferenceEngine


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


class InferenceNodeROS1:
    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._engine: Optional[InferenceEngine] = None

    def run(self) -> None:
        rospy.init_node("rl_inference_node", anonymous=False)

        cfg = self._config
        backend = get_backend(cfg)
        self._engine = InferenceEngine(backend, cfg)
        self._engine.start()

        rospy.loginfo(f"[rl_inference] Backend: {cfg.model.backend} | "
                      f"Model: {cfg.model.path} | Device: {cfg.device.type}")

        info = self._engine.get_info()
        rospy.loginfo(f"[rl_inference] Inputs: {info.input_names} | "
                      f"Outputs: {info.output_names}")

        pub = rospy.Publisher(
            cfg.ros.output_topic,
            Float32MultiArray,
            queue_size=cfg.ros.queue_size,
            latch=cfg.ros.latched,
        )

        latency_pub = None
        if cfg.ros.publish_latency:
            from std_msgs.msg import Float32
            latency_pub = rospy.Publisher(
                cfg.ros.latency_topic, Float32, queue_size=10
            )

        def callback(msg: Float32MultiArray) -> None:
            obs = _parse_float32_msg(msg)
            if obs.ndim == 1:
                obs = obs[np.newaxis, :]

            t0 = time.perf_counter()
            inputs = {cfg.model.input_names[0]: obs}
            outputs = self._engine.infer_sync(inputs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            action = next(iter(outputs.values())).squeeze()
            pub.publish(_make_float32_msg(action))

            if latency_pub is not None:
                from std_msgs.msg import Float32
                latency_pub.publish(Float32(data=float(latency_ms)))

        rospy.Subscriber(
            cfg.ros.input_topic,
            Float32MultiArray,
            callback,
            queue_size=cfg.ros.queue_size,
            buff_size=2**20,
        )

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("[rl_inference] Node ready.")
        rospy.spin()

    def _shutdown(self) -> None:
        if self._engine is not None:
            rospy.loginfo("[rl_inference] Profiler summary:\n" +
                         self._engine.profiler.summary())
            self._engine.stop()
