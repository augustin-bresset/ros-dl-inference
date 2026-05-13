"""ROS 1 inference node implementation."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, MultiArrayLayout

from ..backends import get_backend
from ..core.config import InferenceConfig
from ..core.engine import InferenceEngine

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def _image_msg_to_tensor(msg, target_h: int, target_w: int) -> np.ndarray:
    """Convert sensor_msgs/Image to a normalised [1, 3, H, W] float32 array."""
    import cv2
    from cv_bridge import CvBridge
    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    cv_img = cv2.resize(cv_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = cv_img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = img.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
    return np.ascontiguousarray(img)


def _depth_to_image_msg(depth: np.ndarray, frame_id: str, stamp) -> "Image":
    """Convert a [H, W] float32 depth array to sensor_msgs/Image (32FC1)."""
    from sensor_msgs.msg import Image
    depth = np.squeeze(depth).astype(np.float32)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = depth.shape
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = msg.width * 4
    msg.data = depth.tobytes()
    return msg


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

        latency_pub = None
        if cfg.ros.publish_latency:
            from std_msgs.msg import Float32
            latency_pub = rospy.Publisher(
                cfg.ros.latency_topic, Float32, queue_size=10
            )

        if cfg.ros.input_type == "image":
            self._run_image_mode(cfg, latency_pub)
        else:
            self._run_array_mode(cfg, latency_pub)

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo("[rl_inference] Node ready.")
        rospy.spin()

    # ------------------------------------------------------------------
    # Float32MultiArray mode (generic)
    # ------------------------------------------------------------------

    def _run_array_mode(self, cfg: InferenceConfig, latency_pub) -> None:
        pub = rospy.Publisher(
            cfg.ros.output_topic,
            Float32MultiArray,
            queue_size=cfg.ros.queue_size,
            latch=cfg.ros.latched,
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
            self._maybe_pub_latency(latency_pub, latency_ms)

        rospy.Subscriber(
            cfg.ros.input_topic,
            Float32MultiArray,
            callback,
            queue_size=cfg.ros.queue_size,
            buff_size=2**20,
        )

    # ------------------------------------------------------------------
    # sensor_msgs/Image mode
    # ------------------------------------------------------------------

    def _run_image_mode(self, cfg: InferenceConfig, latency_pub) -> None:
        from sensor_msgs.msg import Image

        input_shape = cfg.model.input_shapes.get(cfg.model.input_names[0], [3, 518, 518])
        _, target_h, target_w = input_shape

        if cfg.ros.output_type == "image":
            pub = rospy.Publisher(
                cfg.ros.output_topic, Image,
                queue_size=cfg.ros.queue_size,
                latch=cfg.ros.latched,
            )
        else:
            pub = rospy.Publisher(
                cfg.ros.output_topic, Float32MultiArray,
                queue_size=cfg.ros.queue_size,
                latch=cfg.ros.latched,
            )

        def callback(msg: Image) -> None:
            try:
                tensor = _image_msg_to_tensor(msg, target_h, target_w)
            except Exception as e:
                rospy.logerr(f"[rl_inference] Image conversion failed: {e}")
                return

            t0 = time.perf_counter()
            inputs = {cfg.model.input_names[0]: tensor}
            outputs = self._engine.infer_sync(inputs)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            result = next(iter(outputs.values()))

            if cfg.ros.output_type == "image":
                pub.publish(_depth_to_image_msg(result, msg.header.frame_id, msg.header.stamp))
            else:
                pub.publish(_make_float32_msg(result.squeeze()))

            self._maybe_pub_latency(latency_pub, latency_ms)

        rospy.Subscriber(
            cfg.ros.input_topic,
            Image,
            callback,
            queue_size=cfg.ros.queue_size,
            buff_size=2**24,
        )

    # ------------------------------------------------------------------

    def _maybe_pub_latency(self, pub, latency_ms: float) -> None:
        if pub is not None:
            from std_msgs.msg import Float32
            pub.publish(Float32(data=float(latency_ms)))

    def _shutdown(self) -> None:
        if self._engine is not None:
            rospy.loginfo("[rl_inference] Profiler summary:\n" +
                         self._engine.profiler.summary())
            self._engine.stop()
