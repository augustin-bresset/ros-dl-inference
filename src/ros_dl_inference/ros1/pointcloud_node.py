"""
ROS 1 segmentation node for torchsparse-based models.

Subscribes to sensor_msgs/PointCloud2, runs the MinkUNet segmentation,
and republishes an annotated cloud with a 'label' field.
"""

from __future__ import annotations

import struct
import time
from typing import Optional

import numpy as np
import rospy
import yaml
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from ..backends.torchsparse_backend import TorchSparseBackend
from ..core.config import InferenceConfig


# ── PointCloud2 conversion ─────────────────────────────────────────────────

_FIELD_DTYPE = {
    PointField.INT8:    np.int8,
    PointField.UINT8:   np.uint8,
    PointField.INT16:   np.int16,
    PointField.UINT16:  np.uint16,
    PointField.INT32:   np.int32,
    PointField.UINT32:  np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


def _build_dtype(fields, point_step: int) -> np.dtype:
    names, formats, offsets = [], [], []
    for f in fields:
        names.append(f.name)
        formats.append(_FIELD_DTYPE.get(f.datatype, np.float32))
        offsets.append(f.offset)
    return np.dtype({"names": names, "formats": formats,
                     "offsets": offsets, "itemsize": point_step})


def pointcloud2_to_xyz_intensity(msg: PointCloud2) -> np.ndarray:
    """Return (N, 4) float32 array [x, y, z, intensity]."""
    dt = _build_dtype(msg.fields, msg.point_step)
    raw = np.frombuffer(bytes(msg.data), dtype=dt)

    xyz = np.column_stack([raw["x"].astype(np.float32),
                           raw["y"].astype(np.float32),
                           raw["z"].astype(np.float32)])

    if "intensity" in raw.dtype.names:
        intensity = raw["intensity"].astype(np.float32)
    else:
        intensity = np.zeros(len(raw), dtype=np.float32)

    valid = np.isfinite(xyz).all(axis=1)
    return np.column_stack([xyz[valid], intensity[valid]])


def make_labeled_cloud(header: Header,
                       points: np.ndarray,
                       labels: np.ndarray,
                       color_map: Optional[dict] = None) -> PointCloud2:
    """
    Build a PointCloud2 with fields x, y, z, label, r, g, b.
    color_map: {int -> (r,g,b)} uint8 tuples. None → no color.
    """
    n = len(labels)
    dt = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("label", "u1"), ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("_pad", "u1"),
    ])
    arr = np.zeros(n, dtype=dt)
    arr["x"] = points[:, 0]
    arr["y"] = points[:, 1]
    arr["z"] = points[:, 2]
    arr["label"] = labels

    if color_map:
        for cls_id, color in color_map.items():
            mask = labels == int(cls_id)
            if mask.any():
                arr["r"][mask] = color[0]
                arr["g"][mask] = color[1]
                arr["b"][mask] = color[2]

    point_step = dt.itemsize
    fields = [
        PointField("x",     0,  PointField.FLOAT32, 1),
        PointField("y",     4,  PointField.FLOAT32, 1),
        PointField("z",     8,  PointField.FLOAT32, 1),
        PointField("label", 12, PointField.UINT8,   1),
        PointField("r",     13, PointField.UINT8,   1),
        PointField("g",     14, PointField.UINT8,   1),
        PointField("b",     15, PointField.UINT8,   1),
    ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.is_dense = True
    msg.is_bigendian = False
    msg.fields = fields
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.data = arr.tobytes()
    return msg


# ── ROS node ───────────────────────────────────────────────────────────────

class SegmentationNodeROS1:
    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._backend: Optional[TorchSparseBackend] = None
        self._color_map: dict = {}

    def run(self) -> None:
        rospy.init_node("lidar_segmentation_node", anonymous=False)
        cfg = self._config

        # Optional color map from labels_cfg parameter
        labels_cfg_path = rospy.get_param("~labels_cfg", "")
        if labels_cfg_path:
            try:
                with open(labels_cfg_path) as f:
                    goose_cfg = yaml.safe_load(f)
                raw_cmap = goose_cfg.get("color_map", {})
                # Convert hex strings to (r,g,b) tuples
                for k, v in raw_cmap.items():
                    h = v.lstrip("#")
                    self._color_map[int(k)] = (
                        int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                    )
                rospy.loginfo(f"[seg] Loaded color map ({len(self._color_map)} classes)")
            except Exception as e:
                rospy.logwarn(f"[seg] Could not load labels_cfg: {e}")

        rospy.loginfo(f"[seg] Loading model from {cfg.model.path} ...")
        self._backend = TorchSparseBackend(cfg)
        self._backend.load()
        rospy.loginfo(f"[seg] Model ready ({cfg.device.type})")

        pub = rospy.Publisher(
            cfg.ros.output_topic, PointCloud2,
            queue_size=cfg.ros.queue_size,
            latch=cfg.ros.latched,
        )

        latency_pub = None
        if cfg.ros.publish_latency:
            from std_msgs.msg import Float32
            latency_pub = rospy.Publisher(cfg.ros.latency_topic, Float32, queue_size=10)

        def callback(msg: PointCloud2) -> None:
            points = pointcloud2_to_xyz_intensity(msg)
            if len(points) == 0:
                return

            t0 = time.perf_counter()
            labels = self._backend.predict_labels(points)
            lat_ms = (time.perf_counter() - t0) * 1000.0

            out_msg = make_labeled_cloud(msg.header, points, labels, self._color_map)
            pub.publish(out_msg)

            if latency_pub is not None:
                from std_msgs.msg import Float32
                latency_pub.publish(Float32(data=float(lat_ms)))

        rospy.Subscriber(
            cfg.ros.input_topic, PointCloud2, callback,
            queue_size=cfg.ros.queue_size, buff_size=2**24,
        )

        rospy.on_shutdown(self._shutdown)
        rospy.loginfo(
            f"[seg] Listening on {cfg.ros.input_topic} → {cfg.ros.output_topic}"
        )
        rospy.spin()

    def _shutdown(self) -> None:
        rospy.loginfo("[seg] Shutting down")
        if self._backend is not None:
            self._backend.unload()
