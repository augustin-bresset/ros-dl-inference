#!/usr/bin/env python3
"""
Offline rosbag inference for off-road-dg-lss segmentation model.

Reads a ROS 1 bag, runs the MinkUNetWithAuxDecoder on every PointCloud2
message, and writes a new bag with an annotated cloud (adds a 'label' field).

Usage:
  python3 scripts/test_rosbag.py \
      --bag     /data/my_recording.bag \
      --config  config/off_road_dg_lss.yaml \
      --topic   /os_cloud_node/points \
      --out     /data/my_recording_seg.bag \
      [--cpu]   [--max-frames 50]

No ROS installation needed — uses the pure-Python `rosbags` library.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml


# ── helpers ────────────────────────────────────────────────────────────────


def _field_offset(fields, name: str) -> tuple[int, str]:
    """Return (byte_offset, numpy_dtype_str) for a PointCloud2 field."""
    _DTYPE = {1: "u1", 2: "i1", 3: "u2", 4: "i2", 5: "u4", 6: "i4",
              7: "f4", 8: "f8"}
    for f in fields:
        if f.name == name:
            return f.offset, _DTYPE[f.datatype]
    raise KeyError(f"Field '{name}' not found in PointCloud2")


def pointcloud2_to_numpy(msg) -> np.ndarray:
    """
    Convert a rosbags PointCloud2 message to (N, 4) float32 [x, y, z, intensity].

    Falls back gracefully when 'intensity' is missing (uses zeros).
    """
    data = bytes(msg.data)
    point_step = msg.point_step
    n = msg.width * msg.height

    fields = msg.fields

    def extract(name: str) -> Optional[np.ndarray]:
        try:
            offset, dtype = _field_offset(fields, name)
        except KeyError:
            return None
        values = np.frombuffer(data, dtype=np.dtype(dtype),
                               count=n, offset=offset).astype(np.float32)
        # Handle row-stride (non-contiguous if point_step > sum of field sizes)
        if point_step != 4:  # cheap check: reparse with stride
            values = np.array(
                [struct.unpack_from(dtype.replace("u", "<u").replace("f", "<f")
                                    .replace("i", "<i"), data,
                                    point_step * i + offset)[0]
                 for i in range(n)],
                dtype=np.float32,
            )
        return values

    # Proper strided extraction using numpy structured array
    dt = np.dtype({"names": [], "formats": [], "offsets": [], "itemsize": point_step})
    present = []
    for name in ("x", "y", "z", "intensity"):
        try:
            off, dtype = _field_offset(fields, name)
            dt = np.dtype({
                "names": dt.names + (name,) if dt.names else (name,),
                "formats": list(dt.formats) + [np.dtype(dtype)] if dt.formats else [np.dtype(dtype)],
                "offsets": list(dt.offsets) + [off] if dt.offsets else [off],
                "itemsize": point_step,
            })
            present.append(name)
        except KeyError:
            pass

    arr = np.frombuffer(data, dtype=dt)
    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    intensity = arr["intensity"].astype(np.float32) if "intensity" in present \
        else np.zeros(n, dtype=np.float32)

    # Filter NaN / inf
    valid = np.isfinite(xyz).all(axis=1)
    if not valid.all():
        xyz = xyz[valid]
        intensity = intensity[valid]

    return np.column_stack([xyz, intensity])


def labels_to_colored_cloud(points: np.ndarray, labels: np.ndarray,
                             color_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,3) float32 xyz and (N,3) uint8 rgb arrays."""
    xyz = points[:, :3]
    colors = np.zeros((len(labels), 3), dtype=np.uint8)
    for cls_id, hex_color in color_map.items():
        mask = labels == int(cls_id)
        if not mask.any():
            continue
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        colors[mask] = [r, g, b]
    return xyz, colors


# ── main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Offline rosbag segmentation inference")
    parser.add_argument("--bag",        required=True,  help="Input .bag file")
    parser.add_argument("--config",     required=True,  help="YAML inference config")
    parser.add_argument("--topic",      default=None,   help="PointCloud2 topic (auto-detect if omitted)")
    parser.add_argument("--out",        default=None,   help="Output .bag file (default: input_seg.bag)")
    parser.add_argument("--cpu",        action="store_true", help="Force CPU inference")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--no-write",   action="store_true",  help="Dry run — no output bag")
    parser.add_argument("--labels-cfg", default=None,
                        help="Path to goose_cfg.yaml for color map (optional)")
    args = parser.parse_args()

    # ── imports that require torchsparse env ──
    try:
        from rosbags.rosbag1 import Reader, Writer
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        sys.exit("rosbags not installed. Run: pip install rosbags")

    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from ros_rl_inference.core.config import InferenceConfig
    from ros_rl_inference.backends.torchsparse_backend import TorchSparseBackend

    # ── load config ──
    cfg = InferenceConfig.from_yaml(args.config)
    if args.cpu:
        cfg.device.type = "cpu"

    print(f"[info] loading model from {cfg.model.path} ...")
    backend = TorchSparseBackend(cfg)
    backend.load()
    print(f"[info] model ready  ({cfg.device.type})")

    # ── load optional color map ──
    color_map: dict = {}
    labels_cfg_path = args.labels_cfg or str(
        Path(__file__).resolve().parents[1] / ".." / "off-road-dg-lss" / "resources" / "goose_cfg.yaml"
    )
    if Path(labels_cfg_path).exists():
        with open(labels_cfg_path) as f:
            goose_cfg = yaml.safe_load(f)
        color_map = goose_cfg.get("color_map", {})
        semantic_map = goose_cfg.get("semantic_map", {})
        print(f"[info] loaded color map ({len(color_map)} classes)")
    else:
        semantic_map = {}
        print("[warn] no color map found — labels will be raw indices")

    bag_path = Path(args.bag)
    out_path  = Path(args.out) if args.out else bag_path.with_suffix("").with_suffix("") \
                                                         .parent / (bag_path.stem + "_seg.bag")

    typestore = get_typestore(Stores.ROS1_NOETIC)

    latencies = []
    frame_idx = 0

    with Reader(str(bag_path)) as reader:
        # ── auto-detect PointCloud2 topic ──
        pc2_connections = [c for c in reader.connections
                           if c.msgtype == "sensor_msgs/msg/PointCloud2"
                           or "PointCloud2" in c.msgtype]
        if not pc2_connections:
            sys.exit("[error] No PointCloud2 topic found in bag")

        if args.topic:
            pc2_connections = [c for c in pc2_connections if c.topic == args.topic]
            if not pc2_connections:
                available = [c.topic for c in reader.connections if "PointCloud2" in c.msgtype]
                sys.exit(f"[error] Topic '{args.topic}' not found. Available: {available}")
        else:
            print(f"[info] auto-selected topic: {pc2_connections[0].topic}")

        target_topic = pc2_connections[0].topic
        all_connections = list(reader.connections)

        writer_ctx = Writer(str(out_path)) if not args.no_write else None

        try:
            if writer_ctx:
                writer_ctx.__enter__()
                # Mirror all connections + add segmented cloud topic
                conn_map = {}
                for c in all_connections:
                    conn_map[c.id] = writer_ctx.add_connection(
                        c.topic, c.msgtype, typestore=typestore
                    )
                seg_conn = writer_ctx.add_connection(
                    target_topic + "_seg",
                    "sensor_msgs/msg/PointCloud2",
                    typestore=typestore,
                )

            for connection, timestamp, rawdata in reader.messages():
                # Pass through all non-target messages unchanged
                if writer_ctx and connection.topic != target_topic:
                    writer_ctx.write(conn_map[connection.id], timestamp, rawdata)
                    continue

                if connection.topic != target_topic:
                    continue

                # ── deserialize ──
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                points = pointcloud2_to_numpy(msg)

                if len(points) == 0:
                    continue

                # ── inference ──
                t0 = time.perf_counter()
                labels = backend.predict_labels(points)  # (N,) uint8
                lat_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(lat_ms)

                n = len(labels)
                print(
                    f"\r[frame {frame_idx:4d}]  pts={n:6d}  lat={lat_ms:6.1f}ms"
                    f"  avg={np.mean(latencies):.1f}ms",
                    end="",
                )

                # ── write original + segmented cloud ──
                if writer_ctx:
                    writer_ctx.write(conn_map[connection.id], timestamp, rawdata)

                    # Build segmented PointCloud2: x y z label r g b
                    xyz, rgb = labels_to_colored_cloud(points, labels, color_map)
                    _write_seg_cloud(writer_ctx, seg_conn, typestore, timestamp,
                                     msg.header, xyz, labels, rgb)

                frame_idx += 1
                if args.max_frames and frame_idx >= args.max_frames:
                    break

        finally:
            if writer_ctx:
                writer_ctx.__exit__(None, None, None)

    print(f"\n[done] {frame_idx} frames processed")
    if latencies:
        print(f"  latency — mean: {np.mean(latencies):.1f}ms  "
              f"p50: {np.percentile(latencies, 50):.1f}ms  "
              f"p95: {np.percentile(latencies, 95):.1f}ms  "
              f"min: {np.min(latencies):.1f}ms  max: {np.max(latencies):.1f}ms")
    if not args.no_write:
        print(f"  output: {out_path}")


# ── PointCloud2 writer helper ───────────────────────────────────────────────


def _write_seg_cloud(writer, conn, typestore, timestamp, header,
                     xyz: np.ndarray, labels: np.ndarray, rgb: np.ndarray):
    """Pack x,y,z,label,r,g,b into a PointCloud2 and write to bag."""
    # Fields: x(f4,0) y(f4,4) z(f4,8) label(u1,12) r(u1,13) g(u1,14) b(u1,15)
    n = len(xyz)
    point_step = 16
    data = np.zeros(n, dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("label", "u1"), ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ("_pad", "u1"),
    ]))
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["label"] = labels
    data["r"] = rgb[:, 0]
    data["g"] = rgb[:, 1]
    data["b"] = rgb[:, 2]

    PointCloud2 = typestore.types["sensor_msgs/msg/PointCloud2"]
    PointField  = typestore.types["sensor_msgs/msg/PointField"]

    fields = [
        PointField(name="x",     offset=0,  datatype=7, count=1),
        PointField(name="y",     offset=4,  datatype=7, count=1),
        PointField(name="z",     offset=8,  datatype=7, count=1),
        PointField(name="label", offset=12, datatype=1, count=1),
        PointField(name="r",     offset=13, datatype=1, count=1),
        PointField(name="g",     offset=14, datatype=1, count=1),
        PointField(name="b",     offset=15, datatype=1, count=1),
    ]

    cloud = PointCloud2(
        header=header,
        height=1,
        width=n,
        is_dense=True,
        is_bigendian=False,
        fields=fields,
        point_step=point_step,
        row_step=point_step * n,
        data=data.tobytes(),
    )

    rawdata = typestore.serialize_ros1(cloud, "sensor_msgs/msg/PointCloud2")
    writer.write(conn, timestamp, rawdata)


if __name__ == "__main__":
    main()
