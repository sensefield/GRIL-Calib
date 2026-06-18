#!/usr/bin/env python3
"""
Compare GRIL-Calib's LiDAR->IMU calibration against the transform stored
in the bag's /tf_static (the vehicle's actual / commissioned calibration).

Outputs the ground-truth transform, our calibration, and the per-axis
errors. Saves a Markdown summary.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import rclpy.serialization
from rosbag2_py import (
    ConverterOptions,
    SequentialReader,
    StorageFilter,
    StorageOptions,
)
from sensor_msgs.msg import Imu, PointCloud2
from tf2_msgs.msg import TFMessage


def open_reader(bag_path: str, topics):
    storage = StorageOptions(uri=bag_path, storage_id="mcap")
    converter = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = SequentialReader()
    reader.open(storage, converter)
    reader.set_filter(StorageFilter(topics=list(topics)))
    return reader


def read_first_frame_ids(bag_path: str, lidar_topic: str, imu_topic: str):
    reader = open_reader(bag_path, [lidar_topic, imu_topic])
    lidar_frame = None
    imu_frame = None
    while reader.has_next() and (lidar_frame is None or imu_frame is None):
        topic, data, _ = reader.read_next()
        if topic == lidar_topic and lidar_frame is None:
            msg = rclpy.serialization.deserialize_message(data, PointCloud2)
            lidar_frame = msg.header.frame_id
        elif topic == imu_topic and imu_frame is None:
            msg = rclpy.serialization.deserialize_message(data, Imu)
            imu_frame = msg.header.frame_id
    return lidar_frame, imu_frame


def quat_to_R(qx, qy, qz, qw):
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (qy * qy + qz * qz),
                 s * (qx * qy - qz * qw),
                 s * (qx * qz + qy * qw)],
            [    s * (qx * qy + qz * qw),
             1 - s * (qx * qx + qz * qz),
                 s * (qy * qz - qx * qw)],
            [    s * (qx * qz - qy * qw),
                 s * (qy * qz + qx * qw),
             1 - s * (qx * qx + qy * qy)],
        ]
    )


def transform_to_matrix(tf):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(
        tf.transform.rotation.x,
        tf.transform.rotation.y,
        tf.transform.rotation.z,
        tf.transform.rotation.w,
    )
    T[:3, 3] = [
        tf.transform.translation.x,
        tf.transform.translation.y,
        tf.transform.translation.z,
    ]
    return T


def read_tf_static(bag_path: str):
    reader = open_reader(bag_path, ["/tf_static"])
    transforms = {}
    while reader.has_next():
        _topic, data, _ = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, TFMessage)
        for tf in msg.transforms:
            transforms[(tf.header.frame_id, tf.child_frame_id)] = (
                transform_to_matrix(tf)
            )
    return transforms


def find_path(transforms, src, dst):
    """BFS that returns T_src_to_dst such that v_dst = T @ v_src.

    A tf_static edge (parent, child, T_edge) means v_parent = T_edge @ v_child.
    So:
      child -> parent traversal uses T_edge as-is (need_inv=False)
      parent -> child traversal uses inv(T_edge)  (need_inv=True)
    Composition is left-multiply: each step transforms current frame to next.
    """
    adj = {}
    for (parent, child), T in transforms.items():
        adj.setdefault(child, []).append((parent, T, False))   # child->parent
        adj.setdefault(parent, []).append((child, T, True))    # parent->child needs inv

    if src not in adj:
        return None, None

    queue = [(src, np.eye(4), [src])]
    visited = {src}
    while queue:
        current, T_acc, path = queue.pop(0)
        if current == dst:
            return T_acc, path
        for nxt, T_edge, need_inv in adj.get(current, []):
            if nxt in visited:
                continue
            visited.add(nxt)
            T_step = np.linalg.inv(T_edge) if need_inv else T_edge
            queue.append((nxt, T_step @ T_acc, path + [nxt]))
    return None, None


def matrix_to_xyz_euler_deg(R):
    """Roll-pitch-yaw matching GRIL-Calib's reported Euler convention.
    Verified against its sample output: roll=atan2(R21,R22),
    pitch=-asin(R20), yaw=atan2(R10,R00)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return np.degrees(np.array([roll, pitch, yaw]))


def parse_gril_result(result_text: str):
    """Parse the homogeneous matrix block at the end of GRIL_Calib_result.txt
    so we always get the exact same R/t the optimizer reported."""
    rows = []
    in_matrix = False
    for line in result_text.splitlines():
        if "Homogeneous Transformation Matrix" in line:
            in_matrix = True
            continue
        if in_matrix:
            nums = re.findall(r"[-+]?\d+\.\d+", line)
            if len(nums) == 4:
                rows.append([float(x) for x in nums])
                if len(rows) == 4:
                    break
    if len(rows) != 4:
        raise ValueError("Could not find 4-row matrix in result file")
    return np.array(rows)


def angle_between_rotations_deg(R_a, R_b):
    R_err = R_a.T @ R_b
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument(
        "--lidar-topic", default="/sensing/lidar/top/pointcloud_raw_ex"
    )
    parser.add_argument("--imu-topic", default="/sensing/imu/imu_data")
    parser.add_argument(
        "--imu-frame-override", default=None,
        help="Override the IMU frame name (e.g. 'imu_back_base_link') "
             "instead of using the IMU message's frame_id.",
    )
    parser.add_argument(
        "--our-result", default="result/GRIL_Calib_result.txt"
    )
    parser.add_argument(
        "--out", default="result/ground_truth_compare.md"
    )
    args = parser.parse_args()

    print(f"[read] frame_ids from {args.bag}", flush=True)
    lidar_frame, imu_frame = read_first_frame_ids(
        args.bag, args.lidar_topic, args.imu_topic
    )
    print(f"  LiDAR frame: {lidar_frame}")
    print(f"  IMU   frame (from msg.header): {imu_frame}")
    if args.imu_frame_override is not None:
        imu_frame = args.imu_frame_override
        print(f"  IMU   frame (overridden):       {imu_frame}")

    if not lidar_frame or not imu_frame:
        raise SystemExit(
            "Could not determine frame_ids; topics may not be in the bag."
        )

    print("[read] /tf_static", flush=True)
    transforms = read_tf_static(args.bag)
    print(f"  {len(transforms)} static transforms")
    for parent, child in sorted(transforms.keys()):
        T = transforms[(parent, child)]
        t = T[:3, 3]
        print(
            f"    {parent}  ->  {child}    "
            f"t=({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})"
        )

    print(f"[search] path {lidar_frame} -> {imu_frame}", flush=True)
    T_LI_truth, path = find_path(transforms, lidar_frame, imu_frame)
    if T_LI_truth is None:
        print("\n(no path) Available static transforms:")
        for parent, child in sorted(transforms.keys()):
            print(f"  {parent}  ->  {child}")
        raise SystemExit(1)

    print("  path: " + " -> ".join(path))

    t_truth = T_LI_truth[:3, 3]
    R_truth = T_LI_truth[:3, :3]
    eul_truth = matrix_to_xyz_euler_deg(R_truth)

    # Our calibration
    our_text = Path(args.our_result).read_text()
    T_LI_ours = parse_gril_result(our_text)
    t_ours = T_LI_ours[:3, 3]
    R_ours = T_LI_ours[:3, :3]
    eul_ours = matrix_to_xyz_euler_deg(R_ours)

    # Errors
    dt = t_ours - t_truth
    dt_norm = float(np.linalg.norm(dt))
    deul = eul_ours - eul_truth
    dR_geodesic = angle_between_rotations_deg(R_truth, R_ours)

    # Console output
    print()
    print("=== Ground truth (bag /tf_static) ===")
    print(
        f"  Translation (m) : {t_truth[0]: .4f}  {t_truth[1]: .4f}  "
        f"{t_truth[2]: .4f}"
    )
    print(
        f"  Rotation (deg)  : roll {eul_truth[0]: .4f}  "
        f"pitch {eul_truth[1]: .4f}  yaw {eul_truth[2]: .4f}"
    )

    print()
    print("=== Our calibration ===")
    print(
        f"  Translation (m) : {t_ours[0]: .4f}  {t_ours[1]: .4f}  "
        f"{t_ours[2]: .4f}"
    )
    print(
        f"  Rotation (deg)  : roll {eul_ours[0]: .4f}  "
        f"pitch {eul_ours[1]: .4f}  yaw {eul_ours[2]: .4f}"
    )

    print()
    print("=== Error (ours - truth) ===")
    print(
        f"  Translation Δ (m): {dt[0]:+.4f}  {dt[1]:+.4f}  "
        f"{dt[2]:+.4f}   |Δt| = {dt_norm:.4f}"
    )
    print(
        f"  Rotation Δ (deg) : roll {deul[0]:+.4f}  "
        f"pitch {deul[1]:+.4f}  yaw {deul[2]:+.4f}"
    )
    print(f"  Geodesic rotation error: {dR_geodesic:.4f} deg")

    # Markdown
    md_lines = []
    md_lines.append("# Ground-truth comparison")
    md_lines.append("")
    md_lines.append(f"- Bag: `{args.bag}`")
    md_lines.append(f"- LiDAR frame: `{lidar_frame}`")
    md_lines.append(f"- IMU   frame: `{imu_frame}`")
    md_lines.append(f"- TF chain: `{' -> '.join(path)}`")
    md_lines.append(f"- Our result file: `{args.our_result}`")
    md_lines.append("")
    md_lines.append("## Values")
    md_lines.append("")
    md_lines.append(
        "| | Truth (tf_static) | Ours (GRIL-Calib) | Error (ours − truth) |"
    )
    md_lines.append("|---|---|---|---|")
    md_lines.append(
        f"| Translation X (m) | {t_truth[0]:+.4f} | {t_ours[0]:+.4f} | "
        f"{dt[0]:+.4f} |"
    )
    md_lines.append(
        f"| Translation Y (m) | {t_truth[1]:+.4f} | {t_ours[1]:+.4f} | "
        f"{dt[1]:+.4f} |"
    )
    md_lines.append(
        f"| Translation Z (m) | {t_truth[2]:+.4f} | {t_ours[2]:+.4f} | "
        f"{dt[2]:+.4f} |"
    )
    md_lines.append(f"| ‖Δt‖ (m) | — | — | **{dt_norm:.4f}** |")
    md_lines.append(
        f"| Roll (deg)  | {eul_truth[0]:+.4f} | {eul_ours[0]:+.4f} | "
        f"{deul[0]:+.4f} |"
    )
    md_lines.append(
        f"| Pitch (deg) | {eul_truth[1]:+.4f} | {eul_ours[1]:+.4f} | "
        f"{deul[1]:+.4f} |"
    )
    md_lines.append(
        f"| Yaw (deg)   | {eul_truth[2]:+.4f} | {eul_ours[2]:+.4f} | "
        f"{deul[2]:+.4f} |"
    )
    md_lines.append(
        f"| Geodesic rotation error (deg) | — | — | **{dR_geodesic:.4f}** |"
    )
    md_lines.append("")
    md_lines.append("## Truth matrix (LiDAR → IMU)")
    md_lines.append("")
    md_lines.append("```")
    for row in T_LI_truth:
        md_lines.append("  " + "  ".join(f"{v: .6f}" for v in row))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Our matrix (LiDAR → IMU)")
    md_lines.append("")
    md_lines.append("```")
    for row in T_LI_ours:
        md_lines.append("  " + "  ".join(f"{v: .6f}" for v in row))
    md_lines.append("```")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md_lines))
    print(f"\n[write] markdown: {out_path}")


if __name__ == "__main__":
    main()
