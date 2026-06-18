#!/usr/bin/env python3
"""
Step A2-c: dynamic-segment-only frame check.

Compares IMU and LiDAR-derived motion signals, but evaluates only dynamic segments
(turning and/or accel-decel rich intervals) to improve X-axis observability.
"""

import argparse
from pathlib import Path

import numpy as np
import rclpy.serialization
from nav_msgs.msg import Odometry
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from sensor_msgs.msg import Imu
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


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    kernel = np.ones(win, dtype=float) / float(win)
    pad = win // 2
    x_pad = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(x_pad, kernel, mode="valid")[: len(x)]


def robust_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    den = np.linalg.norm(a0) * np.linalg.norm(b0)
    if den < 1e-12:
        return float("nan")
    return float(np.dot(a0, b0) / den)


def estimate_lag_seconds(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float) -> float:
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    max_lag = int(round(max_lag_s * fs))
    best_lag = 0
    best = -1e18
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = a0[lag:]
            y = b0[: len(b0) - lag]
        else:
            x = a0[: len(a0) + lag]
            y = b0[-lag:]
        if len(x) < 10:
            continue
        den = np.linalg.norm(x) * np.linalg.norm(y)
        if den < 1e-12:
            continue
        c = float(np.dot(x, y) / den)
        if c > best:
            best = c
            best_lag = lag
    return float(best_lag / fs)


def interp_to_grid(t_src: np.ndarray, x_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.interp(t_grid, t_src, x_src)


def find_existing_topic(bag_path: str, candidates, msg_kind: str):
    for topic in candidates:
        if not topic:
            continue
        reader = open_reader(bag_path, [topic])
        if not reader.has_next():
            continue
        try:
            _topic, data, _ = reader.read_next()
            if msg_kind == "imu":
                rclpy.serialization.deserialize_message(data, Imu)
            else:
                rclpy.serialization.deserialize_message(data, Odometry)
            return topic
        except Exception:
            continue
    return None


def read_imu_signals(bag_path: str, topic: str, max_duration_s: float | None):
    times, gz, ax = [], [], []
    t0 = None
    reader = open_reader(bag_path, [topic])
    while reader.has_next():
        _topic, data, _ = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, Imu)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t0 is None:
            t0 = t
        if max_duration_s is not None and (t - t0) > max_duration_s:
            break
        times.append(t)
        gz.append(msg.angular_velocity.z)
        ax.append(msg.linear_acceleration.x)
    return np.asarray(times), np.asarray(gz), np.asarray(ax)


def read_odom_signals(bag_path: str, topic: str, max_duration_s: float | None):
    times, yaws, vxs = [], [], []
    t0 = None
    reader = open_reader(bag_path, [topic])
    while reader.has_next():
        _topic, data, _ = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, Odometry)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t0 is None:
            t0 = t
        if max_duration_s is not None and (t - t0) > max_duration_s:
            break
        q = msg.pose.pose.orientation
        times.append(t)
        yaws.append(quat_to_yaw(q.x, q.y, q.z, q.w))
        vxs.append(msg.twist.twist.linear.x)
    return np.asarray(times), np.asarray(yaws), np.asarray(vxs)


def read_tf_signals(
    bag_path: str,
    topic: str,
    parent: str,
    child: str,
    max_duration_s: float | None,
):
    times, yaws, poss = [], [], []
    t0 = None
    stop = False
    reader = open_reader(bag_path, [topic])
    while reader.has_next() and not stop:
        _topic, data, _ = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, TFMessage)
        for tf in msg.transforms:
            if tf.header.frame_id != parent or tf.child_frame_id != child:
                continue
            t = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
            if t0 is None:
                t0 = t
            if max_duration_s is not None and (t - t0) > max_duration_s:
                if len(times) > 0:
                    stop = True
                    break
                continue
            tr = tf.transform.translation
            q = tf.transform.rotation
            times.append(t)
            yaws.append(quat_to_yaw(q.x, q.y, q.z, q.w))
            poss.append([tr.x, tr.y, tr.z])

    times = np.asarray(times)
    yaws = np.asarray(yaws)
    poss = np.asarray(poss)
    if len(times) < 3:
        return times, yaws, np.asarray([])

    dt = np.diff(times)
    dt[dt < 1e-6] = 1e-6
    vel_world = np.vstack([np.diff(poss, axis=0) / dt[:, None], [0.0, 0.0, 0.0]])
    cy = np.cos(yaws)
    sy = np.sin(yaws)
    vx_body = cy * vel_world[:, 0] + sy * vel_world[:, 1]
    return times, yaws, vx_body


def sign_agreement(a: np.ndarray, b: np.ndarray, min_abs: float):
    mask = (np.abs(a) >= min_abs) & (np.abs(b) >= min_abs)
    if int(np.sum(mask)) < 10:
        return float("nan"), int(np.sum(mask))
    ok = np.sign(a[mask]) == np.sign(b[mask])
    return float(np.mean(ok)), int(np.sum(mask))


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    out = np.zeros_like(mask, dtype=bool)
    idx = np.flatnonzero(mask)
    for i in idx:
        lo = max(0, i - radius)
        hi = min(len(mask), i + radius + 1)
        out[lo:hi] = True
    return out


def determine_verdict(sign_ratio: float, acc_corr: float):
    if np.isnan(sign_ratio) or np.isnan(acc_corr):
        return "inconclusive"
    if sign_ratio >= 0.8 and acc_corr >= 0.3:
        return "same-x"
    if sign_ratio >= 0.8 and acc_corr <= -0.3:
        return "flipped-x"
    return "inconclusive"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bag")
    p.add_argument("--imu-topic", default="/sensing/imu/imu_data")
    p.add_argument("--odom-topic", default="/aft_mapped_to_init")
    p.add_argument("--odom-topic-candidates", default="/aft_mapped_to_init,/localization/kinematic_state,/odometry/filtered,/odom")
    p.add_argument("--use-tf-fallback", action="store_true")
    p.add_argument("--tf-topic", default="/tf")
    p.add_argument("--tf-parent", default="map")
    p.add_argument("--tf-child", default="base_link")
    p.add_argument("--fs", type=float, default=50.0)
    p.add_argument("--smooth", type=float, default=0.2)
    p.add_argument("--max-lag", type=float, default=0.25)
    p.add_argument("--max-duration", type=float, default=650.0)
    p.add_argument("--sign-threshold", type=float, default=0.02)
    p.add_argument("--dyn-yaw-th", type=float, default=0.06)
    p.add_argument("--dyn-acc-th", type=float, default=0.25)
    p.add_argument("--dyn-pad", type=float, default=0.3)
    p.add_argument("--out", default="result/step_a2_dynamic_segment_check.md")
    args = p.parse_args()

    candidates = [x.strip() for x in args.odom_topic_candidates.split(",") if x.strip()]
    picked_imu = find_existing_topic(args.bag, [args.imu_topic], "imu")
    if picked_imu is None:
        raise SystemExit("IMU topic not readable")
    picked_odom = find_existing_topic(args.bag, [args.odom_topic] + candidates, "odom")

    t_i, imu_gz, imu_ax = read_imu_signals(args.bag, args.imu_topic, args.max_duration)
    if len(t_i) < 10:
        raise SystemExit("IMU too short")

    mode = "odom"
    if picked_odom is not None:
        t_l, yaw_l, vx_l = read_odom_signals(args.bag, picked_odom, args.max_duration)
    else:
        if not args.use_tf_fallback:
            raise SystemExit("No odom topic and tf fallback disabled")
        mode = "tf"
        t_l, yaw_l, vx_l = read_tf_signals(
            args.bag,
            args.tf_topic,
            args.tf_parent,
            args.tf_child,
            args.max_duration,
        )

    if len(t_l) < 10:
        raise SystemExit("LiDAR-side signal too short")

    yaw_rate_l = np.gradient(np.unwrap(yaw_l), t_l)
    fwd_acc_l = np.gradient(vx_l, t_l)

    t0 = max(float(t_i[0]), float(t_l[0]))
    t1 = min(float(t_i[-1]), float(t_l[-1]))
    if t1 - t0 < 20.0:
        raise SystemExit("overlap too short")
    t_grid = np.arange(t0, t1, 1.0 / args.fs)

    imu_gz_g = interp_to_grid(t_i, imu_gz, t_grid)
    imu_ax_g = interp_to_grid(t_i, imu_ax, t_grid)
    yaw_rate_g = interp_to_grid(t_l, yaw_rate_l, t_grid)
    fwd_acc_g = interp_to_grid(t_l, fwd_acc_l, t_grid)

    win = max(1, int(round(args.smooth * args.fs)))
    imu_gz_g = moving_average(imu_gz_g, win)
    imu_ax_g = moving_average(imu_ax_g, win)
    yaw_rate_g = moving_average(yaw_rate_g, win)
    fwd_acc_g = moving_average(fwd_acc_g, win)

    lag = estimate_lag_seconds(imu_gz_g, yaw_rate_g, args.fs, args.max_lag)
    t_shift = t_grid + lag
    valid = (t_shift >= t_grid[0]) & (t_shift <= t_grid[-1])

    imu_gz_a = imu_gz_g[valid]
    imu_ax_a = imu_ax_g[valid]
    yaw_rate_a = np.interp(t_shift[valid], t_grid, yaw_rate_g)
    fwd_acc_a = np.interp(t_shift[valid], t_grid, fwd_acc_g)

    dyn = (np.abs(yaw_rate_a) >= args.dyn_yaw_th) | (np.abs(fwd_acc_a) >= args.dyn_acc_th)
    dyn = dilate_mask(dyn, int(round(args.dyn_pad * args.fs)))

    imu_gz_d = imu_gz_a[dyn]
    imu_ax_d = imu_ax_a[dyn]
    yaw_rate_d = yaw_rate_a[dyn]
    fwd_acc_d = fwd_acc_a[dyn]

    sign_ratio, sign_n = sign_agreement(imu_gz_d, yaw_rate_d, args.sign_threshold)
    yaw_corr = robust_corr(imu_gz_d, yaw_rate_d)
    acc_corr = robust_corr(imu_ax_d, fwd_acc_d)
    verdict = determine_verdict(sign_ratio, acc_corr)

    lines = []
    lines.append("# Step A2-c Dynamic Segment Check")
    lines.append("")
    lines.append(f"- mode: {mode}")
    lines.append(f"- samples(all/aligned/dynamic): {len(t_grid)}/{len(imu_gz_a)}/{len(imu_gz_d)}")
    lines.append(f"- dynamic ratio: {len(imu_gz_d)/max(1,len(imu_gz_a)):.4f}")
    lines.append(f"- lag_s: {lag:+.6f}")
    lines.append(f"- sign_ratio: {sign_ratio:.6f} (n={sign_n})")
    lines.append(f"- yaw_corr: {yaw_corr:.6f}")
    lines.append(f"- acc_corr: {acc_corr:.6f}")
    lines.append(f"- verdict: {verdict}")
    lines.append("")
    lines.append("## Parameters")
    lines.append(f"- dyn_yaw_th: {args.dyn_yaw_th}")
    lines.append(f"- dyn_acc_th: {args.dyn_acc_th}")
    lines.append(f"- dyn_pad: {args.dyn_pad}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))

    print(f"samples(all/aligned/dynamic) = {len(t_grid)}/{len(imu_gz_a)}/{len(imu_gz_d)}")
    print(f"dynamic ratio                = {len(imu_gz_d)/max(1,len(imu_gz_a)):.6f}")
    print(f"gyro_z sign agreement ratio  = {sign_ratio:.6f} (n={sign_n})")
    print(f"corr(acc_x, forward_acc)     = {acc_corr:.6f}")
    print(f"corr(gyro_z, yaw_rate)       = {yaw_corr:.6f}")
    print(f"estimated lag (s)            = {lag:+.6f}")
    print(f"verdict                      = {verdict}")
    print(f"markdown written             = {out}")


if __name__ == "__main__":
    main()
