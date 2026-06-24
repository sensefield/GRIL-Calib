#!/usr/bin/env python3
"""
Step A2: Dynamic check of IMU published frame alignment.

This script compares:
  - IMU gyro z vs LiDAR/odometry yaw rate
  - IMU acc x  vs LiDAR/odometry forward acceleration

It estimates time lag by cross-correlation, then reports:
  1) sign agreement ratio for yaw-rate direction
  2) correlation between acc_x and forward_acc
  3) estimated lag (seconds)
  4) verdict: same-x / flipped-x / inconclusive
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
    da = np.linalg.norm(a0)
    db = np.linalg.norm(b0)
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float(np.dot(a0, b0) / (da * db))


def estimate_lag_seconds(a: np.ndarray, b: np.ndarray, fs: float, max_lag_s: float) -> float:
    """Return lag (seconds) to shift b(t+lag) toward a(t)."""
    if len(a) != len(b) or len(a) < 10:
        return 0.0

    a0 = a - np.mean(a)
    b0 = b - np.mean(b)

    max_lag = int(round(max_lag_s * fs))
    best_lag = 0
    best_score = -np.inf

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

        score = float(np.dot(x, y) / den)
        if score > best_score:
            best_score = score
            best_lag = lag

    return float(best_lag / fs)


def interp_to_grid(t_src: np.ndarray, x_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.interp(t_grid, t_src, x_src)


def find_existing_topic(bag_path: str, candidates, msg_kind: str):
    for topic in candidates:
        reader = open_reader(bag_path, [topic])
        if not reader.has_next():
            continue
        try:
            _topic, data, _ = reader.read_next()
            if msg_kind == "imu":
                rclpy.serialization.deserialize_message(data, Imu)
            elif msg_kind == "odom":
                rclpy.serialization.deserialize_message(data, Odometry)
            else:
                raise ValueError("unknown msg_kind")
            return topic
        except Exception:
            continue
    return None


def read_imu_signals(bag_path: str, topic: str, max_duration_s: float | None = None):
    times = []
    gyro_z = []
    acc_x = []
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
        gyro_z.append(msg.angular_velocity.z)
        acc_x.append(msg.linear_acceleration.x)

    return np.asarray(times), np.asarray(gyro_z), np.asarray(acc_x)


def read_odom_signals(
    bag_path: str,
    topic: str,
    max_duration_s: float | None = None,
):
    times = []
    yaws = []
    vxs = []
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
    max_duration_s: float | None = None,
):
    """Fallback when odom topic is unavailable.

    Returns time, yaw(world->child), and forward speed (child x-axis).
    """
    times = []
    yaws = []
    poss = []
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


def determine_verdict(sign_ratio: float, acc_corr: float):
    if np.isnan(sign_ratio) or np.isnan(acc_corr):
        return "inconclusive"
    if sign_ratio >= 0.8 and acc_corr >= 0.3:
        return "same-x"
    if sign_ratio >= 0.8 and acc_corr <= -0.3:
        return "flipped-x"
    return "inconclusive"


def analyze(
    bag_path: str,
    imu_topic: str,
    odom_topic: str,
    use_tf_fallback: bool,
    tf_topic: str,
    tf_parent: str,
    tf_child: str,
    fs: float,
    smooth_s: float,
    max_lag_s: float,
    sign_threshold: float,
    max_duration_s: float | None,
):
    t_i, imu_gz, imu_ax = read_imu_signals(
        bag_path,
        imu_topic,
        max_duration_s=max_duration_s,
    )
    if len(t_i) < 10:
        raise RuntimeError(f"IMU topic has too few messages: {imu_topic}")

    odom_mode = "odom"
    try:
        t_l, yaw_l, vx_l = read_odom_signals(
            bag_path,
            odom_topic,
            max_duration_s=max_duration_s,
        )
        if len(t_l) < 10:
            raise RuntimeError("odom too short")
    except Exception:
        if not use_tf_fallback:
            raise
        odom_mode = "tf"
        t_l, yaw_l, vx_l = read_tf_signals(
            bag_path,
            tf_topic,
            tf_parent,
            tf_child,
            max_duration_s=max_duration_s,
        )
        if len(t_l) < 10:
            raise RuntimeError(
                "Neither odom topic nor tf fallback produced enough samples"
            )

    yaw_l_u = np.unwrap(yaw_l)
    yaw_rate_l = np.gradient(yaw_l_u, t_l)
    fwd_acc_l = np.gradient(vx_l, t_l)

    t0 = max(float(t_i[0]), float(t_l[0]))
    t1 = min(float(t_i[-1]), float(t_l[-1]))
    if t1 - t0 < 20.0:
        raise RuntimeError("overlap between IMU and odom/tf is too short")

    dt = 1.0 / fs
    t_grid = np.arange(t0, t1, dt)
    if len(t_grid) < 50:
        raise RuntimeError("time grid too short")

    imu_gz_g = interp_to_grid(t_i, imu_gz, t_grid)
    imu_ax_g = interp_to_grid(t_i, imu_ax, t_grid)
    yaw_rate_g = interp_to_grid(t_l, yaw_rate_l, t_grid)
    fwd_acc_g = interp_to_grid(t_l, fwd_acc_l, t_grid)

    win = max(1, int(round(smooth_s * fs)))
    imu_gz_g = moving_average(imu_gz_g, win)
    imu_ax_g = moving_average(imu_ax_g, win)
    yaw_rate_g = moving_average(yaw_rate_g, win)
    fwd_acc_g = moving_average(fwd_acc_g, win)

    lag = estimate_lag_seconds(imu_gz_g, yaw_rate_g, fs, max_lag_s)

    t_shift = t_grid + lag
    valid_shift = (t_shift >= t_grid[0]) & (t_shift <= t_grid[-1])
    imu_gz_a = imu_gz_g[valid_shift]
    imu_ax_a = imu_ax_g[valid_shift]
    yaw_rate_a = np.interp(t_shift[valid_shift], t_grid, yaw_rate_g)
    fwd_acc_a = np.interp(t_shift[valid_shift], t_grid, fwd_acc_g)

    sign_ratio, sign_n = sign_agreement(imu_gz_a, yaw_rate_a, sign_threshold)
    acc_corr = robust_corr(imu_ax_a, fwd_acc_a)
    yaw_corr = robust_corr(imu_gz_a, yaw_rate_a)

    verdict = determine_verdict(sign_ratio, acc_corr)

    return {
        "imu_topic": imu_topic,
        "odom_topic": odom_topic,
        "odom_mode": odom_mode,
        "tf_topic": tf_topic,
        "tf_parent": tf_parent,
        "tf_child": tf_child,
        "n_imu": int(len(t_i)),
        "n_lidar": int(len(t_l)),
        "grid_n": int(len(t_grid)),
        "lag_s": float(lag),
        "sign_ratio": float(sign_ratio),
        "sign_count": int(sign_n),
        "yaw_corr": float(yaw_corr),
        "acc_corr": float(acc_corr),
        "verdict": verdict,
    }


def write_markdown(path: Path, bag: str, r: dict):
    lines = []
    lines.append("# Step A2 Dynamic Frame Check")
    lines.append("")
    lines.append(f"- bag: `{bag}`")
    lines.append(f"- imu_topic: `{r['imu_topic']}`")
    if r["odom_mode"] == "odom":
        lines.append(f"- lidar signal source: odom topic `{r['odom_topic']}`")
    else:
        lines.append(
            "- lidar signal source: tf fallback "
            f"`{r['tf_topic']}` ({r['tf_parent']} -> {r['tf_child']})"
        )
    lines.append(f"- samples: imu={r['n_imu']} lidar={r['n_lidar']} grid={r['grid_n']}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(f"- gyro_z sign agreement ratio: **{r['sign_ratio']:.4f}** (n={r['sign_count']})")
    lines.append(f"- corr(imu_acc_x, lidar_forward_acc): **{r['acc_corr']:.4f}**")
    lines.append(f"- corr(imu_gyro_z, lidar_yaw_rate): **{r['yaw_corr']:.4f}**")
    lines.append(f"- estimated lag: **{r['lag_s']:+.6f} s**")
    lines.append(f"- verdict: **{r['verdict']}**")
    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    lines.append("- sign_ratio >= 0.8 and acc_corr >= +0.3 -> same-x")
    lines.append("- sign_ratio >= 0.8 and acc_corr <= -0.3 -> flipped-x")
    lines.append("- otherwise -> inconclusive")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="Path to mcap bag")
    parser.add_argument("--imu-topic", default="/sensing/imu/imu_data")
    parser.add_argument(
        "--odom-topic",
        default="/aft_mapped_to_init",
        help="Odometry topic for LiDAR-derived motion",
    )
    parser.add_argument(
        "--odom-topic-candidates",
        default=(
            "/aft_mapped_to_init,"
            "/localization/kinematic_state,"
            "/odometry/filtered,"
            "/odom"
        ),
        help="Comma-separated odom candidates; first existing topic is used",
    )
    parser.add_argument("--use-tf-fallback", action="store_true")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--tf-parent", default="map")
    parser.add_argument("--tf-child", default="base_link")
    parser.add_argument("--fs", type=float, default=50.0)
    parser.add_argument("--smooth", type=float, default=0.2, help="moving average window [s]")
    parser.add_argument("--max-lag", type=float, default=0.25, help="max lag search [s]")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=240.0,
        help="Read only the first N seconds from each selected signal source",
    )
    parser.add_argument(
        "--sign-threshold",
        type=float,
        default=0.02,
        help="minimum abs threshold for sign comparison",
    )
    parser.add_argument("--out", default="result/step_a2_dynamic_check.md")
    args = parser.parse_args()

    candidates = [x.strip() for x in args.odom_topic_candidates.split(",") if x.strip()]
    picked_imu = find_existing_topic(args.bag, [args.imu_topic], "imu")
    if picked_imu is None:
        raise SystemExit(f"IMU topic not found/readable: {args.imu_topic}")

    picked_odom = find_existing_topic(args.bag, [args.odom_topic] + candidates, "odom")
    if picked_odom is None and not args.use_tf_fallback:
        raise SystemExit(
            "No readable odom topic found. "
            "Try --use-tf-fallback or pass --odom-topic explicitly."
        )

    if picked_odom is not None:
        args.odom_topic = picked_odom

    result = analyze(
        bag_path=args.bag,
        imu_topic=args.imu_topic,
        odom_topic=args.odom_topic,
        use_tf_fallback=args.use_tf_fallback,
        tf_topic=args.tf_topic,
        tf_parent=args.tf_parent,
        tf_child=args.tf_child,
        fs=args.fs,
        smooth_s=args.smooth,
        max_lag_s=args.max_lag,
        sign_threshold=args.sign_threshold,
        max_duration_s=args.max_duration,
    )

    print(f"gyro_z sign agreement ratio = {result['sign_ratio']:.6f} (n={result['sign_count']})")
    print(f"corr(acc_x, forward_acc)    = {result['acc_corr']:.6f}")
    print(f"corr(gyro_z, yaw_rate)      = {result['yaw_corr']:.6f}")
    print(f"estimated lag (s)           = {result['lag_s']:+.6f}")
    print(f"verdict                     = {result['verdict']}")

    out_path = Path(args.out)
    write_markdown(out_path, args.bag, result)
    print(f"markdown written            = {out_path}")


if __name__ == "__main__":
    main()
