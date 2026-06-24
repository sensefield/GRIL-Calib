#!/usr/bin/env python3
"""
Find the optimal time window in a rosbag for IMU-LiDAR calibration.

Mimics GRIL-Calib's data_sufficiency_assess criterion: at each candidate
window [t0, t0+W], accumulate the rotation Hessian
    H = sum_t  (|omega(t)|^2 I - omega(t) omega(t)^T) * dt
and score the window as the product of the two smallest eigenvalues of H.
A higher score means the three rotation axes are all well-exercised in that
window, which is what gives the calibration its observability.

Outputs Markdown summary + matplotlib plot.
"""

import argparse
from pathlib import Path

import numpy as np
import rclpy.serialization
from rosbag2_py import (
    ConverterOptions,
    SequentialReader,
    StorageFilter,
    StorageOptions,
)
from sensor_msgs.msg import Imu


def read_imu(bag_path: str, topic: str):
    storage = StorageOptions(uri=bag_path, storage_id="mcap")
    converter = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = SequentialReader()
    reader.open(storage, converter)
    reader.set_filter(StorageFilter(topics=[topic]))

    times: list[float] = []
    omegas: list[list[float]] = []
    while reader.has_next():
        _topic, data, _ = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, Imu)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        w = msg.angular_velocity
        times.append(t)
        omegas.append([w.x, w.y, w.z])

    return np.asarray(times), np.asarray(omegas)


def windowed_eigvals(times: np.ndarray, omegas: np.ndarray,
                     window_size_s: float, step_s: float):
    """For each sliding window, return the 3 eigenvalues (ascending) of H."""
    N = len(times)
    # Cumulative sums for O(1) window extraction
    cum_norm_sq = np.zeros(N + 1)
    cum_norm_sq[1:] = np.cumsum(np.einsum("ij,ij->i", omegas, omegas))

    cum_outer = np.zeros((N + 1, 3, 3))
    cum_outer[1:] = np.cumsum(
        omegas[:, :, None] * omegas[:, None, :], axis=0
    )

    dt_avg = float(np.diff(times).mean()) if N > 1 else 1.0 / 200.0

    t0_grid = np.arange(times[0], times[-1] - window_size_s, step_s)
    eigs = np.full((len(t0_grid), 3), np.nan)

    for i, t0 in enumerate(t0_grid):
        a = np.searchsorted(times, t0)
        b = np.searchsorted(times, t0 + window_size_s)
        if b - a < 10:
            continue
        norm_sq_sum = cum_norm_sq[b] - cum_norm_sq[a]
        outer_sum = cum_outer[b] - cum_outer[a]
        H = (norm_sq_sum * np.eye(3) - outer_sum) * dt_avg
        eigs[i] = np.linalg.eigvalsh(H)

    return t0_grid, eigs


def score_from_eigs(eigs: np.ndarray) -> np.ndarray:
    """Product of the two smallest eigenvalues per row."""
    return eigs[:, 0] * eigs[:, 1]


def write_markdown(out_path: Path, args, times, results):
    bag_dur = float(times[-1] - times[0])
    t_start = float(times[0])

    lines = []
    lines.append("# Bag Calibration Window Analysis")
    lines.append("")
    lines.append(f"- Bag: `{args.bag}`")
    lines.append(f"- IMU topic: `{args.topic}`")
    lines.append(f"- IMU messages: {len(times)}")
    lines.append(f"- Bag duration: {bag_dur:.1f} s")
    lines.append(f"- Bag start time (Unix sec): {t_start:.3f}")
    lines.append("")
    lines.append(
        "Score = λ\\_min × λ\\_mid of the angular-velocity Hessian "
        "`H = ∫ (|ω|² I − ω ωᵀ) dt`. "
        "GRIL-Calib's data\\_sufficiency criterion is monotone in this "
        "score, so higher score ⇒ richer observability."
    )
    lines.append("")
    lines.append("## Top 5 windows per size")
    lines.append("")

    for W in args.windows:
        starts_abs, eigs = results[W]
        starts_rel = starts_abs - t_start
        score = score_from_eigs(eigs)

        lines.append(f"### Window size {W:g} s")
        lines.append("")
        lines.append(
            "| Rank | t_start rel (s) | t_start abs (Unix s) | "
            "λ_min | λ_mid | λ_max | score | bag play cmd |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")

        valid = ~np.isnan(score)
        order = np.argsort(np.where(valid, score, -np.inf))[::-1][:5]
        for rank, idx in enumerate(order, 1):
            if not valid[idx]:
                break
            t_rel = starts_rel[idx]
            t_abs = starts_abs[idx]
            cmd = (
                f"`ros2 bag play <bag> --start-offset {t_rel:.1f} "
                f"--playback-until-timestamp {(t_abs + W) * 1e9:.0f}`"
            )
            lines.append(
                f"| {rank} | {t_rel:.1f} | {t_abs:.3f} | "
                f"{eigs[idx, 0]:.3g} | {eigs[idx, 1]:.3g} | "
                f"{eigs[idx, 2]:.3g} | {score[idx]:.4g} | {cmd} |"
            )
        lines.append("")

    # ASCII timeline (use median window size)
    W_mid = args.windows[len(args.windows) // 2]
    starts_abs, eigs = results[W_mid]
    starts_rel = starts_abs - t_start
    score = score_from_eigs(eigs)
    valid = ~np.isnan(score)
    if valid.any():
        score_max = float(np.nanmax(score))
        lines.append(
            f"## ASCII timeline (W={W_mid:g}s, normalized to peak)"
        )
        lines.append("")
        lines.append("```")
        bar_max = 60
        for s_rel, sc in zip(starts_rel, score):
            if np.isnan(sc):
                bar = "?"
            else:
                bar = "#" * int(round(sc / score_max * bar_max))
            lines.append(f"{s_rel:6.1f}s | {bar}")
        lines.append("```")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    return out_path


def write_plot(plot_path: Path, args, times, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_start = float(times[0])
    n = len(args.windows)
    fig, axes = plt.subplots(
        n, 1, figsize=(12, 2.5 * n), sharex=True
    )
    if n == 1:
        axes = [axes]

    for ax, W in zip(axes, args.windows):
        starts_abs, eigs = results[W]
        starts_rel = starts_abs - t_start
        score = score_from_eigs(eigs)
        ax.plot(starts_rel, score, label=f"score (λ_min × λ_mid)", linewidth=1.6)
        ax.plot(starts_rel, eigs[:, 0], "--", alpha=0.5, label="λ_min")
        ax.plot(starts_rel, eigs[:, 1], "--", alpha=0.5, label="λ_mid")

        valid = ~np.isnan(score)
        if valid.any():
            best = int(np.nanargmax(score))
            ax.axvline(
                starts_rel[best], color="r", alpha=0.4,
                label=f"best @ {starts_rel[best]:.0f}s",
            )

        ax.set_ylabel(f"W = {W:g} s")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s, relative to bag start)")
    fig.suptitle(
        "Calibration window richness "
        "(higher score ⇒ better rotation observability)"
    )
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    return plot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", help="Path to mcap bag file")
    parser.add_argument("--topic", default="/sensing/imu/imu_data")
    parser.add_argument(
        "--windows", nargs="+", type=float,
        default=[60, 100, 150, 200, 300],
    )
    parser.add_argument("--step", type=float, default=10.0)
    parser.add_argument("--out", default="result/window_analysis.md")
    parser.add_argument("--plot", default="result/window_analysis.png")
    args = parser.parse_args()

    print(
        f"[read] {args.bag}  topic={args.topic}",
        flush=True,
    )
    times, omegas = read_imu(args.bag, args.topic)
    print(
        f"[read] {len(times)} msgs, "
        f"duration {times[-1] - times[0]:.1f}s",
        flush=True,
    )

    results = {}
    for W in args.windows:
        print(f"[scan] window={W:g}s step={args.step:g}s ...", flush=True)
        starts, eigs = windowed_eigvals(times, omegas, W, args.step)
        results[W] = (starts, eigs)

    md_path = write_markdown(Path(args.out), args, times, results)
    print(f"[write] markdown: {md_path}")

    plot_path = write_plot(Path(args.plot), args, times, results)
    print(f"[write] plot:     {plot_path}")

    # Console top-3 summary for the medium window
    W_mid = args.windows[len(args.windows) // 2]
    starts, eigs = results[W_mid]
    score = score_from_eigs(eigs)
    valid = ~np.isnan(score)
    if valid.any():
        order = np.argsort(np.where(valid, score, -np.inf))[::-1][:3]
        print()
        print(f"=== Top 3 windows (W={W_mid:g}s) ===")
        for rank, idx in enumerate(order, 1):
            t_rel = float(starts[idx] - times[0])
            print(
                f"  {rank}. start={t_rel:6.1f}s  "
                f"eig=({eigs[idx, 0]:.3g}, {eigs[idx, 1]:.3g}, "
                f"{eigs[idx, 2]:.3g})  score={score[idx]:.4g}"
            )


if __name__ == "__main__":
    main()
