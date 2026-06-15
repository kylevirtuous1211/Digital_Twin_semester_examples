"""Filter a Robomimic HDF5 to keep only smooth demonstrations.

Smoothness metric per demo: number of *significant* direction reversals across
the eef_pos trajectory, weighted by axis range. A demo with many micro-reversals
on the same axis (the "stop, go opposite, come back" pattern) gets a high score
and is filtered out.

Usage:
    python scripts/filter_smooth_demos.py \
        --in ./datasets/pickplace_demos.hdf5 \
        --out ./datasets/pickplace_demos_filtered.hdf5 \
        --keep 200 \
        --valid_ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np


def count_significant_reversals(eef: np.ndarray, vel_threshold: float = 0.002) -> int:
    """Count sign changes in per-axis velocity that exceed `vel_threshold` (m/step).

    Filters out micro-jitter near zero velocity (which counts every floating-point
    blip as a reversal). Only counts a flip when both sides cleared the threshold.
    """
    v = np.diff(eef, axis=0)
    total = 0
    for ax in range(3):
        s = v[:, ax]
        # Find indices where |v| > threshold (significant motion)
        sig = np.where(np.abs(s) > vel_threshold)[0]
        if len(sig) < 2:
            continue
        signs = np.sign(s[sig])
        # Count sign changes between consecutive significant samples
        flips = int(np.sum(signs[1:] * signs[:-1] < 0))
        total += flips
    return total


def trajectory_path_ratio(eef: np.ndarray) -> float:
    """Path-length / straight-line-distance ratio (1.0 = perfectly straight).

    Higher ratio = more meandering trajectory."""
    v = np.diff(eef, axis=0)
    path_len = float(np.linalg.norm(v, axis=1).sum())
    direct = float(np.linalg.norm(eef[-1] - eef[0]))
    return path_len / max(direct, 1e-6)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, type=Path)
    p.add_argument("--out", dest="dst", required=True, type=Path)
    p.add_argument("--keep", type=int, default=200, help="Number of smoothest demos to keep.")
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument(
        "--vel_threshold",
        type=float,
        default=0.002,
        help="Velocity threshold (m/step) for counting a reversal as significant.",
    )
    args = p.parse_args()

    with h5py.File(args.src, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo_keys = sorted(
            (k for k in f["data"].keys() if k.startswith("demo_")),
            key=lambda s: int(s.split("_")[1]),
        )

        rows: list[tuple[str, int, float, int]] = []
        for k in demo_keys:
            eef = f["data"][k]["obs"]["eef_pos"][:]
            n_rev = count_significant_reversals(eef, args.vel_threshold)
            ratio = trajectory_path_ratio(eef)
            rows.append((k, n_rev, ratio, eef.shape[0]))

    rows.sort(key=lambda r: (r[1], r[2]))
    keep = rows[: args.keep]
    drop = rows[args.keep :]
    print(f"Read {len(rows)} demos. Keeping {len(keep)}, dropping {len(drop)}.")
    print("Kept demos (best-smoothest):")
    for r in keep[:10]:
        print(f"  {r[0]}  reversals={r[1]:3d}  path/direct={r[2]:.2f}  len={r[3]}")
    print("...")
    for r in keep[-3:]:
        print(f"  {r[0]}  reversals={r[1]:3d}  path/direct={r[2]:.2f}  len={r[3]}")
    if drop:
        print("\nWorst-smoothness demos dropped (first 5):")
        for r in drop[-5:][::-1]:
            print(f"  {r[0]}  reversals={r[1]:3d}  path/direct={r[2]:.2f}  len={r[3]}")

    # Write filtered HDF5 with renumbered demo_0..demo_M.
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    if args.dst.exists():
        args.dst.unlink()
    with h5py.File(args.src, "r") as fin, h5py.File(args.dst, "w") as fout:
        out_data = fout.create_group("data")
        out_data.attrs["env_args"] = json.dumps(env_args)
        for new_idx, (src_key, *_rest) in enumerate(keep):
            src_grp = fin["data"][src_key]
            dst_grp = out_data.create_group(f"demo_{new_idx}")
            dst_grp.create_dataset("actions", data=src_grp["actions"][:])
            dst_obs = dst_grp.create_group("obs")
            for ob_key in src_grp["obs"].keys():
                dst_obs.create_dataset(ob_key, data=src_grp["obs"][ob_key][:])
            if "num_samples" in src_grp.attrs:
                dst_grp.attrs["num_samples"] = int(src_grp.attrs["num_samples"])
            else:
                dst_grp.attrs["num_samples"] = src_grp["actions"].shape[0]
        n_total = len(keep)
        n_valid = max(1, int(n_total * args.valid_ratio))
        train_names = [f"demo_{i}" for i in range(n_total - n_valid)]
        valid_names = [f"demo_{i}" for i in range(n_total - n_valid, n_total)]
        mask = fout.create_group("mask")
        mask.create_dataset("train", data=np.array(train_names, dtype=object))
        mask.create_dataset("valid", data=np.array(valid_names, dtype=object))
    print(f"\nWrote filtered dataset: {args.dst}  ({n_total} demos, train={n_total - n_valid}, valid={n_valid})")


if __name__ == "__main__":
    main()
