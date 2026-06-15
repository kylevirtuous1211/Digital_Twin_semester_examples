"""Per-dimension action normalization for the demo HDF5.

Lab's `train.py --normalize_training_actions` uses a single global min/max
across all action dims, which is a no-op when one dim (gripper) already spans
[-1, 1] — the small position-delta dims stay tiny and BC underweights them.

This script copies the dataset and rewrites each `demo_*/actions[:, j]` column
to [-1, 1] independently per dim j, then saves the per-dim min/max so play
time can un-normalize.

Usage:
    python scripts/perdim_normalize_hdf5.py \
        --in ./datasets/pickplace_demos.hdf5 \
        --out ./datasets/pickplace_demos_perdim_norm.hdf5
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True, type=Path)
    p.add_argument("--out", dest="dst", required=True, type=Path)
    args = p.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.src, args.dst)

    with h5py.File(args.dst, "r+") as f:
        data = f["data"]
        demo_keys = [k for k in data.keys() if k.startswith("demo_")]
        assert demo_keys, "no demo_* groups in dataset"

        all_actions = np.concatenate([data[k]["actions"][:] for k in demo_keys], axis=0)
        a_min = all_actions.min(axis=0)
        a_max = all_actions.max(axis=0)
        span = np.where(a_max - a_min > 1e-12, a_max - a_min, 1.0)

        print("Per-dim action min:", a_min.tolist())
        print("Per-dim action max:", a_max.tolist())
        print("Per-dim span:      ", span.tolist())

        for k in demo_keys:
            acts = data[k]["actions"][:]
            normed = 2.0 * (acts - a_min) / span - 1.0
            del data[k]["actions"]
            data[k].create_dataset("actions", data=normed.astype(np.float32))

        # Stash per-dim params on the data group as attrs so play-time can recover them.
        data.attrs["perdim_action_min"] = a_min.astype(np.float32)
        data.attrs["perdim_action_max"] = a_max.astype(np.float32)

    # Sidecar text file alongside the dataset for convenience.
    sidecar = args.dst.with_suffix(".perdim_norm.txt")
    with sidecar.open("w") as fh:
        fh.write("Per-dimension action min/max used to rescale to [-1, 1].\n")
        fh.write(f"min: {a_min.tolist()}\n")
        fh.write(f"max: {a_max.tolist()}\n")
    print(f"\nWrote normalized dataset: {args.dst}")
    print(f"Wrote sidecar params:    {sidecar}")


if __name__ == "__main__":
    main()
