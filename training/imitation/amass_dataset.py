#!/usr/bin/env python3
"""
AMASS → Quark dataset parser (stub)

Reads AMASS SMPL sequences (.npz) and exports simplified pose/vel arrays
mapped to Quark humanoid joint indices.

Outputs (numpy npz):
  poses.npy  shape [N, T, J]   (joint angles in radians)
  vels.npy   shape [N, T, J]
  root.npy   shape [N, T, 6]  (root pos (3) + root vel (3))

Note: This is a minimal scaffold; SMPL→Quark mapping must be filled.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def collect_amass_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*.npz"):
        # Skip shape files (contain only body shape, no motion)
        if p.name.lower().startswith("shape"):
            continue
        files.append(p)
    return sorted(files)


def smpl_to_quark_angles(smpl_pose: np.ndarray) -> np.ndarray:
    """Map SMPL pose axis-angles to Quark joint order.
    TODO: implement real mapping. For now, return a zero vector of len 18.
    """
    return np.zeros(18, dtype=np.float32)


def process_sequence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    # Expected keys in AMASS npz: 'poses' (T, 156), 'trans' (T,3)
    poses_aa = data.get('poses')  # axis-angle per joint
    trans = data.get('trans')
    if poses_aa is None or trans is None:
        raise ValueError(f"Missing keys in {path}")
    T = poses_aa.shape[0]
    quark_poses = np.stack([smpl_to_quark_angles(poses_aa[t]) for t in range(T)], axis=0)
    quark_vels = np.zeros_like(quark_poses)
    root = np.zeros((T, 6), dtype=np.float32)
    root[:, :3] = trans.astype(np.float32)
    root[1:, 3:6] = root[1:, :3] - root[:-1, :3]
    return quark_poses, quark_vels, root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Path to AMASS root")
    ap.add_argument("--out", type=str, required=True, help="Output folder")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq_out = out / "sequences"
    seq_out.mkdir(parents=True, exist_ok=True)

    files = collect_amass_files(root)
    if not files:
        print("No AMASS .npz files found. Did you extract the dataset?")
        return 1

    count = 0
    for f in files:
        try:
            p, v, r = process_sequence(f)
            # Save each sequence individually to handle variable T
            base = f.stem
            np.save(seq_out / f"{base}_poses.npy", p)
            np.save(seq_out / f"{base}_vels.npy", v)
            np.save(seq_out / f"{base}_root.npy", r)
            count += 1
        except Exception as e:
            print(f"Skipping {f}: {e}")
            continue
    if count == 0:
        print("No valid sequences parsed.")
        return 1
    print(f"Saved {count} sequences to {seq_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


