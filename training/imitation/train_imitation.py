#!/usr/bin/env python3
"""
Motion imitation pretraining (stub): loads processed AMASS arrays and trains a
small policy to match joint poses/velocities (supervised). Later, integrate as
auxiliary reward for RL.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH = True
except Exception:
    TORCH = False


class TinyPolicy(nn.Module):
    def __init__(self, obs_dim: int = 24, act_dim: int = 18):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x):
        return self.net(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Folder with poses.npy, vels.npy, root.npy")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    if not TORCH:
        print("PyTorch required for imitation training")
        return 1

    data = Path(args.data)
    # Support either combined npy files or per-sequence folder
    seq_dir = data / "sequences"
    obs_frames = []
    act_frames = []
    if (data / "poses.npy").exists():
        poses = np.load(data / "poses.npy")  # [N, T, 18]
        vels = np.load(data / "vels.npy")    # [N, T, 18]
        roots = np.load(data / "root.npy")   # [N, T, 6]
        rv = roots[:, :, 3:6]
        obs_np = np.concatenate([rv, vels], axis=-1)  # [N,T,21]
        pad = np.zeros((obs_np.shape[0], obs_np.shape[1], 3), dtype=obs_np.dtype)
        obs_np = np.concatenate([obs_np, pad], axis=-1)  # [N,T,24]
        act_np = poses
        obs_frames.append(obs_np.reshape(-1, 24))
        act_frames.append(act_np.reshape(-1, 18))
    elif seq_dir.exists():
        # Load a subset for speed
        files = sorted(seq_dir.glob("*_root.npy"))
        max_seqs = min(500, len(files))
        for i, rf in enumerate(files[:max_seqs]):
            base = rf.name[:-9]  # remove _root.npy
            try:
                r = np.load(seq_dir / f"{base}_root.npy")  # [T,6]
                v = np.load(seq_dir / f"{base}_vels.npy")  # [T,18]
                p = np.load(seq_dir / f"{base}_poses.npy") # [T,18]
            except Exception:
                continue
            rv = r[:, 3:6]
            obs_np = np.concatenate([rv, v], axis=-1)  # [T,21]
            pad = np.zeros((obs_np.shape[0], 3), dtype=obs_np.dtype)
            obs_np = np.concatenate([obs_np, pad], axis=-1)  # [T,24]
            act_np = p
            obs_frames.append(obs_np)
            act_frames.append(act_np)
        if not obs_frames:
            print("No sequences found in processed/sequences")
            return 1
    else:
        print("No processed AMASS data found (poses.npy or sequences folder).")
        return 1

    obs_t = torch.from_numpy(np.concatenate(obs_frames, axis=0)).float()
    act_t = torch.from_numpy(np.concatenate(act_frames, axis=0)).float()

    policy = TinyPolicy(obs_dim=24, act_dim=18)
    opt = optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    policy.train()
    for epoch in range(args.epochs):
        pred = policy(obs_t)
        loss = loss_fn(pred, act_t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"epoch {epoch+1}/{args.epochs} loss {loss.item():.4f}")

    out = data / "policy_supervised.pt"
    torch.save(policy.state_dict(), out)
    print(f"Saved policy to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


