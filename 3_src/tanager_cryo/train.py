"""Train the cryosphere retrieval emulator on synthetic Tanager spectra.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.train --n 120000 --epochs 60

Everything is seeded and CPU/MPS agnostic; a full run takes a few minutes on an
Apple-silicon laptop. The trained weights, the standardiser, and the band selection are
written to ``4_models/`` so inference never has to recompute the training set.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import synth
from .model import (
    RetrievalNet,
    build_features,
    fit_standardiser,
    gaussian_nll,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "4_models"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=120_000, help="synthetic training spectra")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu0", type=float, default=0.624, help="cos(solar zenith)")
    ap.add_argument("--snr-threshold", type=float, default=synth.DEFAULT_SNR_THRESHOLD)
    ap.add_argument("--out", type=Path, default=MODEL_DIR / "cryo_retrieval.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    t0 = time.time()
    train = synth.generate(
        n=args.n, mu0=args.mu0, seed=args.seed, snr_threshold=args.snr_threshold
    )
    val = synth.generate(
        n=max(args.n // 10, 4_000),
        mu0=args.mu0,
        seed=args.seed + 9973,
        snr_threshold=args.snr_threshold,
    )
    print(
        f"synthetic set: {train.reflectance.shape[0]} train / {val.reflectance.shape[0]} val, "
        f"{train.wavelength.size} bands ({train.wavelength.min():.0f}-"
        f"{train.wavelength.max():.0f} nm) in {time.time() - t0:.1f}s"
    )

    x_tr = build_features(train.reflectance, train.uncertainty)
    x_va = build_features(val.reflectance, val.uncertainty)
    std = fit_standardiser(x_tr, train.params)

    def tens(x, y):
        return TensorDataset(
            torch.from_numpy(std.encode_x(x).astype(np.float32)),
            torch.from_numpy(((y - std.y_mean) / std.y_std).astype(np.float32)),
        )

    dl_tr = DataLoader(tens(x_tr, train.params), batch_size=args.batch, shuffle=True)
    xv, yv = tens(x_va, val.params).tensors
    xv, yv = xv.to(device), yv.to(device)

    net = RetrievalNet(x_tr.shape[1], width=args.width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        run = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mean, logvar = net(xb)
            loss = gaussian_nll(mean, logvar, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            run += loss.item() * xb.shape[0]
        sched.step()

        net.eval()
        with torch.no_grad():
            mean, logvar = net(xv)
            vnll = gaussian_nll(mean, logvar, yv).item()
            vrmse = torch.sqrt(((mean - yv) ** 2).mean(0)).cpu().numpy()

        if vnll < best:
            best = vnll
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "n_features": x_tr.shape[1],
                    "width": args.width,
                    "standardiser": {k: v.tolist() for k, v in std.to_dict().items()},
                    "band_index": train.band_index.tolist(),
                    "wavelength": train.wavelength.tolist(),
                    "param_names": list(synth.PARAM_NAMES),
                    "mu0": args.mu0,
                    "snr_threshold": args.snr_threshold,
                },
                args.out,
            )
        if ep % 5 == 0 or ep == 1:
            print(
                f"  epoch {ep:3d}  train NLL {run / len(dl_tr.dataset):8.4f}  "
                f"val NLL {vnll:8.4f}  val RMSE(std units) {np.round(vrmse, 3)}"
            )

    print(f"\nbest val NLL {best:.4f} -> {args.out}")
    meta = {
        "n_train": args.n,
        "epochs": args.epochs,
        "bands": int(train.wavelength.size),
        "wavelength_min_nm": float(train.wavelength.min()),
        "wavelength_max_nm": float(train.wavelength.max()),
        "snr_threshold": args.snr_threshold,
        "best_val_nll": best,
    }
    (MODEL_DIR / "cryo_retrieval_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
