"""Controlled degradation experiment: Tanager's 221 bands versus Sentinel-2's 10.

Everything except the band set is held fixed -- same synthetic scenes, same forward
model, same architecture, same training schedule, same seeds. The only difference is
whether the network sees 221 contiguous Tanager bands or those same spectra convolved to
Sentinel-2A's L2A surface bands. Any difference in skill is therefore attributable to
spectral resolution rather than to calibration, geolocation, atmospheric correction or
acquisition geometry.

The uncertainty propagation deliberately favours Sentinel-2 (see
``s2compare.propagate_uncertainty``), so the reported hyperspectral advantage is a lower
bound.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.experiment_s2 --n 150000 --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import s2compare, synth
from .model import RetrievalNet, build_features, fit_standardiser, gaussian_nll

ROOT = Path(__file__).resolve().parents[2]


def train_one(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch: int,
    lr: float,
    width: int,
    seed: int,
    device: torch.device,
    label: str,
):
    torch.manual_seed(seed)
    std = fit_standardiser(x, y)
    xt = torch.from_numpy(std.encode_x(x).astype(np.float32))
    yt = torch.from_numpy(((y - std.y_mean) / std.y_std).astype(np.float32))
    xv = torch.from_numpy(std.encode_x(x_val).astype(np.float32)).to(device)
    yv_phys = y_val

    net = RetrievalNet(x.shape[1], width=width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(xt, yt), batch_size=batch, shuffle=True)

    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            mean, logvar = net(xb)
            gaussian_nll(mean, logvar, yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        sched.step()

    net.eval()
    with torch.no_grad():
        mean, logvar = net(xv)
    pred = std.decode_y(mean.cpu().numpy())
    sigma = std.decode_y_sigma(np.exp(0.5 * logvar.cpu().numpy()))
    print(f"  trained {label}: {x.shape[1]} input features")
    return pred, sigma, yv_phys


def skill_table(pred: np.ndarray, truth: np.ndarray, names) -> dict:
    out = {}
    for j, nm in enumerate(names):
        err = pred[:, j] - truth[:, j]
        rec = {
            "rmse": float(np.sqrt((err**2).mean())),
            "r2": float(1.0 - (err**2).mean() / truth[:, j].var()),
        }
        if nm in synth.CONDITIONAL_PARAMS:
            host, thr = synth.CONDITIONAL_PARAMS[nm]
            k = list(names).index(host)
            m = truth[:, k] > thr
            e = pred[m, j] - truth[m, j]
            rec["rmse_conditional"] = float(np.sqrt((e**2).mean()))
            rec["r2_conditional"] = float(1.0 - (e**2).mean() / truth[m, j].var())
        out[nm] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=150_000)
    ap.add_argument("--n-val", type=int, default=30_000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mu0", type=float, default=0.624)
    ap.add_argument("--out", type=Path, default=ROOT / "5_outputs" / "sensor_comparison.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    tr = synth.generate(n=args.n, mu0=args.mu0, seed=args.seed)
    va = synth.generate(n=args.n_val, mu0=args.mu0, seed=args.seed + 9973)
    names = synth.PARAM_NAMES
    print(
        f"synthetic: {args.n} train / {args.n_val} val; "
        f"Tanager {tr.wavelength.size} bands, Sentinel-2 {len(s2compare.S2_L2A_BANDS)} bands"
    )

    results = {}

    x_tr = build_features(tr.reflectance, tr.uncertainty)
    x_va = build_features(va.reflectance, va.uncertainty)
    pred, _, truth = train_one(
        x_tr, tr.params, x_va, va.params, args.epochs, args.batch, args.lr,
        args.width, args.seed, device, f"Tanager ({tr.wavelength.size} bands)",
    )
    results["tanager"] = skill_table(pred, truth, names)

    r_tr = s2compare.convolve(tr.reflectance, tr.wavelength)
    u_tr = s2compare.propagate_uncertainty(tr.uncertainty, tr.wavelength)
    r_va = s2compare.convolve(va.reflectance, va.wavelength)
    u_va = s2compare.propagate_uncertainty(va.uncertainty, va.wavelength)
    pred_s2, _, _ = train_one(
        build_features(r_tr, u_tr), tr.params,
        build_features(r_va, u_va), va.params,
        args.epochs, args.batch, args.lr, args.width, args.seed, device,
        f"Sentinel-2 ({len(s2compare.S2_L2A_BANDS)} bands)",
    )
    results["sentinel2"] = skill_table(pred_s2, truth, names)

    print(f"\n{'parameter':30s} {'Tanager':>12} {'Sentinel-2':>12} {'RMSE ratio':>11}")
    print(f"{'':30s} {'RMSE':>12} {'RMSE':>12} {'S2 / Tanager':>11}")
    summary = {}
    for nm in names:
        key = "rmse_conditional" if "rmse_conditional" in results["tanager"][nm] else "rmse"
        a = results["tanager"][nm][key]
        b = results["sentinel2"][nm][key]
        ratio = b / a if a > 0 else float("nan")
        summary[nm] = {"tanager_rmse": a, "sentinel2_rmse": b, "ratio": ratio, "metric": key}
        print(f"{nm:30s} {a:12.5f} {b:12.5f} {ratio:11.2f}x")

    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "n_bands_tanager": int(tr.wavelength.size),
        "n_bands_sentinel2": len(s2compare.S2_L2A_BANDS),
        "sentinel2_bands": list(s2compare.S2_L2A_BANDS),
        "detail": results,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
