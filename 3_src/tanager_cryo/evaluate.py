"""Evaluate the trained emulator on held-out synthetic spectra.

Two things are reported, and the second is the one that matters.

*Accuracy*, both unconditional and conditional. Pond depth is only defined where there
is a pond; impurity load and grain size are only defined where there is a solid surface.
Reporting a single unconditional RMSE for those parameters mostly measures how often the
parameter was absent, not how well it is retrieved when present.

*Calibration*. A heteroscedastic network is only useful if its stated uncertainty is
honest. We check the standardised residual z = (truth - mean) / sigma: if the predictive
distribution is well calibrated, z has unit variance and the empirical coverage of the
central 68% and 95% intervals matches nominal. An overconfident model -- the usual
failure -- shows z-variance well above 1 and coverage below nominal.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.evaluate --model 4_models/cryo_retrieval.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import synth
from .model import RetrievalNet, Standardiser, build_features

ROOT = Path(__file__).resolve().parents[2]


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = RetrievalNet(ckpt["n_features"], n_params=len(ckpt["param_names"]), width=ckpt["width"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    std = Standardiser(**{k: np.asarray(v) for k, v in ckpt["standardiser"].items()})
    return net, std, ckpt


def predict(net, std, device, reflectance, uncertainty, batch=8192):
    """Return (mean, sigma) in physical units."""
    x = std.encode_x(build_features(reflectance, uncertainty)).astype(np.float32)
    means, sigmas = [], []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            xb = torch.from_numpy(x[i : i + batch]).to(device)
            m, lv = net(xb)
            means.append(m.cpu().numpy())
            sigmas.append(np.exp(0.5 * lv.cpu().numpy()))
    m = np.concatenate(means)
    s = np.concatenate(sigmas)
    return std.decode_y(m), std.decode_y_sigma(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=ROOT / "4_models" / "cryo_retrieval.pt")
    ap.add_argument("--n", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", type=Path, default=ROOT / "5_outputs" / "evaluation.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net, std, ckpt = load_model(args.model, device)
    names = ckpt["param_names"]

    ts = synth.generate(
        n=args.n, mu0=ckpt["mu0"], seed=args.seed, snr_threshold=ckpt["snr_threshold"]
    )
    mean, sigma = predict(net, std, device, ts.reflectance, ts.uncertainty)
    truth = ts.params

    accuracy: dict[str, dict] = {}
    print(f"held-out synthetic spectra: {args.n}\n")
    print(f"{'parameter':30s} {'RMSE':>10} {'bias':>10} {'R2':>7}   {'conditional':>28}")
    for j, nm in enumerate(names):
        err = mean[:, j] - truth[:, j]
        rmse = float(np.sqrt((err**2).mean()))
        bias = float(err.mean())
        var = truth[:, j].var()
        r2 = float(1.0 - (err**2).mean() / var) if var > 0 else float("nan")
        accuracy[nm] = {"rmse": rmse, "bias": bias, "r2": r2}

        cond_txt = ""
        if nm in synth.CONDITIONAL_PARAMS:
            host, thr = synth.CONDITIONAL_PARAMS[nm]
            k = names.index(host)
            m = truth[:, k] > thr
            if m.sum() > 50:
                e = mean[m, j] - truth[m, j]
                v = truth[m, j].var()
                accuracy[nm]["rmse_conditional"] = float(np.sqrt((e**2).mean()))
                accuracy[nm]["r2_conditional"] = float(1.0 - (e**2).mean() / v)
                cond_txt = (
                    f"{host}>{thr}: RMSE {np.sqrt((e**2).mean()):.4f} "
                    f"R2 {1.0 - (e**2).mean() / v:5.3f}"
                )
        print(f"{nm:30s} {rmse:10.4f} {bias:10.4f} {r2:7.3f}   {cond_txt:>28}")

    calibration: dict[str, dict] = {}
    print("\ncalibration of the predictive uncertainty")
    print(f"{'parameter':30s} {'var(z)':>8} {'68% cov':>9} {'95% cov':>9}   verdict")
    for j, nm in enumerate(names):
        z = (truth[:, j] - mean[:, j]) / np.clip(sigma[:, j], 1e-9, None)
        vz = float(z.var())
        c68 = float((np.abs(z) < 1.0).mean())
        c95 = float((np.abs(z) < 1.96).mean())
        calibration[nm] = {"var_z": vz, "coverage_68": c68, "coverage_95": c95}
        if vz > 1.6:
            verdict = "overconfident"
        elif vz < 0.5:
            verdict = "conservative"
        else:
            verdict = "well calibrated"
        print(f"{nm:30s} {vz:8.2f} {c68:9.3f} {c95:9.3f}   {verdict}")
    print("\nnominal coverage: 0.683 and 0.950; var(z) should be near 1.0")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"n": args.n, "seed": args.seed, "model": str(args.model.name),
         "accuracy": accuracy, "calibration": calibration}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
