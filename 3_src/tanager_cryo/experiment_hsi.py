"""Controlled degradation experiment: Tanager versus the EMIT and PRISMA band sets.

Neither EMIT nor PRISMA can acquire the Sirmilik scene -- EMIT stops at the ISS's
±52°, PRISMA at ±70° -- so no real-data comparison is possible at this latitude.
What *can* be answered quantitatively is whether the Arctic observation gap is a
spectral limitation or purely an orbital one: degrade Tanager's bands to each
sensor's published spectral sampling and see how much retrieval skill survives.

Protocol is identical to ``experiment_s2`` -- same synthetic scenes, same forward
model, same architecture, same schedule, same seeds; only the band set changes.
The Tanager baseline is retrained inside this run rather than borrowed from
``sensor_comparison.json``, so every ratio is between arms trained under identical
conditions. All sensors are evaluated over the identical 381–1778 nm interval that
the SNR cut leaves Tanager (the same discipline under which ``s2compare`` excludes
Sentinel-2's B12).

Band models. Neither mission publishes a per-band SRF table the way ESA does for
Sentinel-2, so bands are modelled as Gaussians at the published nominal figures:

- EMIT: ~7.4 nm sampling, 8.5 nm FWHM (on-orbit; Green et al. 2023, RSE).
- PRISMA: ~11 nm sampling, 12 nm FWHM (≤12 nm requirement; Cogliati et al. 2021, RSE).

Synthetic bands whose Gaussian falls mostly into a region Tanager does not cover
(the water-vapour gaps, the interval edges) are dropped by the same ≥60% retained-
response rule ``s2compare`` applies -- those regions are atmospherically opaque for
EMIT and PRISMA too. Uncertainty propagation deliberately favours the degraded
sensor, exactly as in ``s2compare.propagate_uncertainty``, so any Tanager advantage
reported is a lower bound.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.experiment_hsi --n 150000 --epochs 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import synth
from .experiment_s2 import skill_table, train_one
from .model import build_features

ROOT = Path(__file__).resolve().parents[2]

# Published nominal spectral characteristics (sampling interval, FWHM), in nm.
SENSORS = {
    "emit": {"sampling": 7.4, "fwhm": 8.5, "start": 381.0,
             "note": "on-orbit values, Green et al. 2023"},
    "prisma": {"sampling": 11.0, "fwhm": 12.0, "start": 402.0,
               "note": "nominal values, Cogliati et al. 2021"},
}

MIN_RETAINED = 0.60  # same coverage rule as s2compare.convolve


def gaussian_band_weights(
    tanager_wavelength: np.ndarray, sampling: float, fwhm: float, start: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weights, centers) for Gaussian bands laid on the Tanager grid.

    Weights have shape (n_tanager_bands, n_synthetic_bands), columns normalised to
    sum to one. Bands retaining less than ``MIN_RETAINED`` of their continuous
    Gaussian response within Tanager's (gappy) coverage are dropped rather than
    silently distorted.
    """
    wl = np.asarray(tanager_wavelength, dtype=float)
    centers = np.arange(start, wl.max(), sampling)
    sigma = fwhm / 2.3548200450309493  # FWHM -> Gaussian sigma

    raw = np.exp(-0.5 * ((wl[:, None] - centers[None, :]) / sigma) ** 2)
    sampled = np.trapezoid(raw, wl, axis=0)
    full = sigma * np.sqrt(2.0 * np.pi)
    keep = (sampled / full) >= MIN_RETAINED

    w = raw[:, keep]
    return w / np.clip(w.sum(axis=0, keepdims=True), 1e-12, None), centers[keep]


def degrade(
    reflectance: np.ndarray, uncertainty: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convolve reflectance and conservatively propagate uncertainty.

    Independent-error propagation through a weighted mean shrinks the variance,
    which favours the degraded sensor (a real EMIT or PRISMA band is not a
    noise-free average of Tanager bands). Kept deliberately, as in s2compare.
    """
    r = np.asarray(reflectance) @ weights
    u = np.sqrt(np.asarray(uncertainty) ** 2 @ weights**2)
    return r.astype(np.float32), u.astype(np.float32)


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
    ap.add_argument("--out", type=Path, default=ROOT / "5_outputs" / "sensor_comparison_hsi.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")

    tr = synth.generate(n=args.n, mu0=args.mu0, seed=args.seed)
    va = synth.generate(n=args.n_val, mu0=args.mu0, seed=args.seed + 9973)
    names = synth.PARAM_NAMES

    results, band_counts = {}, {"tanager": int(tr.wavelength.size)}

    pred, _, truth = train_one(
        build_features(tr.reflectance, tr.uncertainty), tr.params,
        build_features(va.reflectance, va.uncertainty), va.params,
        args.epochs, args.batch, args.lr, args.width, args.seed, device,
        f"Tanager ({tr.wavelength.size} bands)",
    )
    results["tanager"] = skill_table(pred, truth, names)

    for sensor, spec in SENSORS.items():
        w, centers = gaussian_band_weights(
            tr.wavelength, spec["sampling"], spec["fwhm"], spec["start"]
        )
        band_counts[sensor] = int(centers.size)
        r_tr, u_tr = degrade(tr.reflectance, tr.uncertainty, w)
        r_va, u_va = degrade(va.reflectance, va.uncertainty, w)
        pred_s, _, _ = train_one(
            build_features(r_tr, u_tr), tr.params,
            build_features(r_va, u_va), va.params,
            args.epochs, args.batch, args.lr, args.width, args.seed, device,
            f"{sensor} ({centers.size} bands, {spec['sampling']} nm sampling)",
        )
        results[sensor] = skill_table(pred_s, truth, names)

    summary = {}
    print(f"\n{'parameter':30s} {'Tanager':>10} {'EMIT':>10} {'PRISMA':>10}   RMSE ratios vs Tanager")
    for nm in names:
        key = "rmse_conditional" if "rmse_conditional" in results["tanager"][nm] else "rmse"
        a = results["tanager"][nm][key]
        row = {"metric": key, "tanager_rmse": a}
        line = f"{nm:30s} {a:10.5f}"
        for sensor in SENSORS:
            b = results[sensor][nm][key]
            row[f"{sensor}_rmse"] = b
            row[f"{sensor}_ratio"] = b / a if a > 0 else float("nan")
            line += f" {b:10.5f}"
        summary[nm] = row
        print(line + f"   emit {row['emit_ratio']:.2f}x   prisma {row['prisma_ratio']:.2f}x")

    payload = {
        "config": vars(args) | {"out": str(args.out)},
        "band_models": SENSORS,
        "evaluation_interval_nm": [float(tr.wavelength.min()), float(tr.wavelength.max())],
        "n_bands": band_counts,
        "detail": results,
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
