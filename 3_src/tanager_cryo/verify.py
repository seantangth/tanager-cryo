"""Re-check every number quoted in the technical memo against the pipeline's own outputs.

A memo full of hand-copied figures is a memo that drifts from its code. This script reads
the produced artefacts and asserts each claim, so a reader can confirm in one command that
the prose matches the pipeline. It exits non-zero if anything disagrees.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.verify
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "5_outputs"

_failures: list[str] = []


def check(label: str, actual, expected, tol: float | None = None) -> None:
    if tol is None:
        ok = actual == expected
        shown = f"{actual!r}"
    else:
        ok = abs(float(actual) - float(expected)) <= tol
        shown = f"{float(actual):.4f}"
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {label:52s} {shown}  (memo: {expected})")
    if not ok:
        _failures.append(label)


def main() -> int:
    print("verifying the technical memo against pipeline outputs\n")

    print("observation gap")
    g = json.loads((OUT / "observation_gap.json").read_text())
    s = g["langtang"]["summary"]
    check("EMIT granules covering the Langtang source zone", s["n_observations"], 20)
    check("of those, usable at < 55% cloud", s["n_usable"], 2)
    check("last usable observation before failure", s["last_usable_before_failure"], "2024-04-06")
    a = dt.date.fromisoformat(s["last_usable_before_failure"])
    b = dt.date.fromisoformat(g["langtang"]["failure_date"])
    check("gap, months", (b.year - a.year) * 12 + b.month - a.month, 28)
    check("EMIT observations of Sirmilik", g["sirmilik"]["n_observations"], 0)

    print("\nsensor comparison")
    c = json.loads((OUT / "sensor_comparison.json").read_text())
    check("Tanager bands", c["n_bands_tanager"], 221)
    check("Sentinel-2 L2A bands", c["n_bands_sentinel2"], 10)
    for key, expected in (
        ("log10_absorption_length_m", 13.1),
        ("lap_load", 5.5),
        ("f_pond", 5.2),
        ("f_solid", 4.8),
        ("f_water", 4.1),
        ("pond_depth_m", 1.8),
    ):
        check(f"RMSE ratio, {key}", c["summary"][key]["ratio"], expected, tol=0.06)

    h = json.loads((OUT / "sensor_comparison_hsi.json").read_text())
    check("EMIT synthetic bands", h["n_bands"]["emit"], 150)
    check("PRISMA synthetic bands", h["n_bands"]["prisma"], 100)
    hsi_ratios = [
        h["summary"][nm][f"{s}_ratio"]
        for nm in h["summary"] for s in ("emit", "prisma")
    ]
    check("EMIT/PRISMA ratio minimum", min(hsi_ratios), 0.95, tol=0.01)
    check("EMIT/PRISMA ratio maximum", max(hsi_ratios), 1.03, tol=0.01)

    print("\nscene retrieval")
    import xarray as xr

    d = xr.open_dataset(OUT / "sirmilik_retrieval.nc")
    fwd = d.attrs["forward_model_error_median"]
    inst = d.attrs["instrument_sigma_median"]
    check("forward-model error, reflectance", fwd, 0.0354, tol=0.0008)
    check("instrument sigma, reflectance", inst, 0.0013, tol=0.0002)
    check("ratio, forward-model / instrument", fwd / inst, 27.0, tol=1.5)
    for var, expected in (
        ("f_solid", 0.28), ("f_pond", 0.16), ("f_water", 0.56), ("pond_depth_m", 0.05),
    ):
        check(f"scene median, {var}", float(np.nanmedian(d[var].values)), expected, tol=0.011)
    chi2 = d["reduced_chi2"].values
    outside = d["outside_prior"].values
    okpx = np.isfinite(chi2)
    explained = float(((chi2 <= 4.0) & (outside == 0))[okpx].mean())
    check("pixels explained at chi2_nu <= 4, %", 100 * explained, 30.1, tol=0.15)

    print("\nindependent validation against same-day Sentinel-2")
    v = json.loads((OUT / "s2_validation.json").read_text())
    dm = v["metrics_dark_fraction_linear"]
    check("Sentinel-2 separation, minutes", v["separation_minutes"], 29.2, tol=0.1)
    check("dark-fraction Pearson r", dm["pearson_r"], 0.850, tol=0.01)
    check("dark-fraction bias", dm["bias"], 0.291, tol=0.01)
    check("Tanager mean dark fraction", dm["tanager_mean"], 0.730, tol=0.01)
    check("Sentinel-2 mean dark fraction", dm["sentinel2_mean"], 0.439, tol=0.01)
    check("pond hard-classification r (disclosed limit)",
          v["metrics_hard_classification"]["pond"]["pearson_r"], -0.34, tol=0.01)

    val_nc = OUT / "s2_validation.nc"
    if val_nc.exists():
        from scipy.stats import spearmanr

        d2 = xr.open_dataset(val_nc)
        a = d2["tanager_dark"].values
        b = d2["s2_dark_linear"].values
        ok = np.isfinite(a) & np.isfinite(b)
        check("dark-fraction Spearman rho",
              float(spearmanr(a[ok], b[ok]).statistic), 0.860, tol=0.005)
        edges = np.linspace(0.0, 1.0, 11)
        means = []
        for k in range(10):
            sel = (b[ok] >= edges[k]) & (b[ok] < edges[k + 1])
            if sel.sum() > 50:
                means.append(float(a[ok][sel].mean()))
        check("populated conditional-mean bins", len(means), 10)
        check("binned means strictly monotonic",
              all(x < y for x, y in zip(means, means[1:])), True)
        keep = ~((a[ok] > 0.8) & (b[ok] > 0.8))
        check("Pearson r with high-high cluster removed",
              float(np.corrcoef(a[ok][keep], b[ok][keep])[0, 1]), 0.73, tol=0.01)
    else:
        print("  (s2_validation.nc absent -- run tanager_cryo.s2_validate to enable "
              "the Spearman / monotonicity checks)")

    print("\naccuracy and calibration on held-out synthetic spectra")
    e = json.loads((OUT / "evaluation.json").read_text())
    acc, cal = e["accuracy"], e["calibration"]
    for label, val, expected, tol in (
        ("melt pond fraction RMSE", acc["f_pond"]["rmse"], 0.0062, 0.0002),
        ("snow/ice fraction RMSE", acc["f_solid"]["rmse"], 0.0165, 0.0005),
        ("open water fraction RMSE", acc["f_water"]["rmse"], 0.0185, 0.0005),
        ("absorption length RMSE, conditional",
         acc["log10_absorption_length_m"]["rmse_conditional"], 0.0224, 0.0008),
        ("pond depth RMSE m, conditional",
         acc["pond_depth_m"]["rmse_conditional"], 0.0101, 0.0005),
        ("impurity load RMSE, conditional",
         acc["lap_load"]["rmse_conditional"], 5.62, 0.15),
    ):
        check(label, val, expected, tol=tol)
    vz = [cal[nm]["var_z"] for nm in cal]
    c68 = [cal[nm]["coverage_68"] for nm in cal]
    c95 = [cal[nm]["coverage_95"] for nm in cal]
    check("var(z) minimum", min(vz), 1.05, tol=0.015)
    check("var(z) maximum", max(vz), 1.08, tol=0.015)
    check("68% coverage minimum", min(c68), 0.65, tol=0.01)
    check("68% coverage maximum", max(c68), 0.70, tol=0.01)
    check("95% coverage minimum", min(c95), 0.94, tol=0.01)
    check("95% coverage maximum", max(c95), 0.96, tol=0.01)

    print("\nband selection")
    from . import synth

    wl, _, _ = synth.load_band_selection()
    check("bands retained at SNR > 30", int(wl.size), 221)
    check("shortest retained wavelength, nm", float(wl.min()), 381.0, tol=1.0)
    check("longest retained wavelength, nm", float(wl.max()), 1778.0, tol=1.0)

    print()
    if _failures:
        print(f"{len(_failures)} claim(s) do not match the outputs:")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print("all claims match the pipeline outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
