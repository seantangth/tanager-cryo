"""Build the submission figures.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.figures --all
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from . import s2compare, synth, viz
from .optics import water_absorption

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "5_outputs" / "figures"
OUTDIR = ROOT / "5_outputs"


def _save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


# --- figure 1: retrieval maps -------------------------------------------------
def fig_retrieval_maps(retrieval: Path, scene_visual: Path | None = None) -> Path:
    ds = xr.open_dataset(retrieval)
    panels = [
        ("f_pond", "Melt pond fraction", "", viz.CMAP_SEQ, None),
        ("pond_depth_m", "Melt pond depth", "m", viz.CMAP_SEQ, "f_pond"),
        ("f_solid", "Snow / ice fraction", "", viz.CMAP_SEQ, None),
        ("grain_diameter_mm", "Effective grain diameter", "mm", viz.CMAP_SEQ_ALT, "f_solid"),
        ("f_pond_sigma", "Pond fraction, 1σ", "", viz.CMAP_SEQ_ALT, None),
        ("reduced_chi2", "Goodness of fit  χ²ᵥ", "", viz.CMAP_DIV, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 8.0))
    for ax, (var, title, unit, cmap, gate) in zip(axes.ravel(), panels):
        a = ds[var].values.astype(float)
        if gate is not None:
            a = np.where(ds[gate].values > 0.25, a, np.nan)
        if var == "reduced_chi2":
            vmin, vmax = 0.0, 4.0
        elif var == "grain_diameter_mm":
            # Tight percentiles: a small tail of very long retrieved paths would
            # otherwise flatten the contrast across the range that carries the signal.
            vmin, vmax = viz.robust_limits(a, 5.0, 90.0)
        else:
            vmin, vmax = viz.robust_limits(a)
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7.5, colors=viz.INK_SECONDARY)
        cb.outline.set_visible(False)
        if unit:
            cb.set_label(unit, fontsize=8, color=viz.INK_MUTED)
        if gate is not None:
            ax.text(
                0.02, 0.02, f"shown where {gate} > 0.25",
                transform=ax.transAxes, fontsize=7, color=viz.INK_MUTED,
                va="bottom", ha="left",
            )
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.text(
        0.012, 0.975,
        "Tanager-1 cryosphere retrieval — Sirmilik National Park, Nunavut, 2025-06-06",
        ha="left", va="top", fontsize=13, color=viz.INK,
    )
    fig.text(
        0.012, 0.936,
        f"{ds.attrs.get('n_bands')} bands, 30 m. Blank pixels are cloud, no-data, or not "
        "explained by the three-endmember model — the land in the lower half is exactly "
        "that, and the goodness-of-fit panel finds it unaided.",
        ha="left", va="top", fontsize=8.5, color=viz.INK_MUTED,
    )
    return _save(fig, "fig1_retrieval_maps.png")


# --- figure 2: sensor comparison ----------------------------------------------
def fig_sensor_comparison(summary_json: Path) -> Path:
    payload = json.loads(summary_json.read_text())
    summary = payload["summary"]
    labels = {
        "f_solid": "Snow / ice fraction",
        "f_pond": "Melt pond fraction",
        "f_water": "Open water fraction",
        "log10_absorption_length_m": "Grain size (log₁₀ L)",
        "pond_depth_m": "Melt pond depth",
        "lap_load": "Impurity load",
    }
    names = [k for k in labels if k in summary]
    ratios = np.array([summary[k]["ratio"] for k in names])
    order = np.argsort(ratios)[::-1]
    names = [names[i] for i in order]
    ratios = ratios[order]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ypos = np.arange(len(names))
    ax.barh(ypos, ratios, color=viz.SERIES[0], height=0.62,
            edgecolor=viz.SURFACE, linewidth=2)
    ax.axvline(1.0, color=viz.INK_MUTED, lw=1.0, ls="--")
    ax.text(1.0, -0.62, "  1.0 = no advantage", fontsize=8,
            color=viz.INK_MUTED, va="center", ha="left")
    for y, r in zip(ypos, ratios):
        ax.text(r + max(ratios) * 0.015, y, f"{r:.1f}×", va="center",
                fontsize=9, color=viz.INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[n] for n in names])
    ax.set_xlim(0, max(ratios) * 1.14)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title(
        f"Retrieval error using Sentinel-2's {payload['n_bands_sentinel2']} bands, "
        f"relative to Tanager's {payload['n_bands_tanager']}"
    )
    viz.annotate_units(ax, "RMSE ratio — higher means more is lost by coarsening the spectrum")
    fig.text(
        0.012, -0.02,
        "Identical synthetic scenes, forward model, architecture, schedule and seeds; the only "
        "difference is the band set.\nTanager spectra convolved with the official ESA "
        "Sentinel-2A response functions. Noise propagation favours Sentinel-2, so these are "
        "lower bounds.",
        ha="left", fontsize=7.6, color=viz.INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, "fig2_sensor_comparison.png")


# --- figure 3: why the spectrum matters ---------------------------------------
def fig_why_hyperspectral() -> Path:
    wl, _, _ = synth.load_band_selection()
    a_w = water_absorption(wl)
    depth_1e = 1.0 / np.clip(a_w, 1e-9, None)

    srf_wl, srf, names = s2compare.load_srf()
    centres = [(float((srf_wl * srf[:, j]).sum() / srf[:, j].sum()), names[j])
               for j in range(len(names))]

    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    band = (wl >= 400) & (wl <= 1350)
    ax.semilogy(wl[band], depth_1e[band], color=viz.SERIES[0], label="Tanager — 221 bands")
    for c, nm in centres:
        if 400 <= c <= 1350:
            ax.axvline(c, color=viz.SERIES[1], lw=1.6, alpha=0.85)
    ax.plot([], [], color=viz.SERIES[1], lw=1.6, label="Sentinel-2 band centres")

    ax.axhspan(0.1, 0.5, color=viz.SERIES[2], alpha=0.14)
    ax.text(1290, 0.22, "typical Arctic\nmelt pond depth", fontsize=8,
            color=viz.INK_SECONDARY, ha="right", va="center")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("1/e penetration depth in liquid water (m)")
    # State the ratio rather than "almost no bands" -- the figure plainly shows five
    # Sentinel-2 centres in this window, and a title the figure contradicts is worse than
    # a weaker one that it supports.
    ax.set_title(
        "Melt pond depth is encoded in a 200 nm window that Tanager samples 8x more densely"
    )
    ax.grid(True, which="both")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    n_tan = int(((wl >= 640) & (wl <= 840)).sum())
    n_s2 = sum(1 for c, _ in centres if 640 <= c <= 840)
    fig.text(
        0.012, -0.03,
        f"Between 640 and 840 nm — where water's 1/e penetration depth sweeps through the "
        f"0.1–0.5 m range that Arctic melt ponds occupy —\nTanager has {n_tan} contiguous "
        f"bands and Sentinel-2 has {n_s2}. Absorption from Segelstein (1981).",
        ha="left", fontsize=7.6, color=viz.INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, "fig3_why_hyperspectral.png")


# --- figure 4: observation gap ------------------------------------------------
def fig_observation_gap(gap_json: Path) -> Path:
    import datetime as dt

    g = json.loads(gap_json.read_text())
    recs = g["langtang"]["records"]
    fail = dt.date.fromisoformat(g["langtang"]["failure_date"])
    thr = g["langtang"]["summary"]["usable_cloud_max"]

    dates = [dt.date.fromisoformat(r["time"]) for r in recs]
    clouds = [r["cloud_cover"] if r["cloud_cover"] is not None else np.nan for r in recs]
    usable = [c < thr for c in clouds]

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.scatter([d for d, u in zip(dates, usable) if not u],
               [c for c, u in zip(clouds, usable) if not u],
               s=42, color=viz.INK_MUTED, alpha=0.55, label=f"cloud ≥ {thr:.0f}% — unusable")
    ax.scatter([d for d, u in zip(dates, usable) if u],
               [c for c, u in zip(clouds, usable) if u],
               s=95, color=viz.SERIES[0], zorder=3, label=f"cloud < {thr:.0f}% — usable")
    ax.axvline(fail, color=viz.STATUS["critical"], lw=2.0)
    ax.text(fail, 103, "  2026-08-26\n  rock–ice failure", fontsize=8.5,
            color=viz.STATUS["critical"], va="top", ha="left")

    last_usable = dt.date.fromisoformat(g["langtang"]["summary"]["last_usable_before_failure"])
    ax.annotate(
        "", xy=(fail, 8), xytext=(last_usable, 8),
        arrowprops=dict(arrowstyle="<->", color=viz.INK_SECONDARY, lw=1.3),
    )
    months = (fail.year - last_usable.year) * 12 + fail.month - last_usable.month
    ax.text(last_usable + (fail - last_usable) / 2, 13,
            f"{months} months with no usable observation",
            fontsize=9, color=viz.INK, ha="center")

    ax.set_ylim(0, 108)
    ax.set_ylabel("Scene cloud cover (%)")
    ax.set_title("Every EMIT observation of the Langtang source zone")
    site = g["langtang"]["site"]
    viz.annotate_units(ax, f"{site['lat']:.4f} N, {site['lon']:.4f} E — the point that failed")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    fig.text(
        0.012, -0.04,
        f"{g['langtang']['summary']['n_observations']} EMIT L2A granules cover this point in "
        f"three and a half years; {g['langtang']['summary']['n_usable']} are usable. "
        f"At Sirmilik (73.7 N) the count is structurally zero:\nEMIT stops at ±52° (ISS), "
        "PRISMA at ±70°, and EnMAP's tasked archive holds no systematic sea-ice coverage.",
        ha="left", fontsize=7.6, color=viz.INK_MUTED,
    )
    fig.tight_layout()
    return _save(fig, "fig4_observation_gap.png")


# --- figure 5: calibration ----------------------------------------------------
def fig_calibration(model_path: Path, n: int = 30_000) -> Path:
    import torch

    from .evaluate import load_model, predict

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net, std, ckpt = load_model(model_path, device)
    ts = synth.generate(n=n, mu0=ckpt["mu0"], seed=20260829,
                        snr_threshold=ckpt["snr_threshold"])
    mean, sigma = predict(net, std, device, ts.reflectance, ts.uncertainty)
    names = ckpt["param_names"]
    labels = {
        "f_solid": "Snow / ice fraction", "f_pond": "Melt pond fraction",
        "f_water": "Open water fraction", "log10_absorption_length_m": "Grain size (log₁₀ L)",
        "pond_depth_m": "Melt pond depth", "lap_load": "Impurity load",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    ax = axes[0]
    levels = np.linspace(0.02, 0.99, 40)
    for j, nm in enumerate(names):
        z = np.abs((ts.params[:, j] - mean[:, j]) / np.clip(sigma[:, j], 1e-9, None))
        from scipy.stats import norm
        emp = [(z < norm.ppf(0.5 + lv / 2)).mean() for lv in levels]
        colour = viz.SERIES[j % 3]
        ls = "-" if j < 3 else "--"
        ax.plot(levels, emp, color=colour, ls=ls, lw=1.7, label=labels[nm])
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, lw=1.0, ls=":")
    ax.text(0.62, 0.55, "perfect calibration", rotation=38, fontsize=8,
            color=viz.INK_MUTED, ha="center")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("The network's stated uncertainty is honest")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", ncol=1)

    ax = axes[1]
    vz = [float(((ts.params[:, j] - mean[:, j]) / np.clip(sigma[:, j], 1e-9, None)).var())
          for j in range(len(names))]
    ypos = np.arange(len(names))
    ax.barh(ypos, vz, color=viz.SERIES[0], height=0.6,
            edgecolor=viz.SURFACE, linewidth=2)
    ax.axvline(1.0, color=viz.INK_MUTED, lw=1.0, ls="--")
    ax.text(1.0, len(names) - 0.35, " 1.0", fontsize=8, color=viz.INK_MUTED,
            ha="left", va="center")
    for y, v in zip(ypos, vz):
        ax.text(v + 0.02, y, f"{v:.2f}", va="center", fontsize=9, color=viz.INK)
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[n] for n in names])
    ax.set_xlim(0, max(max(vz) * 1.2, 1.3))
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_title("Variance of the standardised residual")
    viz.annotate_units(ax, "above 1 = overconfident,  below 1 = conservative")
    fig.tight_layout()
    return _save(fig, "fig5_calibration.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--retrieval", type=Path, default=OUTDIR / "sirmilik_retrieval.nc")
    ap.add_argument("--comparison", type=Path, default=OUTDIR / "sensor_comparison.json")
    ap.add_argument("--gap", type=Path, default=OUTDIR / "observation_gap.json")
    ap.add_argument("--model", type=Path, default=ROOT / "4_models" / "cryo_retrieval.pt")
    args = ap.parse_args()

    viz.use_style()
    print("building figures")
    if args.retrieval.exists():
        fig_retrieval_maps(args.retrieval)
    fig_why_hyperspectral()
    if args.gap.exists():
        fig_observation_gap(args.gap)
    if args.model.exists():
        fig_calibration(args.model)
    if args.comparison.exists():
        fig_sensor_comparison(args.comparison)
    else:
        print("  (sensor comparison not yet available)")
    val_nc = OUTDIR / "s2_validation.nc"
    val_js = OUTDIR / "s2_validation.json"
    if val_nc.exists() and val_js.exists():
        fig_s2_validation(val_nc, val_js)
    else:
        print("  (Sentinel-2 validation not yet available)")




# --- figure 6: independent validation against same-day Sentinel-2 ------------------
def fig_s2_validation(nc: Path, metrics_json: Path) -> Path:
    from scipy.stats import spearmanr

    ds = xr.open_dataset(nc)
    m = json.loads(metrics_json.read_text())
    dm = m["metrics_dark_fraction_linear"]

    a = ds["tanager_dark"].values
    b = ds["s2_dark_linear"].values
    ok = np.isfinite(a) & np.isfinite(b)
    # Show Sentinel-2 only where Tanager also has a value, so the two maps are a like
    # for like comparison rather than a comparison of coverage.
    b_shown = np.where(ok, b, np.nan)

    fig = plt.figure(figsize=(12.6, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2], wspace=0.30)

    for i, (arr, title) in enumerate((
        (a, "Tanager  ·  221 bands unmixed at 30 m"),
        (b_shown, "Sentinel-2  ·  10 m NIR, aggregated 3x3"),
    )):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(arr, cmap=viz.CMAP_SEQ, vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7.5, colors=viz.INK_SECONDARY)
        cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist2d(b[ok], a[ok], bins=80, range=[[0, 1], [0, 1]],
              cmap=viz.CMAP_SEQ, cmin=1, alpha=0.85)
    ax.plot([0, 1], [0, 1], color=viz.INK_MUTED, lw=1.2, ls="--")
    ax.text(0.70, 0.65, "1:1", fontsize=8, color=viz.INK_MUTED, rotation=41)

    # The binned conditional mean is the honest summary: it shows the relationship is
    # monotonic while making the compressed range impossible to miss.
    edges = np.linspace(0, 1, 11)
    cx, cy = [], []
    for k in range(10):
        sel = (b[ok] >= edges[k]) & (b[ok] < edges[k + 1])
        if sel.sum() > 50:
            cx.append(0.5 * (edges[k] + edges[k + 1]))
            cy.append(a[ok][sel].mean())
    ax.plot(cx, cy, "o-", color=viz.SERIES[1], lw=2.0, ms=5,
            label="binned mean of Tanager")
    ax.set_xlabel("Sentinel-2 dark fraction")
    ax.set_ylabel("Tanager dark fraction")
    rho = spearmanr(a[ok], b[ok]).statistic
    # Recompute the cluster-robustness number here rather than asserting it: drop the
    # dense high-high corner (both instruments > 0.8) and correlate what remains.
    keep = ~((a[ok] > 0.8) & (b[ok] > 0.8))
    r_nocluster = float(np.corrcoef(a[ok][keep], b[ok][keep])[0, 1])
    monotonic = all(y1 < y2 for y1, y2 in zip(cy, cy[1:]))
    ax.set_title(f"Spearman rho = {rho:+.3f}   ·   Pearson r = {dm['pearson_r']:+.3f}",
                 fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True); ax.set_axisbelow(True)
    ax.legend(loc="lower right")

    fig.text(0.012, 1.06,
             "Independent check: dark-surface fraction, two instruments 29 minutes apart",
             ha="left", va="top", fontsize=12.5, color=viz.INK)
    fig.text(
        0.012, -0.14,
        f"Tanager 2025-06-06 18:12:48 UTC unmixes 221 bands within each 30 m pixel; "
        f"Sentinel-2C 17:43:35 UTC resolves the same ground at 10 m. The relationship is "
        f"{'monotonic' if monotonic else 'NOT monotonic'} across all {len(cx)} bins, and holds\n"
        f"at r = {r_nocluster:+.2f} with the dense high-high cluster (both > 0.8) removed, so "
        f"it is not an artefact of one corner. But Tanager's binned mean spans only "
        f"{min(cy):.2f}-{max(cy):.2f} across Sentinel-2's 0.05-0.95, and "
        f"sits {dm['bias']:+.2f} high on average.\n"
        f"Both follow from one limit: a semi-infinite ice endmember cannot represent thin dark "
        f"ice, so that darkness is attributed to open water instead.",
        ha="left", fontsize=7.6, color=viz.INK_MUTED,
    )
    return _save(fig, "fig6_s2_validation.png")



if __name__ == "__main__":
    main()
