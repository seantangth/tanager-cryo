"""Independent validation of the sub-pixel fractions against same-day Sentinel-2 at 10 m.

The retrieval's central claim is a *sub-pixel* one: that from a single 30 m Tanager
spectrum it can say what proportion of that pixel is solid ice, melt pond and open water.
Validating that against synthetic truth only tests self-consistency. This module tests it
against the real surface, using a completely different information channel.

Why this comparison is fair
---------------------------
- **Same day, 29 minutes apart.** Tanager 2025-06-06 18:12:48 UTC, Sentinel-2C
  17:43:35 UTC. Sea ice drifts metres in half an hour, not kilometres, so the two
  instruments saw the same floes in the same places. An ICESat-2 comparison was attempted
  first and abandoned: the three tracks crossing this scene in June 2025 are flagged
  100% cloud-covered, and the nearest usable one would in any case have been six days
  later, by which time the pack has moved 12-60 km.
- **Exact grid nesting.** Both products sit in EPSG:32617, and Tanager's 30 m pixel edges
  fall on Sentinel-2's 10 m grid lines. Every Tanager pixel maps onto exactly 3x3
  Sentinel-2 pixels, so no resampling is needed and none of the comparison is an artefact
  of interpolation.
- **Different information channel.** Tanager infers the fractions *spectrally*, by
  unmixing 221 bands within one 30 m footprint. Sentinel-2 resolves them *spatially*: at
  10 m most pixels are pure, so counting classified pixels within each 3x3 block gives an
  independent estimate that owes nothing to the forward model being tested.

What this is not
----------------
The Sentinel-2 classification is a classification, not ground truth. It carries its own
errors, particularly on mixed 10 m pixels at floe edges and on thin ice, where pond and
open water are genuinely hard to separate. The comparison bounds agreement between two
independent methods; it does not establish absolute accuracy.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]

# Scene Classification Layer values that mark a Sentinel-2 pixel unusable.
SCL_REJECT = {0, 1, 3, 8, 9, 10}  # no-data, saturated, shadow, cloud med/high, cirrus

CLASS_NAMES = ("solid", "pond", "water")


# Sentinel-2 L2A COGs on Earth Search store reflectance as uint16 DN. The GDAL scale tag
# is left at 1.0, so it cannot be relied on; the conversion is the documented Copernicus
# one, DN / 10000. The BOA offset introduced at processing baseline 04.00 has already been
# applied by the STAC provider (``earthsearch:boa_offset_applied: true``), so no further
# offset is needed here. Trusting the GDAL tag instead put every pixel at a reflectance of
# several thousand and classified the entire scene as solid ice.
S2_DN_SCALE = 1.0e-4


def read_window(href: str, bounds: tuple[float, float, float, float]) -> tuple[np.ndarray, dict]:
    """Windowed read of a COG over the given projected bounds — no full-tile download."""
    with rasterio.open(href) as src:
        win = from_bounds(*bounds, transform=src.transform)
        arr = src.read(1, window=win, boundless=True, fill_value=0)
        prof = {"crs": src.crs, "res": src.res, "nodata": src.nodata}
    return arr, prof


def classify_s2(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Classify 10 m Sentinel-2 pixels into solid ice/snow, melt pond, or open water.

    The discriminators follow the standard optical logic for Arctic summer sea ice, and
    each is physically motivated rather than tuned:

    - **Open water** is dark everywhere. Liquid water absorbs strongly beyond ~700 nm, so
      the near-infrared is near zero, and with no ice floor to scatter light back the
      visible stays dark too.
    - **A melt pond** is water over a scattering ice floor: the near-infrared is still
      killed by the water column, but the visible stays bright because photons reach the
      ice beneath and come back. Depth reddens it — blue survives the column better than
      red — so ponds run blue-dominant.
    - **Solid ice and snow** scatter strongly at all four bands, including the
      near-infrared.

    Returns an int8 array: 0 solid, 1 pond, 2 water, -1 invalid.
    """
    out = np.full(blue.shape, -1, dtype=np.int8)

    # Near-infrared separates anything with a water surface from anything without.
    wet = nir < 0.10
    # A scattering floor beneath the water is what makes a pond bright in the visible.
    bright_vis = green > 0.15
    # Depth reddens: over a pond the blue survives the column better than the red.
    blue_dominant = blue > red * 1.05

    out[valid & ~wet] = 0
    out[valid & wet & bright_vis & blue_dominant] = 1
    out[valid & wet & ~(bright_vis & blue_dominant)] = 2
    return out


def dark_fraction_linear(
    nir: np.ndarray,
    valid: np.ndarray,
    bright_percentile: float = 95.0,
) -> np.ndarray:
    """Threshold-free dark-surface fraction per 10 m pixel, from the near-infrared alone.

    The hard classifier above has a bias that matters. A 10 m Sentinel-2 pixel that is
    half open water still reads NIR ~ 0.17, above the 0.10 cut, and is counted as fully
    solid. Every partially wet pixel is therefore scored as dry, and the classifier's
    total systematically *under*-reports water. Comparing an unmixing against it would
    charge Tanager for that bias.

    So we also compute a threshold-free estimate. In the near-infrared, liquid water is
    effectively black and ice is bright, and reflectance is close to linear in the wet
    fraction, so

        f_dark = (R_bright - R_obs) / (R_bright - R_dark)

    with ``R_bright`` taken from the scene's own bright tail and ``R_dark`` from the
    Fresnel floor. This shares Tanager's inability to separate open water from thin dark
    ice -- both read as "dark in the NIR" -- but it does not share the mixed-pixel bias,
    so between them the two references bracket the truth rather than pointing the same way.
    """
    v = nir[valid]
    r_bright = float(np.percentile(v, bright_percentile))
    r_dark = 0.02
    f = (r_bright - nir) / max(r_bright - r_dark, 1e-6)
    out = np.clip(f, 0.0, 1.0).astype(np.float32)
    out[~valid] = np.nan
    return out


def aggregate_mean_30m(field: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Mean of a 10 m field within each 3x3 block, ignoring NaN."""
    a = field[: ny * 3, : nx * 3].reshape(ny, 3, nx, 3)
    return np.nanmean(a, axis=(1, 3)).astype(np.float32)


def aggregate_to_30m(classes: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Count each class within every 3x3 block, returning fractions of the valid pixels.

    Shape (3, ny, nx), ordered as CLASS_NAMES. Blocks with no valid Sentinel-2 pixel come
    back NaN rather than zero, so an absence of data never masquerades as an absence of
    that surface.
    """
    c = classes[: ny * 3, : nx * 3].reshape(ny, 3, nx, 3)
    frac = np.full((3, ny, nx), np.nan, dtype=np.float32)
    valid_count = (c >= 0).sum(axis=(1, 3)).astype(np.float32)
    ok = valid_count > 0
    for k in range(3):
        cnt = (c == k).sum(axis=(1, 3)).astype(np.float32)
        frac[k][ok] = cnt[ok] / valid_count[ok]
    return frac


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retrieval", type=Path, default=ROOT / "5_outputs" / "sirmilik_retrieval.nc")
    ap.add_argument("--s2-json", type=Path, required=True, help="STAC item JSON with asset hrefs")
    ap.add_argument("--out", type=Path, default=ROOT / "5_outputs" / "s2_validation.json")
    ap.add_argument("--out-nc", type=Path, default=ROOT / "5_outputs" / "s2_validation.nc")
    args = ap.parse_args()

    ret = xr.open_dataset(args.retrieval)
    x, y = ret.x.values, ret.y.values
    res = abs(float(x[1] - x[0]))
    bounds = (x.min() - res / 2, y.min() - res / 2, x.max() + res / 2, y.max() + res / 2)
    ny, nx = ret.sizes["y"], ret.sizes["x"]
    print(f"Tanager grid {ny} x {nx} at {res:.0f} m; bounds {[round(b) for b in bounds]}")

    s2 = json.loads(args.s2_json.read_text())
    print(f"Sentinel-2 {s2['id']} at {s2['properties']['datetime']}")

    bands = {}
    for name in ("blue", "green", "red", "nir"):
        arr, prof = read_window(s2["assets"][name], bounds)
        bands[name] = arr.astype(np.float32) * S2_DN_SCALE
        v = bands[name][arr > 0]
        print(f"  {name:6s} {arr.shape} at {prof['res'][0]:.0f} m   "
              f"reflectance median {np.median(v):.3f}")

    scl, _ = read_window(s2["assets"]["scl"], bounds)
    # SCL ships at 20 m; repeat it up to the 10 m grid before masking.
    if scl.shape != bands["blue"].shape:
        scl = np.kron(scl, np.ones((2, 2), dtype=scl.dtype))[
            : bands["blue"].shape[0], : bands["blue"].shape[1]
        ]
    valid = ~np.isin(scl, list(SCL_REJECT)) & (bands["green"] > 0)
    print(f"  usable Sentinel-2 pixels: {valid.sum():,} of {valid.size:,} "
          f"({100 * valid.mean():.1f}%)")

    classes = classify_s2(bands["blue"], bands["green"], bands["red"], bands["nir"], valid)
    for k, nm in enumerate(CLASS_NAMES):
        print(f"    {nm:6s} {100 * (classes == k).sum() / max(valid.sum(), 1):5.1f}% of usable")

    frac = aggregate_to_30m(classes, ny, nx)

    # Second, independent reference: threshold-free dark fraction from the NIR.
    dark10 = dark_fraction_linear(bands["nir"], valid)
    dark30 = aggregate_mean_30m(dark10, ny, nx)

    tan = {
        "solid": ret["f_solid"].values,
        "pond": ret["f_pond"].values,
        "water": ret["f_water"].values,
    }
    # Tanager's "dark" total is open water plus the pond surface, which is what the NIR
    # sees: a pond is as black as open water beyond ~900 nm.
    tan_dark = np.where(
        np.isfinite(tan["water"]) & np.isfinite(tan["pond"]),
        tan["water"] + tan["pond"], np.nan,
    )
    md = np.isfinite(tan_dark) & np.isfinite(dark30)
    d = tan_dark[md] - dark30[md]
    print(f"\nthreshold-free NIR dark fraction (open water + pond surface)")
    print(f"  n {md.sum():,}   Tanager mean {tan_dark[md].mean():.3f}   "
          f"Sentinel-2 mean {dark30[md].mean():.3f}")
    print(f"  bias {d.mean():+.4f}   RMSE {np.sqrt((d**2).mean()):.4f}   "
          f"r {np.corrcoef(tan_dark[md], dark30[md])[0, 1]:+.3f}")
    dark_metrics = {
        "n": int(md.sum()), "bias": float(d.mean()),
        "rmse": float(np.sqrt((d**2).mean())),
        "pearson_r": float(np.corrcoef(tan_dark[md], dark30[md])[0, 1]),
        "tanager_mean": float(tan_dark[md].mean()),
        "sentinel2_mean": float(dark30[md].mean()),
    }

    results = {}
    print(f"\nhard classification (biased low on water; see dark_fraction_linear)")
    print(f"{'fraction':10s} {'n':>8} {'bias':>9} {'RMSE':>8} {'MAE':>8} {'r':>7}")
    for k, nm in enumerate(CLASS_NAMES):
        a, b = tan[nm], frac[k]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 100:
            print(f"{nm:10s} too few overlapping pixels ({m.sum()})")
            continue
        d = a[m] - b[m]
        r = float(np.corrcoef(a[m], b[m])[0, 1])
        results[nm] = {
            "n": int(m.sum()), "bias": float(d.mean()),
            "rmse": float(np.sqrt((d**2).mean())), "mae": float(np.abs(d).mean()),
            "pearson_r": r,
            "tanager_mean": float(a[m].mean()), "sentinel2_mean": float(b[m].mean()),
        }
        print(f"{nm:10s} {m.sum():8d} {d.mean():+9.4f} {np.sqrt((d**2).mean()):8.4f} "
              f"{np.abs(d).mean():8.4f} {r:7.3f}")

    out = xr.Dataset(coords={"y": ret.y, "x": ret.x, "spatial_ref": ret.spatial_ref})
    for k, nm in enumerate(CLASS_NAMES):
        out[f"s2_f_{nm}"] = (("y", "x"), frac[k])
        out[f"tanager_f_{nm}"] = (("y", "x"), tan[nm].astype(np.float32))
    out["s2_dark_linear"] = (("y", "x"), dark30)
    out["tanager_dark"] = (("y", "x"), tan_dark.astype(np.float32))
    out.attrs.update({
        "sentinel2_id": s2["id"],
        "sentinel2_datetime": s2["properties"]["datetime"],
        "tanager_datetime": "2025-06-06T18:12:48Z",
        "separation_minutes": 29.2,
    })
    args.out_nc.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(args.out_nc, encoding={v: {"zlib": True, "complevel": 5} for v in out.data_vars})
    args.out.write_text(json.dumps(
        {"sentinel2": {"id": s2["id"], "datetime": s2["properties"]["datetime"],
                       "cloud_cover": s2["properties"].get("eo:cloud_cover")},
         "separation_minutes": 29.2,
         "metrics_hard_classification": results,
         "metrics_dark_fraction_linear": dark_metrics}, indent=2))
    print(f"\nwrote {args.out} and {args.out_nc}")


if __name__ == "__main__":
    main()
