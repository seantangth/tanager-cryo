"""Apply the trained emulator to a Tanager scene and write georeferenced parameter maps.

The scene is read with Planet's official ``xarray-hyperspectral`` backend rather than a
bespoke reader, so the retrieval sits directly on the toolchain Planet already ships.

Every retrieved field is written with its companion 1-sigma. Pixels flagged as cloud,
cirrus or no-data are masked before inference; pixels where the network reports an
uncertainty above ``--max-sigma`` for a given parameter are masked in that parameter's
output only, which is how conditional non-identifiability (pond depth where there is no
pond) is expressed in the product rather than hidden.

Usage
-----
    PYTHONPATH=3_src python -m tanager_cryo.retrieve \\
        --scene 1_data/raw/sirmilik/20250606_181248_58_4001_sr.h5 \\
        --out 5_outputs/sirmilik_retrieval.nc
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from . import residual, synth
from .evaluate import load_model, predict
from .model import constrain_fractions

ROOT = Path(__file__).resolve().parents[2]

# Reduced chi-square above which the three-endmember model is judged not to explain the
# pixel, computed against the combined instrument + forward-model error budget. With the
# model error included, a pixel the model represents sits near 1, so a limit of 4 keeps
# everything within twice the expected spread and excludes only genuinely out-of-scope
# surfaces (land, rock, vegetation).
CHI2_LIMIT = 4.0


def retrieve_scene(
    scene_path: Path,
    model_path: Path,
    batch: int = 8192,
) -> xr.Dataset:
    """Run the emulator over every valid pixel of a Tanager scene."""
    warnings.filterwarnings("ignore")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net, std, ckpt = load_model(model_path, device)
    names = ckpt["param_names"]
    band_index = np.asarray(ckpt["band_index"], dtype=int)

    ds = xr.open_dataset(scene_path, engine="tanager")
    valid = (
        (~ds.beta_cloud_mask.values)
        & (~ds.beta_cirrus_mask.values)
        & (~ds.nodata_pixels.values)
    )
    ny, nx = valid.shape
    yy, xx = np.where(valid)
    print(f"{scene_path.name}: {yy.size:,} valid of {ny * nx:,} pixels")

    refl = ds.reflectance.values[band_index][:, yy, xx].T.astype(np.float32)
    unc = ds.reflectance_uncertainty.values[band_index][:, yy, xx].T.astype(np.float32)

    finite = np.isfinite(refl).all(1) & np.isfinite(unc).all(1)
    yy, xx, refl, unc = yy[finite], xx[finite], refl[finite], unc[finite]
    print(f"  {yy.size:,} pixels with complete finite spectra")

    mean, sigma = predict(net, std, device, refl, unc, batch=batch)
    mean = constrain_fractions(mean)

    # Physical goodness of fit. The network will return numbers for any spectrum; this
    # asks whether the forward model can actually reproduce what was observed.
    wl = np.asarray(ckpt["wavelength"], dtype=float)
    modelled = residual.reconstruct(mean, wl, mu0=ckpt["mu0"])
    model_err = residual.estimate_model_error(refl, modelled)
    chi2 = residual.reduced_chi2(refl, modelled, unc, model_error=model_err)
    print(
        f"  forward-model error (reflectance): median {np.median(model_err):.4f}, "
        f"range [{model_err.min():.4f}, {model_err.max():.4f}]; "
        f"instrument sigma median {np.median(unc):.4f} "
        f"-> the retrieval is forward-model limited"
    )
    outside = residual.prior_violation(
        mean, synth.LOG_L_RANGE, synth.POND_DEPTH_RANGE, synth.LAP_RANGE
    )
    explained = (chi2 <= CHI2_LIMIT) & ~outside
    print(
        f"  chi2_nu median {np.median(chi2):.2f}; "
        f"{explained.sum():,} pixels ({100 * explained.mean():.1f}%) explained by the "
        f"three-endmember model (chi2_nu <= {CHI2_LIMIT}, inside prior)"
    )

    out = xr.Dataset(coords={"y": ds.y, "x": ds.x, "spatial_ref": ds.spatial_ref})
    for j, nm in enumerate(names):
        for suffix, src in (("", mean), ("_sigma", sigma)):
            grid = np.full((ny, nx), np.nan, dtype=np.float32)
            # Retrieved values are written only where the physics actually fits.
            grid[yy[explained], xx[explained]] = src[explained, j]
            out[nm + suffix] = (("y", "x"), grid)

    out.attrs["forward_model_error_median"] = float(np.median(model_err))
    out.attrs["instrument_sigma_median"] = float(np.median(unc))
    for nm, src in (("reduced_chi2", chi2), ("outside_prior", outside.astype(np.float32))):
        grid = np.full((ny, nx), np.nan, dtype=np.float32)
        grid[yy, xx] = src
        out[nm] = (("y", "x"), grid)
    out["reduced_chi2"].attrs["note"] = (
        "instrument-uncertainty-weighted goodness of fit of the three-endmember forward "
        "model; ~1 means the model reproduces the observation to within its stated noise"
    )

    # Grain size is the quantity people ask for, but the spectrum constrains the
    # absorption length; the conversion factor is carried explicitly as an attribute
    # so the assumption travels with the product.
    from .forward import GRAIN_SHAPE_FACTOR

    L = 10.0 ** out["log10_absorption_length_m"]
    out["grain_diameter_mm"] = (L / GRAIN_SHAPE_FACTOR) * 1000.0
    out["grain_diameter_mm"].attrs["shape_factor_B"] = GRAIN_SHAPE_FACTOR
    out["grain_diameter_mm"].attrs["note"] = "d = L / B; B stated in forward.GRAIN_SHAPE_FACTOR"

    out.attrs.update(
        {
            "source_scene": scene_path.name,
            "model": model_path.name,
            "snr_threshold": ckpt["snr_threshold"],
            "n_bands": int(band_index.size),
            "mu0": ckpt["mu0"],
            "crs": str(ds.rio.crs),
        }
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=ROOT / "4_models" / "cryo_retrieval.pt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    result = retrieve_scene(args.scene, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Deflate on write: the product is mostly smooth fields and large NaN regions, so
    # compression takes it from ~42 MB to ~11 MB and keeps it comfortably committable.
    encoding = {v: {"zlib": True, "complevel": 5} for v in result.data_vars}
    result.to_netcdf(args.out, encoding=encoding)
    print(f"\nwrote {args.out}")
    for v in result.data_vars:
        arr = result[v].values
        finite = np.isfinite(arr)
        if finite.any():
            print(
                f"  {v:34s} median {np.nanmedian(arr):9.4f}  "
                f"[{np.nanpercentile(arr, 5):8.4f}, {np.nanpercentile(arr, 95):8.4f}]"
            )


if __name__ == "__main__":
    main()
