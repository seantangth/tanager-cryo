"""Quantify what Tanager's spectral resolution buys, against Sentinel-2 as the baseline.

Sentinel-2 is the right comparison for a cryosphere product. It is the sensor operational
snow and sea-ice work actually runs on: free, near-daily at 73 N because the orbits
converge, and 10-20 m. EMIT cannot be the comparison here at all -- the ISS orbit stops
at 52 degrees, so no spaceborne public imaging spectrometer sees this scene. That fact is
itself part of the argument for tasking Tanager to the poles.

Method
------
Rather than compare two differently-calibrated, differently-geolocated products and
confound spectral resolution with everything else, we degrade Tanager to Sentinel-2's
bands and hold everything else fixed:

1. Convolve each Tanager spectrum with the official ESA Sentinel-2A spectral response
   functions (COPE-GSEG-EOPG-TN-15-0007) to synthesise the bands an S2 user would have.
2. Train the same network architecture, on the same synthetic scenes, from those bands.
3. Compare retrieval skill parameter by parameter.

Only the L2A surface-reflectance bands are used. B9 (945 nm) and B10 (1373 nm) are
atmospheric sounding bands and are not delivered in L2A, so including them would
overstate what an S2 user actually has.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_DATA = Path(__file__).resolve().parents[2] / "1_data" / "optical_constants"

# Sentinel-2 L2A surface-reflectance bands. B9 and B10 are atmospheric-only; B12
# (2202 nm) falls outside the SNR-selected Tanager range and is excluded so that both
# sensors are evaluated over an identical spectral interval.
S2_L2A_BANDS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11")


def load_srf(bands: tuple[str, ...] = S2_L2A_BANDS) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (srf_wavelength_nm, srf_matrix, band_names) for the requested bands."""
    z = np.load(_DATA / "s2a_srf.npz", allow_pickle=True)
    wl = z["wavelength"].astype(float)
    srf = z["srf"].astype(float)
    names = [str(n) for n in z["names"]]
    cols = [names.index(b) for b in bands]
    return wl, srf[:, cols], list(bands)


def convolve(
    spectra: np.ndarray,
    tanager_wavelength: np.ndarray,
    bands: tuple[str, ...] = S2_L2A_BANDS,
) -> np.ndarray:
    """Convolve Tanager spectra to Sentinel-2 bands.

    Each output band is the response-weighted mean of the Tanager spectrum, with the
    response resampled onto Tanager's own band centres. Bands whose response falls
    largely outside Tanager's coverage would be biased, so the weight is renormalised by
    the response actually sampled and a band retaining less than 60% of its integrated
    response is rejected outright rather than silently distorted.
    """
    srf_wl, srf, names = load_srf(bands)
    wl = np.asarray(tanager_wavelength, dtype=float)

    weights = np.empty((wl.size, len(names)))
    for j in range(len(names)):
        weights[:, j] = np.interp(wl, srf_wl, srf[:, j], left=0.0, right=0.0)

    full = np.trapezoid(srf, srf_wl, axis=0)
    sampled = np.trapezoid(weights, wl, axis=0)
    retained = sampled / np.clip(full, 1e-12, None)
    keep = retained >= 0.60
    if not keep.all():
        dropped = [n for n, k in zip(names, keep) if not k]
        raise ValueError(
            f"Sentinel-2 bands {dropped} are not sufficiently covered by the Tanager "
            f"band selection (retained fraction {retained[~keep].round(2).tolist()})"
        )

    norm = weights.sum(axis=0, keepdims=True)
    return np.asarray(spectra) @ (weights / np.clip(norm, 1e-12, None))


def propagate_uncertainty(
    uncertainty: np.ndarray,
    tanager_wavelength: np.ndarray,
    bands: tuple[str, ...] = S2_L2A_BANDS,
) -> np.ndarray:
    """Propagate per-band uncertainty through the convolution.

    Treating the Tanager band errors as independent, a weighted mean of n bands reduces
    the variance by the sum of squared weights. This deliberately *favours* the
    degraded sensor -- a real S2 band is not a noise-free average of Tanager bands -- so
    any advantage the hyperspectral retrieval retains is a conservative estimate.
    """
    srf_wl, srf, names = load_srf(bands)
    wl = np.asarray(tanager_wavelength, dtype=float)
    w = np.empty((wl.size, len(names)))
    for j in range(len(names)):
        w[:, j] = np.interp(wl, srf_wl, srf[:, j], left=0.0, right=0.0)
    w = w / np.clip(w.sum(axis=0, keepdims=True), 1e-12, None)
    return np.sqrt(np.asarray(uncertainty) ** 2 @ (w**2))
