"""Optical constants for ice and liquid water, resampled onto Tanager's band centres.

Sources
-------
Ice   : Warren, S. G. and Brandt, R. E. (2008), "Optical constants of ice from the
        ultraviolet to the microwave: A revised compilation", J. Geophys. Res., 113, D14220.
        File: ``IOP_2008_ASCIItable.dat`` (wavelength [um], n, k).
Water : Segelstein, D. J. (1981), "The complex refractive index of water", MSc thesis,
        University of Missouri-Kansas City. Distributed by omlc.org as
        ``segelstein81.txt`` (wavelength [nm], absorption coefficient [1/cm]).

Both files are redistributed in ``1_data/optical_constants/`` and are public-domain /
freely distributed reference data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Repo root is three levels up from this file: 3_src/tanager_cryo/optics.py
_DATA = Path(__file__).resolve().parents[2] / "1_data" / "optical_constants"

ICE_FILE = _DATA / "ice_warren_brandt_2008.dat"
WATER_FILE = _DATA / "water_segelstein_1981.txt"


def _load_ice() -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength_nm, k_ice) from Warren & Brandt (2008), ascending in wavelength."""
    raw = np.loadtxt(ICE_FILE)
    wl_nm = raw[:, 0] * 1000.0  # file is in micrometres
    k = raw[:, 2]
    order = np.argsort(wl_nm)
    return wl_nm[order], k[order]


def _load_water() -> tuple[np.ndarray, np.ndarray]:
    """Return (wavelength_nm, absorption_per_m) from Segelstein (1981).

    The omlc distribution is a two-column table: wavelength [nm], absorption [1/cm].
    A header and a trailing metadata block are skipped by requiring exactly two
    float-parseable fields per line.
    """
    wl, a_cm = [], []
    with open(WATER_FILE) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                w, a = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            wl.append(w)
            a_cm.append(a)
    wl_nm = np.asarray(wl)
    a_per_m = np.asarray(a_cm) * 100.0  # 1/cm -> 1/m
    order = np.argsort(wl_nm)
    return wl_nm[order], a_per_m[order]


def ice_absorption(wavelength_nm: np.ndarray) -> np.ndarray:
    """Bulk absorption coefficient of pure ice, alpha = 4*pi*k/lambda, in 1/m.

    Parameters
    ----------
    wavelength_nm
        Band centres to evaluate at. Values outside the tabulated range are
        clipped to the nearest tabulated endpoint by ``np.interp``.
    """
    wl_tab, k_tab = _load_ice()
    # Interpolate log(k): k spans ~10 orders of magnitude across the VSWIR.
    log_k = np.interp(wavelength_nm, wl_tab, np.log(k_tab))
    k = np.exp(log_k)
    lam_m = np.asarray(wavelength_nm, dtype=float) * 1e-9
    return 4.0 * np.pi * k / lam_m


def water_absorption(wavelength_nm: np.ndarray) -> np.ndarray:
    """Absorption coefficient of pure liquid water in 1/m."""
    wl_tab, a_tab = _load_water()
    return np.exp(np.interp(wavelength_nm, wl_tab, np.log(a_tab)))


def ice_refractive_index(wavelength_nm: np.ndarray) -> np.ndarray:
    """Real part of the refractive index of ice."""
    raw = np.loadtxt(ICE_FILE)
    wl_nm = raw[:, 0] * 1000.0
    order = np.argsort(wl_nm)
    return np.interp(wavelength_nm, wl_nm[order], raw[order, 1])
