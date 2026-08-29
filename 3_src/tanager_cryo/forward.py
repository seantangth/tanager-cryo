"""Forward radiative-transfer model for melting Arctic surfaces at Tanager resolution.

The model produces top-of-canopy (surface) reflectance for the four endmembers that
dominate a summer sea-ice scene, then mixes them linearly at the 30 m pixel scale:

    snow       : semi-infinite, weakly absorbing scattering medium (ART)
    bare ice   : same formalism with a much longer absorption path
    melt pond  : liquid water column over an ice bottom (two-way Beer-Lambert)
    open water : dark, near-Fresnel surface

Snow / ice reflectance
----------------------
We use the asymptotic radiative transfer (ART) result of Kokhanovsky & Zege (2004),
the same formulation underlying the Sentinel-3 SICE snow product (Kokhanovsky et al.,
2019). For a semi-infinite, weakly absorbing medium the spherical albedo is

    r_s(lambda) = exp( -sqrt( alpha(lambda) * L ) )

where ``alpha`` is the bulk absorption coefficient (1/m) and ``L`` the *effective
absorption length* (m). ``L`` is the physically identifiable quantity: it is what the
spectrum actually constrains. It maps to an optical grain diameter through a shape
factor, ``L = B * d``, with ``B`` of order a few for convex grains. We retrieve ``L``
and report ``d`` only through an explicitly stated ``B`` (see ``GRAIN_SHAPE_FACTOR``),
rather than burying the assumption inside the retrieval.

Under direct illumination the plane albedo follows the ART escape function

    r_p(mu0) = r_s ** K0(mu0),      K0(mu) = (3/7) * (1 + 2*mu)

Light-absorbing particles (dust, black carbon, algae) are added as an Angstrom-type
absorption on top of the ice absorption, which is how they manifest: a visible-wavelength
darkening that leaves the near-infrared ice features essentially untouched.
"""

from __future__ import annotations

import numpy as np

from .optics import ice_absorption, water_absorption

# Shape factor relating effective absorption length to optical grain diameter,
# L = B * d. Stated explicitly so the grain-size conversion stays auditable.
GRAIN_SHAPE_FACTOR = 3.62

# Angstrom reference wavelength for light-absorbing particles.
LAP_REFERENCE_NM = 500.0

# Fresnel reflectance of a flat water surface at moderate solar zenith.
WATER_SURFACE_REFLECTANCE = 0.021


def escape_function(mu0: float | np.ndarray) -> np.ndarray:
    """ART escape function K0(mu) = (3/7)(1 + 2*mu)."""
    return (3.0 / 7.0) * (1.0 + 2.0 * np.asarray(mu0, dtype=float))


def lap_absorption(
    wavelength_nm: np.ndarray,
    lap_load: float,
    angstrom_exponent: float = 1.0,
) -> np.ndarray:
    """Angstrom-type absorption from light-absorbing particles, in 1/m.

    ``lap_load`` is the absorption coefficient at ``LAP_REFERENCE_NM``; setting it to
    zero recovers pure ice.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    return lap_load * (wl / LAP_REFERENCE_NM) ** (-angstrom_exponent)


def snow_reflectance(
    wavelength_nm: np.ndarray,
    absorption_length_m: float,
    mu0: float = 1.0,
    lap_load: float = 0.0,
    angstrom_exponent: float = 1.0,
    liquid_water_path_m: float = 0.0,
) -> np.ndarray:
    """Plane albedo of a semi-infinite snow / ice medium.

    Parameters
    ----------
    absorption_length_m
        Effective absorption length ``L``. Fine dry snow is of order 1e-4 m;
        coarse wet snow 1e-3 m; bare glacial or sea ice 1e-2 m and longer.
    mu0
        Cosine of the solar zenith angle.
    lap_load
        Absorption coefficient of light-absorbing particles at 500 nm (1/m).
    liquid_water_path_m
        Path length through interstitial liquid water. Non-zero values deepen the
        980 and 1200 nm water features relative to pure ice, which is what separates
        wet from dry snow.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    alpha = ice_absorption(wl) + lap_absorption(wl, lap_load, angstrom_exponent)
    r_s = np.exp(-np.sqrt(np.clip(alpha * absorption_length_m, 0.0, None)))

    if liquid_water_path_m > 0.0:
        r_s = r_s * np.exp(-water_absorption(wl) * liquid_water_path_m)

    return np.clip(r_s ** escape_function(mu0), 0.0, 1.0)


# A finite-thickness ice endmember was implemented here and removed. The reasoning is
# worth keeping: over pixels Sentinel-2 resolves as continuous ice, the observed spectrum
# has the *shape* of the semi-infinite model at roughly half its amplitude -- the
# observed/modelled ratio is 0.52 at 421 nm and 0.52 at 1032 nm. That reduction is
# essentially grey, and a grey darkening is mathematically indistinguishable from mixing
# in a grey dark surface, which is exactly what the open-water endmember already does.
# Adding a transmittance term therefore bought no new identifiability, only a relabelling;
# and the physically-motivated transmittance shape, exp(-c*sqrt(alpha*L)), collapses far
# too sharply across the visible (T = 0.76 at 421 nm but 0.006 at 661 nm) to reproduce the
# observations, fitting them at RMSE 0.195.
#
# The honest conclusion is a limit, not a missing term: at 30 m over melting sea ice,
# open water and thin dark ice are not separable by passive optics. See the memo.

def melt_pond_reflectance(
    wavelength_nm: np.ndarray,
    depth_m: float,
    bottom_absorption_length_m: float = 5.0e-3,
    mu0: float = 1.0,
    bottom_lap_load: float = 0.0,
) -> np.ndarray:
    """Reflectance of a melt pond: a water column over a scattering ice bottom.

    Two-way attenuation through the column, with the in-water path lengthened by
    refraction toward the vertical (Snell), plus a Fresnel surface term.
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    bottom = snow_reflectance(
        wl,
        absorption_length_m=bottom_absorption_length_m,
        mu0=mu0,
        lap_load=bottom_lap_load,
    )
    # Refraction into water pulls the beam toward nadir; the in-water cosine is
    # always closer to 1 than mu0, so the geometric path stretch is modest.
    mu_w = np.sqrt(1.0 - (1.0 - float(mu0) ** 2) / 1.333**2)
    two_way = np.exp(-2.0 * water_absorption(wl) * depth_m / mu_w)
    return np.clip(WATER_SURFACE_REFLECTANCE + bottom * two_way, 0.0, 1.0)


def open_water_reflectance(
    wavelength_nm: np.ndarray,
    chlorophyll_like: float = 0.0,
) -> np.ndarray:
    """Reflectance of open water: Fresnel surface plus a small visible water-leaving term."""
    wl = np.asarray(wavelength_nm, dtype=float)
    # Water-leaving radiance is confined to the visible; beyond ~750 nm absorption
    # by the water column makes the surface effectively black.
    leaving = chlorophyll_like * np.exp(-((wl - 550.0) ** 2) / (2.0 * 60.0**2))
    nir_kill = np.exp(-water_absorption(wl) * 0.05)
    return np.clip(WATER_SURFACE_REFLECTANCE + leaving * nir_kill, 0.0, 1.0)


def mix(fractions: np.ndarray, endmembers: np.ndarray) -> np.ndarray:
    """Linear spectral mixture.

    Parameters
    ----------
    fractions
        Shape ``(..., n_endmembers)``, non-negative, summing to one.
    endmembers
        Shape ``(n_endmembers, n_bands)``.
    """
    return np.asarray(fractions) @ np.asarray(endmembers)
