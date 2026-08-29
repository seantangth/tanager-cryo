"""Forward-reconstruction residual: an out-of-distribution test with physical meaning.

A retrieval trained on synthetic spectra will happily return numbers for a pixel its
forward model cannot represent. On the Sirmilik scene this is not hypothetical: the
southern half is snow-covered land and exposed terrain, and the three-endmember model
(solid ice/snow, melt pond, open water) has no rock, soil or vegetation. Forced to
explain those pixels, the network drove the absorption length far outside its training
prior -- retrieved grain diameters reached 1 metre and impurity loads went negative.

Rather than widen the prior until the nonsense is hidden inside it, we test whether the
retrieval actually explains the observed spectrum. Push the retrieved parameters back
through the same forward model, and compare:

    chi2_nu = (1 / n_bands) * sum_b [ (R_obs(b) - R_model(b)) / sigma_obs(b) ]^2

Because ``sigma_obs`` is Tanager's own per-band uncertainty, this is a properly weighted
goodness of fit, not an unweighted RMSE: it asks whether the mismatch is large *relative
to what the instrument claims it can measure*. A pixel the model represents well sits
near chi2_nu ~ 1. Land, cloud shadow, and anything else outside the three-endmember
world sits far above it, and is excluded from the product with a stated reason.
"""

from __future__ import annotations

import numpy as np

from . import forward as F


def reconstruct(
    params: np.ndarray,
    wavelength: np.ndarray,
    mu0: float,
    chunk: int = 50_000,
) -> np.ndarray:
    """Rebuild spectra from retrieved parameters, vectorised over pixels.

    ``params`` columns follow ``synth.PARAM_NAMES``:
    (f_solid, f_pond, f_water, log10_L, pond_depth_m, lap_load).

    The optical constants are interpolated once and the whole forward model is then
    evaluated as array arithmetic. A per-pixel Python loop over a full scene is roughly
    three orders of magnitude slower and makes the residual test impractical to run,
    which would defeat its purpose.
    """
    from .optics import ice_absorption, water_absorption

    wl = np.asarray(wavelength, dtype=float)
    a_ice = ice_absorption(wl)[None, :]
    a_w = water_absorption(wl)[None, :]
    lap_shape = (wl / F.LAP_REFERENCE_NM)[None, :] ** -1.0
    k0 = float(F.escape_function(mu0))
    mu_w = np.sqrt(1.0 - (1.0 - float(mu0) ** 2) / 1.333**2)
    water_spec = F.open_water_reflectance(wl)[None, :]

    n = params.shape[0]
    out = np.empty((n, wl.size), dtype=np.float32)

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        p = params[start:stop]
        f_solid = p[:, 0:1]
        f_pond = p[:, 1:2]
        f_water = p[:, 2:3]
        L = np.clip(10.0 ** p[:, 3:4], 1e-6, 1.0)
        depth = np.clip(p[:, 4:5], 0.0, 5.0)
        lap = np.clip(p[:, 5:6], 0.0, None)

        alpha = a_ice + lap * lap_shape
        solid = np.clip(np.exp(-np.sqrt(np.clip(alpha * L, 0.0, None))) ** k0, 0.0, 1.0)

        # The pond floor carries no impurity load: ``melt_pond_reflectance`` leaves
        # ``bottom_lap_load`` at zero, and the synthetic training set was generated that
        # way. Adding ``lap`` here would silently make the residual test inconsistent
        # with the model the network was trained to invert.
        L_bottom = np.minimum(L * 8.0, 5e-2)
        bottom = np.clip(
            np.exp(-np.sqrt(np.clip(a_ice * L_bottom, 0.0, None))) ** k0, 0.0, 1.0
        )
        pond = np.clip(
            F.WATER_SURFACE_REFLECTANCE + bottom * np.exp(-2.0 * a_w * depth / mu_w),
            0.0,
            1.0,
        )

        out[start:stop] = (f_solid * solid + f_pond * pond + f_water * water_spec).astype(
            np.float32
        )
    return out


def estimate_model_error(
    observed: np.ndarray,
    modelled: np.ndarray,
    quantile: float = 0.5,
) -> np.ndarray:
    """Estimate per-band forward-model error from the residual distribution.

    Tanager's ``surface_reflectance_uncertainty`` describes the *instrument*: over
    Sirmilik it is 0.001-0.010 in reflectance, i.e. 0.5-1.8% relative through the
    visible and near-infrared. The three-endmember ART model is nowhere near that
    accurate -- it reproduces observed sea-ice reflectance to about 0.02-0.06.

    That gap is the useful measurement here: **the retrieval is forward-model limited,
    not instrument limited.** Tanager resolves structure that a simplified model cannot
    yet exploit. Weighting a goodness-of-fit test by instrument sigma alone therefore
    rejects every pixel in the scene, because it asks a 5% model to agree with a 1%
    measurement.

    We instead estimate the model error empirically, per band, as a robust spread of the
    residuals (scaled MAD, which is insensitive to the land pixels the model genuinely
    cannot represent), and add it in quadrature to the instrument term.
    """
    resid = observed - modelled
    med = np.median(resid, axis=0, keepdims=True)
    mad = np.median(np.abs(resid - med), axis=0)
    return 1.4826 * mad


def reduced_chi2(
    observed: np.ndarray,
    modelled: np.ndarray,
    sigma: np.ndarray,
    model_error: np.ndarray | None = None,
) -> np.ndarray:
    """Per-pixel reduced chi-square against the combined error budget.

    ``sigma`` is the instrument uncertainty; ``model_error`` the per-band forward-model
    error. They are added in quadrature, which is the standard error budget for an
    inversion: a fit can only be asked to agree to the accuracy of the poorer of the two.
    """
    total = np.clip(sigma, 1e-8, None)
    if model_error is not None:
        total = np.sqrt(total**2 + model_error[None, :] ** 2)
    resid = (observed - modelled) / total
    return (resid**2).mean(axis=1)


def prior_violation(
    params: np.ndarray,
    log_l_range: tuple[float, float],
    depth_range: tuple[float, float],
    lap_range: tuple[float, float],
) -> np.ndarray:
    """Flag pixels whose retrieval left the training prior.

    Extrapolation beyond the prior is not automatically wrong, but it is untested: the
    network never saw such spectra. Flagging it keeps that distinction visible in the
    product instead of burying it.
    """
    log_L, depth, lap = params[:, 3], params[:, 4], params[:, 5]
    pad_L = 0.1 * (log_l_range[1] - log_l_range[0])
    return (
        (log_L < log_l_range[0] - pad_L)
        | (log_L > log_l_range[1] + pad_L)
        | (depth < 0.0)
        | (depth > depth_range[1] * 1.5)
        | (lap < 0.0)
        | (lap > lap_range[1] * 1.5)
    )
