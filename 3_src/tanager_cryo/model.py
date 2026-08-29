"""Heteroscedastic retrieval emulator: reflectance + instrument sigma -> parameters + sigma.

The network inverts the forward model in ``forward.py``. It is deliberately small: the
mapping is smooth and low-dimensional, the training set is synthetic, and a large model
would only overfit the forward model's own idealisations.

What makes it more than a regression demo is the second input and the second output.

*Second input*: every Tanager scene ships ``surface_reflectance_uncertainty``, a
per-pixel, per-band 1-sigma. Feeding it alongside the reflectance lets the network
condition on measurement quality that varies pixel to pixel -- across a sea-ice scene
that varies enormously, because bright snow and near-black open water are measured with
very different signal-to-noise.

*Second output*: the network emits a predictive log-variance per parameter, trained
under a Gaussian negative log-likelihood. This matters here because several parameters
are conditionally unidentifiable: pond depth means nothing in a pixel with no pond, and
impurity load means nothing in a pixel of open water. A point-estimate network would
emit confident nonsense in those pixels. A heteroscedastic one reports that it does not
know, and the retrieval can be masked on its own stated uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .synth import PARAM_NAMES

# Clamp on predicted log-variance, for numerical stability of the NLL.
LOGVAR_MIN, LOGVAR_MAX = -12.0, 4.0


@dataclass
class Standardiser:
    """Feature and target standardisation, fitted on the training set."""

    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def encode_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def decode_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean

    def decode_y_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """Map a standard-space sigma back to physical units."""
        return sigma * self.y_std

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }


def build_features(reflectance: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
    """Concatenate reflectance with log-uncertainty.

    The log is taken because the uncertainty spans two orders of magnitude across the
    VSWIR; on a linear scale the SWIR bands would contribute almost nothing to the
    input and the network would be blind to exactly the quality variation it needs.
    """
    return np.concatenate([reflectance, np.log(np.clip(uncertainty, 1e-8, None))], axis=1)


class RetrievalNet(nn.Module):
    """MLP emitting a mean and a log-variance for each retrieved parameter."""

    def __init__(self, n_features: int, n_params: int = len(PARAM_NAMES), width: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width // 2),
            nn.GELU(),
        )
        self.head_mean = nn.Linear(width // 2, n_params)
        self.head_logvar = nn.Linear(width // 2, n_params)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head_mean(h), self.head_logvar(h).clamp(LOGVAR_MIN, LOGVAR_MAX)


def gaussian_nll(
    mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Mean Gaussian negative log-likelihood, up to an additive constant."""
    return (0.5 * (logvar + (target - mean) ** 2 / logvar.exp())).mean()


def fit_standardiser(x: np.ndarray, y: np.ndarray) -> Standardiser:
    return Standardiser(
        x_mean=x.mean(0),
        x_std=x.std(0) + 1e-8,
        y_mean=y.mean(0),
        y_std=y.std(0) + 1e-8,
    )


def constrain_fractions(params: np.ndarray) -> np.ndarray:
    """Project the three endmember fractions back onto the simplex.

    The network regresses in an unconstrained space, so raw output can sit slightly
    outside [0, 1] or fail to sum to one. Clip-and-renormalise is the cheapest
    projection that preserves the ordering the network predicted.
    """
    out = params.copy()
    fr = np.clip(out[:, :3], 0.0, None)
    total = fr.sum(1, keepdims=True)
    out[:, :3] = np.divide(fr, total, out=np.full_like(fr, 1.0 / 3.0), where=total > 1e-8)
    return out
