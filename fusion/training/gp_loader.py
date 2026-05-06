"""
gp_loader.py
------------
Helpers for loading a saved SVGP bundle and running single-sample inference.

The bundle format is produced by train_gp.py and contains:
  model          : SVGPModel (gpytorch)
  likelihood     : GaussianLikelihood
  feature_names  : list[str]
  x_mean, x_std  : float32 arrays, shape (n_features,)
  y_mean, y_std  : float32 arrays, shape (1,)
  sigma_ref      : float -- 90th percentile predictive std on training data (mOhm)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_gp_bundle(path: str | Path) -> dict[str, Any]:
    """
    Load a saved SVGP bundle from disk.

    The bundle contains custom gpytorch objects so weights_only=False is
    required. 
    """
    # weights_only=False is needed because the bundle contains custom GP model classes,
    # not just plain tensors.
    return torch.load(Path(path), map_location="cpu", weights_only=False)


@torch.no_grad()
def predict_single(bundle: dict[str, Any], x_raw: np.ndarray) -> tuple[float, float]:
    """
    Run a single-sample GP prediction and return (mean, variance) in physical units.

    Standardises x_raw using the stored x_mean/x_std, runs the GP forward pass
    through the likelihood, then converts the output back to physical units
    (mOhm for resistance residuals). Designed for online inference where one
    sample arrives per step.

    Parameters
    ----------
    bundle : loaded GP bundle (from load_gp_bundle)
    x_raw  : shape (n_features,), in original un-normalised units

    Returns
    -------
    mu  : predicted residual mean (mOhm)
    var : predicted residual variance (mOhm^2, always >= 0)
    """
    x_mean = bundle["x_mean"]   # training-set feature means
    x_std  = bundle["x_std"]    # training-set feature standard deviations
    y_mean = float(bundle["y_mean"].flatten()[0])   # training-set target mean
    y_std  = float(bundle["y_std"].flatten()[0])    # training-set target std

    # Normalise input to zero mean, unit variance (same as during training)
    x_s = (x_raw.astype(np.float32) - x_mean) / x_std
    xt  = torch.tensor(x_s[None, :], dtype=torch.float32)   # shape: (1, n_features)

    model = bundle["model"]
    lik   = bundle["likelihood"]
    model.eval()
    lik.eval()

    # Run one forward pass through the GP + Gaussian likelihood
    pred  = lik(model(xt))
    mu_s  = pred.mean.item()                     # predicted mean (normalised)
    var_s = max(pred.variance.item(), 0.0)       # predicted variance (normalised, clipped >= 0)

    # Convert back to physical units (mOhm)
    mu  = mu_s * y_std + y_mean
    var = var_s * (y_std ** 2)
    return mu, var
