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
from torch.utils.data import DataLoader, TensorDataset


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


@torch.no_grad()
def predict_single_certainty(bundle: dict[str, Any], x_raw: np.ndarray) -> tuple[float, float, float, float]:
    """
    Single-sample GP prediction with certainty signals.

    Returns
    -------
    mu       : predicted residual mean (mOhm)
    var      : predicted residual variance (mOhm^2)
    norm_var : epistemic variance divided by prior outputscale, in [0, 1]
               0 means the model is confident (in-distribution), 1 means uncertain (OOD)
    ind_dist : min L2 distance to nearest inducing point in standardised space
    """
    x_mean = bundle["x_mean"]
    x_std  = bundle["x_std"]
    y_mean = float(bundle["y_mean"].flatten()[0])
    y_std  = float(bundle["y_std"].flatten()[0])

    x_s = (x_raw.astype(np.float32) - x_mean) / x_std
    xt  = torch.tensor(x_s[None, :], dtype=torch.float32)

    model = bundle["model"]
    lik   = bundle["likelihood"]
    model.eval()
    lik.eval()

    f_dist = model(xt)
    pred   = lik(f_dist)

    mu  = pred.mean.item() * y_std + y_mean
    var = max(pred.variance.item(), 0.0) * (y_std ** 2)

    k_scale  = model.covar_module.outputscale.item()
    norm_var = float((f_dist.variance.item() / k_scale).real)
    norm_var = max(0.0, min(1.0, norm_var))

    Z = model.variational_strategy.inducing_points.detach()
    x_proj = model.feature_extractor(xt).detach() if hasattr(model, "feature_extractor") else xt
    ind_dist = float(torch.cdist(x_proj, Z).min().item())

    return mu, var, norm_var, ind_dist


@torch.no_grad()
def predict_svgp_certainty(model, likelihood, X, batch_size=4096):
    """
    Batch GP prediction returning mean and two certainty signals.

    Both signals rise when the model is operating outside its training
    distribution and fall toward zero when it is confident.

    Parameters
    ----------
    model      : SVGPModel or DeepKernelSVGPModel (from load_gp_bundle)
    likelihood : GaussianLikelihood (from load_gp_bundle)
    X          : (N, D) array or tensor of standardised inputs
    batch_size : number of rows to process at once

    Returns
    -------
    mu       : (N,) predicted mean in standardised output units
    norm_var : (N,) epistemic variance divided by prior outputscale, in [0, 1]
               0 means model is confident (in-distribution)
               1 means model is uncertain (out-of-distribution)
    ind_dist : (N,) min L2 distance to the nearest inducing point
               in standardised input space, or latent space for deep kernel
    """
    model.eval()
    likelihood.eval()

    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32).to(device) if isinstance(X, np.ndarray) else X.to(device)

    Z       = model.variational_strategy.inducing_points.detach().cpu()
    k_scale = model.covar_module.outputscale.item()

    loader = DataLoader(TensorDataset(X_t), batch_size=batch_size, shuffle=False)

    means, norm_vars, ind_dists = [], [], []
    for (xb,) in loader:
        f_dist = model(xb)
        means.append(likelihood(f_dist).mean.detach().cpu())
        norm_vars.append((f_dist.variance.detach().cpu() / k_scale).clamp(0.0, 1.0))
        x_proj = model.feature_extractor(xb).detach().cpu() if hasattr(model, "feature_extractor") else xb.detach().cpu()
        ind_dists.append(torch.cdist(x_proj, Z).min(dim=1).values)

    return torch.cat(means), torch.cat(norm_vars), torch.cat(ind_dists)
