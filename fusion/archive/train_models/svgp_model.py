"""
archive/train_models/svgp_model.py
-----------------------------------
DO NOT MODIFY THIS FILE

This file must be at exactly this path (fusion/archive/train_models/svgp_model.py)
so that torch.load can find the SVGPModel class when loading a saved GP bundle.

When Python saves a class (via torch.save), it stores the full module path of
the class. The saved GP bundles (gp_el*.pt) stores the class as
  "fusion.archive.train_models.svgp_model.SVGPModel"

When you call torch.load later, Python follows that saved path to find the class.
If this file is missing or moved, torch.load will give ModuleNotFoundError.

To change anything about the GP model architecture, modify train_gp.py instead
and retrain -- do not edit SVGPModel directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import gpytorch
import torch


#      KernelSpec: a simple config object that specifies which kernel to use
# ---------------------------------------------------------------------------

@dataclass
class KernelSpec:
    """
    Describes which covariance kernel the SVGP should use.

    name     : kernel type -- "rbf", "matern12", "matern32", "matern52", "linear"
    ard      : if True, each input feature gets its own length-scale (recommended)
    add      : optionally add a second kernel on top (rarely used)
    nn_hidden: hidden layer size for deep kernel (only used when name="deep")
    nn_out   : output dimension for deep kernel (only used when name="deep")
    """
    name: str
    ard: bool = True
    add: Optional["KernelSpec"] = None
    nn_hidden: int = 32
    nn_out: int = 8


def _build_kernel(spec: KernelSpec, input_dim: int) -> gpytorch.kernels.Kernel:
    """Build a gpytorch kernel from a KernelSpec description."""
    if spec.name == "rbf":
        # Radial Basis Function (Gaussian) kernel -- very smooth predictions
        base = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim if spec.ard else None)
    elif spec.name == "matern12":
        # Matern 1/2 kernel -- rough, non-differentiable predictions
        base = gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=input_dim if spec.ard else None)
    elif spec.name == "matern32":
        # Matern 3/2 kernel -- once differentiable; good balance for physical signals
        base = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim if spec.ard else None)
    elif spec.name == "matern52":
        # Matern 5/2 kernel -- twice differentiable; smoother than 3/2
        base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim if spec.ard else None)
    elif spec.name == "linear":
        # Linear kernel -- the GP becomes a Bayesian linear model
        base = gpytorch.kernels.LinearKernel()
    else:
        raise ValueError(f"Unknown kernel type: {spec.name}")

    # ScaleKernel wraps the base kernel with a learned output scale
    kernel = gpytorch.kernels.ScaleKernel(base)
    # Optional additive second kernel (e.g. matern32 + linear)
    if spec.add is not None:
        kernel = kernel + _build_kernel(spec.add, input_dim)
    return kernel



             # SVGPModel: the main GP model used for R-correction
# ---------------------------------------------------------------------------

class SVGPModel(gpytorch.models.ApproximateGP):
    """
    Sparse Variational Gaussian Process (SVGP) model.

    Instead of using all N training points (which would be O(N^3)), the SVGP
    uses M << N "inducing points" as a compressed summary of the training data.
    The inducing points are initialised from a random subset of the training data
    and are then optimised together with the kernel hyperparameters.

    This model predicts a scalar residual delta = R_real - R_arx.
    Inputs are the 38 GP features, output is the additive R correction (mOhm).

    DO NOT MODIFY: the architecture as this must match what the saved .pt bundles expects.
    """

    def __init__(self, inducing_points: torch.Tensor, kernel_spec: KernelSpec) -> None:
        # Variational distribution over the inducing point function values
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)   # number of inducing points M
        )
        # Strategy that ties the full GP to the inducing approximation
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_dist,
            learn_inducing_locations=True,   # inducing points move during training
        )
        super().__init__(variational_strategy)
        # Constant mean function (learned constant offset)
        self.mean_module  = gpytorch.means.ConstantMean()
        # Covariance function built from KernelSpec
        self.covar_module = _build_kernel(kernel_spec, input_dim=inducing_points.size(1))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """Compute the GP prior distribution at input locations x."""
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


     # DeepKernelSVGPModel: SVGP with a neural-network feature extractor
# (not used by default, included for comparison and  compatibility with saved bundles)
# ---------------------------------------------------------------------------

class DeepKernelSVGPModel(gpytorch.models.ApproximateGP):
    """
    Deep Kernel Learning SVGP.

    A small neural network maps raw inputs to a lower-dimensional latent space,
    and the GP kernel operates in that latent space. This can capture nonlinear
    feature interactions but takes longer to train.

    Not used by the current training scripts, included only so that bundles
    trained with this architecture can still be loaded.
    """

    def __init__(
        self,
        inducing_points_init: torch.Tensor,
        input_dim: int,
        nn_hidden: int = 32,
        nn_out: int = 8,
    ) -> None:
        # Small MLP: input_dim -> nn_hidden -> nn_hidden/2 -> nn_out
        _fe = torch.nn.Sequential(
            torch.nn.Linear(input_dim, nn_hidden), torch.nn.ReLU(),
            torch.nn.Linear(nn_hidden, max(nn_hidden // 2, nn_out)), torch.nn.ReLU(),
            torch.nn.Linear(max(nn_hidden // 2, nn_out), nn_out),
        )
        # Project inducing points through the (untrained) MLP so they live in latent space
        with torch.no_grad():
            Z = _fe.to(inducing_points_init.device)(inducing_points_init.float()).detach()

        variational_dist     = gpytorch.variational.CholeskyVariationalDistribution(Z.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, Z, variational_dist, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.feature_extractor = _fe
        self.mean_module  = gpytorch.means.ConstantMean()
        # Matern 3/2 kernel in the nn_out-dimensional latent space
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=nn_out)
        )

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        # Project raw input through the MLP before passing to the GP
        return super().__call__(self.feature_extractor(x), *args, **kwargs)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )
