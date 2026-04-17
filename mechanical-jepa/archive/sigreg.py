"""
SIGReg: Sketched Isotropic Gaussian Regularization.

From LeJEPA (arXiv:2511.08544, Nov 2025) / LeWorldModel (arXiv:2603.19312, Mar 2026).

Key idea: Regularize encoder outputs to be distributed like an isotropic Gaussian.
This replaces EMA, stop-gradient, and VICReg entirely for preventing collapse.

Algorithm:
1. Project embeddings onto M random unit vectors (sketch)
2. For each projection: compute Epps-Pulley normality test statistic
3. Loss encourages each projection to be standard normal
4. If all projections are normal, the full distribution is (approximately) isotropic Gaussian

References:
- LeJEPA paper: https://arxiv.org/abs/2511.08544
- LeWorldModel: https://arxiv.org/abs/2603.19312
- Epps-Pulley test: Epps & Pulley (1983), a smooth test for normality

Note: The Epps-Pulley test uses the characteristic function of the normal distribution.
For a sample x_1,...,x_n, the test statistic is related to:
    T = sum over i,j of exp(-0.5*(x_i - x_j)^2) vs expected under normality

Simpler differentiable proxy (used here):
    - Mean matching: E[x] ≈ 0
    - Variance matching: Var[x] ≈ 1
    - Skewness penalty: E[x^3] ≈ 0
    - Kurtosis matching: E[x^4] ≈ 3
This is a method-of-moments normality test, differentiable and efficient.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.

    Projects embeddings onto random directions and penalizes deviation
    from standard normal distribution in each projected direction.

    This prevents collapse (embeddings becoming constant/degenerate) by
    encouraging the embedding distribution to fill the space isotropically.

    Args:
        embed_dim: Embedding dimension
        n_projections: Number of random directions to project onto (M in paper)
        method: 'moments' (differentiable moment matching) or 'mmd' (maximum mean discrepancy)
        sigma: Kernel bandwidth for MMD (if method='mmd')
    """

    def __init__(
        self,
        embed_dim: int = 512,
        n_projections: int = 64,
        method: str = 'moments',
        sigma: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_projections = n_projections
        self.method = method
        self.sigma = sigma

        # Fixed random projection directions (not trained)
        # Shape: (n_projections, embed_dim)
        projections = torch.randn(n_projections, embed_dim)
        projections = F.normalize(projections, dim=-1)
        self.register_buffer('projections', projections)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.

        Args:
            z: (B, D) or (B, N, D) embeddings
                B = batch size, N = sequence length, D = embed_dim

        Returns:
            sigreg_loss: scalar regularization loss
        """
        if z.dim() == 3:
            # Flatten batch and sequence dimensions
            z = z.reshape(-1, z.shape[-1])  # (B*N, D)

        B, D = z.shape

        if B < 4:
            # Not enough samples for meaningful statistics
            return torch.tensor(0.0, device=z.device, requires_grad=False)

        if self.method == 'moments':
            return self._moments_loss(z)
        elif self.method == 'mmd':
            return self._mmd_loss(z)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _moments_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Moment-matching loss for normality on random projections.

        For an isotropic Gaussian with mean 0 and std 1:
        - 1st moment (mean): 0
        - 2nd moment (variance): 1
        - 3rd moment (skewness): 0
        - 4th moment (kurtosis): 3

        This is a differentiable method-of-moments normality test.
        """
        B, D = z.shape

        # Project embeddings: (B, n_projections)
        projections = self.projections  # (M, D)
        z_proj = z @ projections.T  # (B, M)

        # Normalize projections to have global mean 0, std 1
        # This handles the case where the embedding scale is not unit
        z_proj_mean = z_proj.mean(dim=0, keepdim=True)  # (1, M)
        z_proj_std = z_proj.std(dim=0, keepdim=True).clamp(min=1e-6)  # (1, M)
        z_norm = (z_proj - z_proj_mean) / z_proj_std  # (B, M) standardized

        # Moment penalties
        # 1st moment (mean should be 0 — after standardization it IS 0)
        # 2nd moment (std should be 1 — after standardization it IS 1)
        # Key invariance check: std of UNSTANDARDIZED projections should not be too small
        # (prevents collapse where all projections have nearly zero variance)

        # Variance penalty: penalize std < 1 per projection direction
        # This is the VICReg-style variance term applied to projections
        proj_std = z_proj.std(dim=0)  # (M,)
        variance_loss = F.relu(1.0 - proj_std).mean()

        # 3rd moment (skewness should be 0 for Gaussian)
        skew = (z_norm ** 3).mean(dim=0)  # (M,)
        skewness_loss = (skew ** 2).mean()

        # 4th moment (kurtosis should be 3 for Gaussian, excess kurtosis = 0)
        kurt = (z_norm ** 4).mean(dim=0)  # (M,)
        kurtosis_loss = ((kurt - 3.0) ** 2).mean()

        # Total loss (scale factors from empirical tuning)
        loss = variance_loss + 0.1 * skewness_loss + 0.01 * kurtosis_loss

        return loss

    def _mmd_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maximum Mean Discrepancy between embedding distribution and standard Gaussian.

        Uses RBF kernel: k(x, y) = exp(-||x-y||^2 / (2*sigma^2))

        MMD^2 = E[k(z,z')] - 2*E[k(z,w)] + E[k(w,w')]
        where z ~ encoder, w ~ N(0, I)
        """
        B, D = z.shape

        # Sample from standard Gaussian with same shape
        w = torch.randn_like(z)

        # Project onto random directions to make computation tractable
        z_proj = z @ self.projections.T  # (B, M)
        w_proj = w @ self.projections.T  # (B, M)

        def rbf_kernel(x, y, sigma):
            """RBF kernel on projected space."""
            # Pairwise squared distances: (B_x, B_y)
            xx = (x ** 2).sum(dim=1, keepdim=True)  # (B_x, 1)
            yy = (y ** 2).sum(dim=1, keepdim=True)  # (B_y, 1)
            xy = x @ y.T  # (B_x, B_y)
            dist_sq = xx + yy.T - 2 * xy  # (B_x, B_y)
            return torch.exp(-dist_sq / (2 * sigma ** 2))

        K_zz = rbf_kernel(z_proj, z_proj, self.sigma)
        K_zw = rbf_kernel(z_proj, w_proj, self.sigma)
        K_ww = rbf_kernel(w_proj, w_proj, self.sigma)

        # Remove diagonal (self-comparisons) for unbiased MMD
        mask = 1 - torch.eye(B, device=z.device)
        mmd_sq = (K_zz * mask).sum() / (B * (B - 1))
        mmd_sq -= 2 * K_zw.mean()
        mmd_sq += (K_ww * mask).sum() / (B * (B - 1))

        return mmd_sq.clamp(min=0)


def sigreg_loss(z: torch.Tensor, embed_dim: int, n_projections: int = 64,
                method: str = 'moments') -> torch.Tensor:
    """
    Functional interface for SIGReg loss.

    Creates fresh random projections each call (no persistent state).
    Use the SIGReg module if you want fixed projections across calls.
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.shape[-1])

    B, D = z.shape
    if B < 4:
        return torch.tensor(0.0, device=z.device)

    # Random projections
    projections = F.normalize(torch.randn(n_projections, D, device=z.device), dim=-1)
    z_proj = z @ projections.T  # (B, M)

    # Variance loss: penalize small std per direction
    proj_std = z_proj.std(dim=0)
    variance_loss = F.relu(1.0 - proj_std).mean()

    # Standardize
    z_norm = (z_proj - z_proj.mean(dim=0, keepdim=True)) / (z_proj.std(dim=0, keepdim=True) + 1e-6)

    # Skewness
    skew = (z_norm ** 3).mean(dim=0)
    skewness_loss = (skew ** 2).mean()

    # Kurtosis
    kurt = (z_norm ** 4).mean(dim=0)
    kurtosis_loss = ((kurt - 3.0) ** 2).mean()

    return variance_loss + 0.1 * skewness_loss + 0.01 * kurtosis_loss
