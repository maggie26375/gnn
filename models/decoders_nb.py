"""
Negative Binomial Decoder for gene expression prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class NBDecoder(nn.Module):
    """
    Negative Binomial Decoder that predicts gene expression counts.
    
    The negative binomial distribution is commonly used for modeling
    count data with overdispersion (variance > mean), which is typical
    in gene expression data.
    
    The decoder outputs two parameters:
    - mu: mean of the negative binomial distribution
    - theta: dispersion parameter (inverse of overdispersion)
    """
    
    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
        hidden_dims: List[int] = [512, 512, 512],
        dropout: float = 0.1,
    ):
        """
        Initialize Negative Binomial Decoder.
        
        Args:
            latent_dim: Dimension of latent representation
            gene_dim: Number of genes to predict
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self._gene_dim = gene_dim
        self.latent_dim = latent_dim
        
        # Build MLP for mean (mu) prediction
        mu_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            mu_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final layer to predict mean
        mu_layers.append(nn.Linear(prev_dim, gene_dim))
        self.mu_net = nn.Sequential(*mu_layers)
        
        # Dispersion parameter (theta) - can be learned per gene
        # Initialize with positive values (inverse of overdispersion)
        self.theta_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], gene_dim),
        )
        
    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to predict NB distribution parameters.
        
        Args:
            latent: Latent representation [*, latent_dim]
            
        Returns:
            mu: Mean parameter [*, gene_dim]
            theta: Dispersion parameter [*, gene_dim]
        """
        # Predict mean (ensure positive with softplus)
        mu = torch.nn.functional.softplus(self.mu_net(latent))
        
        # Predict dispersion (ensure positive with softplus)
        theta = torch.nn.functional.softplus(self.theta_net(latent))
        
        return mu, theta
    
    def gene_dim(self) -> int:
        """Return the number of genes."""
        return self._gene_dim


def nb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative Binomial Negative Log-Likelihood loss.
    
    Computes the negative log-likelihood of observed counts x under
    a negative binomial distribution with parameters mu (mean) and
    theta (dispersion).
    
    Args:
        x: Observed counts [*, gene_dim]
        mu: Predicted mean [*, gene_dim]
        theta: Dispersion parameter [*, gene_dim]
        eps: Small constant for numerical stability
        
    Returns:
        Negative log-likelihood [*]
    """
    # Ensure positive parameters
    mu = mu + eps
    theta = theta + eps
    
    # Negative binomial log-likelihood
    # log P(x | mu, theta) = log Gamma(x + theta) - log Gamma(theta) - log Gamma(x + 1)
    #                        + theta * log(theta) - theta * log(theta + mu)
    #                        + x * log(mu) - x * log(theta + mu)
    
    log_theta_mu_eps = torch.log(theta + mu + eps)
    
    # Using the relationship: log Gamma(n+1) = log(n!)
    # For computational efficiency, we use the following formulation:
    nll = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1.0 + eps)
    )
    
    # Return negative log-likelihood (flip sign)
    return -nll.mean(dim=-1)

