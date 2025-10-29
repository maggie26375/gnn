"""
Gene expression decoders for mapping latent representations to gene counts.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional


class FinetuneVCICountsDecoder(nn.Module):
    """
    Decoder for fine-tuning VCI (Variational Causal Inference) counts prediction.
    
    This decoder maps latent representations to gene expression counts,
    optionally with gene-specific parameters.
    """
    
    def __init__(
        self,
        genes: Optional[List[str]] = None,
        latent_dim: int = 512,
        hidden_dims: List[int] = [1024, 1024],
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        """
        Initialize the VCI Counts Decoder.
        
        Args:
            genes: List of gene names (optional, used for gene_dim)
            latent_dim: Dimension of latent representation
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu', etc.)
        """
        super().__init__()
        
        # Store gene information
        if genes is not None:
            self._gene_dim = len(genes)
            self.genes = genes
        else:
            self._gene_dim = 2000  # Default
            self.genes = None
        
        self.latent_dim = latent_dim
        
        # Get activation function
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "elu":
            act_fn = nn.ELU()
        else:
            act_fn = nn.GELU()
        
        # Build decoder network
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                act_fn,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final layer to predict gene counts
        layers.append(nn.Linear(prev_dim, self._gene_dim))
        
        self.decoder_net = nn.Sequential(*layers)
        
        # Optional: gene-specific bias terms
        self.gene_bias = nn.Parameter(torch.zeros(self._gene_dim))
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to gene counts.
        
        Args:
            latent: Latent representation [*, latent_dim]
            
        Returns:
            counts: Predicted gene counts [*, gene_dim]
        """
        # Pass through decoder network
        counts = self.decoder_net(latent)
        
        # Add gene-specific bias
        counts = counts + self.gene_bias
        
        # Ensure non-negative counts (softplus for smooth gradients)
        counts = torch.nn.functional.softplus(counts)
        
        return counts
    
    def gene_dim(self) -> int:
        """Return the number of genes."""
        return self._gene_dim


class SimpleCountsDecoder(nn.Module):
    """
    Simple linear decoder for counts prediction.
    """
    
    def __init__(
        self,
        latent_dim: int,
        gene_dim: int,
    ):
        """
        Initialize simple counts decoder.
        
        Args:
            latent_dim: Dimension of latent representation
            gene_dim: Number of genes
        """
        super().__init__()
        
        self._gene_dim = gene_dim
        self.decoder = nn.Linear(latent_dim, gene_dim)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to counts.
        
        Args:
            latent: Latent representation [*, latent_dim]
            
        Returns:
            counts: Predicted counts [*, gene_dim]
        """
        counts = self.decoder(latent)
        return torch.nn.functional.softplus(counts)
    
    def gene_dim(self) -> int:
        """Return the number of genes."""
        return self._gene_dim

