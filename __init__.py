"""
SE+ST Combined Model Package

A combined State Embedding (SE) and State Transition (ST) model for cross-cell-type
perturbation prediction in single-cell genomics.

This package provides:
1. SE_ST_CombinedModel: The main combined model
2. SE inference utilities
3. Training and inference scripts
4. Configuration files

Usage:
    from se_st_combined.models.se_st_combined import SE_ST_CombinedModel
    from se_st_combined.utils.se_st_utils import load_se_st_model
"""

__version__ = "0.1.0"
__author__ = "maggie26375"
__email__ = "your-email@example.com"

# Use lazy imports to avoid loading all dependencies at import time
# Users can import specific modules as needed:
#   from se_st_combined.models.se_st_combined import SE_ST_CombinedModel
#   from se_st_combined.models.state_transition import StateTransitionPerturbationModel
#   from se_st_combined.models.decoders import FinetuneVCICountsDecoder
#   from se_st_combined.models.decoders_nb import NBDecoder

__all__ = [
    "__version__",
]
