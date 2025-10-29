"""
SE+ST Combined Model for Cross-Cell-Type Perturbation Prediction

This model combines State Embedding (SE) and State Transition (ST) models:
1. Uses pre-trained SE model to encode cells into universal state embeddings
2. Uses ST model to predict perturbation effects in the state embedding space
3. Enables better cross-cell-type generalization by learning cell-type-agnostic representations
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import PerturbationModel
from .state_transition import StateTransitionPerturbationModel

logger = logging.getLogger(__name__)


class SE_ST_CombinedModel(PerturbationModel):
    """
    Combined SE+ST model for cross-cell-type perturbation prediction.
    
    Architecture:
    1. SE Encoder: Converts raw gene expression to universal state embeddings
    2. ST Predictor: Predicts perturbation effects in state embedding space
    3. Decoder: Maps predictions back to gene expression space
    
    This design enables:
    - Cell-type-agnostic perturbation modeling
    - Better cross-cell-type generalization
    - Leveraging pre-trained SE representations
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        se_model_path: str,
        se_checkpoint_path: str,
        se_embed_key: str = "X_state",
        freeze_se_model: bool = True,
        st_hidden_dim: int = 672,
        st_cell_set_len: int = 128,
        predict_residual: bool = True,
        distributional_loss: str = "energy",
        transformer_backbone_key: str = "llama",
        transformer_config_override: dict = None,
        **kwargs,
    ):
        """
        Initialize SE+ST Combined Model.
        
        Args:
            input_dim: Dimension of input gene expression
            hidden_dim: Hidden dimension for the combined model
            output_dim: Dimension of output (gene expression)
            pert_dim: Dimension of perturbation embeddings
            se_model_path: Path to SE model directory
            se_checkpoint_path: Path to SE model checkpoint
            se_embed_key: Key for SE embeddings in obsm
            freeze_se_model: Whether to freeze SE model parameters
            st_hidden_dim: Hidden dimension for ST model
            st_cell_set_len: Cell sequence length for ST model
            predict_residual: Whether to predict residual in ST model
            distributional_loss: Loss function for ST model
            transformer_backbone_key: Transformer backbone for ST model
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            **kwargs,
        )
        
        # Store SE model configuration
        self.se_model_path = se_model_path
        self.se_checkpoint_path = se_checkpoint_path
        self.se_embed_key = se_embed_key
        self.freeze_se_model = freeze_se_model
        self.st_hidden_dim = st_hidden_dim
        self.st_cell_set_len = st_cell_set_len
        self.transformer_config_override = transformer_config_override
        
        # Initialize SE model
        self._load_se_model()
        
        # State embedding dimension (from SE model) - must be set before building ST model
        self.state_dim = self.se_model.output_dim if hasattr(self.se_model, 'output_dim') else 512
        
        # Initialize ST model (uses state_dim)
        self._build_st_model(
            predict_residual=predict_residual,
            distributional_loss=distributional_loss,
            transformer_backbone_key=transformer_backbone_key,
            transformer_config_override=transformer_config_override,
        )
        
        logger.info(f"SE+ST Combined Model initialized:")
        logger.info(f"  - SE model: {se_model_path}")
        logger.info(f"  - State embedding dim: {self.state_dim}")
        logger.info(f"  - ST hidden dim: {st_hidden_dim}")
        logger.info(f"  - Freeze SE: {freeze_se_model}")
    
    def _load_se_model(self):
        """Load pre-trained SE model."""
        try:
            # Note: This requires the SE model to be available
            # You may need to install the SE model separately or provide the inference module
            from se_st_combined.utils.se_inference import SEInference
            
            # Load SE model
            # Use input_dim (gene dimension) to create SE model
            self.se_inference = SEInference(input_dim=self.input_dim, output_dim=512)
            self.se_inference.load_model(self.se_checkpoint_path)
            self.se_model = self.se_inference.model
            
            # Freeze SE model if specified
            if self.freeze_se_model:
                for param in self.se_model.parameters():
                    param.requires_grad = False
                logger.info("SE model parameters frozen")
            
            logger.info(f"SE model loaded successfully from {self.se_checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load SE model: {e}")
            raise
    
    def _build_networks(self):
        """Build networks - required by PerturbationModel base class."""
        # This method is required by the abstract base class but the actual
        # network building is done by the ST model
        pass
    
    def _build_st_model(self, predict_residual: bool, distributional_loss: str, transformer_backbone_key: str, transformer_config_override: dict = None):
        """Build ST model for state embedding space."""
        # Use override config from checkpoint if available, otherwise calculate
        if transformer_config_override:
            logger.info(f"Using transformer config from checkpoint: {transformer_config_override}")
            num_heads = transformer_config_override.get('num_attention_heads', 8)
            head_dim = transformer_config_override.get('head_dim', self.st_hidden_dim // num_heads)
        else:
            # Calculate transformer configuration based on hidden_dim
            # For Llama: hidden_dim must be divisible by num_attention_heads
            # Common configurations: 672 = 8*84 or 12*56
            if self.st_hidden_dim % 8 == 0:
                num_heads = 8
                head_dim = self.st_hidden_dim // 8
            elif self.st_hidden_dim % 12 == 0:
                num_heads = 12
                head_dim = self.st_hidden_dim // 12
            else:
                # Fallback: find largest divisor <= 16
                for n in range(16, 0, -1):
                    if self.st_hidden_dim % n == 0:
                        num_heads = n
                        head_dim = self.st_hidden_dim // n
                        break
            logger.info(f"Calculated transformer config: hidden_dim={self.st_hidden_dim}, num_heads={num_heads}, head_dim={head_dim}")
        
        # ST model configuration
        st_config = {
            'input_dim': self.state_dim,  # Use state embedding dimension
            'hidden_dim': self.st_hidden_dim,
            'output_dim': self.output_dim,
            'pert_dim': self.pert_dim,
            'predict_residual': predict_residual,
            'distributional_loss': distributional_loss,
            'transformer_backbone_key': transformer_backbone_key,
            'cell_set_len': self.st_cell_set_len,
            'n_encoder_layers': 4,
            'n_decoder_layers': 4,
            'dropout': 0.1,
            'lr': 1e-4,
            'embed_key': None,  # We're working in embedding space, not gene space
            'output_space': 'gene',  # Output space
            'gene_dim': self.output_dim,  # Gene dimension
            'batch_dim': None,  # Batch dimension (optional)
            'transformer_backbone_kwargs': {
                'hidden_size': self.st_hidden_dim,
                'intermediate_size': self.st_hidden_dim * 4,
                'num_hidden_layers': 4,
                'num_attention_heads': num_heads,  # Dynamically calculated
                'num_key_value_heads': num_heads,  # Same as num_attention_heads for standard attention
                'head_dim': head_dim,  # Dynamically calculated
                'use_cache': False,
                'attention_dropout': 0.0,
                'hidden_dropout': 0.0,
                'layer_norm_eps': 1e-6,
                'pad_token_id': 0,
                'bos_token_id': 1,
                'eos_token_id': 2,
                'tie_word_embeddings': False,
                'rotary_dim': 0,
                'use_rotary_embeddings': False,
            }
        }
        
        # Initialize ST model
        self.st_model = StateTransitionPerturbationModel(**st_config)
        logger.info("ST model initialized for state embedding space")
    
    def encode_cells_to_state(self, cell_expressions: torch.Tensor) -> torch.Tensor:
        """
        Encode raw cell expressions to state embeddings using SE model.
        
        Args:
            cell_expressions: Raw gene expression [B, N_genes]
            
        Returns:
            state_embeddings: State embeddings [B, state_dim]
        """
        with torch.no_grad() if self.freeze_se_model else torch.enable_grad():
            # Use SE model to encode cells
            # Note: This is a simplified interface - actual implementation may need
            # to handle batch processing and data format conversion
            state_embeddings = self.se_model.encode_cells(cell_expressions)
            return state_embeddings
    
    def forward(self, batch: Dict[str, torch.Tensor], padded: bool = True) -> torch.Tensor:
        """
        Forward pass through SE+ST combined model.
        
        Args:
            batch: Input batch containing:
                - ctrl_cell_emb: Control cell expressions [B*S, N_genes]
                - pert_emb: Perturbation embeddings [B*S, pert_dim]
                - pert_cell_emb: Target perturbed expressions [B*S, N_genes]
            padded: Whether batch is padded
            
        Returns:
            predictions: Predicted perturbed expressions [B*S, N_genes]
        """
        # Step 1: Encode control cells to state embeddings
        ctrl_expressions = batch["ctrl_cell_emb"]  # [B*S, N_genes]
        
        # Debug logging
        logger.debug(f"SE_ST_CombinedModel.forward:")
        logger.debug(f"  Input ctrl_expressions shape: {ctrl_expressions.shape}")
        logger.debug(f"  Expected: [B*S={ctrl_expressions.shape[0]}, N_genes={self.input_dim}]")
        
        state_embeddings = self.encode_cells_to_state(ctrl_expressions)  # [B*S, state_dim]
        
        logger.debug(f"  Output state_embeddings shape: {state_embeddings.shape}")
        logger.debug(f"  Expected: [B*S={ctrl_expressions.shape[0]}, state_dim={self.state_dim}]")
        logger.debug(f"  pert_emb shape: {batch['pert_emb'].shape}")
        
        # Step 2: Create new batch with state embeddings
        st_batch = batch.copy()
        st_batch["ctrl_cell_emb"] = state_embeddings
        
        # Step 3: Forward through ST model
        predictions = self.st_model.forward(st_batch, padded=padded)
        
        return predictions
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True) -> torch.Tensor:
        """Training step for SE+ST combined model."""
        # Forward pass
        predictions = self.forward(batch, padded=padded)
        
        # Compute loss
        target = batch["pert_cell_emb"]
        
        if padded:
            predictions = predictions.reshape(-1, self.st_cell_set_len, self.output_dim)
            target = target.reshape(-1, self.st_cell_set_len, self.output_dim)
        else:
            predictions = predictions.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)
        
        # Use ST model's loss function
        loss = self.st_model.loss_fn(predictions, target).nanmean()
        
        # Log metrics
        self.log("train_loss", loss)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, padded: bool = True) -> torch.Tensor:
        """Validation step for SE+ST combined model."""
        # Forward pass
        predictions = self.forward(batch, padded=padded)
        
        # Compute loss
        target = batch["pert_cell_emb"]
        
        if padded:
            predictions = predictions.reshape(-1, self.st_cell_set_len, self.output_dim)
            target = target.reshape(-1, self.st_cell_set_len, self.output_dim)
        else:
            predictions = predictions.reshape(1, -1, self.output_dim)
            target = target.reshape(1, -1, self.output_dim)
        
        # Use ST model's loss function
        loss = self.st_model.loss_fn(predictions, target).nanmean()
        
        # Log metrics
        self.log("val_loss", loss)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers for SE+ST model."""
        # Only optimize ST model parameters if SE is frozen
        if self.freeze_se_model:
            optimizer = torch.optim.Adam(self.st_model.parameters(), lr=self.lr)
        else:
            # Optimize both SE and ST if SE is not frozen
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        return optimizer
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Prediction step for inference."""
        return self.forward(batch, padded=False)

