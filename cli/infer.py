"""
SE+ST Combined Model Inference CLI

Usage:
    se-st-infer \
        --checkpoint competition/se_st_combined_40k_v2/final_model.ckpt \
        --adata /data/competition_val_template.h5ad \
        --output competition/prediction.h5ad \
        --pert-col target_gene \
        --se-model-path SE-600M \
        --perturbation-features /data/ESM2_pert_features.pt
"""

import argparse
import logging
from pathlib import Path
import sys

import anndata
import h5py
import numpy as np
import torch
from tqdm import tqdm

from se_st_combined.models.se_st_combined import SE_ST_CombinedModel

logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG to see detailed logs
    format='[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


def _infer_transformer_config_from_checkpoint(state_dict: dict, hparams: dict) -> dict:
    """
    Infer transformer configuration from checkpoint weights.
    
    This ensures we use the EXACT same configuration as during training,
    not a guessed one.
    """
    config = {}
    
    # Try to get q_proj weight to infer dimensions
    q_proj_key = 'st_model.transformer_backbone.layers.0.self_attn.q_proj.weight'
    if q_proj_key in state_dict:
        q_proj_shape = state_dict[q_proj_key].shape
        out_features, in_features = q_proj_shape
        
        # For standard self-attention: out_features = in_features = hidden_size
        # For multi-head attention: num_heads * head_dim = hidden_size
        st_hidden_dim = hparams.get('st_hidden_dim', in_features)
        
        # Infer num_heads and head_dim
        # Common patterns: hidden_dim = num_heads * head_dim
        # Try common num_heads values
        for num_heads in [8, 12, 16, 6, 4]:
            if st_hidden_dim % num_heads == 0:
                head_dim = st_hidden_dim // num_heads
                config['num_attention_heads'] = num_heads
                config['num_key_value_heads'] = num_heads
                config['head_dim'] = head_dim
                logger.info(f"Inferred from checkpoint: num_heads={num_heads}, head_dim={head_dim}")
                break
    
    return config


def load_model_from_checkpoint(checkpoint_path: str, se_model_path: str) -> tuple:
    """Load SE+ST Combined Model from checkpoint and return model + hyperparameters."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load the checkpoint to inspect it
    # Note: weights_only=False is safe here because we trust our own checkpoints
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract hyperparameters from checkpoint
    if 'hyper_parameters' in ckpt:
        hparams = ckpt['hyper_parameters']
        logger.info(f"Found hyperparameters in checkpoint: {list(hparams.keys())}")
    else:
        logger.warning("No hyperparameters found in checkpoint, using defaults")
        hparams = {}
    
    # Log key dimensions
    input_dim = hparams.get('input_dim', 18080)
    output_dim = hparams.get('output_dim', 18080)
    logger.info(f"Model dimensions: input={input_dim}, output={output_dim}")
    
    # Infer transformer configuration from checkpoint weights
    state_dict = ckpt.get('state_dict', {})
    transformer_config = _infer_transformer_config_from_checkpoint(state_dict, hparams)
    
    # Merge inferred config into hparams
    if transformer_config:
        logger.info(f"Inferred transformer config from checkpoint: {transformer_config}")
        hparams['_inferred_transformer_config'] = transformer_config
    
    # Create model with correct dimensions
    # Pass inferred transformer config if available
    model_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': hparams.get('hidden_dim', 512),
        'output_dim': output_dim,
        'pert_dim': hparams.get('pert_dim', 1280),
        'se_model_path': se_model_path,
        'se_checkpoint_path': hparams.get('se_checkpoint_path', f"{se_model_path}/se600m_epoch15.ckpt"),
        'st_cell_set_len': hparams.get('st_cell_set_len', 128),
        'st_hidden_dim': hparams.get('st_hidden_dim', 672),
        'predict_residual': hparams.get('predict_residual', True),
        'distributional_loss': hparams.get('distributional_loss', 'energy'),
        'transformer_backbone_key': hparams.get('transformer_backbone_key', 'llama'),
    }
    
    # Add inferred transformer config if available
    if '_inferred_transformer_config' in hparams:
        model_kwargs['transformer_config_override'] = hparams['_inferred_transformer_config']
    
    model = SE_ST_CombinedModel(**model_kwargs)
    
    # Load state dict
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'], strict=False)
        logger.info("Model weights loaded successfully")
    else:
        raise ValueError("No state_dict found in checkpoint")
    
    model.eval()
    return model, hparams


def load_perturbation_features(features_path: str) -> dict:
    """Load ESM2 perturbation features."""
    logger.info(f"Loading perturbation features from {features_path}")
    features = torch.load(features_path, map_location='cpu')
    logger.info(f"Loaded {len(features)} perturbation features")
    return features


def select_hvg_genes(adata: anndata.AnnData, n_top_genes: int) -> anndata.AnnData:
    """
    Select highly variable genes (HVG) from AnnData.
    
    Args:
        adata: Input AnnData with all genes
        n_top_genes: Number of top HVG to select
        
    Returns:
        AnnData with only HVG genes selected
    """
    logger.info(f"Selecting top {n_top_genes} highly variable genes...")
    logger.info(f"Original data: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Calculate gene variance
    from scipy.sparse import issparse
    
    if issparse(adata.X):
        # For sparse matrix, compute variance efficiently
        gene_mean = np.array(adata.X.mean(axis=0)).flatten()
        gene_mean_sq = np.array(adata.X.power(2).mean(axis=0)).flatten()
        gene_var = gene_mean_sq - gene_mean ** 2
    else:
        # For dense matrix
        gene_var = np.var(adata.X, axis=0)
    
    # Select top variance genes
    top_var_indices = np.argsort(gene_var)[-n_top_genes:]
    hvg_gene_names = adata.var_names[top_var_indices]
    
    # Subset to HVG
    adata_hvg = adata[:, hvg_gene_names].copy()
    
    logger.info(f"✅ Selected HVG data: {adata_hvg.n_obs} cells × {adata_hvg.n_vars} genes")
    
    return adata_hvg, hvg_gene_names


def run_inference(
    model: SE_ST_CombinedModel,
    adata: anndata.AnnData,
    pert_features: dict,
    pert_dim: int = 1280,
    cell_sentence_len: int = 128,
    hvg_gene_names: list = None,
    pert_col: str = "target_gene",
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> anndata.AnnData:
    """
    Run inference on an AnnData object.
    
    Args:
        model: Trained SE+ST Combined Model
        adata: Input AnnData with perturbation metadata (should be HVG-selected)
        pert_features: Dictionary mapping perturbation names to embeddings
        pert_dim: Dimension of perturbation embeddings (from model hyperparameters)
        cell_sentence_len: Length of cell sentence (default 128)
        hvg_gene_names: List of HVG gene names (for mapping back to full space)
        pert_col: Column name for perturbation in adata.obs
        batch_size: Batch size for inference
        device: Device to run inference on
        
    Returns:
        AnnData object with predicted expression values
    """
    model = model.to(device)
    model.eval()
    
    logger.info(f"Running inference on {adata.n_obs} cells")
    logger.info(f"Device: {device}")
    logger.info(f"Perturbation embedding dim: {pert_dim}, Cell sentence length: {cell_sentence_len}")
    
    # Get unique perturbations
    unique_perts = adata.obs[pert_col].unique()
    logger.info(f"Found {len(unique_perts)} unique perturbations")
    
    # Prepare predictions storage (use float16 to save space)
    predictions = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float16)
    
    # Group cells by perturbation for efficient batch processing
    with torch.no_grad():
        for pert_name in tqdm(unique_perts, desc="Processing perturbations"):
            # Get indices for this perturbation
            pert_mask = adata.obs[pert_col] == pert_name
            pert_indices = np.where(pert_mask)[0]
            
            if len(pert_indices) == 0:
                continue
            
            # Get perturbation embedding
            pert_name_str = pert_name.decode('utf-8') if isinstance(pert_name, bytes) else pert_name
            
            # Try to find embedding (case-insensitive)
            pert_emb = None
            for key in pert_features.keys():
                if key.lower() == pert_name_str.lower():
                    pert_emb = pert_features[key]
                    break
            
            if pert_emb is None:
                logger.warning(f"No embedding found for {pert_name_str}, using zeros")
                pert_emb = torch.zeros(pert_dim)
            
            if not isinstance(pert_emb, torch.Tensor):
                pert_emb = torch.tensor(pert_emb)
            
            # CRITICAL: Ensure pert_emb matches expected pert_dim
            if pert_emb.shape[0] != pert_dim:
                logger.warning(f"Perturbation embedding dimension mismatch!")
                logger.warning(f"  Expected: {pert_dim}, Got: {pert_emb.shape[0]}")
                logger.warning(f"  Truncating or padding to match model expectation")
                
                if pert_emb.shape[0] > pert_dim:
                    # Truncate if too large
                    pert_emb = pert_emb[:pert_dim]
                else:
                    # Pad with zeros if too small
                    padding = torch.zeros(pert_dim - pert_emb.shape[0], dtype=pert_emb.dtype)
                    pert_emb = torch.cat([pert_emb, padding])
            
            pert_emb = pert_emb.to(device)
            
            # Get expression data for these cells
            if hasattr(adata.X, 'toarray'):
                X = adata.X[pert_indices].toarray()
            else:
                X = adata.X[pert_indices]
            
            # Process in batches
            for i in range(0, len(pert_indices), batch_size):
                end_idx = min(i + batch_size, len(pert_indices))
                batch_X = X[i:end_idx]
                
                # Convert to tensor
                batch_X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
                
                # Create cell sentence by repeating each cell 128 times (to match training)
                # Shape: [curr_batch_size, cell_sentence_len, gene_dim]
                batch_X_tensor = batch_X_tensor.unsqueeze(1).repeat(1, cell_sentence_len, 1)
                
                # Flatten to [curr_batch_size*cell_sentence_len, gene_dim] as model expects
                curr_batch_size = batch_X_tensor.shape[0]
                batch_X_tensor = batch_X_tensor.reshape(-1, batch_X_tensor.shape[2])
                
                # Repeat perturbation embedding for each cell in sentence
                batch_pert_emb = pert_emb.unsqueeze(0).repeat(curr_batch_size * cell_sentence_len, 1)
                
                # Run inference
                try:
                    # Model expects a batch dictionary
                    batch_dict = {
                        'ctrl_cell_emb': batch_X_tensor,
                        'pert_emb': batch_pert_emb,
                    }
                    
                    # Always log shapes for debugging
                    logger.info(f"🔍 Batch {pert_name_str} [{i}:{end_idx}]:")
                    logger.info(f"  curr_batch_size: {curr_batch_size}, cell_sentence_len: {cell_sentence_len}")
                    logger.info(f"  ctrl_cell_emb: {batch_dict['ctrl_cell_emb'].shape}")
                    logger.info(f"  pert_emb: {batch_dict['pert_emb'].shape}")
                    logger.info(f"  Expected total cells: {curr_batch_size * cell_sentence_len}")
                    
                    pred = model(batch_dict)
                    
                    logger.info(f"  ✅ Model output: {pred.shape}")
                    
                    # Predictions are [curr_batch_size*cell_sentence_len, gene_dim]
                    # Reshape to [curr_batch_size, cell_sentence_len, gene_dim] and average
                    pred = pred.reshape(curr_batch_size, cell_sentence_len, -1)
                    pred = pred.mean(dim=1)  # Average over cell_sentence_len → [curr_batch_size, gene_dim]
                    
                    # Store predictions
                    pred_np = pred.cpu().numpy()
                    predictions[pert_indices[i:end_idx]] = pred_np
                    
                except Exception as e:
                    logger.error(f"❌ Error processing batch for {pert_name_str}: {e}")
                    logger.error(f"   Batch details:")
                    logger.error(f"     curr_batch_size: {curr_batch_size}")
                    logger.error(f"     cell_sentence_len: {cell_sentence_len}")
                    logger.error(f"     ctrl_cell_emb shape: {batch_X_tensor.shape}")
                    logger.error(f"     pert_emb shape: {batch_pert_emb.shape}")
                    logger.error(f"   Full traceback:", exc_info=True)
                    # Use input as fallback
                    predictions[pert_indices[i:end_idx]] = batch_X
    
    # Create output AnnData with sparse matrix to save space
    from scipy.sparse import csr_matrix
    
    logger.info(f"Converting predictions to sparse matrix...")
    # scipy.sparse doesn't support float16, convert to float32
    logger.info(f"   Converting from float16 to float32 (scipy requirement)...")
    predictions_float32 = predictions.astype(np.float32)
    predictions_sparse = csr_matrix(predictions_float32)
    
    output_adata = anndata.AnnData(
        X=predictions_sparse,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    
    logger.info(f"Inference completed! Predictions shape: {predictions.shape}")
    logger.info(f"Sparse matrix density: {predictions_sparse.nnz / (predictions.shape[0] * predictions.shape[1]) * 100:.2f}%")
    return output_adata


def main():
    parser = argparse.ArgumentParser(description="SE+ST Combined Model Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to input AnnData (.h5ad file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output predictions (.h5ad file)"
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="target_gene",
        help="Column name for perturbation in adata.obs"
    )
    parser.add_argument(
        "--se-model-path",
        type=str,
        required=True,
        help="Path to SE model directory"
    )
    parser.add_argument(
        "--perturbation-features",
        type=str,
        required=True,
        help="Path to ESM2 perturbation features (.pt file)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--hvg-genes",
        type=str,
        default=None,
        help="Path to pickle file with HVG gene names (optional). If provided, uses these genes instead of computing HVG."
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    adata_path = Path(args.adata)
    if not adata_path.exists():
        logger.error(f"AnnData file not found: {adata_path}")
        sys.exit(1)
    
    pert_features_path = Path(args.perturbation_features)
    if not pert_features_path.exists():
        logger.error(f"Perturbation features not found: {pert_features_path}")
        sys.exit(1)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load model and get hyperparameters
    model, hparams = load_model_from_checkpoint(
        str(checkpoint_path),
        args.se_model_path
    )
    
    # Load perturbation features
    pert_features = load_perturbation_features(str(pert_features_path))
    
    # Load input data
    logger.info(f"Loading input data from {adata_path}")
    adata_full = anndata.read_h5ad(adata_path)
    logger.info(f"Loaded {adata_full.n_obs} cells x {adata_full.n_vars} genes")
    
    # Select HVG if model was trained on subset of genes
    model_input_dim = hparams.get('input_dim', adata_full.n_vars)
    
    if model_input_dim < adata_full.n_vars:
        logger.info(f"Model expects {model_input_dim} genes, but data has {adata_full.n_vars}")
        
        # Check if HVG genes list is provided
        if args.hvg_genes:
            logger.info(f"Loading HVG genes from {args.hvg_genes}")
            import pickle
            with open(args.hvg_genes, 'rb') as f:
                hvg_gene_names = pickle.load(f)
            
            logger.info(f"Loaded {len(hvg_gene_names)} HVG genes from file")
            
            # Verify all HVG genes exist in the data
            missing_genes = set(hvg_gene_names) - set(adata_full.var_names)
            if missing_genes:
                logger.warning(f"⚠️ {len(missing_genes)} HVG genes not found in data (will be skipped)")
                hvg_gene_names = [g for g in hvg_gene_names if g in adata_full.var_names]
            
            # Subset to HVG genes
            adata_hvg = adata_full[:, hvg_gene_names].copy()
            logger.info(f"✅ Using provided HVG: {adata_hvg.shape}")
        else:
            logger.info("Computing HVG using variance method...")
            adata_hvg, hvg_gene_names = select_hvg_genes(adata_full, model_input_dim)
    else:
        logger.info(f"Using all {adata_full.n_vars} genes (no HVG selection needed)")
        adata_hvg = adata_full
        hvg_gene_names = adata_full.var_names.tolist()
    
    # Run inference with correct dimensions from hyperparameters
    output_adata = run_inference(
        model=model,
        adata=adata_hvg,
        pert_features=pert_features,
        pert_dim=hparams.get('pert_dim', 1280),
        cell_sentence_len=hparams.get('st_cell_set_len', 128),
        hvg_gene_names=hvg_gene_names,
        pert_col=args.pert_col,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    # Map back to full gene space if HVG was used
    if model_input_dim < adata_full.n_vars:
        logger.info(f"Mapping predictions back to full gene space ({adata_full.n_vars} genes)...")
        from scipy.sparse import lil_matrix
        
        # Create full predictions matrix (sparse for efficiency)
        # Note: scipy.sparse doesn't support float16, use float32
        full_pred = lil_matrix((adata_full.n_obs, adata_full.n_vars), dtype=np.float32)
        
        # Get HVG indices in the original full matrix
        hvg_indices = [adata_full.var_names.get_loc(gene) for gene in hvg_gene_names]
        
        # Convert output to dense if sparse
        from scipy.sparse import issparse
        if issparse(output_adata.X):
            output_dense = output_adata.X.toarray()
        else:
            output_dense = output_adata.X
        
        # Fill HVG positions
        full_pred[:, hvg_indices] = output_dense
        
        # Convert to CSR for efficiency
        full_pred = full_pred.tocsr()
        
        # Create final output with full gene space
        output_adata = anndata.AnnData(
            X=full_pred,
            obs=adata_full.obs.copy(),
            var=adata_full.var.copy(),
        )
        
        logger.info(f"✅ Mapped to full space: {output_adata.shape}")
    else:
        # Ensure we use the original obs/var
        output_adata.obs = adata_full.obs.copy()
        output_adata.var = adata_full.var.copy()
    
    # Save predictions with light compression (faster)
    logger.info(f"Saving predictions to {output_path}")
    logger.info("Saving with lzf compression (faster than gzip)...")
    output_adata.write_h5ad(output_path, compression="lzf")
    logger.info("✅ Inference completed successfully!")
    
    # Show file size
    import os
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
