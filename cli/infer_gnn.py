#!/usr/bin/env python3
"""
Generate predictions for competition submission using trained GNN model.

This script loads a trained model and generates predictions on the
competition validation template data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import anndata as ad
import logging
from tqdm import tqdm
import glob

from gnn.models.gnn_perturbation import GNN_PerturbationModel
from gnn.utils.string_network_loader import load_string_network_for_hvgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gene_names(data_dir: str):
    """Load gene names from h5ad files."""
    import os

    possible_files = [
        f"{data_dir}/competition_val_template.h5ad",
        f"{data_dir}/competition_train.h5ad",
        f"{data_dir}/k562.h5ad",
        f"{data_dir}/jurkat.h5ad",
    ]

    for filepath in possible_files:
        if os.path.exists(filepath):
            logger.info(f"Loading gene names from {filepath}")
            adata = ad.read_h5ad(filepath)
            gene_names = adata.var_names.tolist()
            logger.info(f"✅ Loaded {len(gene_names)} genes")
            return gene_names

    h5ad_files = glob.glob(os.path.join(data_dir, "*.h5ad"))
    if len(h5ad_files) > 0:
        filepath = h5ad_files[0]
        logger.info(f"Found h5ad file: {filepath}")
        adata = ad.read_h5ad(filepath)
        gene_names = adata.var_names.tolist()
        logger.info(f"✅ Loaded {len(gene_names)} genes")
        return gene_names

    logger.error(f"❌ No h5ad files found in {data_dir}")
    return []


def infer_gnn(
    checkpoint_path: str,
    adata_path: str,
    output_path: str,
    pert_features_path: str,
    data_dir: str = "/data",
    batch_size: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Generate predictions using trained GNN model.

    Args:
        checkpoint_path: Path to trained model checkpoint
        adata_path: Path to competition validation template (.h5ad)
        output_path: Where to save predictions (.h5ad)
        pert_features_path: Path to perturbation features
        data_dir: Data directory for loading gene names
        batch_size: Batch size for inference
        device: Device to use
    """
    logger.info("="*60)
    logger.info("GNN Inference for Competition Submission")
    logger.info("="*60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Input data: {adata_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Device: {device}")

    # 1. Load gene names for STRING network
    logger.info("\n1. Loading gene names...")
    hvg_gene_names = load_gene_names(data_dir)
    logger.info(f"Loaded {len(hvg_gene_names)} genes")

    # 2. Load STRING network
    logger.info("\n2. Loading STRING network...")
    edge_index, gene_to_idx = load_string_network_for_hvgs(
        hvg_gene_names=hvg_gene_names,
        cache_dir=f"{data_dir}/string_cache",
        confidence_threshold=0.4,
    )
    logger.info(f"✅ STRING network: {edge_index.shape[1]} edges, {len(gene_to_idx)} genes")

    # 3. Load model
    logger.info("\n3. Loading trained model...")
    try:
        # Decoder configuration (must match training config!)
        decoder_cfg = {
            'latent_dim': 512,
            'gene_dim': 18080,
            'hidden_dims': [1024, 1024, 512],
            'dropout': 0.1,
            'residual_decoder': False,
        }

        model = GNN_PerturbationModel(
            # Basic
            input_dim=18080,
            hidden_dim=512,
            output_dim=512,
            pert_dim=1280,
            # SE
            se_model_path="SE-600M",
            se_checkpoint_path="SE-600M/se600m_epoch15.ckpt",
            freeze_se_model=True,
            # ST
            st_hidden_dim=512,
            st_cell_set_len=128,
            # GNN
            use_gnn=True,
            gene_network_edge_index=edge_index,
            gene_to_idx=gene_to_idx,
            gnn_hidden_dim=128,
            gnn_layers=3,
            gnn_type="gcn",
            gnn_dropout=0.1,
            # Training
            lr=1e-4,
            # Decoder (NEW! - required to load checkpoint with decoder)
            decoder_cfg=decoder_cfg,
            gene_decoder_bool=True,
        )

        # Load checkpoint weights
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 4. Load competition validation template
    logger.info("\n4. Loading competition validation template...")
    adata = ad.read_h5ad(adata_path)
    logger.info(f"✅ Loaded data: {adata.shape[0]} cells, {adata.shape[1]} genes")

    # 5. Load perturbation features
    logger.info("\n5. Loading perturbation features...")
    pert_features = torch.load(pert_features_path)
    logger.info(f"✅ Loaded {len(pert_features)} perturbation features")

    # 6. Prepare input data
    logger.info("\n6. Preparing input data...")

    # Get control cells (non-targeting)
    control_mask = adata.obs['target_gene'] == 'non-targeting'
    control_cells = adata[control_mask].X.toarray() if hasattr(adata[control_mask].X, 'toarray') else adata[control_mask].X
    logger.info(f"Found {control_cells.shape[0]} control cells")

    # Group by perturbation
    unique_perts = [p for p in adata.obs['target_gene'].unique() if p != 'non-targeting']
    logger.info(f"Found {len(unique_perts)} unique perturbations")

    # 7. Generate predictions
    logger.info("\n7. Generating predictions...")

    predictions_list = []
    cell_indices = []

    with torch.no_grad():
        for pert in tqdm(unique_perts, desc="Processing perturbations"):
            # Get perturbation embedding
            if pert not in pert_features:
                logger.warning(f"Perturbation {pert} not found in features, skipping")
                continue

            pert_emb = pert_features[pert]
            if not isinstance(pert_emb, torch.Tensor):
                pert_emb = torch.tensor(pert_emb)

            # Get cells for this perturbation
            pert_mask = adata.obs['target_gene'] == pert
            pert_indices = np.where(pert_mask)[0]

            if len(pert_indices) == 0:
                continue

            # Sample control cells (same number as perturbed cells)
            n_samples = min(len(pert_indices), len(control_cells))
            ctrl_sample_indices = np.random.choice(len(control_cells), n_samples, replace=False)
            ctrl_samples = control_cells[ctrl_sample_indices]

            # Convert to tensors
            ctrl_tensor = torch.tensor(ctrl_samples, dtype=torch.float32).to(device)
            pert_tensor = pert_emb.unsqueeze(0).expand(n_samples, -1).to(device)

            # Create batch
            batch = {
                'ctrl_cell_emb': ctrl_tensor,
                'pert_emb': pert_tensor,
            }

            # Forward pass
            pred = model(batch, padded=False)

            # Decode from state space (512) to gene space (18080)
            if hasattr(model, 'gene_decoder') and model.gene_decoder is not None:
                logger.info("Decoding state embeddings to gene expression space...")
                gene_pred = model.gene_decoder(pred)  # [n_samples, 18080]
                predictions_list.append(gene_pred.cpu().numpy())
            else:
                logger.warning("No gene decoder found! Saving state embeddings (512-dim) instead.")
                predictions_list.append(pred.cpu().numpy())

            cell_indices.extend(pert_indices[:n_samples])

    # 8. Combine predictions
    logger.info("\n8. Combining predictions...")
    all_predictions = np.concatenate(predictions_list, axis=0)
    logger.info(f"Generated {all_predictions.shape[0]} predictions")

    # 9. Create output AnnData
    logger.info("\n9. Creating output file...")

    # For competition submission, we need to match the template structure
    # Create a copy of the template and fill in predictions
    adata_out = adata.copy()

    # Fill predictions into the corresponding cells
    if all_predictions.shape[1] == adata_out.shape[1]:
        # Predictions are in gene expression space (18080-dim) - CORRECT!
        logger.info(f"✅ Predictions are in gene expression space ({all_predictions.shape[1]} genes)")
        logger.info("Filling predictions into X matrix...")

        # Initialize X with zeros (for cells we don't predict)
        import scipy.sparse as sp
        if sp.issparse(adata_out.X):
            adata_out.X = adata_out.X.toarray()

        # Fill in predictions
        for i, cell_idx in enumerate(cell_indices):
            adata_out.X[cell_idx] = all_predictions[i]

        logger.info(f"✅ Filled predictions for {len(cell_indices)} cells")
    else:
        # Predictions are in state embedding space (512-dim) - NOT IDEAL
        logger.warning(f"⚠️  Prediction dimension ({all_predictions.shape[1]}) != gene dimension ({adata_out.shape[1]})")
        logger.warning("Predictions are in state embedding space. Decoder was not used.")
        logger.warning("Saving state embeddings to obsm['gnn_predictions']...")

        # Save as obsm
        adata_out.obsm['gnn_predictions'] = np.zeros((adata_out.shape[0], all_predictions.shape[1]))
        for i, cell_idx in enumerate(cell_indices):
            adata_out.obsm['gnn_predictions'][cell_idx] = all_predictions[i]

    # 10. Save output
    logger.info(f"\n10. Saving predictions to {output_path}...")
    adata_out.write_h5ad(output_path)
    logger.info("✅ Predictions saved!")

    logger.info("\n" + "="*60)
    logger.info("✅ Inference complete!")
    logger.info("="*60)
    logger.info(f"Predictions saved to: {output_path}")
    logger.info(f"Total cells predicted: {len(cell_indices)}")
    logger.info(f"Prediction shape: {all_predictions.shape}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate competition predictions using trained GNN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--adata", type=str, required=True, help="Path to competition validation template (.h5ad)")
    parser.add_argument("--output", type=str, required=True, help="Output path for predictions (.h5ad)")
    parser.add_argument("--pert-features", "--perturbation-features", dest="pert_features", type=str, default="/data/ESM2_pert_features.pt", help="Perturbation features")
    parser.add_argument("--data-dir", dest="data_dir", type=str, default="/data", help="Data directory")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Run inference
    infer_gnn(
        checkpoint_path=args.checkpoint,
        adata_path=args.adata,
        output_path=args.output,
        pert_features_path=args.pert_features,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=device,
    )


if __name__ == "__main__":
    main()
