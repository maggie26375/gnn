#!/usr/bin/env python3
"""
Evaluate GNN model with decoder in both state and gene expression spaces.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from gnn.models.gnn_perturbation import GNN_PerturbationModel
from gnn.cli.train import SE_ST_DataModule
from gnn.utils.string_network_loader import load_string_network_for_hvgs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gene_names(data_dir: str):
    """Load gene names from h5ad files."""
    import anndata as ad
    import os
    import glob

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


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, space_name: str) -> dict:
    """Compute evaluation metrics."""
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]

    metrics = {
        'space': space_name,
        'mse': mean_squared_error(target_flat, pred_flat),
        'mae': mean_absolute_error(target_flat, pred_flat),
        'rmse': np.sqrt(mean_squared_error(target_flat, pred_flat)),
        'r2': r2_score(target_flat, pred_flat),
    }

    # Pearson and Spearman correlation
    if len(pred_flat) > 0:
        try:
            pearson_r, pearson_p = pearsonr(pred_flat, target_flat)
            spearman_r, spearman_p = spearmanr(pred_flat, target_flat)
            metrics['pearson_r'] = pearson_r
            metrics['pearson_p'] = pearson_p
            metrics['spearman_r'] = spearman_r
            metrics['spearman_p'] = spearman_p
        except Exception as e:
            logger.warning(f"Failed to compute correlation: {e}")
            metrics['pearson_r'] = np.nan
            metrics['spearman_r'] = np.nan

    # Per-sample correlation
    per_sample_corr = []
    for i in range(predictions.shape[0]):
        pred_sample = predictions[i]
        target_sample = targets[i]
        mask = ~(np.isnan(pred_sample) | np.isnan(target_sample))
        if mask.sum() > 1:
            try:
                corr, _ = pearsonr(pred_sample[mask], target_sample[mask])
                per_sample_corr.append(corr)
            except:
                pass

    if len(per_sample_corr) > 0:
        metrics['mean_per_sample_pearson'] = np.mean(per_sample_corr)
        metrics['median_per_sample_pearson'] = np.median(per_sample_corr)

    return metrics


def evaluate_model(
    checkpoint_path: str,
    data_dir: str,
    toml_config: str,
    pert_features: str,
    output_dir: str,
    batch_size: int = 8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Evaluate model in both state and gene expression spaces."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("GNN Model Evaluation with Decoder")
    logger.info("="*60)

    # 1. Load gene names and STRING network
    hvg_gene_names = load_gene_names(data_dir)
    edge_index, gene_to_idx = load_string_network_for_hvgs(
        hvg_gene_names=hvg_gene_names,
        cache_dir=f"{data_dir}/string_cache",
        confidence_threshold=0.4,
    )

    # 2. Load model with decoder
    logger.info("\n2. Loading model with decoder...")
    decoder_cfg = {
        'latent_dim': 512,
        'gene_dim': 18080,
        'hidden_dims': [1024, 1024, 512],
        'dropout': 0.1,
        'residual_decoder': False,
    }

    model = GNN_PerturbationModel(
        input_dim=18080, hidden_dim=512, output_dim=512, pert_dim=1280,
        se_model_path="SE-600M", se_checkpoint_path="SE-600M/se600m_epoch15.ckpt",
        freeze_se_model=True, st_hidden_dim=512, st_cell_set_len=128,
        use_gnn=True, gene_network_edge_index=edge_index, gene_to_idx=gene_to_idx,
        gnn_hidden_dim=128, gnn_layers=3, gnn_type="gcn", gnn_dropout=0.1,
        lr=1e-4, decoder_cfg=decoder_cfg, gene_decoder_bool=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    has_decoder = hasattr(model, 'gene_decoder') and model.gene_decoder is not None
    logger.info(f"✅ Model loaded (has decoder: {has_decoder})")

    # 3. Setup data
    datamodule = SE_ST_DataModule(
        toml_config_path=toml_config,
        perturbation_features_file=pert_features,
        batch_size=batch_size,
        num_workers=4,
        val_split=val_split,
        test_split=test_split,
    )
    datamodule.setup(stage="test")

    if datamodule.test_dataset is not None and len(datamodule.test_dataset) > 0:
        dataloader = datamodule.test_dataloader()
        split_name = "test"
    else:
        datamodule.setup(stage="fit")
        dataloader = datamodule.val_dataloader()
        split_name = "validation"

    logger.info(f"Using {split_name} set")

    # 4. Run inference
    logger.info(f"\n4. Running inference...")
    state_predictions = []
    state_targets = []
    gene_predictions = []
    gene_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass (state space predictions)
            state_pred = model(batch, padded=True)

            # Get state space targets
            state_target = batch["pert_cell_emb"]
            if len(state_target.shape) == 3:
                state_target = state_target.reshape(-1, state_target.shape[-1])

            # Encode to state space if needed
            if state_target.shape[-1] == model.input_dim:
                state_target = model.encode_cells_to_state(state_target)

            # Store state space results
            state_predictions.append(state_pred.cpu().numpy())
            state_targets.append(state_target.cpu().numpy())

            # Decode to gene expression space if decoder available
            if has_decoder:
                gene_pred = model.gene_decoder(state_pred)

                # Get gene space targets
                if "pert_cell_counts" in batch:
                    gene_target = batch["pert_cell_counts"]
                else:
                    # Use original gene expression
                    gene_target = batch["pert_cell_emb"]

                if len(gene_target.shape) == 3:
                    gene_target = gene_target.reshape(-1, gene_target.shape[-1])

                gene_predictions.append(gene_pred.cpu().numpy())
                gene_targets.append(gene_target.cpu().numpy())

    # 5. Concatenate results
    state_preds = np.concatenate(state_predictions, axis=0)
    state_targs = np.concatenate(state_targets, axis=0)

    logger.info(f"\n5. Computing metrics...")

    # State space metrics
    state_metrics = compute_metrics(state_preds, state_targs, "State Space (512-dim)")

    logger.info("\n" + "="*60)
    logger.info("STATE SPACE (512-dim) Metrics:")
    logger.info("="*60)
    for name, value in state_metrics.items():
        if name != 'space':
            logger.info(f"  {name}: {value:.6f}")

    # Gene space metrics (if decoder exists)
    if has_decoder and len(gene_predictions) > 0:
        gene_preds = np.concatenate(gene_predictions, axis=0)
        gene_targs = np.concatenate(gene_targets, axis=0)

        gene_metrics = compute_metrics(gene_preds, gene_targs, "Gene Expression (18080-dim)")

        logger.info("\n" + "="*60)
        logger.info("GENE EXPRESSION SPACE (18080-dim) Metrics:")
        logger.info("="*60)
        for name, value in gene_metrics.items():
            if name != 'space':
                logger.info(f"  {name}: {value:.6f}")

        # Save combined metrics
        metrics_df = pd.DataFrame([state_metrics, gene_metrics])
    else:
        logger.warning("\nNo gene decoder - skipping gene expression space evaluation")
        metrics_df = pd.DataFrame([state_metrics])

    # 6. Save results
    metrics_path = output_dir / "decoder_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\n✅ Metrics saved to {metrics_path}")

    logger.info("\n" + "="*60)
    logger.info("✅ Evaluation complete!")
    logger.info("="*60)

    return metrics_df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/data")
    parser.add_argument("--toml_config", type=str, default="/data/starter.toml.simple")
    parser.add_argument("--pert_features", type=str, default="/data/ESM2_pert_features.pt")
    parser.add_argument("--output_dir", type=str, default="./decoder_evaluation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        toml_config=args.toml_config,
        pert_features=args.pert_features,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        device=device,
    )


if __name__ == "__main__":
    main()
