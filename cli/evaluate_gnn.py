#!/usr/bin/env python3
"""
Evaluate trained GNN model on test set.

This script:
1. Loads the trained model checkpoint
2. Evaluates on test set (or validation set)
3. Computes metrics: MSE, Pearson correlation, etc.
4. Saves results to CSV and generates plots
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
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


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted values [N, D]
        targets: Ground truth values [N, D]

    Returns:
        Dictionary of metrics
    """
    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    # Remove NaN values
    mask = ~(np.isnan(pred_flat) | np.isnan(target_flat))
    pred_flat = pred_flat[mask]
    target_flat = target_flat[mask]

    metrics = {
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

    # Per-sample correlation (average across samples)
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
    """
    Evaluate trained GNN model.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Data directory
        toml_config: TOML config path
        pert_features: Perturbation features path
        output_dir: Where to save results
        batch_size: Batch size for evaluation
        val_split: Validation split ratio
        test_split: Test split ratio
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("GNN Model Evaluation")
    logger.info("="*60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Output dir: {output_dir}")
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

    # 3. Load model from checkpoint
    logger.info("\n3. Loading model from checkpoint...")
    try:
        # First, create model with same architecture as training
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
        )

        # Then load checkpoint weights
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

    # 4. Setup data module
    logger.info("\n4. Setting up data module...")
    datamodule = SE_ST_DataModule(
        toml_config_path=toml_config,
        perturbation_features_file=pert_features,
        batch_size=batch_size,
        num_workers=4,
        val_split=val_split,
        test_split=test_split,
    )
    datamodule.setup(stage="test")
    logger.info("✅ Data module ready")

    # Use test dataloader if available, otherwise val
    if datamodule.test_dataset is not None and len(datamodule.test_dataset) > 0:
        dataloader = datamodule.test_dataloader()
        split_name = "test"
        logger.info(f"Using test set: {len(datamodule.test_dataset)} samples")
    else:
        datamodule.setup(stage="fit")
        dataloader = datamodule.val_dataloader()
        split_name = "validation"
        logger.info(f"Using validation set: {len(datamodule.val_dataset)} samples")

    # 5. Run inference
    logger.info(f"\n5. Running inference on {split_name} set...")
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward pass
            predictions = model(batch, padded=True)
            targets = batch["pert_cell_emb"]

            # Flatten if needed
            if len(targets.shape) == 3:
                targets = targets.reshape(-1, targets.shape[-1])

            # Encode target to state embedding space (same as training)
            if targets.shape[-1] == model.input_dim:
                targets = model.encode_cells_to_state(targets)

            # Store results
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    logger.info(f"✅ Inference complete: {predictions.shape[0]} samples")

    # 6. Compute metrics
    logger.info("\n6. Computing metrics...")
    metrics = compute_metrics(predictions, targets)

    logger.info("\n" + "="*60)
    logger.info("Evaluation Results:")
    logger.info("="*60)
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.6f}")

    # 7. Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\n✅ Metrics saved to {metrics_path}")

    # 8. Generate plots
    logger.info("\n7. Generating plots...")

    # Plot 1: Scatter plot (sample subset for clarity)
    plt.figure(figsize=(10, 8))
    sample_size = min(10000, len(predictions.flatten()))
    indices = np.random.choice(len(predictions.flatten()), sample_size, replace=False)
    plt.scatter(
        targets.flatten()[indices],
        predictions.flatten()[indices],
        alpha=0.3,
        s=1
    )
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    plt.xlabel("Ground Truth", fontsize=12)
    plt.ylabel("Prediction", fontsize=12)
    plt.title(f"Predictions vs Ground Truth\nPearson r = {metrics.get('pearson_r', 0):.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_plot.png", dpi=150)
    plt.close()
    logger.info("  ✅ Scatter plot saved")

    # Plot 2: Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(targets.flatten(), bins=50, alpha=0.7, label="Ground Truth", density=True)
    axes[0].hist(predictions.flatten(), bins=50, alpha=0.7, label="Predictions", density=True)
    axes[0].set_xlabel("Value", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Distribution Comparison", fontsize=14)
    axes[0].legend()

    # Residuals
    residuals = predictions.flatten() - targets.flatten()
    axes[1].hist(residuals, bins=50, alpha=0.7, color='orange')
    axes[1].set_xlabel("Residual (Pred - True)", fontsize=12)
    axes[1].set_ylabel("Count", fontsize=12)
    axes[1].set_title(f"Residuals Distribution\nMAE = {metrics.get('mae', 0):.4f}", fontsize=14)
    axes[1].axvline(0, color='red', linestyle='--', lw=2)

    plt.tight_layout()
    plt.savefig(output_dir / "distribution_plots.png", dpi=150)
    plt.close()
    logger.info("  ✅ Distribution plots saved")

    # 9. Save predictions
    logger.info("\n8. Saving predictions...")
    pred_df = pd.DataFrame({
        'sample_idx': np.repeat(np.arange(predictions.shape[0]), predictions.shape[1]),
        'feature_idx': np.tile(np.arange(predictions.shape[1]), predictions.shape[0]),
        'prediction': predictions.flatten(),
        'target': targets.flatten(),
        'residual': residuals,
    })
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"✅ Predictions saved to {pred_path}")

    logger.info("\n" + "="*60)
    logger.info("✅ Evaluation complete!")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Metrics: evaluation_metrics.csv")
    logger.info(f"  - Predictions: predictions.csv")
    logger.info(f"  - Plots: scatter_plot.png, distribution_plots.png")

    return metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained GNN model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/data", help="Data directory")
    parser.add_argument("--toml_config", type=str, default="/data/starter.toml.simple", help="TOML config")
    parser.add_argument("--pert_features", type=str, default="/data/ESM2_pert_features.pt", help="Perturbation features")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test split")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Run evaluation
    metrics = evaluate_model(
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
