"""
Simplified GNN training script that reuses existing SE-ST data setup.
Just adds GNN on top of working SE-ST configuration.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import logging
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from gnn.models.gnn_perturbation import GNN_PerturbationModel
from gnn.utils.string_network_loader import load_string_network_for_hvgs
from gnn.cli.train import SE_ST_DataModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Simplified training - just specify the essential arguments.

    Usage:
        python -m gnn.cli.train_gnn_simple \\
            --data_dir /data \\
            --toml_config /data/starter.toml \\
            --pert_features /data/ESM2_pert_features.pt
    """
    import argparse

    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="/data")
    parser.add_argument("--toml_config", type=str, default="/data/starter.toml")
    parser.add_argument("--pert_features", type=str, default="/data/ESM2_pert_features.pt")

    # GNN arguments
    parser.add_argument("--use_gnn", type=bool, default=True)
    parser.add_argument("--gnn_type", type=str, default="gcn", choices=["gcn", "gat", "sage"])
    parser.add_argument("--gnn_hidden_dim", type=int, default=128)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--string_confidence", type=float, default=0.4)

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_early_stopping", action="store_true", help="Disable early stopping")

    # Data split arguments (for auto-splitting train data into train/val/test)
    parser.add_argument("--val_split", type=float, default=0.0, help="Fraction of train data to use for validation (0.0-1.0)")
    parser.add_argument("--test_split", type=float, default=0.0, help="Fraction of train data to use for testing (0.0-1.0)")

    # SE model arguments
    parser.add_argument("--se_model_path", type=str, default="SE-600M")
    parser.add_argument("--se_checkpoint", type=str, default="SE-600M/se600m_epoch15.ckpt")

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("GNN-Enhanced Perturbation Prediction Training")
    logger.info("="*60)
    logger.info(f"Data config: {args.toml_config}")
    logger.info(f"Use GNN: {args.use_gnn}")
    logger.info(f"GNN type: {args.gnn_type}")
    logger.info(f"GNN layers: {args.gnn_layers}")
    logger.info(f"GNN hidden dim: {args.gnn_hidden_dim}")

    # 1. Load gene names
    logger.info("Loading gene names from data...")
    hvg_gene_names = load_gene_names(args.data_dir)
    logger.info(f"Loaded {len(hvg_gene_names)} genes")

    # 2. Load STRING network (REQUIRED for GNN mode)
    edge_index, gene_to_idx = None, None
    if args.use_gnn:
        if len(hvg_gene_names) == 0:
            logger.error("❌ Cannot use GNN: No gene names loaded from data!")
            logger.error("Please check that h5ad files exist in data_dir")
            raise ValueError("No gene names available for STRING network loading")

        logger.info("Loading STRING network...")
        try:
            edge_index, gene_to_idx = load_string_network_for_hvgs(
                hvg_gene_names=hvg_gene_names,
                cache_dir=f"{args.data_dir}/string_cache",
                confidence_threshold=args.string_confidence,
            )
            logger.info(f"✅ Loaded STRING network: {edge_index.shape[1]} edges, {len(gene_to_idx)} genes")
        except Exception as e:
            logger.error(f"❌ Failed to load STRING network: {e}")
            logger.error("GNN mode requires STRING network. Training aborted.")
            raise RuntimeError(f"STRING network loading failed: {e}") from e

    # 3. Create model with decoder configuration
    logger.info("Creating GNN perturbation model with gene decoder...")

    # Decoder configuration for state -> gene expression
    decoder_cfg = {
        'latent_dim': 512,      # Output dimension from ST model
        'gene_dim': 18080,      # Number of genes
        'hidden_dims': [1024, 1024, 512],  # Decoder layers
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
        se_model_path=args.se_model_path,
        se_checkpoint_path=args.se_checkpoint,
        freeze_se_model=True,
        # ST
        st_hidden_dim=512,
        st_cell_set_len=128,
        # GNN
        use_gnn=args.use_gnn,
        gene_network_edge_index=edge_index,
        gene_to_idx=gene_to_idx,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        gnn_type=args.gnn_type,
        gnn_dropout=0.1,
        # Training
        lr=args.lr,
        # Decoder (NEW!)
        decoder_cfg=decoder_cfg,
        gene_decoder_bool=True,
    )
    logger.info(f"✅ Model created (GNN enabled: {args.use_gnn})")

    # 4. Create data module
    logger.info("Creating data module...")

    # Log split configuration
    if args.val_split > 0 or args.test_split > 0:
        logger.info(f"Auto-split enabled:")
        logger.info(f"  - Validation split: {args.val_split:.1%}")
        logger.info(f"  - Test split: {args.test_split:.1%}")
        logger.info(f"  - Actual train: {1 - args.val_split - args.test_split:.1%}")

    datamodule = SE_ST_DataModule(
        toml_config_path=args.toml_config,
        perturbation_features_file=args.pert_features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
    )
    logger.info("✅ Data module created")

    # 5. Setup trainer
    logger.info("Setting up trainer...")

    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="gnn-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
    ]

    # Add early stopping if not disabled
    if not args.disable_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                mode="min",
            )
        )
        logger.info("Early stopping enabled (patience=10)")
    else:
        logger.info("Early stopping DISABLED - will train for full epochs/steps")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        val_check_interval=100,
        gradient_clip_val=1.0,
        precision="16-mixed",
        devices=1,
        accelerator="gpu",
        callbacks=callbacks,
        logger=TensorBoardLogger(
            save_dir="logs",
            name="gnn_experiment",
        ),
    )
    logger.info("✅ Trainer ready")

    # 6. Train
    logger.info("Starting training...")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Batch size: {args.batch_size}")

    trainer.fit(model, datamodule)

    logger.info("="*60)
    logger.info("✅ Training completed successfully!")
    logger.info("="*60)

    return model


def load_gene_names(data_dir: str):
    """Load gene names from h5ad files."""
    import anndata as ad
    import os
    import glob

    # First try specific known files
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
            logger.info(f"✅ Loaded {len(gene_names)} genes from {os.path.basename(filepath)}")
            return gene_names

    # If none found, search for any h5ad file
    logger.info(f"Searching for any h5ad files in {data_dir}...")
    h5ad_files = glob.glob(os.path.join(data_dir, "*.h5ad"))

    if len(h5ad_files) > 0:
        filepath = h5ad_files[0]
        logger.info(f"Found h5ad file: {filepath}")
        adata = ad.read_h5ad(filepath)
        gene_names = adata.var_names.tolist()
        logger.info(f"✅ Loaded {len(gene_names)} genes from {os.path.basename(filepath)}")
        return gene_names

    logger.error(f"❌ No h5ad files found in {data_dir}")
    return []


if __name__ == "__main__":
    main()
