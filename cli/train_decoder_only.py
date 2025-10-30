#!/usr/bin/env python3
"""
Train only the gene decoder while keeping the main model frozen.

This script:
1. Loads a trained GNN model with frozen weights
2. Trains only the decoder to map state embeddings back to gene expression
3. Uses the actual gene expression data (pert_cell_emb) as targets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from lightning.pytorch import Trainer, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

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
            adata = ad.read_h5ad(filepath)
            return adata.var_names.tolist()

    h5ad_files = glob.glob(os.path.join(data_dir, "*.h5ad"))
    if len(h5ad_files) > 0:
        adata = ad.read_h5ad(h5ad_files[0])
        return adata.var_names.tolist()

    return []


class DecoderOnlyModule(LightningModule):
    """Lightning module that trains only the decoder."""

    def __init__(self, full_model: GNN_PerturbationModel, lr: float = 1e-3):
        super().__init__()
        self.full_model = full_model
        self.lr = lr

        # Freeze everything except decoder
        for param in self.full_model.parameters():
            param.requires_grad = False

        # Unfreeze only decoder
        if hasattr(self.full_model, 'gene_decoder') and self.full_model.gene_decoder is not None:
            for param in self.full_model.gene_decoder.parameters():
                param.requires_grad = True
            logger.info("✅ Decoder unfrozen for training")
        else:
            raise ValueError("Model does not have a gene_decoder!")

        self.loss_fn = nn.MSELoss()

    def forward(self, batch):
        """Forward pass: main model (frozen) + decoder (trainable)."""
        # Get state predictions from frozen main model
        with torch.no_grad():
            state_pred = self.full_model(batch, padded=True)

        # Decode to gene expression (trainable)
        gene_pred = self.full_model.gene_decoder(state_pred)

        return gene_pred

    def training_step(self, batch, batch_idx):
        """Training step - only decoder is trained."""
        gene_pred = self(batch)

        # Target is the original gene expression
        gene_target = batch["pert_cell_emb"]
        if len(gene_target.shape) == 3:
            gene_target = gene_target.reshape(-1, gene_target.shape[-1])

        # Loss
        loss = self.loss_fn(gene_pred, gene_target)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        gene_pred = self(batch)

        gene_target = batch["pert_cell_emb"]
        if len(gene_target.shape) == 3:
            gene_target = gene_target.reshape(-1, gene_target.shape[-1])

        loss = self.loss_fn(gene_pred, gene_target)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Only optimize decoder parameters."""
        # Only get decoder parameters
        decoder_params = [p for p in self.full_model.gene_decoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(decoder_params, lr=self.lr)
        return optimizer


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train only the gene decoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/data")
    parser.add_argument("--toml_config", type=str, default="/data/starter.toml.simple")
    parser.add_argument("--pert_features", type=str, default="/data/ESM2_pert_features.pt")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--test_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Training Decoder Only (Main Model Frozen)")
    logger.info("="*60)

    # 1. Load gene names and STRING network
    logger.info("\n1. Loading gene names...")
    hvg_gene_names = load_gene_names(args.data_dir)

    logger.info("\n2. Loading STRING network...")
    edge_index, gene_to_idx = load_string_network_for_hvgs(
        hvg_gene_names=hvg_gene_names,
        cache_dir=f"{args.data_dir}/string_cache",
        confidence_threshold=0.4,
    )

    # 2. Load full model from checkpoint
    logger.info("\n3. Loading trained model from checkpoint...")
    decoder_cfg = {
        'latent_dim': 512,
        'gene_dim': 18080,
        'hidden_dims': [1024, 1024, 512],
        'dropout': 0.1,
        'residual_decoder': False,
    }

    full_model = GNN_PerturbationModel(
        input_dim=18080, hidden_dim=512, output_dim=512, pert_dim=1280,
        se_model_path="SE-600M", se_checkpoint_path="SE-600M/se600m_epoch15.ckpt",
        freeze_se_model=True, st_hidden_dim=512, st_cell_set_len=128,
        use_gnn=True, gene_network_edge_index=edge_index, gene_to_idx=gene_to_idx,
        gnn_hidden_dim=128, gnn_layers=3, gnn_type="gcn", gnn_dropout=0.1,
        lr=1e-4, decoder_cfg=decoder_cfg, gene_decoder_bool=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    full_model.load_state_dict(checkpoint['state_dict'])
    logger.info("✅ Model loaded")

    # 3. Create decoder-only training module
    logger.info("\n4. Creating decoder-only training module...")
    decoder_module = DecoderOnlyModule(full_model, lr=args.lr)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in decoder_module.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in decoder_module.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    # 4. Setup data
    logger.info("\n5. Setting up data...")
    datamodule = SE_ST_DataModule(
        toml_config_path=args.toml_config,
        perturbation_features_file=args.pert_features,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    # 5. Setup trainer
    logger.info("\n6. Setting up trainer...")
    callbacks = [
        ModelCheckpoint(
            dirpath="decoder_checkpoints",
            filename="decoder-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            mode="min",
        ),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=TensorBoardLogger("logs", name="decoder_only"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
        gradient_clip_val=1.0,
        val_check_interval=100,
    )

    # 6. Train
    logger.info("\n7. Starting decoder training...")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Learning rate: {args.lr}")

    trainer.fit(decoder_module, datamodule)

    logger.info("\n" + "="*60)
    logger.info("✅ Decoder training complete!")
    logger.info("="*60)
    logger.info("Best checkpoint saved in: decoder_checkpoints/")


if __name__ == "__main__":
    main()
