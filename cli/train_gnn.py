"""
Training script for GNN-based perturbation prediction model.
"""

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gnn_perturbation import GNN_PerturbationModel
from utils.string_network_loader import load_string_network_for_hvgs
from cli.train import SE_ST_DataModule  # Reuse existing data module

logger = logging.getLogger(__name__)


def train_gnn_model(
    data_dir: str = "/data",
    batch_size: int = 16,
    max_epochs: int = 100,
    lr: float = 1e-4,
    # GNN-specific parameters
    use_gnn: bool = True,
    gnn_hidden_dim: int = 512,
    gnn_layers: int = 2,
    gnn_type: str = "gcn",  # "gcn", "gat", "sage"
    string_confidence: float = 0.4,
    # SE-ST parameters
    se_model_path: str = "SE-600M",
    se_checkpoint_path: str = "SE-600M/se600m_epoch15.ckpt",
    resume_from_checkpoint: str = None,
) -> GNN_PerturbationModel:
    """
    Train GNN-based perturbation model.

    Args:
        data_dir: Data directory
        batch_size: Batch size
        max_epochs: Maximum epochs
        lr: Learning rate
        use_gnn: Whether to use GNN
        gnn_hidden_dim: GNN hidden dimension
        gnn_layers: Number of GNN layers
        gnn_type: GNN type
        string_confidence: STRING confidence threshold
        se_model_path: SE model path
        se_checkpoint_path: SE checkpoint path
        resume_from_checkpoint: Resume from checkpoint

    Returns:
        Trained model
    """

    # 1. Load gene regulatory network from STRING
    logger.info("Loading STRING gene regulatory network...")

    # TODO: Get HVG gene names from your dataset
    # For now, using a placeholder - you should load actual gene names
    hvg_gene_names = load_hvg_names_from_data(data_dir)

    if use_gnn:
        try:
            edge_index, gene_to_idx = load_string_network_for_hvgs(
                hvg_gene_names=hvg_gene_names,
                cache_dir=f"{data_dir}/string_cache",
                confidence_threshold=string_confidence,
            )
            logger.info(f"Loaded STRING network: {edge_index.shape[1]} edges, {len(gene_to_idx)} genes")
        except Exception as e:
            logger.warning(f"Failed to load STRING network: {e}. Falling back to SE-ST mode.")
            use_gnn = False
            edge_index = None
            gene_to_idx = None
    else:
        edge_index = None
        gene_to_idx = None

    # 2. Create model
    model = GNN_PerturbationModel(
        # Basic parameters
        input_dim=18080,
        hidden_dim=512,
        output_dim=512,
        pert_dim=1280,
        # SE model
        se_model_path=se_model_path,
        se_checkpoint_path=se_checkpoint_path,
        freeze_se_model=True,
        # ST model
        st_hidden_dim=512,
        st_cell_set_len=128,
        # GNN parameters
        use_gnn=use_gnn,
        gene_network_edge_index=edge_index,
        gene_to_idx=gene_to_idx,
        gnn_hidden_dim=gnn_hidden_dim,
        gnn_layers=gnn_layers,
        gnn_type=gnn_type,
        # Training
        lr=lr,
    )

    # 3. Create data module
    datamodule = SE_ST_DataModule(
        toml_config_path=f"{data_dir}/starter.toml",
        perturbation_features_file=f"{data_dir}/ESM2_pert_features.pt",
        batch_size=batch_size,
        num_workers=4,
    )

    # 4. Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="gnn-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
        ),
    ]

    # 5. Setup logger
    logger_tb = TensorBoardLogger(
        save_dir="logs",
        name="gnn_experiment",
    )

    # 6. Trainer configuration
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger_tb,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        precision=16,
        devices=1,
        accelerator="gpu",
    )

    # 7. Train
    if resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.fit(model, datamodule, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(model, datamodule)

    return model


def load_hvg_names_from_data(data_dir: str) -> list:
    """
    Load highly variable gene names from data.

    Args:
        data_dir: Data directory

    Returns:
        List of gene names
    """
    import anndata as ad
    import os

    # Try to load from one of the h5ad files
    possible_files = [
        f"{data_dir}/competition_train.h5ad",
        f"{data_dir}/k562.h5ad",
        f"{data_dir}/jurkat.h5ad",
    ]

    for filepath in possible_files:
        if os.path.exists(filepath):
            logger.info(f"Loading gene names from {filepath}")
            adata = ad.read_h5ad(filepath)
            gene_names = adata.var_names.tolist()
            logger.info(f"Loaded {len(gene_names)} gene names")
            return gene_names

    logger.warning("No data file found to extract gene names. Using empty list.")
    return []


@hydra.main(version_base=None, config_path="../configs", config_name="gnn_config")
def main(cfg: DictConfig) -> None:
    """Main training function."""

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Get resume checkpoint path (if any)
    resume_ckpt = cfg.get("resume_from_checkpoint", None)

    # Train model
    model = train_gnn_model(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        max_epochs=cfg.training.max_epochs,
        lr=cfg.training.optimizer.lr,
        use_gnn=cfg.model.use_gnn,
        gnn_hidden_dim=cfg.model.gnn_hidden_dim,
        gnn_layers=cfg.model.gnn_layers,
        gnn_type=cfg.model.gnn_type,
        string_confidence=cfg.model.string_confidence,
        se_model_path=cfg.model.se_model_path,
        se_checkpoint_path=cfg.model.se_checkpoint_path,
        resume_from_checkpoint=resume_ckpt,
    )

    logger.info("Training completed successfully!")

    return model


if __name__ == "__main__":
    main()
