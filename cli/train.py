"""
Training script for SE+ST Combined Model with Hydra configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

# Import model classes
from se_st_combined.models.se_st_combined import SE_ST_CombinedModel
from se_st_combined.models.state_transition import StateTransitionPerturbationModel

# Import data utilities
from se_st_combined.data.perturbation_dataset import PerturbationDataset, collate_perturbation_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variable to disable Hydra struct checking
import os
os.environ['HYDRA_FULL_ERROR'] = '1'


class SE_ST_DataModule(LightningDataModule):
    """Data module for SE+ST training with Hydra configuration."""
    
    def __init__(
        self,
        toml_config_path: str,
        perturbation_features_file: str,
        batch_size: int = 16,
        num_workers: int = 4,
        batch_col: str = "batch_var",
        pert_col: str = "target_gene",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        **kwargs
    ):
        super().__init__()
        self.toml_config_path = toml_config_path
        self.perturbation_features_file = perturbation_features_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.batch_col = batch_col
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.kwargs = kwargs
        
        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: str = None):
        """Setup data for training/validation."""
        logger.info(f"Setting up data from {self.toml_config_path}")
        
        # Create train dataset
        if stage == "fit" or stage is None:
            logger.info("Creating training dataset...")
            self.train_dataset = PerturbationDataset(
                toml_config_path=self.toml_config_path,
                perturbation_features_file=self.perturbation_features_file,
                split="train",
                batch_col=self.batch_col,
                pert_col=self.pert_col,
                cell_type_key=self.cell_type_key,
                control_pert=self.control_pert,
            )
            logger.info(f"Training dataset created with {len(self.train_dataset)} samples")
            
            # Create validation dataset
            logger.info("Creating validation dataset...")
            self.val_dataset = PerturbationDataset(
                toml_config_path=self.toml_config_path,
                perturbation_features_file=self.perturbation_features_file,
                split="val",
                batch_col=self.batch_col,
                pert_col=self.pert_col,
                cell_type_key=self.cell_type_key,
                control_pert=self.control_pert,
            )
            
            # If no val data, use a subset of train data for validation
            if len(self.val_dataset) == 0:
                logger.warning("No validation data found, using test split for validation")
                self.val_dataset = PerturbationDataset(
                    toml_config_path=self.toml_config_path,
                    perturbation_features_file=self.perturbation_features_file,
                    split="test",
                    batch_col=self.batch_col,
                    pert_col=self.pert_col,
                    cell_type_key=self.cell_type_key,
                    control_pert=self.control_pert,
                )
            
            logger.info(f"Validation dataset created with {len(self.val_dataset)} samples")
        
        # Create test dataset
        if stage == "test" or stage is None:
            logger.info("Creating test dataset...")
            self.test_dataset = PerturbationDataset(
                toml_config_path=self.toml_config_path,
                perturbation_features_file=self.perturbation_features_file,
                split="test",
                batch_col=self.batch_col,
                pert_col=self.pert_col,
                cell_type_key=self.cell_type_key,
                control_pert=self.control_pert,
            )
            logger.info(f"Test dataset created with {len(self.test_dataset)} samples")
        
    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataset is None:
            logger.error("Train dataset not initialized! Call setup() first.")
            return DataLoader([], batch_size=self.batch_size, num_workers=0)
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_perturbation_batch,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataset is None:
            logger.error("Val dataset not initialized! Call setup() first.")
            return DataLoader([], batch_size=self.batch_size, num_workers=0)
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_perturbation_batch,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        if self.test_dataset is None:
            logger.error("Test dataset not initialized! Call setup() first.")
            return DataLoader([], batch_size=self.batch_size, num_workers=0)
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_perturbation_batch,
            pin_memory=True,
        )


def setup_logger(cfg: DictConfig):
    """Setup experiment logger"""
    loggers = []
    
    # TensorBoard logger (always enabled)
    tb_logger = TensorBoardLogger(
        save_dir=cfg.get('output_dir', './outputs'),
        name=cfg.get('name', 'se_st_training'),
    )
    loggers.append(tb_logger)
    
    # W&B logger (optional)
    if cfg.get('wandb', {}).get('enabled', False):
        wandb_logger = WandbLogger(
            project=cfg.wandb.get('project', 'se-st-combined'),
            entity=cfg.wandb.get('entity', None),
            name=cfg.get('name', 'se_st_training'),
            save_dir=cfg.get('output_dir', './outputs'),
            tags=cfg.wandb.get('tags', []),
        )
        loggers.append(wandb_logger)
        logger.info("Weights & Biases logging enabled")
    
    return loggers


def setup_callbacks(cfg: DictConfig):
    """Setup training callbacks"""
    callbacks = []
    
    output_dir = Path(cfg.get('output_dir', './outputs'))
    experiment_name = cfg.get('name', 'se_st_training')
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / experiment_name / "checkpoints",
        filename="step={step}-val_loss={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=cfg.get('save_top_k', 3),
        save_last=False,  # ← Don't save last.ckpt every validation (saves disk space!)
        verbose=True,
        every_n_train_steps=cfg.training.get('ckpt_every_n_steps', 1000),
        save_on_train_epoch_end=False,  # Only save on validation, not every epoch
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback (optional)
    if cfg.training.get('early_stopping', False):
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.training.get('patience', 10),
            verbose=True,
        )
        callbacks.append(early_stop_callback)
    
    return callbacks


def train_main(cfg: DictConfig):
    """
    Main training function with Hydra configuration.
    
    Usage:
        # Using default config
        se-st-train
        
        # Override config parameters
        se-st-train model.kwargs.st_hidden_dim=1024 training.batch_size=16
        
        # Use different config file
        se-st-train --config-name custom_config
    """
    # Disable struct mode to allow flexible configuration
    OmegaConf.set_struct(cfg, False)
    
    logger.info("=" * 60)
    logger.info("SE+ST Combined Model Training with Hydra")
    logger.info("=" * 60)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Create output directory
    output_dir = Path(cfg.get('output_dir', './outputs'))
    experiment_name = cfg.get('name', 'se_st_training')
    full_output_dir = output_dir / experiment_name
    full_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {full_output_dir}")
    
    # Save the full configuration
    config_save_path = full_output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(cfg, f)
    logger.info(f"Configuration saved to: {config_save_path}")
    
    # Setup loggers
    loggers = setup_logger(cfg)
    
    # Setup callbacks
    callbacks = setup_callbacks(cfg)
    
    # Initialize trainer
    trainer = Trainer(
        max_steps=cfg.training.get('max_steps', 40000),
        max_epochs=cfg.training.get('max_epochs', None),
        accelerator=cfg.get('accelerator', 'auto'),
        devices=cfg.get('devices', 'auto'),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.get('log_every_n_steps', 100),
        val_check_interval=cfg.training.get('val_check_interval', 1000),
        precision=cfg.training.get('precision', '16-mixed' if torch.cuda.is_available() else '32'),
        gradient_clip_val=cfg.training.get('gradient_clip_val', None),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),
    )
    
    logger.info("Trainer initialized")
    logger.info(f"Device: {cfg.get('accelerator', 'auto')}")
    
    # Initialize model based on config
    logger.info("Initializing model...")
    model_name = cfg.get('model', {}).get('name', 'se_st_combined')
    model_kwargs = OmegaConf.to_container(cfg.get('model', {}).get('kwargs', {}), resolve=True)
    
    if model_name == 'se_st_combined':
        logger.info("Creating SE_ST_CombinedModel")
        model = SE_ST_CombinedModel(**model_kwargs)
    else:
        logger.info("Creating StateTransitionPerturbationModel")
        model = StateTransitionPerturbationModel(**model_kwargs)
    
    logger.info(f"Model initialized: {model.__class__.__name__}")
    
    # Initialize data module
    logger.info("Initializing data module...")
    data_kwargs = OmegaConf.to_container(cfg.get('data', {}).get('kwargs', {}), resolve=True)
    
    # Add batch_size from training config if not in data kwargs
    if 'batch_size' not in data_kwargs:
        data_kwargs['batch_size'] = cfg.training.get('batch_size', 16)
    
    datamodule = SE_ST_DataModule(**data_kwargs)
    logger.info("Data module initialized")
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Max steps: {cfg.training.get('max_steps', 40000)}")
    logger.info(f"Batch size: {data_kwargs.get('batch_size', 16)}")
    
    try:
        trainer.fit(model, datamodule)
        logger.info("Training completed successfully!")
        
        # Save final model
        final_model_path = full_output_dir / "final_model.ckpt"
        trainer.save_checkpoint(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        return 1


@hydra.main(version_base=None, config_path="../configs", config_name="se_st_combined")
def main(cfg: DictConfig):
    """Hydra entry point that disables struct before calling train_main"""
    # Must disable struct mode IMMEDIATELY before any config access
    OmegaConf.set_struct(cfg, False)
    return train_main(cfg)


if __name__ == "__main__":
    sys.exit(main())

