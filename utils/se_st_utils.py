"""
Utility functions for SE+ST Combined Model

This module provides helper functions for working with the SE+ST combined model,
including data preprocessing, model loading, and inference utilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import scanpy as sc
from pathlib import Path

logger = logging.getLogger(__name__)


def preprocess_data_for_se_st(
    adata_path: str,
    se_model_path: str,
    se_checkpoint_path: str,
    output_path: str,
    batch_size: int = 1000,
    device: str = "cuda"
) -> str:
    """
    Preprocess data by converting gene expressions to SE embeddings.
    
    Args:
        adata_path: Path to input AnnData file
        se_model_path: Path to SE model directory
        se_checkpoint_path: Path to SE model checkpoint
        output_path: Path to save preprocessed data
        batch_size: Batch size for processing
        device: Device to use for computation
        
    Returns:
        Path to preprocessed data file
    """
    try:
        from ...emb.inference import Inference
        
        # Load SE model
        logger.info(f"Loading SE model from {se_checkpoint_path}")
        se_inference = Inference()
        se_inference.load_model(se_checkpoint_path)
        
        # Load data
        logger.info(f"Loading data from {adata_path}")
        adata = sc.read_h5ad(adata_path)
        
        # Convert to SE embeddings
        logger.info("Converting gene expressions to SE embeddings...")
        se_inference.transform(
            input_path=adata_path,
            output_path=output_path,
            batch_size=batch_size
        )
        
        logger.info(f"Preprocessed data saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise


def create_se_st_batch(
    ctrl_expressions: torch.Tensor,
    pert_embeddings: torch.Tensor,
    se_model,
    device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    """
    Create a batch for SE+ST model by converting expressions to SE embeddings.
    
    Args:
        ctrl_expressions: Control cell expressions [N_cells, N_genes]
        pert_embeddings: Perturbation embeddings [N_cells, pert_dim]
        se_model: Pre-trained SE model
        device: Device to use for computation
        
    Returns:
        Batch dictionary with SE embeddings
    """
    # Move to device
    ctrl_expressions = ctrl_expressions.to(device)
    pert_embeddings = pert_embeddings.to(device)
    
    # Convert to SE embeddings
    with torch.no_grad():
        se_embeddings = se_model.encode_cells(ctrl_expressions)
    
    # Create batch
    batch = {
        "ctrl_cell_emb": se_embeddings,
        "pert_emb": pert_embeddings,
    }
    
    return batch


def load_se_st_model(
    model_dir: str,
    checkpoint_path: str,
    se_model_path: str,
    se_checkpoint_path: str,
    device: str = "cuda"
):
    """
    Load SE+ST combined model from checkpoint.
    
    Args:
        model_dir: Directory containing model configuration
        checkpoint_path: Path to model checkpoint
        se_model_path: Path to SE model directory
        se_checkpoint_path: Path to SE model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded SE+ST model
    """
    try:
        from .se_st_combined import SE_ST_CombinedModel
        import torch
        
        # Load model configuration
        config_path = Path(model_dir) / "config.yaml"
        if config_path.exists():
            from omegaconf import OmegaConf
            config = OmegaConf.load(config_path)
        else:
            # Use default configuration
            config = {
                'input_dim': 2000,
                'hidden_dim': 512,
                'output_dim': 2000,
                'pert_dim': 1280,
                'se_model_path': se_model_path,
                'se_checkpoint_path': se_checkpoint_path,
            }
        
        # Initialize model
        model = SE_ST_CombinedModel(**config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        # Move to device
        model = model.to(device)
        model.eval()
        
        logger.info(f"SE+ST model loaded from {checkpoint_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading SE+ST model: {e}")
        raise


def predict_perturbation_effects(
    model,
    ctrl_expressions: torch.Tensor,
    pert_embeddings: torch.Tensor,
    batch_size: int = 1000,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Predict perturbation effects using SE+ST model.
    
    Args:
        model: Loaded SE+ST model
        ctrl_expressions: Control cell expressions [N_cells, N_genes]
        pert_embeddings: Perturbation embeddings [N_cells, pert_dim]
        batch_size: Batch size for prediction
        device: Device to use for computation
        
    Returns:
        Predicted perturbed expressions [N_cells, N_genes]
    """
    model.eval()
    predictions = []
    
    # Process in batches
    n_cells = ctrl_expressions.shape[0]
    for i in range(0, n_cells, batch_size):
        end_idx = min(i + batch_size, n_cells)
        
        # Get batch
        batch_ctrl = ctrl_expressions[i:end_idx].to(device)
        batch_pert = pert_embeddings[i:end_idx].to(device)
        
        # Create batch
        batch = {
            "ctrl_cell_emb": batch_ctrl,
            "pert_emb": batch_pert,
        }
        
        # Predict
        with torch.no_grad():
            batch_pred = model.forward(batch, padded=False)
            predictions.append(batch_pred.cpu())
    
    # Concatenate predictions
    predictions = torch.cat(predictions, dim=0)
    return predictions


def evaluate_cross_cell_type_performance(
    model,
    test_data: Dict[str, torch.Tensor],
    cell_types: List[str],
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate cross-cell-type performance of SE+ST model.
    
    Args:
        model: Loaded SE+ST model
        test_data: Dictionary containing test data for each cell type
        cell_types: List of cell types to evaluate
        device: Device to use for computation
        
    Returns:
        Dictionary with performance metrics for each cell type
    """
    results = {}
    
    for cell_type in cell_types:
        if cell_type not in test_data:
            logger.warning(f"No test data for cell type: {cell_type}")
            continue
        
        # Get test data
        data = test_data[cell_type]
        ctrl_expressions = data["ctrl_expressions"].to(device)
        pert_embeddings = data["pert_embeddings"].to(device)
        true_perturbations = data["true_perturbations"].to(device)
        
        # Predict
        predictions = predict_perturbation_effects(
            model, ctrl_expressions, pert_embeddings, device=device
        )
        
        # Compute metrics
        mse = torch.nn.functional.mse_loss(predictions, true_perturbations).item()
        mae = torch.nn.functional.l1_loss(predictions, true_perturbations).item()
        
        # Compute correlation
        pred_flat = predictions.flatten().cpu().numpy()
        true_flat = true_perturbations.flatten().cpu().numpy()
        correlation = np.corrcoef(pred_flat, true_flat)[0, 1]
        
        results[cell_type] = {
            "mse": mse,
            "mae": mae,
            "correlation": correlation
        }
        
        logger.info(f"Cell type {cell_type}: MSE={mse:.4f}, MAE={mae:.4f}, Corr={correlation:.4f}")
    
    return results


def compare_with_baseline(
    se_st_results: Dict[str, float],
    baseline_results: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Compare SE+ST results with baseline model results.
    
    Args:
        se_st_results: Results from SE+ST model
        baseline_results: Results from baseline model
        
    Returns:
        Comparison results showing improvements
    """
    comparison = {}
    
    for cell_type in se_st_results:
        if cell_type not in baseline_results:
            continue
        
        se_st_metrics = se_st_results[cell_type]
        baseline_metrics = baseline_results[cell_type]
        
        # Compute improvements
        mse_improvement = (baseline_metrics["mse"] - se_st_metrics["mse"]) / baseline_metrics["mse"] * 100
        mae_improvement = (baseline_metrics["mae"] - se_st_metrics["mae"]) / baseline_metrics["mae"] * 100
        corr_improvement = (se_st_metrics["correlation"] - baseline_metrics["correlation"]) / baseline_metrics["correlation"] * 100
        
        comparison[cell_type] = {
            "mse_improvement": mse_improvement,
            "mae_improvement": mae_improvement,
            "correlation_improvement": corr_improvement,
            "se_st_mse": se_st_metrics["mse"],
            "baseline_mse": baseline_metrics["mse"],
            "se_st_correlation": se_st_metrics["correlation"],
            "baseline_correlation": baseline_metrics["correlation"],
        }
    
    return comparison


def save_results(
    results: Dict,
    output_path: str
):
    """
    Save evaluation results to file.
    
    Args:
        results: Results dictionary to save
        output_path: Path to save results
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Example of how to use the utilities
    
    # 1. Preprocess data
    preprocess_data_for_se_st(
        adata_path="data/input.h5ad",
        se_model_path="SE-600M",
        se_checkpoint_path="SE-600M/se600m_epoch15.ckpt",
        output_path="data/preprocessed.h5ad"
    )
    
    # 2. Load model
    model = load_se_st_model(
        model_dir="competition_se_st/se_st_first_run",
        checkpoint_path="competition_se_st/se_st_first_run/checkpoints/step=20000.ckpt",
        se_model_path="SE-600M",
        se_checkpoint_path="SE-600M/se600m_epoch15.ckpt"
    )
    
    # 3. Make predictions
    # (This would require actual data)
    # predictions = predict_perturbation_effects(model, ctrl_expressions, pert_embeddings)
    
    print("SE+ST utilities loaded successfully!")

