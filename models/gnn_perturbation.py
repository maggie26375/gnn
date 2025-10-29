"""
GNN-based Perturbation Prediction Model

This model integrates Gene Regulatory Networks (from STRING database) with
the SE+ST architecture to improve perturbation prediction by explicitly
modeling gene-gene interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import logging

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logging.warning("torch_geometric not available. Install with: pip install torch-geometric")

from .se_st_combined import SE_ST_CombinedModel

logger = logging.getLogger(__name__)


class GeneRegulatoryGNN(nn.Module):
    """
    GNN module for propagating perturbation effects through gene regulatory network.

    Supports multiple GNN architectures:
    - GCN (Graph Convolutional Network)
    - GAT (Graph Attention Network)
    - GraphSAGE
    """

    def __init__(
        self,
        gene_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        gnn_type: str = "gcn",  # "gcn", "gat", "sage"
        dropout: float = 0.1,
        use_edge_weights: bool = True,
    ):
        """
        Initialize GNN module.

        Args:
            gene_dim: Input gene feature dimension
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ("gcn", "gat", "sage")
            dropout: Dropout rate
            use_edge_weights: Whether to use edge weights from STRING
        """
        super().__init__()

        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required. Install with: pip install torch-geometric")

        self.gene_dim = gene_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.use_edge_weights = use_edge_weights

        # Build GNN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = gene_dim if i == 0 else hidden_dim

            if self.gnn_type == "gcn":
                self.convs.append(GCNConv(in_dim, hidden_dim))
            elif self.gnn_type == "gat":
                self.convs.append(GATConv(in_dim, hidden_dim, heads=1))
            elif self.gnn_type == "sage":
                self.convs.append(SAGEConv(in_dim, hidden_dim))
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}")

            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GNN.

        Args:
            x: Node features [num_genes, gene_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)

        Returns:
            Node representations [num_genes, hidden_dim]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # GNN convolution
            if self.use_edge_weights and edge_weight is not None:
                x = conv(x, edge_index, edge_weight)
            else:
                x = conv(x, edge_index)

            # Normalization
            x = norm(x)

            # Activation (except last layer)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)

        return x


class GNN_PerturbationModel(SE_ST_CombinedModel):
    """
    GNN-enhanced SE+ST model for perturbation prediction.

    Architecture:
    1. SE Encoder: genes → cell state embeddings
    2. GNN: Propagate perturbation through gene regulatory network
    3. Transformer: Model cell-cell interactions
    4. Decoder: state → perturbed gene expression
    """

    def __init__(
        self,
        # GNN-specific parameters
        gene_network_edge_index: Optional[torch.Tensor] = None,
        gene_network_edge_weight: Optional[torch.Tensor] = None,
        gene_to_idx: Optional[Dict[str, int]] = None,
        use_gnn: bool = True,
        gnn_hidden_dim: int = 512,
        gnn_layers: int = 2,
        gnn_type: str = "gcn",
        gnn_dropout: float = 0.1,
        # Inherited parameters
        **kwargs
    ):
        """
        Initialize GNN perturbation model.

        Args:
            gene_network_edge_index: Gene regulatory network [2, num_edges]
            gene_network_edge_weight: Edge weights [num_edges]
            gene_to_idx: Mapping from gene names to indices
            use_gnn: Whether to use GNN (if False, falls back to SE-ST)
            gnn_hidden_dim: Hidden dimension for GNN
            gnn_layers: Number of GNN layers
            gnn_type: Type of GNN ("gcn", "gat", "sage")
            gnn_dropout: Dropout rate for GNN
            **kwargs: Parameters for SE_ST_CombinedModel
        """
        super().__init__(**kwargs)

        self.use_gnn = use_gnn and TORCH_GEOMETRIC_AVAILABLE

        if self.use_gnn:
            # Initialize GNN
            self.gnn = GeneRegulatoryGNN(
                gene_dim=self.hidden_dim,
                hidden_dim=gnn_hidden_dim,
                num_layers=gnn_layers,
                gnn_type=gnn_type,
                dropout=gnn_dropout,
            )

            # Gene network (will be registered as buffer)
            if gene_network_edge_index is not None:
                self.register_buffer('gene_network_edge_index', gene_network_edge_index)
            else:
                self.gene_network_edge_index = None

            if gene_network_edge_weight is not None:
                self.register_buffer('gene_network_edge_weight', gene_network_edge_weight)
            else:
                self.gene_network_edge_weight = None

            self.gene_to_idx = gene_to_idx

            # Projection layer (if GNN output dim != ST hidden dim)
            if gnn_hidden_dim != self.st_hidden_dim:
                self.gnn_projection = nn.Linear(gnn_hidden_dim, self.st_hidden_dim)
            else:
                self.gnn_projection = nn.Identity()

            logger.info(f"Initialized GNN perturbation model with {gnn_type} GNN")
        else:
            logger.warning("GNN disabled or torch_geometric not available. Using SE-ST only.")

    def apply_gnn_to_cells(
        self,
        cell_states: torch.Tensor,
        perturbed_gene_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Apply GNN to each cell's gene expression.

        Args:
            cell_states: Cell state embeddings [batch_size, hidden_dim] - 2D tensor
            perturbed_gene_names: List of perturbed gene names

        Returns:
            GNN-processed cell states [batch_size, st_hidden_dim] - same shape as input
        """
        if not self.use_gnn or self.gene_network_edge_index is None:
            return cell_states

        # Expect 2D input: [batch_size, hidden_dim]
        if len(cell_states.shape) != 2:
            raise ValueError(f"Expected 2D input [batch_size, hidden_dim], got {cell_states.shape}")

        device = cell_states.device

        # Move gene network to device
        edge_index = self.gene_network_edge_index.to(device)
        edge_weight = self.gene_network_edge_weight.to(device) if self.gene_network_edge_weight is not None else None

        # Process each cell
        processed_cells = []

        for i in range(cell_states.shape[0]):
            cell_state = cell_states[i]  # [hidden_dim]

            # Optionally zero out perturbed genes
            if perturbed_gene_names is not None and self.gene_to_idx is not None:
                for gene_name in perturbed_gene_names:
                    if gene_name in self.gene_to_idx:
                        gene_idx = self.gene_to_idx[gene_name]
                        if gene_idx < cell_state.shape[0]:
                            cell_state[gene_idx] = 0.0

            # Expand to [num_genes, hidden_dim] for GNN
            # For simplicity, we use cell_state as node features
            # In practice, you might want to create per-gene features
            num_genes = len(self.gene_to_idx) if self.gene_to_idx else cell_state.shape[0]

            # Create node features (simplified: repeat cell state)
            # You can improve this by creating gene-specific features
            node_features = cell_state.unsqueeze(0).expand(num_genes, -1)

            # Apply GNN
            gnn_output = self.gnn(node_features, edge_index, edge_weight)

            # Aggregate (mean pooling)
            aggregated = gnn_output.mean(dim=0)

            # Project to ST hidden dim
            projected = self.gnn_projection(aggregated)

            processed_cells.append(projected)

        # Stack to get [batch_size, st_hidden_dim]
        processed_cells = torch.stack(processed_cells)

        return processed_cells

    def forward(self, batch: Dict[str, torch.Tensor], padded: bool = True) -> torch.Tensor:
        """
        Forward pass with GNN integration.

        Args:
            batch: Input batch with keys:
                - ctrl_cell_emb: Control cell expression [B*S, gene_dim]
                - pert_emb: Perturbation embedding [B*S, pert_dim]
                - perturbed_gene_names: (optional) List of perturbed gene names
            padded: Whether batch is padded

        Returns:
            Predictions [B*S, output_dim]
        """
        # 1. SE Encoder: genes → cell state
        cell_states = self.encode_cells_to_state(
            batch["ctrl_cell_emb"]
        )  # [B*S, hidden_dim]

        # 2. GNN: Propagate through gene network
        if self.use_gnn:
            perturbed_genes = batch.get("perturbed_gene_names", None)
            cell_states = self.apply_gnn_to_cells(cell_states, perturbed_genes)

        # 3. ST Model: Transformer + Decoder
        # Create new batch with processed cell states
        st_batch = batch.copy()
        st_batch["ctrl_cell_emb"] = cell_states

        predictions = self.st_model.forward(st_batch, padded=padded)

        return predictions

    def set_gene_network(
        self,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        gene_to_idx: Optional[Dict[str, int]] = None,
    ):
        """
        Set or update gene regulatory network.

        Args:
            edge_index: [2, num_edges]
            edge_weight: [num_edges] (optional)
            gene_to_idx: Gene name to index mapping
        """
        self.register_buffer('gene_network_edge_index', edge_index)

        if edge_weight is not None:
            self.register_buffer('gene_network_edge_weight', edge_weight)

        self.gene_to_idx = gene_to_idx

        logger.info(f"Updated gene network: {edge_index.shape[1]} edges, {len(gene_to_idx) if gene_to_idx else 'unknown'} genes")
