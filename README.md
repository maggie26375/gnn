# GNN-based Perturbation Prediction

This repository implements a Graph Neural Network (GNN) enhanced version of the SE+ST model for cellular perturbation prediction, integrating gene regulatory networks from the STRING database.

## Features

- **STRING Network Integration**: Automatically downloads and processes gene regulatory networks from STRING database
- **Multiple GNN Architectures**: Supports GCN, GAT, and GraphSAGE
- **SE+ST Compatibility**: Built on top of the proven SE+ST architecture
- **Flexible Configuration**: Easy to switch between pure SE-ST and GNN-enhanced modes

## Architecture

```
Input (Control Cells)
    ↓
SE Encoder (genes → cell state)
    ↓
GNN (propagate through gene regulatory network)
    ↓
Transformer (cell-cell interactions)
    ↓
Decoder (state → perturbed genes)
    ↓
Output (Perturbed Cells)
```

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch lightning hydra-core

# PyTorch Geometric (for GNN)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Data processing
pip install anndata scanpy pandas numpy requests
```

### Quick Start

```bash
# Clone repository
git clone https://github.com/maggie26375/gnn.git
cd gnn

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training with GNN

```bash
# Train GNN model with STRING network
python -m gnn.cli.train_gnn \
    model.use_gnn=true \
    model.gnn_type=gcn \
    model.gnn_layers=2 \
    model.string_confidence=0.4 \
    training.max_epochs=100
```

### Training without GNN (Pure SE-ST)

```bash
# Disable GNN to use pure SE-ST
python -m gnn.cli.train_gnn \
    model.use_gnn=false \
    training.max_epochs=100
```

### Configuration Options

Key parameters in `configs/gnn_config.yaml`:

```yaml
model:
  # GNN settings
  use_gnn: true              # Enable/disable GNN
  gnn_type: "gcn"           # GNN architecture: "gcn", "gat", "sage"
  gnn_layers: 2             # Number of GNN layers
  gnn_hidden_dim: 512       # Hidden dimension
  string_confidence: 0.4    # STRING confidence threshold (0-1)

  # SE-ST settings
  se_model_path: "SE-600M"
  freeze_se_model: true
  st_hidden_dim: 512
```

## STRING Database

The model automatically downloads gene regulatory networks from STRING:
- **Species**: Human (Homo sapiens, taxonomy ID 9606)
- **Network Type**: Physical interactions (direct protein-protein interactions)
- **Confidence**: Configurable threshold (default 0.4)
- **Cache**: Downloaded files are cached in `data/string_cache/`

### Manual STRING Network Loading

```python
from gnn.utils.string_network_loader import load_string_network_for_hvgs

# Load STRING network for your genes
edge_index, gene_to_idx = load_string_network_for_hvgs(
    hvg_gene_names=your_gene_list,
    cache_dir="./data/string_cache",
    confidence_threshold=0.4
)
```

## Model Comparison

| Model | Description | Use Case |
|-------|-------------|----------|
| **SE-ST** | Transformer-based, no explicit network | General purpose, fast training |
| **GNN-enhanced** | SE-ST + gene regulatory network | Biologically-informed, interpretable |

## Performance Tips

1. **GNN Type Selection**:
   - GCN: Fastest, good for large networks
   - GAT: Attention mechanism, slower but more expressive
   - GraphSAGE: Good for inductive learning

2. **STRING Confidence**:
   - Lower (0.15-0.4): More edges, more noise
   - Higher (0.5-0.9): Fewer edges, higher quality

3. **Number of Layers**:
   - 1-2 layers: Local neighborhood
   - 3-4 layers: Broader network propagation

## Directory Structure

```
gnn/
├── models/
│   ├── gnn_perturbation.py      # GNN model
│   ├── se_st_combined.py        # SE+ST base model
│   └── ...
├── utils/
│   ├── string_network_loader.py # STRING database loader
│   └── ...
├── cli/
│   ├── train_gnn.py             # Training script
│   └── ...
├── configs/
│   └── gnn_config.yaml          # Configuration
└── README.md
```

## Troubleshooting

### torch_geometric Installation Issues

If you encounter issues installing PyTorch Geometric:

```bash
# Check your PyTorch version
python -c "import torch; print(torch.__version__)"

# Install PyG for your specific PyTorch version
# Replace cu118 with your CUDA version (cu102, cu113, cu118, etc.)
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### STRING Network Download Fails

If STRING download fails:
1. Check internet connection
2. Try manual download from: https://string-db.org/cgi/download
3. Place file in `data/string_cache/` directory

### Out of Memory (OOM)

If training runs out of memory:
- Reduce batch_size: `training.batch_size=8`
- Reduce GNN layers: `model.gnn_layers=1`
- Use smaller GNN: `model.gnn_hidden_dim=256`

## Citation

If you use this code, please cite:

```bibtex
@software{gnn_perturbation,
  title={GNN-based Perturbation Prediction},
  author={Your Name},
  year={2025},
  url={https://github.com/maggie26375/gnn}
}
```

## License

MIT License

## Acknowledgments

- STRING database: https://string-db.org/
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- SE+ST model: Original implementation
