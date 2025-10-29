#!/bin/bash
# Training script with automatic batch size fallback for CUDA OOM errors

set -e

DATA_DIR="${1:-/data}"
TOML_CONFIG="${2:-/data/starter.toml.working}"
PERT_FEATURES="${3:-/data/ESM2_pert_features.pt}"

echo "=================================================="
echo "GNN Training with Automatic Batch Size Fallback"
echo "=================================================="
echo "Data dir: $DATA_DIR"
echo "TOML config: $TOML_CONFIG"
echo "Pert features: $PERT_FEATURES"
echo ""

# Export PYTHONPATH
export PYTHONPATH=/workspace:$PYTHONPATH

# Try batch sizes in descending order
BATCH_SIZES=(8 4 2 1)

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo ""
    echo "=== Attempting training with batch_size=$BATCH_SIZE ==="
    echo ""

    # Run training
    if python -m gnn.cli.train_gnn_simple \
        --data_dir "$DATA_DIR" \
        --toml_config "$TOML_CONFIG" \
        --pert_features "$PERT_FEATURES" \
        --gnn_hidden_dim 128 \
        --gnn_layers 3 \
        --gnn_type gcn \
        --string_confidence 0.4 \
        --batch_size $BATCH_SIZE \
        --max_epochs 50 \
        --max_steps 80000 \
        --num_workers 4; then

        echo ""
        echo "=================================================="
        echo "✅ Training completed successfully with batch_size=$BATCH_SIZE"
        echo "=================================================="
        exit 0
    else
        EXIT_CODE=$?
        echo ""
        echo "⚠️  Training failed with batch_size=$BATCH_SIZE (exit code: $EXIT_CODE)"

        # Check if it's likely an OOM error
        if [ $BATCH_SIZE -gt 1 ]; then
            echo "Retrying with smaller batch size..."
        else
            echo ""
            echo "=================================================="
            echo "❌ Training failed even with batch_size=1"
            echo "Consider:"
            echo "  - Reducing gnn_hidden_dim (currently 128)"
            echo "  - Reducing gnn_layers (currently 3)"
            echo "  - Using a smaller model"
            echo "=================================================="
            exit $EXIT_CODE
        fi
    fi
done

echo ""
echo "=================================================="
echo "❌ All batch sizes failed"
echo "=================================================="
exit 1
