#!/usr/bin/env python3
"""
Fix TOML configuration to properly split data into train/val/test.

This script reads the current TOML config and creates a new one with proper splits.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import toml
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_toml_splits(
    input_toml: str = "/data/starter.toml.working",
    output_toml: str = None,
    strategy: str = "simple"
):
    """
    Fix TOML splits for train/val/test.

    Args:
        input_toml: Path to input TOML file
        output_toml: Path to output TOML file (default: input_toml)
        strategy: Split strategy
            - "simple": Use specific datasets for each split
            - "auto": Automatically split based on size
    """
    if output_toml is None:
        output_toml = input_toml

    logger.info("="*60)
    logger.info("Fixing TOML Train/Val/Test Splits")
    logger.info("="*60)
    logger.info(f"Input: {input_toml}")
    logger.info(f"Output: {output_toml}")
    logger.info(f"Strategy: {strategy}")

    # Load existing TOML
    logger.info("\n1. Loading existing TOML...")
    try:
        config = toml.load(input_toml)
        logger.info(f"✅ Loaded TOML config")
    except Exception as e:
        logger.error(f"❌ Failed to load TOML: {e}")
        return False

    # Check current configuration
    logger.info("\n2. Current configuration:")
    if 'training' in config:
        logger.info(f"   [training] section: {len(config['training'])} datasets")
        for name, split in config['training'].items():
            logger.info(f"     - {name}: {split}")

    if 'zeroshot' in config:
        logger.info(f"   [zeroshot] section: {len(config['zeroshot'])} datasets")
        for name, split in config['zeroshot'].items():
            logger.info(f"     - {name}: {split}")

    # Create new split configuration
    logger.info("\n3. Creating new split configuration...")

    if strategy == "simple":
        # Simple strategy: Split datasets manually
        # train: competition_train, k562, hepg2 (3 datasets)
        # val: jurkat, k562_gwps (2 datasets)
        # test: rpe1 (1 dataset for zero-shot)

        new_config = config.copy()
        new_config['training'] = {
            'competition_train': 'train',
            'k562': 'train',
            'hepg2': 'train',
            'jurkat': 'val',
            'k562_gwps': 'val',
            'rpe1': 'test',
        }

        # Remove zeroshot section (already included in training)
        if 'zeroshot' in new_config:
            del new_config['zeroshot']

        logger.info("   New split configuration:")
        logger.info("   Train (3 datasets): competition_train, k562, hepg2")
        logger.info("   Val (2 datasets): jurkat, k562_gwps")
        logger.info("   Test (1 dataset): rpe1")

    else:
        logger.error(f"❌ Unknown strategy: {strategy}")
        return False

    # Save new TOML
    logger.info(f"\n4. Saving new TOML to {output_toml}...")
    try:
        with open(output_toml, 'w') as f:
            toml.dump(new_config, f)
        logger.info(f"✅ Saved new TOML config")
    except Exception as e:
        logger.error(f"❌ Failed to save TOML: {e}")
        return False

    # Print the new config
    logger.info("\n" + "="*60)
    logger.info("New TOML Configuration:")
    logger.info("="*60)
    with open(output_toml, 'r') as f:
        print(f.read())

    logger.info("\n" + "="*60)
    logger.info("✅ Success! You can now train with:")
    logger.info("="*60)
    print(f"""
python -m gnn.cli.train_gnn_simple \\
    --data_dir /data \\
    --toml_config {output_toml} \\
    --pert_features /data/ESM2_pert_features.pt \\
    --gnn_hidden_dim 128 \\
    --gnn_layers 3 \\
    --gnn_type gcn \\
    --string_confidence 0.4 \\
    --batch_size 8 \\
    --max_epochs 50 \\
    --max_steps 80000 \\
    --num_workers 4
""")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/data/starter.toml.working",
        help="Input TOML file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output TOML file (default: overwrite input)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="simple",
        choices=["simple", "auto"],
        help="Split strategy"
    )

    args = parser.parse_args()

    success = fix_toml_splits(
        input_toml=args.input,
        output_toml=args.output,
        strategy=args.strategy
    )

    if not success:
        sys.exit(1)
