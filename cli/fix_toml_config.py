"""
Fix TOML configuration to point to actual data files.
This script scans for H5 files and updates the TOML config.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import glob
import toml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_toml_config(data_dir="/data", toml_path="/data/starter.toml", output_path=None):
    """
    Fix TOML config to use actual H5 files found in data directory.

    Args:
        data_dir: Data directory to scan
        toml_path: Original TOML file
        output_path: Where to save fixed TOML (defaults to original path + .fixed)
    """

    if output_path is None:
        output_path = toml_path + ".fixed"

    logger.info("="*60)
    logger.info("Fixing TOML Configuration")
    logger.info("="*60)

    # 1. Scan for H5 files
    logger.info(f"\n1. Scanning {data_dir} for H5 files...")
    h5_files = glob.glob(os.path.join(data_dir, "*.h5"))

    if len(h5_files) == 0:
        logger.error(f"❌ No H5 files found in {data_dir}")
        return False

    logger.info(f"✅ Found {len(h5_files)} H5 files:")
    for f in h5_files:
        logger.info(f"   - {os.path.basename(f)}")

    # 2. Load original TOML
    logger.info(f"\n2. Loading original TOML from {toml_path}")
    try:
        config = toml.load(toml_path)
        logger.info(f"✅ Loaded TOML config")
    except Exception as e:
        logger.error(f"❌ Failed to load TOML: {e}")
        return False

    # 3. Update dataset paths
    logger.info(f"\n3. Updating dataset configurations...")

    if 'datasets' not in config:
        config['datasets'] = {}

    # Create dataset entry for each H5 file
    for h5_file in h5_files:
        basename = os.path.basename(h5_file)
        dataset_name = basename.replace('.h5', '')

        # Update or create dataset entry
        config['datasets'][dataset_name] = basename
        logger.info(f"   ✅ {dataset_name} → {basename}")

    # 4. Save fixed TOML
    logger.info(f"\n4. Saving fixed TOML to {output_path}")
    try:
        with open(output_path, 'w') as f:
            toml.dump(config, f)
        logger.info(f"✅ Saved fixed TOML")
    except Exception as e:
        logger.error(f"❌ Failed to save TOML: {e}")
        return False

    # 5. Show the fixed config
    logger.info("\n" + "="*60)
    logger.info("Fixed TOML Content:")
    logger.info("="*60)
    with open(output_path, 'r') as f:
        print(f.read())

    logger.info("\n" + "="*60)
    logger.info("✅ Success! Use this command to train:")
    logger.info("="*60)

    cmd = f"""
export PYTHONPATH=/workspace:$PYTHONPATH

python -m gnn.cli.train_gnn_simple \\
    --data_dir {data_dir} \\
    --toml_config {output_path} \\
    --pert_features {data_dir}/ESM2_pert_features.pt \\
    --gnn_hidden_dim 128 \\
    --gnn_layers 3 \\
    --gnn_type gcn \\
    --string_confidence 0.4 \\
    --batch_size 8 \\
    --max_epochs 50 \\
    --max_steps 80000 \\
    --num_workers 4
"""
    print(cmd)

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data")
    parser.add_argument("--toml_path", type=str, default="/data/starter.toml")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    success = fix_toml_config(args.data_dir, args.toml_path, args.output)

    if not success:
        sys.exit(1)
