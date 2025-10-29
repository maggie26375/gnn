"""
Find the correct data configuration by scanning actual files.
This will generate the proper training command.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_data_files(data_dir="/data"):
    """Scan data directory and find all relevant files."""

    logger.info("="*60)
    logger.info("Scanning for data files...")
    logger.info("="*60)

    results = {
        'h5_files': [],
        'h5ad_files': [],
        'toml_files': [],
        'pert_files': [],
    }

    # Find all H5 files
    h5_patterns = [
        os.path.join(data_dir, "*.h5"),
        os.path.join(data_dir, "**/*.h5"),
    ]
    for pattern in h5_patterns:
        results['h5_files'].extend(glob.glob(pattern, recursive=True))

    # Find all H5AD files
    h5ad_patterns = [
        os.path.join(data_dir, "*.h5ad"),
        os.path.join(data_dir, "**/*.h5ad"),
    ]
    for pattern in h5ad_patterns:
        results['h5ad_files'].extend(glob.glob(pattern, recursive=True))

    # Find TOML files
    results['toml_files'] = glob.glob(os.path.join(data_dir, "*.toml"))

    # Find perturbation feature files
    pert_patterns = [
        os.path.join(data_dir, "*pert*.pt"),
        os.path.join(data_dir, "*ESM*.pt"),
    ]
    for pattern in pert_patterns:
        results['pert_files'].extend(glob.glob(pattern))

    # Deduplicate
    for key in results:
        results[key] = sorted(list(set(results[key])))

    return results


def suggest_training_command(data_dir="/data"):
    """Suggest the correct training command based on found files."""

    files = find_data_files(data_dir)

    logger.info(f"\n📁 Found files:")
    logger.info(f"   H5 files: {len(files['h5_files'])}")
    for f in files['h5_files']:
        logger.info(f"      - {f}")

    logger.info(f"   H5AD files: {len(files['h5ad_files'])}")
    for f in files['h5ad_files']:
        logger.info(f"      - {f}")

    logger.info(f"   TOML files: {len(files['toml_files'])}")
    for f in files['toml_files']:
        logger.info(f"      - {f}")

    logger.info(f"   Perturbation features: {len(files['pert_files'])}")
    for f in files['pert_files']:
        logger.info(f"      - {f}")

    # Determine best h5ad file for gene names
    h5ad_file = None
    if len(files['h5ad_files']) > 0:
        # Prefer competition_val_template or similar
        for f in files['h5ad_files']:
            if 'competition' in os.path.basename(f).lower():
                h5ad_file = f
                break
        if not h5ad_file:
            h5ad_file = files['h5ad_files'][0]

    # Determine toml file
    toml_file = files['toml_files'][0] if len(files['toml_files']) > 0 else None

    # Determine pert file
    pert_file = None
    if len(files['pert_files']) > 0:
        # Prefer ESM2_pert_features
        for f in files['pert_files']:
            if 'ESM2' in os.path.basename(f):
                pert_file = f
                break
        if not pert_file:
            pert_file = files['pert_files'][0]

    logger.info("\n" + "="*60)
    logger.info("📋 Recommended Configuration:")
    logger.info("="*60)
    logger.info(f"Gene names from: {h5ad_file}")
    logger.info(f"TOML config: {toml_file}")
    logger.info(f"Pert features: {pert_file}")

    if not toml_file or not pert_file or not h5ad_file:
        logger.error("\n❌ Missing required files!")
        if not h5ad_file:
            logger.error("   - No H5AD file found (needed for gene names)")
        if not toml_file:
            logger.error("   - No TOML config found")
        if not pert_file:
            logger.error("   - No perturbation features found")
        return

    logger.info("\n" + "="*60)
    logger.info("✅ Suggested Training Command:")
    logger.info("="*60)

    cmd = f"""
export PYTHONPATH=/workspace:$PYTHONPATH

python -m gnn.cli.train_gnn_simple \\
    --data_dir {data_dir} \\
    --toml_config {toml_file} \\
    --pert_features {pert_file} \\
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

    logger.info("\n" + "="*60)
    logger.info("💡 Notes:")
    logger.info("="*60)
    logger.info("1. Gene names will be loaded from the h5ad file")
    logger.info("2. STRING network will be downloaded automatically")
    logger.info("3. Training data comes from TOML config datasets")
    logger.info("4. If you see 'Found 0 H5 files', check TOML dataset paths")
    logger.info("\n✅ Copy and run the command above!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data")
    args = parser.parse_args()

    suggest_training_command(args.data_dir)
