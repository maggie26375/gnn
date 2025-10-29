"""
Diagnostic script to check data configuration and availability.
Run this first to verify your data setup before training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import os
import toml
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_data_setup(data_dir: str = "/data", toml_path: str = "/data/starter.toml"):
    """Check if data is properly configured."""

    logger.info("="*60)
    logger.info("Data Configuration Diagnosis")
    logger.info("="*60)

    # 1. Check data directory
    logger.info(f"\n1. Checking data directory: {data_dir}")
    if os.path.exists(data_dir):
        logger.info(f"   ✅ Directory exists")
        files = os.listdir(data_dir)
        logger.info(f"   Found {len(files)} files/directories:")
        for f in sorted(files)[:20]:  # Show first 20
            logger.info(f"      - {f}")
        if len(files) > 20:
            logger.info(f"      ... and {len(files)-20} more")
    else:
        logger.error(f"   ❌ Directory does not exist!")
        return False

    # 2. Check TOML config
    logger.info(f"\n2. Checking TOML config: {toml_path}")
    if os.path.exists(toml_path):
        logger.info(f"   ✅ TOML file exists")
        try:
            config = toml.load(toml_path)
            logger.info(f"   Config sections: {list(config.keys())}")

            # Show dataset info
            if 'datasets' in config:
                logger.info(f"   Datasets defined: {list(config['datasets'].keys())}")
                for dataset_name, dataset_config in config['datasets'].items():
                    logger.info(f"\n   Dataset: {dataset_name}")

                    # Handle both dict and string configs
                    if isinstance(dataset_config, dict):
                        logger.info(f"      path: {dataset_config.get('path', 'NOT SET')}")
                        logger.info(f"      train_split: {dataset_config.get('train_split', 'NOT SET')}")
                        logger.info(f"      val_split: {dataset_config.get('val_split', 'NOT SET')}")
                        logger.info(f"      test_split: {dataset_config.get('test_split', 'NOT SET')}")
                        data_path = dataset_config.get('path', '')
                    elif isinstance(dataset_config, str):
                        logger.info(f"      path (string): {dataset_config}")
                        data_path = dataset_config
                    else:
                        logger.warning(f"      ⚠️ Unknown config type: {type(dataset_config)}")
                        continue

                    # Check if files exist
                    if data_path:
                        full_path = os.path.join(data_dir, data_path)
                        if os.path.exists(full_path):
                            logger.info(f"      ✅ Data path exists: {full_path}")
                            # List H5 files
                            h5_files = glob.glob(os.path.join(full_path, "**/*.h5"), recursive=True)
                            h5ad_files = glob.glob(os.path.join(full_path, "**/*.h5ad"), recursive=True)
                            logger.info(f"      Found {len(h5_files)} .h5 files")
                            logger.info(f"      Found {len(h5ad_files)} .h5ad files")

                            if len(h5_files) > 0:
                                logger.info(f"      Example .h5 files:")
                                for f in h5_files[:5]:
                                    logger.info(f"         - {f}")
                            if len(h5ad_files) > 0:
                                logger.info(f"      Example .h5ad files:")
                                for f in h5ad_files[:5]:
                                    logger.info(f"         - {f}")
                        else:
                            logger.error(f"      ❌ Data path does not exist: {full_path}")
            else:
                logger.warning(f"   ⚠️ No 'datasets' section in config")

        except Exception as e:
            logger.error(f"   ❌ Failed to parse TOML: {e}")
            return False
    else:
        logger.error(f"   ❌ TOML file does not exist!")
        return False

    # 3. Check perturbation features
    logger.info(f"\n3. Checking perturbation features")
    pert_files = glob.glob(os.path.join(data_dir, "*pert*.pt"))
    pert_files += glob.glob(os.path.join(data_dir, "*ESM*.pt"))
    if len(pert_files) > 0:
        logger.info(f"   ✅ Found {len(pert_files)} perturbation feature files:")
        for f in pert_files:
            size_mb = os.path.getsize(f) / (1024*1024)
            logger.info(f"      - {os.path.basename(f)} ({size_mb:.1f} MB)")
    else:
        logger.warning(f"   ⚠️ No perturbation feature files found")

    # 4. Check for h5ad files (for gene names)
    logger.info(f"\n4. Checking for h5ad files (needed for gene names)")
    h5ad_files = glob.glob(os.path.join(data_dir, "*.h5ad"))
    if len(h5ad_files) > 0:
        logger.info(f"   ✅ Found {len(h5ad_files)} h5ad files:")
        for f in h5ad_files:
            size_mb = os.path.getsize(f) / (1024*1024)
            logger.info(f"      - {os.path.basename(f)} ({size_mb:.1f} MB)")

            # Try to load and show gene count
            try:
                import anndata as ad
                adata = ad.read_h5ad(f)
                logger.info(f"         Genes: {adata.n_vars}, Cells: {adata.n_obs}")
            except Exception as e:
                logger.warning(f"         Could not read: {e}")
    else:
        logger.error(f"   ❌ No h5ad files found in {data_dir}")
        logger.error(f"   GNN requires gene names from h5ad files!")
        return False

    logger.info("\n" + "="*60)
    logger.info("Diagnosis complete!")
    logger.info("="*60)
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/data")
    parser.add_argument("--toml_config", type=str, default="/data/starter.toml")
    args = parser.parse_args()

    success = diagnose_data_setup(args.data_dir, args.toml_config)

    if success:
        logger.info("\n✅ Data setup looks good!")
    else:
        logger.error("\n❌ Data setup has issues. Please fix before training.")
        sys.exit(1)
