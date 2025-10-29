"""
SE+ST Combined Model - Perturbation Dataset

This module implements the data loading and pairing logic for the SE+ST combined model.
Heavily inspired by STATE's cell-load implementation.
"""

import logging
import glob
import re
from pathlib import Path
from typing import Dict, List, Set, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import tomli


logger = logging.getLogger(__name__)


def safe_decode_array(arr) -> np.ndarray:
    """
    Decode any byte-strings in arr to UTF-8 and cast all entries to Python str.
    
    Args:
        arr: array-like of bytes or other objects
    Returns:
        np.ndarray[str]: decoded strings
    """
    decoded = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            decoded.append(x.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(x))
    return np.array(decoded, dtype=str)


class H5MetadataCache:
    """Cache for H5 file metadata to avoid repeated disk reads."""
    
    def __init__(
        self,
        h5_path: str,
        pert_col: str = "target_gene",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        batch_col: str = "batch_var",
    ):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            obs = f["obs"]
            
            # Read categories and decode to strings
            self.pert_categories = safe_decode_array(obs[pert_col]["categories"][:])
            self.cell_type_categories = safe_decode_array(obs[cell_type_key]["categories"][:])
            
            # Handle batch column (can be categorical or numeric)
            batch_ds = obs[batch_col]
            if "categories" in batch_ds:
                self.batch_categories = safe_decode_array(batch_ds["categories"][:])
                self.batch_codes = batch_ds["codes"][:].astype(np.int32)
            else:
                raw = batch_ds[:]
                self.batch_categories = raw.astype(str)
                self.batch_codes = raw.astype(np.int32)
            
            # Read codes
            self.pert_codes = obs[pert_col]["codes"][:].astype(np.int32)
            self.cell_type_codes = obs[cell_type_key]["codes"][:].astype(np.int32)
            
            # Find control perturbation code
            idx = np.where(self.pert_categories == control_pert)[0]
            if idx.size == 0:
                logger.warning(
                    f"control_pert='{control_pert}' not found in {pert_col} categories. "
                    f"Available: {sorted(set(self.pert_categories))[:10]}"
                )
                # Try case-insensitive match
                control_pert_lower = control_pert.lower()
                for i, cat in enumerate(self.pert_categories):
                    if cat.lower() == control_pert_lower:
                        idx = np.array([i])
                        logger.info(f"Found control using case-insensitive match: '{cat}'")
                        break
                
                if idx.size == 0:
                    raise ValueError(
                        f"control_pert='{control_pert}' not found (even case-insensitive). "
                        f"Available: {sorted(set(self.pert_categories))[:20]}"
                    )
            
            self.control_pert_code = int(idx[0])
            self.control_mask = self.pert_codes == self.control_pert_code
            
            self.n_cells = len(self.pert_codes)


class PerturbationDataset(Dataset):
    """
    Dataset class for loading perturbation data from H5 files.
    Each sample is a (control_cell, perturbed_cell, perturbation_embedding) pair.
    """
    
    def __init__(
        self,
        toml_config_path: str,
        perturbation_features_file: str,
        split: str = "train",
        batch_col: str = "batch_var",
        pert_col: str = "target_gene",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        cell_sentence_len: int = 128,  # Number of cells per sentence (STATE default)
        random_seed: int = 42,
        embed_key: str = None,  # Not used for now, but kept for compatibility
        **kwargs,  # For backward compatibility (n_ctrl_samples, n_pert_samples)
    ):
        """
        Initialize a perturbation dataset.
        
        Args:
            toml_config_path: Path to TOML configuration file
            perturbation_features_file: Path to perturbation features (ESM2 embeddings)
            split: Data split - 'train', 'val', or 'test'
            batch_col: H5 obs column for batches
            pert_col: H5 obs column for perturbations
            cell_type_key: H5 obs column for cell types
            control_pert: Perturbation treated as control
            cell_sentence_len: Number of cells per sentence (STATE default: 128)
            random_seed: Random seed for reproducibility
            embed_key: Key under obsm for embeddings (not used yet)
            **kwargs: Backward compatibility (n_ctrl_samples, n_pert_samples ignored)
        """
        super().__init__()
        self.toml_config_path = Path(toml_config_path)
        self.perturbation_features_file = Path(perturbation_features_file)
        self.split = split
        self.batch_col = batch_col
        self.pert_col = pert_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.cell_sentence_len = cell_sentence_len
        self.embed_key = embed_key
        self.rng = np.random.default_rng(random_seed)
        
        # Load configuration
        self.config = self._load_toml_config()
        
        # Load perturbation embeddings (ESM2 features)
        self.pert_embedding_map = self._load_perturbation_embeddings()

        # Compute mean embedding for fallback (when perturbation not found)
        if len(self.pert_embedding_map) > 0:
            all_embeddings = torch.stack(list(self.pert_embedding_map.values()))
            self.mean_embedding = all_embeddings.mean(dim=0)
            logger.info(f"✅ Computed mean embedding with shape {self.mean_embedding.shape} for fallback")
        else:
            self.mean_embedding = None
            logger.warning("No embeddings loaded - fallback embedding will be None!")

        # Storage for sentence metadata (indices only, not data!)
        # Each item: {h5_path, data_key, pert_indices, ctrl_indices, pert_emb, metadata}
        self.sentence_specs: List[Dict] = []
        
        # Create sentence specifications from H5 files
        self._setup_datasets()
        
        logger.info(f"✅ Dataset created with {len(self.sentence_specs)} sentences for split '{self.split}'")
        if len(self.sentence_specs) == 0:
            logger.error("❌ No sentences created!")
    
    def _load_toml_config(self) -> Dict:
        """Load TOML configuration file."""
        with open(self.toml_config_path, "rb") as f:
            config = tomli.load(f)
        return config
    
    def _load_perturbation_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Load perturbation embeddings from torch file.
        Returns a dictionary mapping perturbation names to embeddings.
        """
        pert_dict = torch.load(self.perturbation_features_file, map_location="cpu")
        logger.info(f"Loaded {len(pert_dict)} perturbation embeddings")
        logger.info(f"Sample keys: {list(pert_dict.keys())[:5]}")
        
        # Validate missing perturbations will be handled later
        return pert_dict
    
    def _expand_file_pattern(self, pattern: str) -> List[str]:
        """
        Expand brace patterns like {a,b,c} into multiple patterns.
        """
        def expand_single_brace(text: str) -> List[str]:
            match = re.search(r"\{([^}]+)\}", text)
            if not match:
                return [text]
            
            before = text[:match.start()]
            after = text[match.end():]
            options = match.group(1).split(",")
            
            results = []
            for option in options:
                new_text = before + option.strip() + after
                results.extend(expand_single_brace(new_text))
            return results
        
        return expand_single_brace(pattern)
    
    def _find_dataset_files(self, dataset_path: str) -> List[Path]:
        """Find all H5 files matching the dataset path pattern."""
        expanded_patterns = self._expand_file_pattern(dataset_path)
        files = []
        
        for pattern in expanded_patterns:
            if any(char in pattern for char in "*?[]{}"):
                # Use glob for patterns
                matching_files = sorted(glob.glob(pattern))
                files.extend([Path(f) for f in matching_files])
            else:
                # Direct path
                p = Path(pattern)
                if p.exists():
                    if p.is_file():
                        files.append(p)
                    else:
                        # Directory - search for H5 files
                        files.extend(sorted(p.glob("*.h5")))
                        files.extend(sorted(p.glob("*.h5ad")))
        
        return files
    
    def _setup_datasets(self):
        """
        Set up datasets according to TOML configuration and split.
        """
        for dataset_name, dataset_path in self.config.get("datasets", {}).items():
            # Check if this dataset should be loaded for current split
            training_config = self.config.get("training", {})
            dataset_split = training_config.get(dataset_name, "train")
            
            # For now, always load the dataset if it's marked as "train" in training config
            # We'll handle zeroshot splits at the cell type level inside _load_and_pair_single_h5
            if dataset_split == "train":
                logger.info(f"Loading dataset: {dataset_name} for split: {self.split}")
                
                # Find H5 files
                files = self._find_dataset_files(dataset_path)
                logger.info(f"Found {len(files)} H5 files for {dataset_name}")
                
                for h5_file in files:
                    logger.info(f"Processing {h5_file.name}...")
                    try:
                        self._load_and_pair_single_h5(str(h5_file), dataset_name)
                    except Exception as e:
                        logger.error(f"Error processing {h5_file}: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                logger.debug(f"Skipping dataset {dataset_name} (configured for split: {dataset_split})")
    
    def _load_and_pair_single_h5(self, h5_path: str, dataset_name: str):
        """
        Load a single H5 file and create control/perturbation pairs.
        Now also checks zeroshot configuration to filter cells by split.
        """
        # Load metadata
        cache = H5MetadataCache(
            h5_path,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            control_pert=self.control_pert,
            batch_col=self.batch_col,
        )
        
        # Determine which data to use and get shape (but DON'T load full matrix yet!)
        with h5py.File(h5_path, "r") as f:
            if self.embed_key and f"obsm/{self.embed_key}" in f:
                data_key = f"obsm/{self.embed_key}"
                n_cells, n_genes = f[data_key].shape
                logger.info(f"Will use {data_key} with shape ({n_cells}, {n_genes})")
            elif "obsm/X_hvg" in f:
                data_key = "obsm/X_hvg"
                n_cells, n_genes = f[data_key].shape
                logger.info(f"Will use {data_key} with shape ({n_cells}, {n_genes})")
            else:
                X_ds = f["X"]
                attrs = dict(X_ds.attrs) if hasattr(X_ds, 'attrs') else {}
                if attrs.get('encoding-type') == 'csr_matrix' or (isinstance(X_ds, h5py.Group) and 'data' in X_ds and 'indices' in X_ds and 'indptr' in X_ds):
                    # Handle CSR format (both with encoding-type and without)
                    logger.info(f"Detected CSR format in {Path(h5_path).name}")
                    data_key = "X"
                    n_cells = X_ds["indptr"].shape[0] - 1
                    n_genes = 18080  # Based on your data structure
                    logger.info(f"Will use CSR X with shape ({n_cells}, {n_genes})")
                else:
                    data_key = "X"
                    n_cells, n_genes = X_ds.shape
                    logger.info(f"Will use X with shape ({n_cells}, {n_genes})")
        
        logger.info(f"Processing {n_cells} cells from {Path(h5_path).name}")
        
        # Organize cells by (perturbation, batch)
        cells_by_pert_batch: Dict[tuple, List[int]] = {}
        for cell_idx in range(n_cells):
            pert_code = cache.pert_codes[cell_idx]
            pert_name = cache.pert_categories[pert_code]
            batch_code = cache.batch_codes[cell_idx]
            batch_name = cache.batch_categories[batch_code]
            cell_type_code = cache.cell_type_codes[cell_idx]
            cell_type = cache.cell_type_categories[cell_type_code]
            
            key = (pert_name, batch_name, cell_type)
            if key not in cells_by_pert_batch:
                cells_by_pert_batch[key] = []
            cells_by_pert_batch[key].append(cell_idx)
        
        # Get unique perturbations (excluding control)
        unique_perts = set()
        unique_cell_types = set()
        for (pert, batch, ct), indices in cells_by_pert_batch.items():
            unique_cell_types.add(ct)
            if pert != self.control_pert:
                unique_perts.add(pert)

        logger.info(f"Found {len(unique_perts)} unique perturbations (excluding control)")
        logger.info(f"🔍 Cell types in {Path(h5_path).name}: {sorted(unique_cell_types)}")
        
        # Get zeroshot configuration
        zeroshot_config = self.config.get("zeroshot", {})
        
        # Build a map of cell type to split (case-insensitive)
        celltype_splits = {}
        celltype_splits_lower = {}  # lowercase version for case-insensitive matching
        for key, split_val in zeroshot_config.items():
            # key format: "dataset_name.celltype"
            if "." in key:
                parts = key.split(".", 1)
                if parts[0] == dataset_name:
                    celltype_splits[parts[1]] = split_val
                    celltype_splits_lower[parts[1].lower()] = split_val
        
        logger.info(f"Zeroshot cell type splits: {celltype_splits}")
        logger.info(f"🎯 Current dataset split: {self.split}")

        # Create pairs for each perturbation
        n_pairs_created = 0
        for pert_name in unique_perts:
            # Get perturbation embedding
            pert_emb = self._get_pert_embedding(pert_name)
            if pert_emb is None:
                continue
            
            # Find all perturbed cells for this perturbation (across all batches/cell types)
            all_pert_cells = []
            for (p, batch, ct), indices in cells_by_pert_batch.items():
                if p == pert_name:
                    # Check if this cell type should be included in current split
                    # Try exact match first, then case-insensitive
                    ct_split = celltype_splits.get(ct)
                    if ct_split is None:
                        ct_split = celltype_splits_lower.get(ct.lower(), "train")  # default to train

                    if ct_split == self.split:
                        all_pert_cells.extend([(idx, batch, ct) for idx in indices])
                    else:
                        # Debug: show why cells are being filtered out
                        if len(indices) > 0:
                            logger.debug(f"  Skipping {len(indices)} cells of type '{ct}' for pert '{pert_name}' (ct_split={ct_split}, current_split={self.split})")

            if len(all_pert_cells) == 0:
                continue
            
            # Note: We can still create sentences even with fewer cells by sampling with replacement
            # This matches STATE's behavior
            
            # Group perturbed cells by (batch, cell_type) to find matching controls
            pert_cells_by_context = {}
            for idx, batch, ct in all_pert_cells:
                key = (batch, ct)
                if key not in pert_cells_by_context:
                    pert_cells_by_context[key] = []
                pert_cells_by_context[key].append(idx)
            
            # Try to create a sentence from cells with the same batch/cell type
            for (batch_name, cell_type), pert_indices in pert_cells_by_context.items():
                # Check split (case-insensitive)
                ct_split = celltype_splits.get(cell_type)
                if ct_split is None:
                    ct_split = celltype_splits_lower.get(cell_type.lower(), "train")
                if ct_split != self.split:
                    continue
                
                # Find corresponding control cells
                ctrl_key = (self.control_pert, batch_name, cell_type)
                ctrl_cells = cells_by_pert_batch.get(ctrl_key, [])
                
                # Need at least 1 cell for both pert and ctrl
                if len(pert_indices) == 0 or len(ctrl_cells) == 0:
                    continue
                
                # ✅ MEMORY FIX: Store only indices, not data!
                # Data will be loaded on-demand in __getitem__
                sentence_spec = {
                    'h5_path': h5_path,
                    'data_key': data_key,
                    'pert_indices': pert_indices,  # All available perturbed cell indices
                    'ctrl_indices': ctrl_cells,    # All available control cell indices
                    'pert_emb': pert_emb,
                    'perturbation': pert_name,
                    'cell_type': cell_type,
                    'batch': batch_name,
                }
                self.sentence_specs.append(sentence_spec)
                n_pairs_created += 1
                
                # Create additional sentence specs (up to 10) if enough cells
                # This gives multiple views of the same perturbation/context
                max_additional = min(10, len(pert_indices) // self.cell_sentence_len, len(ctrl_cells) // self.cell_sentence_len)
                for _ in range(max_additional):
                    # Reuse the same spec (random sampling happens in __getitem__)
                    self.sentence_specs.append(sentence_spec.copy())
                    n_pairs_created += 1
        
        logger.info(f"Created {n_pairs_created} pairs from {Path(h5_path).name}")
    
    def _get_pert_embedding(self, pert_name: str) -> Optional[torch.Tensor]:
        """
        Get perturbation embedding by name.
        Handles case variations and uses mean embedding as fallback for missing perturbations.

        This ensures validation/test sets don't get empty even if some perturbations
        (like TAZ, YAP1) are missing from the ESM2 embedding file.
        """
        # Try exact match first
        if pert_name in self.pert_embedding_map:
            return self.pert_embedding_map[pert_name]

        # Try case variations
        for key in self.pert_embedding_map.keys():
            if key.lower() == pert_name.lower():
                logger.debug(f"Found embedding for '{pert_name}' using case-insensitive match: '{key}'")
                return self.pert_embedding_map[key]

        # Try uppercase (common for gene names)
        if pert_name.upper() in self.pert_embedding_map:
            return self.pert_embedding_map[pert_name.upper()]

        # Try lowercase
        if pert_name.lower() in self.pert_embedding_map:
            return self.pert_embedding_map[pert_name.lower()]

        # Fallback: use mean embedding
        if self.mean_embedding is not None:
            logger.warning(f"⚠️ No embedding found for '{pert_name}', using mean embedding as fallback")
            return self.mean_embedding
        else:
            logger.warning(f"❌ No embedding found for '{pert_name}' and no fallback available, skipping")
            return None
    
    def __len__(self) -> int:
        return len(self.sentence_specs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Dynamically create sentence by reading H5 on-demand."""
        spec = self.sentence_specs[idx]
        
        # Helper function to read cells with H5py's strict indexing requirements
        def read_cells_from_h5(X, indices, n_samples):
            """
            Read cells from H5 dataset. H5py requires:
            1. Indices must be sorted
            2. No duplicate indices
            
            So we read unique indices, then replicate as needed.
            """
            # Sample (with replacement if needed)
            replace = len(indices) < n_samples
            sampled = np.random.choice(indices, size=n_samples, replace=replace)
            
            # Get unique indices and their positions
            unique_indices, inverse_indices = np.unique(sampled, return_inverse=True)
            
            # Read unique cells (H5py is happy with sorted unique indices)
            unique_cells = X[unique_indices.tolist(), :]
            
            # Convert to numpy if needed
            if hasattr(unique_cells, 'toarray'):
                unique_cells = unique_cells.toarray()
            else:
                unique_cells = np.array(unique_cells)
            
            # Replicate cells according to original sampling
            cells = unique_cells[inverse_indices]
            return cells
        
        # Helper function to read cells from dense numpy array
        def read_cells_from_h5_dense(X_dense, indices, n_samples):
            """
            Read cells from dense numpy array.
            """
            # Sample (with replacement if needed)
            replace = len(indices) < n_samples
            sampled = np.random.choice(indices, size=n_samples, replace=replace)
            
            # Read cells directly from dense array
            cells = X_dense[sampled, :]
            return cells
        
        # Helper function to read cells from CSR format (based on STATE's method)
        def read_cells_from_csr(h5_file, indices, n_samples):
            """
            Read cells from CSR format using STATE's method.
            """
            # Sample (with replacement if needed)
            replace = len(indices) < n_samples
            sampled = np.random.choice(indices, size=n_samples, replace=replace)
            
            cells = []
            for idx in sampled:
                # Use STATE's CSR reading logic
                indptr = h5_file["/X/indptr"]
                start_ptr = indptr[idx]
                end_ptr = indptr[idx + 1]
                
                if start_ptr == end_ptr:
                    # Empty row
                    cell_data = np.zeros(18080, dtype=np.float32)
                else:
                    sub_data = h5_file["/X/data"][start_ptr:end_ptr]
                    sub_indices = h5_file["/X/indices"][start_ptr:end_ptr]
                    
                    # Create dense representation
                    cell_data = np.zeros(18080, dtype=np.float32)
                    cell_data[sub_indices] = sub_data
                
                cells.append(cell_data)
            
            return np.array(cells)
        
        # Open H5 file and read only the required cells
        with h5py.File(spec['h5_path'], 'r') as h5_file:
            X = h5_file[spec['data_key']]
            
            # Handle CSR format using STATE's method
            attrs = dict(X.attrs) if hasattr(X, 'attrs') else {}
            if attrs.get('encoding-type') == 'csr_matrix' or (isinstance(X, h5py.Group) and 'data' in X and 'indices' in X and 'indptr' in X):
                # Use STATE's CSR reading method
                pert_cells = read_cells_from_csr(h5_file, spec['pert_indices'], self.cell_sentence_len)
                ctrl_cells = read_cells_from_csr(h5_file, spec['ctrl_indices'], self.cell_sentence_len)
            else:
                # Read perturbed cells
                pert_cells = read_cells_from_h5(X, spec['pert_indices'], self.cell_sentence_len)
                
                # Read control cells
                ctrl_cells = read_cells_from_h5(X, spec['ctrl_indices'], self.cell_sentence_len)
        
        return {
            'ctrl_cell_emb': torch.from_numpy(ctrl_cells).float(),  # [cell_sentence_len, gene_dim]
            'pert_cell_emb': torch.from_numpy(pert_cells).float(),  # [cell_sentence_len, gene_dim]
            'pert_emb': spec['pert_emb'].unsqueeze(0).expand(self.cell_sentence_len, -1),  # [cell_sentence_len, pert_dim]
            'perturbation': spec['perturbation'],
            'cell_type': spec['cell_type'],
            'batch': spec['batch'],
        }


def collate_perturbation_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for perturbation batches with cell sentences.
    
    Each item in batch already contains [cell_sentence_len, dim] tensors.
    We just need to stack them into [batch_size, cell_sentence_len, dim].
    
    Args:
        batch: List of sentence dictionaries from PerturbationDataset
        
    Returns:
        Batched dictionary with stacked tensors
    """
    if len(batch) == 0:
        return {}
    
    # Stack all sentence tensors
    # Each item['ctrl_cell_emb'] has shape [cell_sentence_len, gene_dim]
    # After stacking: [batch_size, cell_sentence_len, gene_dim]
    collated = {
        'ctrl_cell_emb': torch.stack([item['ctrl_cell_emb'] for item in batch]),
        'pert_cell_emb': torch.stack([item['pert_cell_emb'] for item in batch]),
        'pert_emb': torch.stack([item['pert_emb'] for item in batch]),  # Fixed: pert_emb not pert_embedding
        'perturbation': [item['perturbation'] for item in batch],
        'cell_type': [item['cell_type'] for item in batch],
        'batch': [item['batch'] for item in batch],
    }
    
    return collated
