"""
STRING Database Network Loader

This module provides utilities to download and process gene regulatory networks
from the STRING database for use in GNN-based perturbation prediction models.
"""

import os
import gzip
import requests
import pandas as pd
import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class STRINGNetworkLoader:
    """
    Load and process gene regulatory networks from STRING database.

    STRING (Search Tool for the Retrieval of Interacting Genes/Proteins) is a
    comprehensive database of known and predicted protein-protein interactions.
    """

    def __init__(
        self,
        species: str = "9606",  # 9606 = Homo sapiens
        cache_dir: str = "./data/string_cache",
        confidence_threshold: float = 0.4,
    ):
        """
        Initialize STRING network loader.

        Args:
            species: NCBI taxonomy identifier (9606 for human, 10090 for mouse)
            cache_dir: Directory to cache downloaded STRING files
            confidence_threshold: Minimum confidence score (0-1) for edges
        """
        self.species = species
        self.cache_dir = cache_dir
        self.confidence_threshold = confidence_threshold

        os.makedirs(cache_dir, exist_ok=True)

        # STRING database URLs
        self.base_url = "https://stringdb-static.org/download"
        self.version = "v12.0"

    def download_string_network(
        self,
        network_type: str = "physical",  # "physical" or "full"
    ) -> str:
        """
        Download STRING network file.

        Args:
            network_type: "physical" (direct interactions) or "full" (all)

        Returns:
            Path to downloaded file
        """
        if network_type == "physical":
            filename = f"{self.species}.protein.physical.links.{self.version}.txt.gz"
            url_subdir = "protein.physical.links"
        else:
            filename = f"{self.species}.protein.links.{self.version}.txt.gz"
            url_subdir = "protein.links"

        filepath = os.path.join(self.cache_dir, filename)

        # Check if already downloaded
        if os.path.exists(filepath):
            logger.info(f"Using cached STRING network: {filepath}")
            return filepath

        # Download
        url = f"{self.base_url}/{url_subdir}.{self.version}/{filename}"
        logger.info(f"Downloading STRING network from {url}")

        # Disable SSL verification to avoid certificate issues
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded to {filepath}")
        return filepath

    def load_network(
        self,
        network_type: str = "physical",
        gene_names: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, int], pd.DataFrame]:
        """
        Load STRING network and convert to PyTorch format.

        Args:
            network_type: "physical" or "full"
            gene_names: List of gene names to filter (e.g., HVGs from your dataset)

        Returns:
            edge_index: [2, num_edges] tensor
            gene_to_idx: Dict mapping gene names to indices
            edge_data: DataFrame with edge information
        """
        # Download if needed
        filepath = self.download_string_network(network_type)

        # Load data
        logger.info("Loading STRING network...")
        with gzip.open(filepath, 'rt') as f:
            df = pd.read_csv(f, sep=' ')

        logger.info(f"Loaded {len(df)} edges from STRING")

        # Filter by confidence
        df = df[df['combined_score'] >= self.confidence_threshold * 1000]
        logger.info(f"After confidence filtering (>={self.confidence_threshold}): {len(df)} edges")

        # Convert STRING IDs to gene names
        df['protein1'] = df['protein1'].str.replace(f'{self.species}.', '', regex=False)
        df['protein2'] = df['protein2'].str.replace(f'{self.species}.', '', regex=False)

        # Map to gene symbols if provided
        if gene_names is not None:
            # Create mapping (this is simplified - in practice you'd use biomart or similar)
            gene_set = set(gene_names)

            # Filter edges to only include genes in our dataset
            df = df[
                df['protein1'].isin(gene_set) &
                df['protein2'].isin(gene_set)
            ]

            logger.info(f"After filtering to {len(gene_names)} genes: {len(df)} edges")

        # Create gene to index mapping
        all_genes = list(set(df['protein1'].tolist() + df['protein2'].tolist()))
        gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}

        # Convert to edge_index format
        edge_list = []
        for _, row in df.iterrows():
            source_idx = gene_to_idx[row['protein1']]
            target_idx = gene_to_idx[row['protein2']]

            # Add both directions (undirected graph)
            edge_list.append([source_idx, target_idx])
            edge_list.append([target_idx, source_idx])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        logger.info(f"Created graph with {len(gene_to_idx)} nodes and {edge_index.shape[1]} edges")

        return edge_index, gene_to_idx, df

    def create_gene_network_from_data(
        self,
        perturbation_data: pd.DataFrame,
        fold_change_threshold: float = 0.5,
        min_observations: int = 3,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Infer gene regulatory network from perturbation data.

        This complements STRING network with data-driven edges.

        Args:
            perturbation_data: DataFrame with columns ['target_gene', 'affected_gene', 'fold_change']
            fold_change_threshold: Minimum |fold_change| to consider
            min_observations: Minimum number of experiments showing the effect

        Returns:
            edge_index: [2, num_edges] tensor
            gene_to_idx: Dict mapping gene names to indices
        """
        # Group by (target, affected) pairs
        grouped = perturbation_data.groupby(['target_gene', 'affected_gene'])

        edges = []
        for (target, affected), group in grouped:
            # Check if effect is consistent
            fold_changes = group['fold_change'].values

            if len(fold_changes) >= min_observations:
                mean_fc = np.mean(fold_changes)
                std_fc = np.std(fold_changes)

                # Strong and consistent effect
                if abs(mean_fc) > fold_change_threshold and std_fc < 0.3:
                    edges.append((target, affected, mean_fc))

        logger.info(f"Inferred {len(edges)} edges from perturbation data")

        # Create mapping
        all_genes = list(set([e[0] for e in edges] + [e[1] for e in edges]))
        gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}

        # Convert to edge_index
        edge_list = [[gene_to_idx[e[0]], gene_to_idx[e[1]]] for e in edges]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        return edge_index, gene_to_idx

    def merge_networks(
        self,
        string_edge_index: torch.Tensor,
        string_gene_to_idx: Dict[str, int],
        data_edge_index: torch.Tensor,
        data_gene_to_idx: Dict[str, int],
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Merge STRING network with data-inferred network.

        Args:
            string_edge_index, string_gene_to_idx: STRING network
            data_edge_index, data_gene_to_idx: Data-inferred network

        Returns:
            merged_edge_index: Combined network
            merged_gene_to_idx: Combined gene mapping
        """
        # Create unified gene mapping
        all_genes = list(set(
            list(string_gene_to_idx.keys()) +
            list(data_gene_to_idx.keys())
        ))
        unified_gene_to_idx = {gene: idx for idx, gene in enumerate(all_genes)}

        # Remap STRING edges
        string_edges_remapped = []
        for i in range(string_edge_index.shape[1]):
            source_gene = [g for g, idx in string_gene_to_idx.items() if idx == string_edge_index[0, i].item()][0]
            target_gene = [g for g, idx in string_gene_to_idx.items() if idx == string_edge_index[1, i].item()][0]

            string_edges_remapped.append([
                unified_gene_to_idx[source_gene],
                unified_gene_to_idx[target_gene]
            ])

        # Remap data edges
        data_edges_remapped = []
        for i in range(data_edge_index.shape[1]):
            source_gene = [g for g, idx in data_gene_to_idx.items() if idx == data_edge_index[0, i].item()][0]
            target_gene = [g for g, idx in data_gene_to_idx.items() if idx == data_edge_index[1, i].item()][0]

            data_edges_remapped.append([
                unified_gene_to_idx[source_gene],
                unified_gene_to_idx[target_gene]
            ])

        # Combine and deduplicate
        all_edges = string_edges_remapped + data_edges_remapped
        unique_edges = list(set([tuple(e) for e in all_edges]))

        merged_edge_index = torch.tensor(unique_edges, dtype=torch.long).t()

        logger.info(f"Merged network: {len(unified_gene_to_idx)} nodes, {merged_edge_index.shape[1]} edges")

        return merged_edge_index, unified_gene_to_idx


def load_string_network_for_hvgs(
    hvg_gene_names: List[str],
    cache_dir: str = "./data/string_cache",
    confidence_threshold: float = 0.4,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Convenience function to load STRING network for a set of HVGs.

    Args:
        hvg_gene_names: List of highly variable gene names
        cache_dir: Cache directory
        confidence_threshold: Minimum confidence score

    Returns:
        edge_index: [2, num_edges] tensor
        gene_to_idx: Dict mapping gene names to indices
    """
    loader = STRINGNetworkLoader(
        cache_dir=cache_dir,
        confidence_threshold=confidence_threshold
    )

    edge_index, gene_to_idx, _ = loader.load_network(
        network_type="physical",
        gene_names=hvg_gene_names
    )

    return edge_index, gene_to_idx
