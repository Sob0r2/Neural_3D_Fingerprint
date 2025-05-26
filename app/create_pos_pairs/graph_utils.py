import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


def read_graph(mol_path: str) -> Data:
    """
    Read graph representation from disk.

    Args:
        mol_path (str): Path to molecule folder.

    Returns:
        Data: PyTorch Geometric Data object.
    """
    node_df = pd.read_parquet(os.path.join(mol_path, "nodes.parquet"))
    edge_matrix = np.load(os.path.join(mol_path, "edges.npy"), allow_pickle=True)

    return build_graph(node_df, edge_matrix)


def build_graph(node_df, edge_matrix) -> Data:
    """
    Create PyG Data object.

    Args:
        node_df: Graph nodes definitions
        edge_matrix: Edge definitions

    Returns:
        Data: PyTorch Geometric Data object.
    """
    x = torch.tensor(node_df.values, dtype=torch.float)
    edge_index, edge_attr = [], []

    for edge in edge_matrix:
        src, tgt, *attr = edge
        edge_index += [[src, tgt], [tgt, src]]
        edge_attr += [attr, attr]

    return Data(
        x=x,
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
