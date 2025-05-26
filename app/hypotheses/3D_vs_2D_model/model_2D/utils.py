import pandas as pd
import pickle 
import json
import os
import sys

import torch
from torch_geometric.data import Data

from rdkit import Chem
from rdkit.Chem import AllChem

from dotenv import load_dotenv
load_dotenv()

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(parent_dir)

from data_preprocessing.features import get_atom_features

DATA_PATH = os.getenv("DATA_PATH")
BOND_TYPE_MAP = {
    0.33: 0,  # single
    0.66: 1,  # double
    1.0: 2,   # triple
    0.5: 3    # aromatic
}

def load_bond_stats(json_path):
    with open(json_path, "r") as f:
        stats = json.load(f)
    min_vals = torch.tensor([stats[str(k)]["min"] for k in BOND_TYPE_MAP], dtype=torch.float32)
    max_vals = torch.tensor([stats[str(k)]["max"] for k in BOND_TYPE_MAP], dtype=torch.float32)
    return min_vals, max_vals

def compute_and_scale_bond_counts(mol, num_nodes, min_vals, max_vals):
    bond_counts = torch.zeros((num_nodes, 4), dtype=torch.float32)
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_type = round(bond.GetBondTypeAsDouble(), 2)

        if bond_type in BOND_TYPE_MAP:
            idx = BOND_TYPE_MAP[bond_type]
            bond_counts[begin, idx] += 1.0
            bond_counts[end, idx] += 1.0

    denom = max_vals - min_vals
    denom[denom == 0] = 1.0
    bond_counts = (bond_counts - min_vals) / denom
    return bond_counts

def mol_to_edge_index(mol):
    edge_index = []
    for bond in mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([start, end])
        edge_index.append([end, start])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def change_graph_to_2D(final_df_path):
    with open(final_df_path, "rb") as f:
        final_df = pickle.load(f)

    processed_df = {}
    all_bond_counts = []

    for key, val in final_df.items():
        x = val.x[:, :-3]  # Drop XYZ
        edge_index = val.edge_index
        edge_attr = val.edge_attr

        num_nodes = x.size(0)
        bond_counts = torch.zeros((num_nodes, 4), dtype=torch.float32)
        bond_type_ids = edge_attr[:, 0]

        for j in range(edge_index.size(1)):
            src = edge_index[0, j].item()
            dst = edge_index[1, j].item()
            bond_type = round(bond_type_ids[j].item(), 2)

            if bond_type in BOND_TYPE_MAP:
                idx = BOND_TYPE_MAP[bond_type]
                bond_counts[src, idx] += 1.0
                bond_counts[dst, idx] += 1.0

        x_new = torch.cat([x, bond_counts], dim=1)
        processed_df[key] = Data(x=x_new, edge_index=edge_index)
        all_bond_counts.append(bond_counts)

    all_bond_counts_tensor = torch.cat(all_bond_counts, dim=0)
    min_vals = all_bond_counts_tensor.min(dim=0, keepdim=True).values
    max_vals = all_bond_counts_tensor.max(dim=0, keepdim=True).values

    bond_stats = {}
    for bond_value, idx in BOND_TYPE_MAP.items():
        bond_stats[str(bond_value)] = {
            "min": float(min_vals[0, idx].item()),
            "max": float(max_vals[0, idx].item())
        }

    with open(os.path.join(DATA_PATH, "2D_test_min_max.json"), "w") as f:
        json.dump(bond_stats, f, indent=4)

    denom = max_vals - min_vals
    denom[denom == 0] = 1.0

    def scale_bond_counts_tensor(x):
        base = x[:, :-4]
        bond = x[:, -4:]
        bond_scaled = (bond - min_vals) / denom
        return torch.cat([base, bond_scaled], dim=1)

    for key, data_obj in processed_df.items():
        data_obj.x = scale_bond_counts_tensor(data_obj.x)

    return processed_df, (min_vals, max_vals)

def preprocess_df(df):
    atom_options = {6: "C", 7: "N", 8: "O", 9: "F"}
    for key, label in atom_options.items():
        df[label] = (df["atomic_num"] == key).astype(int)
    df.drop(columns=["atomic_num"], inplace=True)

    features = {
        "num_bonds": 4,
        "hybridization": 4,
        "aromatic": 1,
        "chirality": 2,
        "valence": 4,
        "in_ring": 1,
    }

    for f, max_val in features.items():
        df[f] = df[f].clip(0, max_val) / max_val

    return df

def create_df_from_mol(smiles, bond_min_vals, bond_max_vals):
    mol = Chem.MolFromSmiles(smiles)
    node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    nodes_df = preprocess_df(pd.DataFrame(node_features))

    bond_counts = compute_and_scale_bond_counts(mol, len(nodes_df), bond_min_vals, bond_max_vals)
    x = torch.tensor(nodes_df.values, dtype=torch.float32)
    x = torch.cat([x, bond_counts], dim=1)

    edge_index = mol_to_edge_index(mol)
    return mol, x, edge_index

def smiles_to_2D_fp(smiles, model, device, scaler_path):
    try:
        bond_min_vals, bond_max_vals = load_bond_stats(os.path.join(DATA_PATH, scaler_path))
        _, x, edge_index = create_df_from_mol(smiles, bond_min_vals, bond_max_vals)
        data = Data(x=x, edge_index=edge_index).to(device)
        output = model(data)
        return output.view(-1)
    except Exception:
        return torch.full((1024,), float('nan')).to(device)
