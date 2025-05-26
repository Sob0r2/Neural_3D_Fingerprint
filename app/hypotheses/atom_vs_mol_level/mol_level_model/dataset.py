import json
import random
import os
import pandas as pd
from torch_geometric.data import Dataset
import torch

class MolLevelDataset(Dataset):
    """
    Dataset for molecule descriptors triplets used in contrastive learning.
    Each sample contains an anchor, a positive, and multiple negatives.
    """

    def __init__(self, similar_mol_path, scaler, batch_size=512):
        super().__init__()
        with open(similar_mol_path, "r") as f:
            self.similar_mols = json.load(f)
        self.mol_list = list(self.similar_mols.keys())
        self.batch_size = batch_size
        self.scaler = scaler

    def _get_descriptor(self, path):
        folder = os.path.dirname(path)
        desc_path = os.path.join(folder, "descriptor.json")
        with open(desc_path) as f:
            desc = json.load(f)
        desc.pop("SMILES", None)
        desc.pop("file_path", None)
        desc["GETAWAY_1"] = 0
        for col in desc.keys():
            desc[col] = (desc[col] - self.scaler[col]["mean"]) / self.scaler[col]["std"]
        return torch.tensor(list(desc.values()), dtype=torch.float32)

    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, idx):
        anchor = self.mol_list[idx]
        for _ in range(20):
            try:
                positive = random.choice(self.similar_mols[anchor])
                positive_data = self._get_descriptor(positive)
                break
            except:
                continue
        else:
            raise ValueError(f"Could not find valid positive for anchor: {anchor}")

        try:
            anchor_data = self._get_descriptor(anchor)
        except:
            if idx + 1 < len(self.mol_list):
                anchor = self.mol_list[idx + 1]
                anchor_data = self._get_descriptor(anchor)
            else:
                raise IndexError("Anchor fallback failed")

        candidates = [mol for mol in self.mol_list if mol not in [anchor, positive]]
        random.shuffle(candidates)

        negative_data = []
        for neg in candidates:
            if len(negative_data) >= self.batch_size - 2:
                break
            try:
                negative_data.append(self._get_descriptor(neg))
            except:
                continue

        if len(negative_data) < self.batch_size - 2:
            raise ValueError("Not enough negative samples found")

        return anchor_data, positive_data, torch.stack(negative_data)
