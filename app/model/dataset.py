import json
import random

from torch_geometric.data import Dataset


class MoleculeGraphDataset(Dataset):
    """
    Dataset for molecule graph triplets used in contrastive learning.
    Each sample contains an anchor, a positive, and multiple negatives.
    """

    def __init__(self, similar_mol_path, final_df, batch_size=512):
        super().__init__()
        with open(similar_mol_path, "r") as f:
            self.similar_mols = json.load(f)
        self.mol_list = list(self.similar_mols.keys())
        self.final_df = final_df
        self.batch_size = batch_size

    def __len__(self):
        return len(self.mol_list)

    def __getitem__(self, idx):
        anchor = self.mol_list[idx]
        for _ in range(20):
            try:
                positive = random.choice(self.similar_mols[anchor])
                positive_data = self.final_df[positive]
                break
            except:
                continue
        else:
            raise ValueError(f"Could not find valid positive for anchor: {anchor}")

        try:
            anchor_data = self.final_df[anchor]
        except:
            if idx + 1 < len(self.mol_list):
                anchor = self.mol_list[idx + 1]
                anchor_data = self.final_df[anchor]
            else:
                raise IndexError("Anchor fallback failed")

        candidates = [mol for mol in self.mol_list if mol not in [anchor, positive]]
        random.shuffle(candidates)

        negative_data = []
        for neg in candidates:
            if len(negative_data) >= self.batch_size - 2:
                break
            try:
                negative_data.append(self.final_df[neg])
            except:
                continue

        if len(negative_data) < self.batch_size - 2:
            raise ValueError("Not enough negative samples found")

        return anchor_data, positive_data, negative_data
