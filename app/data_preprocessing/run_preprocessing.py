import json
import os

import numpy as np
import pandas as pd
from descriptors import calculate_descriptors_v2
from dotenv import load_dotenv
from features import get_atom_features, get_edge_features
from rdkit import Chem
from utils import add_conformer_to_mol, create_pos_df, preprocess_df

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
input_folder = os.path.join(DATA_PATH, "qm9_data_json")
output_folder = os.path.join(DATA_PATH, "qm9_data")
means_and_stds = os.path.join(DATA_PATH, "means_and_stds.json")

with open(means_and_stds) as f:
    scaler = json.load(f)


def create_df_from_mol(smiles, atoms):
    """
    Constructs molecule, node and edge features from SMILES and atom coordinates.
    """
    mol = Chem.MolFromSmiles(smiles)
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    pos_df = create_pos_df(atoms)
    node_df, norm_pos_df = preprocess_df(pd.DataFrame(atom_features), pos_df, scaler)
    mol = add_conformer_to_mol(mol, norm_pos_df)
    edges = get_edge_features(mol)
    return mol, node_df, edges


def main():
    os.makedirs(output_folder, exist_ok=True)

    for idx, file in enumerate(os.listdir(input_folder)):
        path = os.path.join(input_folder, file)
        with open(path, "r") as f:
            data = json.load(f)

        mol = Chem.MolFromSmiles(data["smiles"])
        if mol and len(mol.GetAtoms()) > 1:
            mol, nodes, edges = create_df_from_mol(data["smiles"], data["atoms"])
            final_path = os.path.join(
                output_folder, data["smiles"].replace("/", "").replace("\\", "")
            )
            os.makedirs(final_path, exist_ok=True)

            nodes.to_parquet(os.path.join(final_path, "nodes.parquet"))
            np.save(os.path.join(final_path, "edges"), edges)
            Chem.MolToMolFile(mol, os.path.join(final_path, "molecule.mol"))

            descriptor = calculate_descriptors_v2(
                data["smiles"], mol, data["homo"], data["lumo"]
            )
            descriptor["file_path"] = os.path.join(final_path, "molecule.mol")
            with open(os.path.join(final_path, "descriptor.json"), "w") as f:
                json.dump(descriptor, f)

        if idx % 1000 == 0:
            print(f"Processed {idx} molecules")


if __name__ == "__main__":
    main()
