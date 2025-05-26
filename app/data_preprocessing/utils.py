import numpy as np
import pandas as pd
from rdkit import Chem


def normalize_df(df, scaler):
    """
    Normalizes x, y, z coordinates using precomputed means and stds.
    """
    df["x"] = (df["x"] - scaler["x_mean"]) / scaler["x_std"]
    df["y"] = (df["y"] - scaler["y_mean"]) / scaler["y_std"]
    df["z"] = (df["z"] - scaler["z_mean"]) / scaler["z_std"]
    return df


def create_pos_df(atoms):
    """
    Creates a DataFrame of 3D coordinates, excluding hydrogens.
    """
    return pd.DataFrame(
        [{"x": x, "y": y, "z": z} for symbol, x, y, z in atoms if symbol != "H"]
    )


def add_conformer_to_mol(mol, df):
    """
    Adds 3D coordinates as a conformer to an RDKit molecule.
    """
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, row in df.iterrows():
        conf.SetAtomPosition(i, (row["x"], row["y"], row["z"]))
    mol.AddConformer(conf, assignId=True)
    return mol


def preprocess_df(df, positions, scaler):
    """
    Adds atom type one-hot features and normalized coordinates.
    """
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

    positions = normalize_df(positions, scaler).reset_index()
    df = df.join(pd.DataFrame(positions, columns=["x", "y", "z"]))
    return df, pd.DataFrame(positions, columns=["x", "y", "z"])
