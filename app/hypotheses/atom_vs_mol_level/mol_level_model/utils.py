import os
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def load_descriptor(descriptor_path):
    try:
        with open(descriptor_path) as f:
            descriptor = json.load(f)
        descriptor.pop("SMILES", None)
        descriptor.pop("file_path", None)
        return descriptor
    except Exception as e:
        print(f"Failed to load {descriptor_path}: {e}")
        return None

def create_descriptor_df(folder):
    paths = [
        os.path.join(folder, mol, "descriptor.json")
        for mol in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, mol))
    ]

    with ThreadPoolExecutor() as executor:
        descriptors = list(executor.map(load_descriptor, paths))

    descriptors = [d for d in descriptors if d is not None]

    return pd.DataFrame(descriptors)

def calculate_mean_and_std(folder):
    df = create_descriptor_df(folder)
    df = df.select_dtypes(include=["float64", "int64"])
    scaler = {
        col: {
            "mean": df[col].mean(),
            "std": df[col].std() if df[col].std() != 0 else 1.0 
        }
        for col in df.columns
    }

    return scaler

def approximate_homo_lumo(mol):
    mw = Descriptors.MolWt(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    logp = Descriptors.MolLogP(mol)
    num_pi_bonds = len([b for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 2.0])
    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_atoms = mol.GetNumAtoms()

    homo = -5.5 + 0.005 * mw - 0.02 * num_pi_bonds + 0.01 * logp - 0.003 * tpsa
    lumo = -1.0 + 0.003 * mw + 0.01 * num_rings - 0.02 * num_atoms + 0.002 * tpsa

    return homo, lumo
