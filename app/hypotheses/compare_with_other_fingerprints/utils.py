import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors
from rdkit.Geometry import Point3D
import pandas as pd
import torch

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(parent_dir)

from data_preprocessing.features import get_atom_features, get_edge_features
from data_preprocessing.utils import preprocess_df, add_conformer_to_mol
from create_pos_pairs.graph_utils import build_graph

from dotenv import load_dotenv
load_dotenv()

def is_valid_smiles(smiles):
    """
    Check if a SMILES string is chemically valid and can be embedded in 3D space.

    Parameters:
        smiles (str): SMILES string of the molecule.

    Returns:
        bool: True if the molecule is valid and 3D coordinates can be embedded, False otherwise.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    return AllChem.EmbedMolecule(mol, randomSeed=0) == 0


def smiles_to_ecfp(smiles, radius=2, nbits=1024):
    """
    Convert a SMILES string to an Extended-Connectivity Fingerprint (ECFP).

    Parameters:
        smiles (str): SMILES string of the molecule.
        radius (int): Radius of the fingerprint.
        nbits (int): Size of the fingerprint vector.

    Returns:
        np.ndarray: Numpy array representing the ECFP fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_maccs(smiles):
    """
    Convert a SMILES string to a MACCS fingerprint.

    Parameters:
        smiles (str): SMILES string of the molecule.

    Returns:
        np.ndarray: 167-bit MACCS fingerprint as a numpy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(167)
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_rdkit(smiles, nbits=1024):
    """
    Convert a SMILES string to an RDKit fingerprint.

    Parameters:
        smiles (str): SMILES string of the molecule.
        nbits (int): Size of the fingerprint vector.

    Returns:
        np.ndarray: RDKit fingerprint as a numpy array.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nbits)
    fp = Chem.RDKFingerprint(mol, fpSize=nbits)
    arr = np.zeros((nbits,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_rdf(mol_input):
    """
    Generate the Radial Distribution Function (RDF) descriptor for a molecule.

    Parameters:
        mol_input (str or tuple): Either a SMILES string or a tuple containing:
                                  (SMILES string, list of (symbol, x, y, z) positions)

    Returns:
        np.ndarray: RDF descriptor (256 elements) as a numpy array.
    """
    if isinstance(mol_input, str):
        # SMILES input mode
        smiles = mol_input
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(256)

        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=42)

        # If embedding fails, generate random 3D coordinates
        if success != 0:
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(mol.GetNumAtoms()):
                conf.SetAtomPosition(i, Point3D(
                    random.uniform(-3, 3),
                    random.uniform(-3, 3),
                    random.uniform(-3, 3),
                ))
            print("random")
            mol.RemoveAllConformers()
            mol.AddConformer(conf)
    else:
        # Tuple input: (smiles, manual positions)
        smiles, pos = mol_input
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(256)

        mol = Chem.AddHs(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())

        idx_map = {}
        i = 0
        for j, (symbol, x, y, z) in enumerate(pos):
            if symbol == 'H':
                continue  # Skip hydrogens in manual position assignment
            idx_map[j] = i
            conf.SetAtomPosition(i, Point3D(x, y, z))
            i += 1

        mol.RemoveAllConformers()
        mol.AddConformer(conf)

    # Remove explicit hydrogens before computing descriptors
    mol = Chem.RemoveHs(mol)

    # Calculate RDF descriptor
    rdf_fp = rdMolDescriptors.CalcRDF(mol, confId=0)
    return np.array(rdf_fp)


def smiles_to_random(smiles):
    return np.random.randint(0, 2, size=1024)


def create_pos_df(atoms):
    arr = []
    for i,atom in enumerate(atoms):
        if atom[0] != "H":
            res = {}
            res["x"] = atom[1]
            res["y"] = atom[2]
            res["z"] = atom[3]
            arr.append(res)
    return pd.DataFrame(arr)


def create_df_from_mol(rec, scaler, has_pos):
    if has_pos:
        smiles, atoms = rec[0], rec[1]
        mol = Chem.MolFromSmiles(smiles)
        node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
        atoms = create_pos_df(atoms)
    else:
        smiles = rec
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if success != 0:
            raise ValueError("Cannot create conformer for: " + smiles)
        AllChem.UFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        atoms = []
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "H":
                pos = conf.GetAtomPosition(atom.GetIdx())
                atoms.append({
                    "x": pos.x,
                    "y": pos.y,
                    "z": pos.z
                })
        atoms = pd.DataFrame(atoms)
        mol = Chem.RemoveHs(mol)
        node_features = [get_atom_features(atom) for atom in mol.GetAtoms()]

    nodes, positions = preprocess_df(pd.DataFrame(node_features), atoms, scaler)
    mol = add_conformer_to_mol(mol, positions)
    edges = get_edge_features(mol)
    return mol, nodes, edges


def smiles_to_3D(smiles, model, scaler, has_pos = True):
    try:
        _, nodes, edges = create_df_from_mol(smiles, scaler, has_pos)
        record = build_graph(nodes, edges).to("cuda")
        output = model(record)
        return output.view(-1)  # 1D tensor must be returned
    except Exception:
        return torch.full((1024,), float('nan')).to("cuda")