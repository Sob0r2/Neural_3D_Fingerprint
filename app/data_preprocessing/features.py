import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem


def get_atom_features(atom):
    """Extract atomic features from an RDKit atom object."""
    return {
        "atomic_num": atom.GetAtomicNum(),
        "num_bonds": len(atom.GetBonds()),
        "hybridization": int(atom.GetHybridization()),
        "aromatic": int(atom.GetIsAromatic()),
        "chirality": int(atom.GetChiralTag()),
        "valence": atom.GetTotalValence(),
        "in_ring": int(atom.IsInRing()),
    }


def get_edge_features(mol):
    """Extract bond features including bond order, distance, and angles to XYZ axes."""
    edges = []
    axes = {
        "X": np.array([1, 0, 0]),
        "Y": np.array([0, 1, 0]),
        "Z": np.array([0, 0, 1]),
    }
    conf = mol.GetConformer()
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        pos_i = np.array(conf.GetAtomPosition(i))
        pos_j = np.array(conf.GetAtomPosition(j))
        vec = pos_j - pos_i
        dist = np.linalg.norm(vec)

        if dist == 0:
            continue  # Avoid division by zero

        angles = {
            f"angle_{axis}": np.degrees(
                np.arccos(np.clip(np.dot(vec, vec_axis) / dist, -1.0, 1.0))
            )
            for axis, vec_axis in axes.items()
        }

        bond_order = {
            rdchem.BondType.SINGLE: 0.33,
            rdchem.BondType.DOUBLE: 0.66,
            rdchem.BondType.TRIPLE: 1.0,
            rdchem.BondType.AROMATIC: 0.5,
        }.get(bond.GetBondType(), 0)

        edges.append(
            [
                i,
                j,
                bond_order,
                dist,
                angles["angle_X"],
                angles["angle_Y"],
                angles["angle_Z"],
            ]
        )

    edges = np.array(edges)
    edges[:, 4:] /= 180  # Normalize angles

    return edges
