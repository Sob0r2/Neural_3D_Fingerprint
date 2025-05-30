import json
import os

import faiss
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdShapeHelpers


def build_faiss_index(features: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build FAISS L2 index.

    Args:
        features (np.ndarray): Matrix of feature vectors.

    Returns:
        faiss.IndexFlatL2: FAISS index.
    """
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)
    return index


def calculate_similarity(mol1, mol2):
    """
    Calculate 2D and 3D similarity between two molecules.

    Returns:
        (float, float): Tanimoto 2D, Tanimoto 3D similarity
    """
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
    sim2d = DataStructs.TanimotoSimilarity(fp1, fp2)

    try:
        sim3d = 1.0 - rdShapeHelpers.ShapeTanimotoDist(mol1, mol2)
    except Exception:
        sim3d = 0.0

    return sim2d, sim3d


def find_most_similar_mols(df, indices):
    """
    Find top-5 most similar molecules based on combined 2D + 3D similarity.

    Args:
        df (pd.DataFrame): DataFrame containing 'file_path' column.
        indices (List[int]): List of FAISS nearest neighbor indices.

    Returns:
        List[str]: List of file paths to the top-5 most similar molecules.
    """
    mol1 = Chem.MolFromMolFile(df.iloc[indices[0]]["file_path"], removeHs=False)

    scores = []
    for idx in indices[1:]:
        mol2 = Chem.MolFromMolFile(df.iloc[idx]["file_path"], removeHs=False)
        sim2d, sim3d = calculate_similarity(mol1, mol2)
        scores.append(sim2d + 1.1 * sim3d)

    sorted_indices = np.argsort(scores)[::-1][:5]
    return [df.iloc[indices[i + 1]]["file_path"] for i in sorted_indices]


def generate_similarity_dict(df, faiss_indices, output_path):
    """
    Generate and save the dictionary mapping molecule -> similar molecules.

    Args:
        df (pd.DataFrame): Input DataFrame.
        faiss_indices (np.ndarray): FAISS neighbor indices.
        output_path (str): Path to save the JSON result.
    """
    result = {}
    for i, idx in enumerate(faiss_indices):
        if i % 100 == 0:
            print(f"Processing molecule {i}")
        result[df.iloc[idx[0]]["file_path"]] = find_most_similar_mols(df, idx)

    with open(output_path, "w") as f:
        json.dump(result, f)

    return result
