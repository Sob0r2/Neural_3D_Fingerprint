import copy
import pickle

from graph_utils import read_graph
from preprocessing import load_qm9_descriptors, scale_features
from similarity import build_faiss_index, generate_similarity_dict

from config import FINAL_DF_DICT_PATH, QM9_DATA_PATH, SIMILAR_MOL_PATH

df = load_qm9_descriptors(QM9_DATA_PATH)
df_scaled = scale_features(df)

features = copy.deepcopy(df_scaled).drop(["SMILES", "file_path"], axis=1)
index = build_faiss_index(features.values)

_, faiss_indices = index.search(features.values, 200)
similar_mol_dict = generate_similarity_dict(df, faiss_indices, SIMILAR_MOL_PATH)

final_df_dict = {}
for i, key in enumerate(similar_mol_dict):
    print(f"Processing graph {i}")
    mol_dir = key.replace("molecule.mol", "")
    final_df_dict[key] = read_graph(mol_dir)

with open(FINAL_DF_DICT_PATH, "wb") as f:
    pickle.dump(final_df_dict, f)
