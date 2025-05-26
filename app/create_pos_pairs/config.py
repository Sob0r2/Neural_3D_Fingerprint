import os

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv("DATA_PATH")
QM9_DATA_PATH = os.path.join(DATA_PATH, "qm9_data")
SIMILAR_MOL_PATH = os.path.join(DATA_PATH, "similar_mol.json")
FINAL_DF_DICT_PATH = os.path.join(DATA_PATH, "final_df_dict.pkl")
