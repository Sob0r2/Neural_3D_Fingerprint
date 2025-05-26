import json
import os

import pandas as pd


def load_qm9_descriptors(data_path: str) -> pd.DataFrame:
    """
    Load descriptor.json files from the QM9 dataset and return a DataFrame.

    Args:
        data_path (str): Path to the QM9 directory.

    Returns:
        pd.DataFrame: Combined descriptor data.
    """
    series_list = []
    for i, dir_name in enumerate(os.listdir(data_path)):
        file_path = os.path.join(data_path, dir_name, "descriptor.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        series_list.append(pd.Series(data))
        if i % 1000 == 0:
            print(f"Processed {i} files")

    return pd.DataFrame(series_list)


def scale_features(
    df: pd.DataFrame, exclude_cols=("SMILES", "file_path")
) -> pd.DataFrame:
    """
    Standardize features (zero mean, unit variance).

    Args:
        df (pd.DataFrame): Input dataframe.
        exclude_cols (tuple): Columns to exclude from scaling.

    Returns:
        pd.DataFrame: Scaled feature dataframe.
    """
    df_scaled = df.copy()
    for col in df_scaled.drop(columns=list(exclude_cols)).columns:
        df_scaled[col] = (df_scaled[col] - df_scaled[col].mean()) / df_scaled[col].std()
    return df_scaled
