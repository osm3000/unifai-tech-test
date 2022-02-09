import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np


def transform_fedas(df: pd.DataFrame, col_label: str) -> pd.DataFrame:
    """
    Break the fedas number into its sub-categories
    """
    df[f"{col_label}_0"] = (
        df[col_label].apply(lambda x: int(str(x)[0])).astype("category")
    )
    df[f"{col_label}_1"] = (
        df[col_label].apply(lambda x: int(str(x)[1:3])).astype("category")
    )
    df[f"{col_label}_2"] = (
        df[col_label].apply(lambda x: int(str(x)[3:5])).astype("category")
    )
    df[f"{col_label}_3"] = (
        df[col_label].apply(lambda x: int(str(x)[5])).astype("category")
    )

    df.drop(columns=[col_label], inplace=True)

    return df


def label_encode_columns(
    df: pd.DataFrame, list_of_training_features: dict
) -> pd.DataFrame:
    encoders = {}
    columns = list(list_of_training_features.keys())
    for col in columns:
        if list_of_training_features[col] == "categorical":
            print(f"Label encoder: {col}")
            le = LabelEncoder().fit(df[col])
            df[col] = le.transform(df[col])
            encoders[col] = le
    return df, encoders


def standard_normalization(
    df: pd.DataFrame, list_of_training_features: dict
) -> pd.DataFrame:
    for column in list_of_training_features:
        if list_of_training_features[column] == "float64":
            df[column] = (df[column] - df[column].mean()) / df[column].std()

    return df


def inverse_transform_fedas(predictions: np.ndarray) -> list:
    """
    Generate the final fedas labels
    """
    final_data = []
    for row in predictions:
        row_string = ""
        for value in row:
            row_string += str(value)

        final_data.append([row_string])

    return final_data