import copy
import time
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

SCORES = Tuple[np.ndarray, np.ndarray]
FEATURES_LABELS = Tuple[pd.DataFrame, pd.DataFrame]
ARITHEMETIC_OPERATIONS = Optional[Union[int, float]]


def get_features_labels(
    df: pd.DataFrame,
    target_column: str,
) -> FEATURES_LABELS:
    """
    Splits the dataframe into features and labels
    """
    feature_columns = [
        column for column in df.columns.tolist() if column != target_column
    ]
    X = df.loc[:, feature_columns]
    y = df.loc[:, target_column]
    X = X.fillna(df.mean())
    return X, y


def time_predict(model: Any, df: pd.DataFrame) -> float:
    """
    Helper function to time the run of a single batch prediction of a model
    """
    start = time.time()
    model.predict(df)
    end = time.time()
    return end - start


def get_predictions(
    model: Any,
    data: pd.DataFrame,
    data_copy: pd.DataFrame,
) -> SCORES:
    """
    Helper function to get the prediction of the models for the dataframe with a
    change in any value and a dataframe without a change in any value
    a model
    """
    changed_df = pd.DataFrame.from_dict([data_copy], orient="columns")
    unchanged_df = pd.DataFrame.from_dict([data], orient="columns")

    changed_X, _ = get_features_labels(changed_df, target_column="SalePrice")
    unchanged_X, _ = get_features_labels(unchanged_df, target_column="SalePrice")

    changed_score = model.predict(changed_X)
    unchanged_score = model.predict(unchanged_X)
    return changed_score, unchanged_score


def get_test_case(
    data: pd.DataFrame,
    model: Any,
    key: str,
    add: ARITHEMETIC_OPERATIONS = None,
    multiply: ARITHEMETIC_OPERATIONS = None,
) -> SCORES:
    """
    Helper method to create the different test cases that alter the keys in a
    dictionary
    """
    # Create a deepcopy of the data which will contain the dataframe that will
    # have an altered value
    data_copy = copy.deepcopy(data)
    value = data.get(key)
    if add:
        data_copy[key] = value + add
    elif multiply:
        data_copy[key] = value * multiply
    else:
        raise Exception("Please pass either an addition or multiply value")
    changed_score, unchanged_score = get_predictions(model, data, data_copy)
    return changed_score, unchanged_score
