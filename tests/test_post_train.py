import numpy as np
import pandas as pd
import copy
import math
import time
import pytest


def get_features_labels(df: pd.DataFrame, target_column: str):
    """
    Splits the dataframe into features and labels
    """
    feature_columns = [
        column for column in df.columns.tolist() if column != target_column
    ]
    X = df.loc[:, feature_columns]
    y = df.loc[:, target_column]
    X = X.fillna(X.mean())
    return X, y


def time_predict(model, df):
    """
    Helper function to time the run of a single batch prediction of a model
    """
    start = time.time()
    model.predict(df)
    end = time.time()
    return end - start


def test_config(request):
    """Make sure that the config is being loaded correctly"""
    score = request.config.getini("rmse")
    assert int(score) == 40_000


def get_predictions(
    model, data, data_copy,
):
    """
    Helper function to compare predictions of two different dataframes using
    a model
    """
    changed_df = pd.DataFrame.from_dict([data_copy], orient="columns")
    unchanged_df = pd.DataFrame.from_dict([data], orient="columns")

    changed_X, _ = get_features_labels(changed_df, target_column="SalePrice")
    unchanged_X, _ = get_features_labels(unchanged_df, target_column="SalePrice")

    changed_score = model.predict(changed_X)
    unchanged_score = model.predict(unchanged_X)
    return changed_score, unchanged_score


def get_test_case(data, model, key, add=None, multiply=None):
    """
    Helper method to create the different test cases that alter the keys in a
    dictionary
    """
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


# Keeping all except 1 feature at a time the same
# Changing these features a bit should not result in a noticeable difference
# in the models prediction with the ground truth
@pytest.mark.parametrize(
    "kwargs", [({"key": "YearBuilt", "add": 1}), ({"key": "LotFrontage", "add": 5}),]
)
def test_invariance_tests(kwargs, model, dummy_house):
    changed_score, unchanged_score = get_test_case(dummy_house, model, **kwargs)
    # check that there's about max $5k difference between unchanged and changed
    # house prices
    # $5k is something I feel makes sense, obviously domain knowledge plays
    # a big role in coming up with these test parameters
    assert math.isclose(changed_score, unchanged_score, rel_tol=5e3)


# Keeping all except 1 feature at a time the same
# Chaning these features a bit should result in a notieceable difference
# TODO: Add in argument to parametrize to handle cases where a feature should
# negatively impact price
@pytest.mark.parametrize(
    "kwargs",
    [
        # 1. Increasing overall quality should increase the SalePrice
        ({"key": "LotArea", "multiply": 1.5}),
        # 2. Having a garage with a bigger capacity should increase the SalePrice
        ({"key": "GarageCars", "add": 2}),
        # 3. Better OverallCond should increase the SalePrice
        ({"key": "OverallCond", "add": 5}),
    ],
)
def test_directional_expectation_tests(
    kwargs, model, dummy_house,
):
    changed_score, unchanged_score = get_test_case(dummy_house, model, **kwargs)
    assert changed_score > unchanged_score


def test_model_inference_times(request, dataset, model):
    X, _ = get_features_labels(dataset, target_column="SalePrice")
    latency = np.array([time_predict(model, X) for _ in range(100)])
    latency_p99 = np.quantile(latency, 0.99)
    inference_time = float(request.config.getini("inference_time"))
    assert (
        latency_p99 < inference_time
    ), f"Prediction time at the 99th percentile should be < {inference_time} but was {latency_p99}"


def test_model_metric(request, model_metrics):
    current_score = model_metrics.get("rmse")
    rmse = request.config.getini("rmse")
    assert int(current_score) <= int(rmse)
