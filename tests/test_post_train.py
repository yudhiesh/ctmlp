import numpy as np
import pandas as pd
import copy
import math
import time


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


def test_config(request):
    score = request.config.getini("rmse")
    assert int(score) == 40_000


# Keeping all except 1 feature at a time the same
# Changing these features a bit should not result in a noticeable difference
# in the models prediction with the ground truth
def test_invariance_tests(model, dummy_house):
    # Changing the year built a bit shouldn't really increase a houses value
    dummy_house_copy = copy.deepcopy(dummy_house)
    year_built = dummy_house.get("YearBuilt")
    dummy_house_copy["YearBuilt"] = year_built + 1
    changed_df = pd.DataFrame.from_dict([dummy_house], orient="columns")
    unchanged_df = pd.DataFrame.from_dict([dummy_house_copy], orient="columns")
    changed_X, _ = get_features_labels(changed_df, target_column="SalePrice")
    unchanged_X, _ = get_features_labels(unchanged_df, target_column="SalePrice")
    changed_score = model.predict(changed_X)
    unchanged_score = model.predict(unchanged_X)
    # check that there's about max $5k difference between unchanged and changed
    # house prices
    # $5k is something I feel makes sense, obviously domain knowledge plays a big
    # role in coming up with these test parameters
    assert math.isclose(changed_score, unchanged_score, rel_tol=5e3)


# Keeping all except 1 feature at a time the same
# Chaning these features a bit should result in a notieceable difference
def test_directional_expectation_tests():
    ...


def time_predict(model, df):
    """
    Helper function to time the run of a single batch prediction of a model
    """
    start = time.time()
    model.predict(df)
    end = time.time()
    return end - start


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
