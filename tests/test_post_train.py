import math

import numpy as np
import pytest

from helpers.utils import get_features_labels, get_test_case, time_predict


@pytest.mark.parametrize(
    "kwargs",
    [
        # 1. Having a house built later should not impact the SalePrice
        ({"key": "YearBuilt", "add": 1}),
    ],
)
def test_invariance_tests(kwargs, model, dummy_house):
    """
    Keeping all except 1 feature at a time the same
    Changing these features a bit should not result in a noticeable difference
    in the models prediction with the ground truth
    """
    changed_score, unchanged_score = get_test_case(
        dummy_house,
        model,
        **kwargs,
    )
    # check that there's about max $5k difference between unchanged and changed
    # house prices
    # $5k is something I feel makes sense, obviously domain knowledge plays
    # a big role in coming up with these test parameters
    assert math.isclose(
        changed_score,
        unchanged_score,
        rel_tol=5e3,
    )


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
    kwargs,
    model,
    dummy_house,
):
    """
    Keeping all except 1 feature at a time the same
    Chaning these features a bit should result in a notieceable difference
    """
    changed_score, unchanged_score = get_test_case(
        dummy_house,
        model,
        **kwargs,
    )
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
