def test_config(request):
    score = request.config.getini("rmse")
    assert int(score) == 40_000


def test_invariance_tests(dummy_house):
    ...


def test_directional_expectation_tests():
    ...


def test_model_inference_times():
    ...


def test_model_metric(request, model_metrics):
    current_score = model_metrics.get("rmse")
    rmse = request.config.getini("rmse")
    assert int(current_score) <= int(rmse)
