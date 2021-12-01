import json
import pandas as pd
import pickle

import pytest


def pytest_addoption(parser):
    """
    pytest hook that is used to make values from the pytest.ini available to all
    test
    """
    parser.addini(
        "rmse",
        "Min RMSE score for a model to past post-train test",
    )
    parser.addini(
        "inference_time",
        "Max inference time for the model to be making predictions at the 99th percentile",
    )


@pytest.fixture
def dummy_house():
    """
    Sample dataset that will be used to alter and make prediction on
    """
    return {
        "Id": 34,
        "MSSubClass": 20,  # Identifies the type of dwelling involved in the sale.
        "LotFrontage": 70.0,  # Linear feet of street connected to property
        "LotArea": 10552,  # Lot size in square feet
        "OverallQual": 5,  # Rates the overall material and finish of the house
        "OverallCond": 5,  # Rates the overall condition of the house
        "YearBuilt": 1959,  # Original construction date
        "YearRemodAdd": 1959,  # Remodel date (same as construction date if no remodeling or additions)
        "MasVnrArea": 0.0,  # Masonry veneer area in square feet
        "BsmtFinSF1": 1018,  # Type 1 finished square feet
        "BsmtFinSF2": 0,  # Type 2 finished square feet
        "BsmtUnfSF": 380,  #  Unfinished square feet of basement area
        "TotalBsmtSF": 1398,  # Total square feet of basement area
        "1stFlrSF": 1700,  # First Floor square feet
        "2ndFlrSF": 0,  # Second floor square feet
        "LowQualFinSF": 0,  # Low quality finished square feet (all floors)
        "GrLivArea": 1700,  # Above grade (ground) living area square feet
        "BsmtFullBath": 0,  # Basement full bathrooms
        "BsmtHalfBath": 1,  # Basement half bathrooms
        "FullBath": 1,  # Full bathrooms above grade
        "HalfBath": 1,  #  Half baths above grade
        "BedroomAbvGr": 4,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 6,  # Total rooms above grade (does not include bathrooms)
        "Fireplaces": 1,  # Number of fireplaces
        "GarageYrBlt": 1959.0,  # Year garage was built
        "GarageCars": 2,  # Size of garage in car capacity
        "GarageArea": 447,  # Size of garage in square feet
        "WoodDeckSF": 0,  # Wood deck area in square feet
        "OpenPorchSF": 38,  # Open porch area in square feet
        "EnclosedPorch": 0,  # Enclosed porch area in square feet
        "3SsnPorch": 0,  # Three season porch area in square feet
        "ScreenPorch": 0,  # Screen porch area in square feet
        "PoolArea": 0,  # Pool area in square feet
        "MiscVal": 0,  # $Value of miscellaneous feature
        "MoSold": 4,  # Month Sold (MM)
        "YrSold": 2010,  # Year Sold (YYYY)
        "SalePrice": 165500,
    }


@pytest.fixture
def model():
    """
    Fixture that loads the already trained model
    """
    filename = "./models/model.pkl"
    return pickle.load(open(filename, "rb"))


@pytest.fixture
def model_metrics():
    """
    Fixture that loads the metrics that are saved of the current model
    """
    filename = "./test_score.json"
    with open(filename) as f:
        metrics = json.load(f)
    return metrics


@pytest.fixture
def dataset():
    """
    Fixture that loads a bigger dataset
    """
    filename = "./data/raw/test.csv"
    return pd.read_csv(filename)
