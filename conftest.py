import json
import pickle

import pytest


def pytest_addoption(parser):
    parser.addini("rmse", "Min RMSE score for a model to past post-train test")
    parser.addini(
        "inference_time", "Max inference time for the model to be making predictions"
    )


@pytest.fixture
def dummy_house():
    return {
        "Id": {33: 34},
        "MSSubClass": {33: 20},  # Identifies the type of dwelling involved in the sale.
        "LotFrontage": {33: 70.0},  # Linear feet of street connected to property
        "LotArea": {33: 10552},  # Lot size in square feet
        "OverallQual": {33: 5},  # Rates the overall material and finish of the house
        "OverallCond": {33: 5},  # Rates the overall condition of the house
        "YearBuilt": {33: 1959},  # Original construction date
        "YearRemodAdd": {
            33: 1959
        },  # Remodel date (same as construction date if no remodeling or additions)
        "MasVnrArea": {33: 0.0},  # Masonry veneer area in square feet
        "BsmtFinSF1": {33: 1018},  # Type 1 finished square feet
        "BsmtFinSF2": {33: 0},  # Type 2 finished square feet
        "BsmtUnfSF": {33: 380},  #  Unfinished square feet of basement area
        "TotalBsmtSF": {33: 1398},  # Total square feet of basement area
        "1stFlrSF": {33: 1700},  # First Floor square feet
        "2ndFlrSF": {33: 0},  # Second floor square feet
        "LowQualFinSF": {33: 0},  # Low quality finished square feet (all floors)
        "GrLivArea": {33: 1700},  # Above grade (ground) living area square feet
        "BsmtFullBath": {33: 0},  # Basement full bathrooms
        "BsmtHalfBath": {33: 1},  # Basement half bathrooms
        "FullBath": {33: 1},  # Full bathrooms above grade
        "HalfBath": {33: 1},  #  Half baths above grade
        "BedroomAbvGr": {33: 4},
        "KitchenAbvGr": {33: 1},
        "TotRmsAbvGrd": {33: 6},  # Total rooms above grade (does not include bathrooms)
        "Fireplaces": {33: 1},  # Number of fireplaces
        "GarageYrBlt": {33: 1959.0},  # Year garage was built
        "GarageCars": {33: 2},  # Size of garage in car capacity
        "GarageArea": {33: 447},  # Size of garage in square feet
        "WoodDeckSF": {33: 0},  # Wood deck area in square feet
        "OpenPorchSF": {33: 38},  # Open porch area in square feet
        "EnclosedPorch": {33: 0},  # Enclosed porch area in square feet
        "3SsnPorch": {33: 0},  # Three season porch area in square feet
        "ScreenPorch": {33: 0},  # Screen porch area in square feet
        "PoolArea": {33: 0},  # Pool area in square feet
        "MiscVal": {33: 0},  # $Value of miscellaneous feature
        "MoSold": {33: 4},  # Month Sold (MM)
        "YrSold": {33: 2010},  # Year Sold (YYYY)
        "SalePrice": {33: 165500},
    }


@pytest.fixture
def load_model():
    filename = "./models/model.pkl"
    model = pickle.load(open(filename, "rb"))
    return model


@pytest.fixture
def model_metrics():
    filename = "./test_score.json"
    with open(filename) as f:
        metrics = json.load(f)
    return metrics

