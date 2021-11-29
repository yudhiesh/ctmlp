import os
from typing import Any, Tuple
import pandas as pd
import numpy as np
import logging
import click
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

FEATURES_LABLES = Tuple[pd.DataFrame, pd.DataFrame]
TRAIN_TEST = Tuple[pd.DataFrame, pd.DataFrame]
TRAIN_VALID_FEATURES_LABLES = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]

logging.basicConfig(level=logging.INFO)


class TrainModel:
    def __init__(self, train_path: str, test_path: str, **kwargs) -> None:
        self.train_path = train_path
        self.test_path = test_path
        self.target_column = kwargs.get("target_column", "SalePrice")
        self.batch_size = kwargs.get("batch_size", 32)
        self.seed = kwargs.get("seed", 42)
        self.save_dir = kwargs.get("save_dir", "../models")

        if not any([self.train_path, self.test_path]) or not any(
            os.path.exists(path)
            for path in [
                self.train_path,
                self.test_path,
            ]
        ):
            raise Exception(
                "Incorrect path for train or test was passed when training model",
            )

    def load_dataset(self) -> TRAIN_TEST:
        """Load train and test datasets"""
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        return (train, test)  # type: ignore

    def get_features_labels(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> FEATURES_LABLES:
        """
        Splits the dataframe into features and labels
        """
        feature_columns = [
            column for column in df.columns.tolist() if column != target_column
        ]
        X = df.loc[:, feature_columns]
        y = df.loc[:, target_column]
        X = self.process_features(df=X)
        return X, y

    def get_train_valid(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> TRAIN_VALID_FEATURES_LABLES:
        """
        Extracts the features and labels from the train and valid datasets
        """
        X_train, y_train = self.get_features_labels(
            train, target_column=self.target_column
        )
        X_valid, y_test = self.get_features_labels(
            test, target_column=self.target_column
        )
        return (X_train, y_train, X_valid, y_test)

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna(df.mean())
        return df

    def is_data_leaking(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> bool:
        """
        Check that there is no data leaking into the test dataset
        """
        if train.equals(test):
            raise Exception("Both dataframes are equal!")
        # Merging both dfs together
        # Possible values in indicator are: ['left_only', 'right_only', 'both']
        # 'both' means that the rows exist in both dfs
        exists = (
            pd.merge(
                train["Id"],
                test["Id"],
                on=["Id"],
                how="outer",
                indicator="exist",
            )
            .exist.unique()
            .tolist()
        )
        if "both" in exists:
            return True
        return False

    def is_overfitting_batch(self):
        train, _ = self.load_dataset()
        batch = train.sample(
            n=self.batch_size,
            random_state=self.seed,
        )
        batch_X, batch_y = self.get_features_labels(
            df=batch,
            target_column=self.target_column,
        )
        model = self.get_model()
        model.fit(batch_X, batch_y)
        y_pred = model.predict(batch_X)
        if not np.allclose(y_pred, batch_y):
            return False
        return True

    def get_model(self) -> Any:
        """Returns the current model class that is being used"""
        return LinearRegression()

    def run_training(self):
        train, test = self.load_dataset()
        assert (
            not self.is_data_leaking(train, test) and self.is_overfitting_batch()
        ), "Failed pre-train test"
        logging.info("Passed pre-training tests, starting training")
        X_train, y_train, X_valid, y_test = self.get_train_valid(train, test)
        # TODO:
        # Train model and save to models dir
        # Run post train tests


if __name__ == "__main__":

    TrainModel(
        train_path="./data/raw/train.csv",
        test_path="./data/raw/test.csv",
    ).run_training()
