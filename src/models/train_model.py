import logging
import os
import pickle
import json
from typing import Any, Tuple
import click
from pathlib import Path

import numpy as np
import pandas as pd
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
        self.batch_size = int(kwargs.get("batch_size", 32))
        self.seed = int(kwargs.get("seed", 42))
        self.save_dir = kwargs.get("save_dir", "models")

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
        X_valid, y_valid = self.get_features_labels(
            test, target_column=self.target_column
        )
        return (X_train, y_train, X_valid, y_valid)

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic feature processor that fills NaN values with mean"""
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
        logging.info("Checking if there is data leakage")
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
        return "both" in exists

    def is_overfitting_batch(self) -> bool:
        """
        Checks whether the model is able to overfit a single batch of data
        """
        logging.info("Checking of the model is able to overfit a batch of data")
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

    def run_training(self) -> None:
        """Runs training of a model which then saves the metrics and model"""
        train, test = self.load_dataset()
        assert (
            not self.is_data_leaking(
                train,
                test,
            )
            and self.is_overfitting_batch()
        ), "Failed pre-train test"
        logging.info("Passed pre-training tests, starting training")
        X_train, y_train, X_test, y_test = self.get_train_valid(train, test)
        model = self.get_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = round(
            mean_squared_error(
                y_true=y_test,
                y_pred=y_pred,
                squared=False,
            ),
            2,
        )
        logging.info(f"Completed training with RMSE: {rmse}")
        metric = {"rmse": rmse}
        self.save_model(model)
        with open("test_score.json", "w") as file:
            json.dump(metric, file)
        logging.info("Saved test_score")
        logging.info("Completed training & testing")

    def save_model(self, data: Any) -> None:
        """Saves model to models/model.pkl"""
        filename = "model.pkl"
        save_path = Path(self.save_dir).resolve() / filename
        try:
            pickle.dump(
                data,
                open(
                    save_path.as_posix(),
                    "wb",
                ),
            )
        except pickle.PickleError:
            logging.error("Unable to pickle data")
            raise
        logging.info(f"Saved data to : {save_path.as_posix()}")


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option(
    "--train_path",
    type=str,
    required=True,
    help="Path to the train data",
)
@click.option(
    "--test_path",
    type=str,
    required=True,
    help="Path to the test data",
)
@click.pass_context
def main(ctx, train_path, test_path):
    kwargs = dict(
        [item.strip("--").split("=") for item in ctx.args],
    )
    TrainModel(
        train_path=train_path,
        test_path=test_path,
        **kwargs,
    ).run_training()


if __name__ == "__main__":
    main()
