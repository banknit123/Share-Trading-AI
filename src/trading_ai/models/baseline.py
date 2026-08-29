from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "ret_1",
    "ret_5",
    "ret_20",
    "macd",
    "rsi_14",
    "volatility_20",
    "volume_ratio_20",
    "trend_strength",
]


@dataclass
class Prediction:
    probability_up: float
    expected_return: float


class BaselineDirectionModel:
    """Simple interpretable baseline model for walk-forward testing."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        self.avg_up_return = 0.0
        self.avg_down_return = 0.0

    def fit(self, df: pd.DataFrame) -> "BaselineDirectionModel":
        x = df[FEATURE_COLUMNS]
        y = (df["future_return_1"] > 0).astype(int)
        if y.nunique() < 2:
            raise ValueError("Training data must contain both up and down observations")
        self.pipeline.fit(x, y)
        self.avg_up_return = float(df.loc[y == 1, "future_return_1"].mean())
        self.avg_down_return = float(df.loc[y == 0, "future_return_1"].mean())
        return self

    def predict_one(self, row: pd.Series) -> Prediction:
        x = row[FEATURE_COLUMNS].to_frame().T
        p_up = float(self.pipeline.predict_proba(x)[0, 1])
        exp_ret = p_up * self.avg_up_return + (1 - p_up) * self.avg_down_return
        return Prediction(probability_up=p_up, expected_return=float(exp_ret))
