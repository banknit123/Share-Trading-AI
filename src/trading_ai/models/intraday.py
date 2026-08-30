from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "macd_fast", "rsi_14",
    "volatility_12", "volume_ratio_12", "range_pct", "trend_6_24",
    "minute_sin", "minute_cos",
]


@dataclass
class IntradayPrediction:
    probability_up: float
    expected_return: float


class IntradayDirectionModel:
    def __init__(self, target_column: str) -> None:
        self.target_column = target_column
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced")),
        ])
        self.avg_up_return = 0.0
        self.avg_down_return = 0.0

    def fit(self, df: pd.DataFrame) -> "IntradayDirectionModel":
        y = (df[self.target_column] > 0).astype(int)
        if y.nunique() < 2:
            raise ValueError("Training data must contain both up and down observations")
        self.pipeline.fit(df[FEATURE_COLUMNS], y)
        self.avg_up_return = float(df.loc[y == 1, self.target_column].mean())
        self.avg_down_return = float(df.loc[y == 0, self.target_column].mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        out = df.copy()
        out["probability_up"] = probs
        out["expected_return"] = (
            probs * self.avg_up_return + (1 - probs) * self.avg_down_return
        )
        return out
