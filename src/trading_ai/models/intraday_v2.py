from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ema_spread_6_18", "ema_spread_18_36",
    "rsi_14", "range_pct", "volatility_12", "volatility_36", "volume_ratio_12",
    "volume_z_36", "price_z_36", "position_36", "vwap_distance", "minute_sin", "minute_cos",
]


class IntradayModelV2:
    def __init__(self, target_column: str, kind: str = "logistic") -> None:
        self.target_column = target_column
        self.kind = kind
        if kind == "logistic":
            self.model = Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ])
        elif kind == "hgb":
            self.model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42,
            )
        else:
            raise ValueError(f"Unsupported model kind: {kind}")
        self.avg_up_return = 0.0
        self.avg_down_return = 0.0

    def fit(self, df: pd.DataFrame) -> "IntradayModelV2":
        y = (df[self.target_column] > 0).astype(int)
        if y.nunique() < 2:
            raise ValueError("Training data must contain both up and down observations")
        self.model.fit(df[FEATURE_COLUMNS], y)
        self.avg_up_return = float(df.loc[y == 1, self.target_column].mean())
        self.avg_down_return = float(df.loc[y == 0, self.target_column].mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.model.predict_proba(df[FEATURE_COLUMNS])[:, 1]
        out = df.copy()
        out["probability_up"] = probs
        out["expected_return"] = probs * self.avg_up_return + (1 - probs) * self.avg_down_return
        return out
