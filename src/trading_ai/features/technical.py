from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    volume = out["Volume"] if "Volume" in out else pd.Series(index=out.index, dtype=float)

    out["ret_1"] = close.pct_change()
    out["ret_5"] = close.pct_change(5)
    out["ret_20"] = close.pct_change(20)
    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["rsi_14"] = _rsi(close)
    out["volatility_20"] = out["ret_1"].rolling(20).std() * np.sqrt(252)
    out["volume_ratio_20"] = volume / volume.rolling(20).mean()
    out["trend_strength"] = (out["sma_10"] / out["sma_50"]) - 1
    out["future_return_1"] = close.shift(-1) / close - 1
    return out.replace([np.inf, -np.inf], np.nan).dropna()
