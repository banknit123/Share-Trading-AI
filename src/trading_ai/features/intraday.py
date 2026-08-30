from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON_BARS = {"10m": 2, "30m": 6, "60m": 12}  # assumes 5-minute candles


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create 5-minute features and 10/30/60-minute forward-return targets."""
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    out["ret_1"] = close.pct_change(1)
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["ret_12"] = close.pct_change(12)
    out["ema_6"] = close.ewm(span=6, adjust=False).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_24"] = close.ewm(span=24, adjust=False).mean()
    out["macd_fast"] = out["ema_6"] - out["ema_24"]
    out["rsi_14"] = _rsi(close, 14)
    out["volatility_12"] = out["ret_1"].rolling(12).std()
    out["volume_ratio_12"] = volume / volume.rolling(12).mean()
    out["range_pct"] = (high - low) / close.replace(0, np.nan)
    out["trend_6_24"] = out["ema_6"] / out["ema_24"] - 1

    idx = pd.DatetimeIndex(out.index)
    minute_of_day = idx.hour * 60 + idx.minute
    out["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    for name, bars in HORIZON_BARS.items():
        out[f"future_return_{name}"] = close.shift(-bars) / close - 1

    return out.replace([np.inf, -np.inf], np.nan).dropna()
