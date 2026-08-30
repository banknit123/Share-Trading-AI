from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_BARS = {"30m": 6, "60m": 12}


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_intraday_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    typical = (high + low + close) / 3.0
    session = pd.Series(out.index.date, index=out.index)
    pv = typical * volume
    out["vwap"] = pv.groupby(session).cumsum() / volume.groupby(session).cumsum().replace(0, np.nan)
    out["vwap_distance"] = close / out["vwap"] - 1

    out["ret_1"] = close.pct_change()
    out["ret_3"] = close.pct_change(3)
    out["ret_6"] = close.pct_change(6)
    out["ret_12"] = close.pct_change(12)
    out["ema_6"] = close.ewm(span=6, adjust=False).mean()
    out["ema_18"] = close.ewm(span=18, adjust=False).mean()
    out["ema_36"] = close.ewm(span=36, adjust=False).mean()
    out["ema_spread_6_18"] = out["ema_6"] / out["ema_18"] - 1
    out["ema_spread_18_36"] = out["ema_18"] / out["ema_36"] - 1
    out["rsi_14"] = _rsi(close)
    out["range_pct"] = (high - low) / close.replace(0, np.nan)
    out["volatility_12"] = out["ret_1"].rolling(12).std()
    out["volatility_36"] = out["ret_1"].rolling(36).std()
    out["volume_ratio_12"] = volume / volume.rolling(12).mean()
    out["volume_z_36"] = (volume - volume.rolling(36).mean()) / volume.rolling(36).std().replace(0, np.nan)

    roll_mean = close.rolling(36).mean()
    roll_std = close.rolling(36).std().replace(0, np.nan)
    out["price_z_36"] = (close - roll_mean) / roll_std
    out["position_36"] = (close - low.rolling(36).min()) / (high.rolling(36).max() - low.rolling(36).min()).replace(0, np.nan)

    minutes = out.index.hour * 60 + out.index.minute
    out["minute_sin"] = np.sin(2 * np.pi * minutes / (24 * 60))
    out["minute_cos"] = np.cos(2 * np.pi * minutes / (24 * 60))

    for name, bars in HORIZON_BARS.items():
        out[f"future_return_{name}"] = close.shift(-bars) / close - 1

    return out.replace([np.inf, -np.inf], np.nan).dropna()
