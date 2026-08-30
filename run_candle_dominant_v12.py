from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.intraday_v2 import add_intraday_features_v2
from run_historical_multimodal_v10 import load_events, add_news_features, NEWS_FEATURES

ROUND_TRIP_COST = 0.0012
HORIZONS = {"60m": 12, "120m": 24}

W_CANDLE = 0.70
W_MARKET = 0.20
W_NEWS = 0.10

CANDLE_FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_6", "ret_9", "ret_12", "ret_18", "ret_24",
    "vwap_distance", "ema_spread_6_18", "ema_spread_18_36", "ema_spread_6_36",
    "ema6_slope_3", "ema18_slope_6", "rsi_14", "range_pct",
    "body_pct", "upper_wick_pct", "lower_wick_pct", "close_location",
    "bullish_3", "bullish_6", "bullish_12", "momentum_consistency_6", "momentum_consistency_12",
    "volatility_6", "volatility_12", "volatility_36", "vol_expansion_6_36",
    "volume_ratio_6", "volume_ratio_12", "volume_z_36", "up_volume_share_6", "up_volume_share_12",
    "price_z_36", "position_12", "position_36", "breakout_12", "breakdown_12",
]

MARKET_FEATURES = [
    "nifty_ret_1", "nifty_ret_3", "nifty_ret_6", "nifty_ret_12",
    "nifty_vol_12", "nifty_vwap_distance",
    "rel_ret_1", "rel_ret_3", "rel_ret_6", "rel_ret_12",
]


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    idx = idx.tz_convert("UTC").floor("5min")
    out.index = idx
    if out.index.has_duplicates:
        out = out.groupby(level=0).last()
    return out.sort_index()


def add_candle_features(raw: pd.DataFrame) -> pd.DataFrame:
    out = add_intraday_features_v2(raw)
    close = out["Close"].astype(float)
    open_ = out["Open"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float)

    for n in (2, 9, 18, 24):
        out[f"ret_{n}"] = close.pct_change(n)

    out["ema_spread_6_36"] = out["ema_6"] / out["ema_36"] - 1
    out["ema6_slope_3"] = out["ema_6"].pct_change(3)
    out["ema18_slope_6"] = out["ema_18"].pct_change(6)

    rng = (high - low).replace(0, np.nan)
    out["body_pct"] = (close - open_) / open_.replace(0, np.nan)
    out["upper_wick_pct"] = (high - np.maximum(open_, close)) / close.replace(0, np.nan)
    out["lower_wick_pct"] = (np.minimum(open_, close) - low) / close.replace(0, np.nan)
    out["close_location"] = (close - low) / rng

    bullish = (close > open_).astype(float)
    out["bullish_3"] = bullish.rolling(3).mean()
    out["bullish_6"] = bullish.rolling(6).mean()
    out["bullish_12"] = bullish.rolling(12).mean()
    sign = np.sign(out["ret_1"])
    out["momentum_consistency_6"] = sign.rolling(6).mean()
    out["momentum_consistency_12"] = sign.rolling(12).mean()

    out["volatility_6"] = out["ret_1"].rolling(6).std()
    out["vol_expansion_6_36"] = out["volatility_6"] / out["volatility_36"].replace(0, np.nan)
    out["volume_ratio_6"] = volume / volume.rolling(6).mean().replace(0, np.nan)

    up_vol = volume.where(close.diff() > 0, 0.0)
    out["up_volume_share_6"] = up_vol.rolling(6).sum() / volume.rolling(6).sum().replace(0, np.nan)
    out["up_volume_share_12"] = up_vol.rolling(12).sum() / volume.rolling(12).sum().replace(0, np.nan)

    out["position_12"] = (close - low.rolling(12).min()) / (
        high.rolling(12).max() - low.rolling(12).min()
    ).replace(0, np.nan)
    prev_high12 = high.rolling(12).max().shift(1)
    prev_low12 = low.rolling(12).min().shift(1)
    out["breakout_12"] = close / prev_high12 - 1
    out["breakdown_12"] = close / prev_low12 - 1

    return out.replace([np.inf, -np.inf], np.nan)


def benchmark_features() -> pd.DataFrame:
    """Build a robust contemporaneous market proxy from the liquid NSE universe.

    The Yahoo NIFTY 5-minute series can arrive with timestamp conventions that do
    not line up with the equity series. To prevent the market layer from vanishing,
    v12 uses the cross-sectional median of the same liquid NSE universe as a broad
    market proxy. Only current/past bars are used; no future information enters.
    """
    pieces = []
    for symbol in DEFAULT_CONFIG.universe:
        raw = _to_utc(download_history(symbol, period="60d", interval="5m"))
        f = add_intraday_features_v2(raw).sort_index()
        sub = f[["ret_1", "ret_3", "ret_6", "ret_12", "volatility_12", "vwap_distance"]].copy()
        pieces.append(sub)

    if not pieces:
        raise RuntimeError("Could not build market proxy: no universe data")

    stack = pd.concat(pieces)
    proxy = stack.groupby(level=0).median(numeric_only=True).sort_index()
    proxy = proxy.rename(columns={
        "ret_1": "nifty_ret_1",
        "ret_3": "nifty_ret_3",
        "ret_6": "nifty_ret_6",
        "ret_12": "nifty_ret_12",
        "volatility_12": "nifty_vol_12",
        "vwap_distance": "nifty_vwap_distance",
    })
    return proxy


def _align_benchmark(stock: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    aligned = bench.reindex(stock.index, method="nearest", tolerance=pd.Timedelta(minutes=3))
    return stock.join(aligned)


def build_panel(events_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bench = benchmark_features()
    print(f"Market proxy bars: {len(bench)} using {len(DEFAULT_CONFIG.universe)} NSE stocks")
    frames = []
    for symbol in DEFAULT_CONFIG.universe:
        raw = _to_utc(download_history(symbol, period="60d", interval="5m"))
        f = add_candle_features(raw).sort_index()
        before_align = len(f)
        f = _align_benchmark(f, bench)
        aligned_market = int(f["nifty_ret_1"].notna().sum()) if len(f) else 0

        f["rel_ret_1"] = f["ret_1"] - f["nifty_ret_1"]
        f["rel_ret_3"] = f["ret_3"] - f["nifty_ret_3"]
        f["rel_ret_6"] = f["ret_6"] - f["nifty_ret_6"]
        f["rel_ret_12"] = f["ret_12"] - f["nifty_ret_12"]
        f = add_news_features(f, events_by_symbol.get(symbol))
        f["symbol"] = symbol
        f["timestamp"] = f.index
        for name, bars in HORIZONS.items():
            f[f"future_{name}"] = f["Close"].shift(-bars) / f["Close"] - 1.0
        frames.append(f)
        print(
            f"{symbol:15} rows={len(f):5d} stock_bars={before_align:5d} market_aligned={aligned_market:5d} "
            f"news60={(f['news_count_60m'] > 0).sum():4d} ret6_avg={f['ret_6'].mean():+.3%}"
        )

    if not frames:
        raise RuntimeError("No stock frames were built")

    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    needed = CANDLE_FEATURES + MARKET_FEATURES + NEWS_FEATURES + [f"future_{h}" for h in HORIZONS]
    missing_all = [c for c in needed if c not in panel.columns or panel[c].notna().sum() == 0]
    if missing_all:
        raise RuntimeError(f"Features with zero usable observations: {missing_all}")

    panel = panel.dropna(subset=needed).sort_values(["timestamp", "symbol"])
    if panel.empty:
        raise RuntimeError(
            "Panel became empty after feature filtering. Check market_aligned counts and feature diagnostics above."
        )
    return panel


def split_time(panel: pd.DataFrame):
    times = np.array(sorted(panel["timestamp"].unique()))
    if len(times) < 10:
        raise RuntimeError(f"Too few unique timestamps for train/validation/test split: {len(times)}")
    t1 = times[int(len(times) * 0.60)]
    t2 = times[int(len(times) * 0.80)]
    return (
        panel[panel.timestamp < t1].copy(),
        panel[(panel.timestamp >= t1) & (panel.timestamp < t2)].copy(),
        panel[panel.timestamp >= t2].copy(),
    )


def fit_model(train: pd.DataFrame, features: list[str], target: str):
    y = (train[target] > ROUND_TRIP_COST).astype(int)
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(train[features], y)
    return model


def prob(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(df[features])[:, 1]


def evaluate(p: np.ndarray, df: pd.DataFrame, target: str) -> dict:
    ret = df[target].to_numpy()
    y = (ret > ROUND_TRIP_COST).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    rank_ic = pd.Series(p).corr(pd.Series(ret), method="spearman")
    cutoff = np.quantile(p, 0.90)
    top = ret[p >= cutoff] - ROUND_TRIP_COST
    return {
        "auc": float(auc),
        "rank_ic": float(rank_ic),
        "n": int(len(top)),
        "net": float(np.mean(top)) if len(top) else 0.0,
        "win": float(np.mean(top > 0)) if len(top) else 0.0,
    }


def fmt(label: str, m: dict) -> str:
    return (
        f"{label:26} auc={m['auc']:.3f} rank_ic={m['rank_ic']:+.3f} "
        f"top10_n={m['n']:4d} top10_net={m['net']:+.3%} top10_win={m['win']:.2%}"
    )


def main() -> None:
    print("Share-Trading-AI v12 candle-dominant weighted validation")
    print("No trades are placed.")
    print(f"Decision architecture: {W_CANDLE:.0%} candle + {W_MARKET:.0%} market + {W_NEWS:.0%} news")
    print("Candles remain the primary signal; news is only a secondary context layer.")
    print("Market context uses a contemporaneous equal-weight/median liquid-NSE proxy.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)

    events = load_events()
    panel = build_panel(events)
    train, val, test = split_time(panel)
    print(f"\nRows: train={len(train)} validation={len(val)} untouched_test={len(test)}")

    passes = 0
    for horizon in HORIZONS:
        target = f"future_{horizon}"
        candle_model = fit_model(train, CANDLE_FEATURES, target)
        market_model = fit_model(train, MARKET_FEATURES, target)
        news_model = fit_model(train, NEWS_FEATURES, target)

        cv = prob(candle_model, val, CANDLE_FEATURES)
        ct = prob(candle_model, test, CANDLE_FEATURES)
        mv = prob(market_model, val, MARKET_FEATURES)
        mt = prob(market_model, test, MARKET_FEATURES)
        nv = prob(news_model, val, NEWS_FEATURES)
        nt = prob(news_model, test, NEWS_FEATURES)

        weighted_v = W_CANDLE * cv + W_MARKET * mv + W_NEWS * nv
        weighted_t = W_CANDLE * ct + W_MARKET * mt + W_NEWS * nt

        candle_v = evaluate(cv, val, target)
        candle_t = evaluate(ct, test, target)
        weighted_val = evaluate(weighted_v, val, target)
        weighted_test = evaluate(weighted_t, test, target)

        print(f"\n{horizon} horizon")
        print(fmt("CANDLE_ONLY VALIDATION", candle_v))
        print(fmt("CANDLE_ONLY TEST", candle_t))
        print(fmt("70/20/10 VALIDATION", weighted_val))
        print(fmt("70/20/10 TEST", weighted_test))
        print(
            "CONTEXT LIFT: "
            f"val_auc={weighted_val['auc']-candle_v['auc']:+.3f} "
            f"test_auc={weighted_test['auc']-candle_t['auc']:+.3f} "
            f"val_rank_ic={weighted_val['rank_ic']-candle_v['rank_ic']:+.3f} "
            f"test_rank_ic={weighted_test['rank_ic']-candle_t['rank_ic']:+.3f} "
            f"val_top10_net={weighted_val['net']-candle_v['net']:+.3%} "
            f"test_top10_net={weighted_test['net']-candle_t['net']:+.3%}"
        )

        robust = (
            weighted_val["net"] > 0
            and weighted_test["net"] > 0
            and weighted_val["rank_ic"] > 0
            and weighted_test["rank_ic"] > 0
            and weighted_val["win"] >= 0.50
            and weighted_test["win"] >= 0.50
        )
        passes += int(robust)
        print("HORIZON GATE:", "PASS FOR REPEATED WALK-FORWARD" if robust else "FAIL - RESEARCH ONLY")

    print("\nV12 overall research gate:")
    if passes:
        print("At least one candle-dominant horizon passed this single split. Next step is repeated walk-forward validation, not live trading.")
    else:
        print("No horizon passed. Keep candle dominance, but redesign candle/market features rather than increasing news weight.")
    print("Live execution remains disabled regardless of this run.")


if __name__ == "__main__":
    main()
