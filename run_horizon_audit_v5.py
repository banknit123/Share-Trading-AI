from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.intraday_v2 import add_intraday_features_v2

ROUND_TRIP_COST = 0.0012
HORIZONS = {"60m": 12, "120m": 24, "240m": 48}
FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "vwap_distance",
    "ema_spread_6_18", "ema_spread_18_36", "rsi_14", "range_pct",
    "volatility_12", "volatility_36", "volume_ratio_12", "volume_z_36",
    "price_z_36", "position_36", "rs_1", "rs_3", "rs_6", "rs_12",
    "benchmark_ret_1", "benchmark_ret_6", "benchmark_vol_12",
]


def build_panel() -> pd.DataFrame:
    benchmark_raw = download_history(DEFAULT_CONFIG.benchmark, period="60d", interval="5m")
    b = benchmark_raw[["Close"]].copy()
    b["benchmark_ret_1"] = b["Close"].pct_change()
    b["benchmark_ret_3"] = b["Close"].pct_change(3)
    b["benchmark_ret_6"] = b["Close"].pct_change(6)
    b["benchmark_ret_12"] = b["Close"].pct_change(12)
    b["benchmark_vol_12"] = b["benchmark_ret_1"].rolling(12).std()
    for name, bars in HORIZONS.items():
        b[f"benchmark_future_{name}"] = b["Close"].shift(-bars) / b["Close"] - 1
    b = b.drop(columns=["Close"])

    frames = []
    for symbol in DEFAULT_CONFIG.universe:
        raw = download_history(symbol, period="60d", interval="5m")
        f = add_intraday_features_v2(raw).join(b, how="inner")
        f["rs_1"] = f["ret_1"] - f["benchmark_ret_1"]
        f["rs_3"] = f["ret_3"] - f["benchmark_ret_3"]
        f["rs_6"] = f["ret_6"] - f["benchmark_ret_6"]
        f["rs_12"] = f["ret_12"] - f["benchmark_ret_12"]
        for name, bars in HORIZONS.items():
            f[f"future_return_{name}"] = f["Close"].shift(-bars) / f["Close"] - 1
            f[f"excess_future_{name}"] = f[f"future_return_{name}"] - f[f"benchmark_future_{name}"]
        f["symbol"] = symbol
        frames.append(f)

    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    need = FEATURES + [f"future_return_{h}" for h in HORIZONS] + [f"excess_future_{h}" for h in HORIZONS]
    panel = panel.dropna(subset=need)
    panel["timestamp"] = panel.index
    return panel.sort_values(["timestamp", "symbol"])


def split_by_time(panel: pd.DataFrame):
    times = np.array(sorted(panel["timestamp"].unique()))
    n = len(times)
    t1 = times[int(n * 0.60)]
    t2 = times[int(n * 0.80)]
    train = panel[panel["timestamp"] < t1].copy()
    val = panel[(panel["timestamp"] >= t1) & (panel["timestamp"] < t2)].copy()
    test = panel[panel["timestamp"] >= t2].copy()
    return train, val, test


def rank_ic(prob: np.ndarray, values: pd.Series) -> float:
    a = pd.Series(prob).rank(pct=True)
    b = values.reset_index(drop=True).rank(pct=True)
    return float(a.corr(b))


def audit_horizon(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, horizon: str) -> None:
    target = f"excess_future_{horizon}"
    raw_target = f"future_return_{horizon}"
    y_train = (train[target] > ROUND_TRIP_COST).astype(int)
    if y_train.nunique() < 2:
        print(f"{horizon}: training labels only one class")
        return

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[FEATURES], y_train)

    print(f"\n{horizon} horizon")
    for label, df in (("VALIDATION", val), ("UNTOUCHED TEST", test)):
        prob = model.predict_proba(df[FEATURES])[:, 1]
        y = (df[target] > ROUND_TRIP_COST).astype(int)
        auc = roc_auc_score(y, prob) if y.nunique() > 1 else float("nan")
        ric_excess = rank_ic(prob, df[target])
        ric_raw = rank_ic(prob, df[raw_target])
        scored = df[["timestamp", "symbol", target, raw_target, "benchmark_ret_6", "rs_6"]].copy()
        scored["probability"] = prob

        # Evaluate only candidate snapshots that pass the unchanged regime + RS filters.
        eligible = scored[(scored["benchmark_ret_6"] > 0) & (scored["rs_6"] > 0)]
        if eligible.empty:
            print(f"  {label}: auc={auc:.3f} rank_ic_excess={ric_excess:.3f} rank_ic_raw={ric_raw:.3f} eligible=0")
            continue

        top = eligible.sort_values(["timestamp", "probability", "rs_6"], ascending=[True, False, False]).groupby("timestamp", as_index=False).first()
        gross = top[raw_target]
        net = gross - ROUND_TRIP_COST
        q80 = top["probability"].quantile(0.80)
        high = top[top["probability"] >= q80]
        high_net = high[raw_target] - ROUND_TRIP_COST

        print(
            f"  {label}: auc={auc:.3f} rank_ic_excess={ric_excess:.3f} rank_ic_raw={ric_raw:.3f} "
            f"top_count={len(top)} gross_avg={gross.mean():.3%} net_avg={net.mean():.3%} "
            f"top20_count={len(high)} top20_net_avg={high_net.mean():.3%}"
        )


def main() -> None:
    print("Share-Trading-AI v5 horizon audit")
    print("No trades are placed. This compares whether longer horizons improve signal quality.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)

    panel = build_panel()
    train, val, test = split_by_time(panel)
    print(f"Rows: train={len(train)} validation={len(val)} test={len(test)}")

    for horizon in HORIZONS:
        audit_horizon(train, val, test, horizon)

    print("\nInterpretation:")
    print("- Prefer horizons with positive validation AND untouched-test rank IC and positive top-candidate net averages.")
    print("- If all horizons remain weak/negative, stop price-only ML and add genuinely new information such as news/event signals.")
    print("- This script is diagnostic only; live execution remains disabled.")


if __name__ == "__main__":
    main()
