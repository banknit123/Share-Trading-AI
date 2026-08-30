from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.intraday_v2 import add_intraday_features_v2

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "vwap_distance",
    "ema_spread_6_18", "ema_spread_18_36", "rsi_14", "range_pct",
    "volatility_12", "volatility_36", "volume_ratio_12", "volume_z_36",
    "price_z_36", "position_36", "rs_1", "rs_3", "rs_6", "rs_12",
    "benchmark_ret_1", "benchmark_ret_6", "benchmark_vol_12",
]

HORIZON = "30m"
BARS = 6
ROUND_TRIP_COST = 0.0012
THRESHOLDS = (0.55, 0.60, 0.65, 0.70)
MIN_VALIDATION_TRADES = 8


def build_panel() -> pd.DataFrame:
    benchmark_raw = download_history(DEFAULT_CONFIG.benchmark, period="60d", interval="5m")
    b = benchmark_raw[["Close"]].copy()
    b["benchmark_ret_1"] = b["Close"].pct_change()
    b["benchmark_ret_3"] = b["Close"].pct_change(3)
    b["benchmark_ret_6"] = b["Close"].pct_change(6)
    b["benchmark_ret_12"] = b["Close"].pct_change(12)
    b["benchmark_vol_12"] = b["benchmark_ret_1"].rolling(12).std()
    b["benchmark_future_30m"] = b["Close"].shift(-BARS) / b["Close"] - 1
    b = b.drop(columns=["Close"])

    frames: list[pd.DataFrame] = []
    for symbol in DEFAULT_CONFIG.universe:
        raw = download_history(symbol, period="60d", interval="5m")
        f = add_intraday_features_v2(raw).join(b, how="inner")
        f["rs_1"] = f["ret_1"] - f["benchmark_ret_1"]
        f["rs_3"] = f["ret_3"] - f["benchmark_ret_3"]
        f["rs_6"] = f["ret_6"] - f["benchmark_ret_6"]
        f["rs_12"] = f["ret_12"] - f["benchmark_ret_12"]
        f["excess_future_30m"] = f["future_return_30m"] - f["benchmark_future_30m"]
        f["symbol"] = symbol
        frames.append(f)

    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=FEATURES + ["future_return_30m", "excess_future_30m"])
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


def fit_model(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    y = (train["excess_future_30m"] > ROUND_TRIP_COST).astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Training labels contain only one class")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[FEATURES], y)
    return model


def score(model, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["probability"] = model.predict_proba(out[FEATURES])[:, 1]
    return out


def simulate(scored: pd.DataFrame, threshold: float):
    equity = 1.0
    returns: list[float] = []
    trades: list[tuple[pd.Timestamp, str, float, float]] = []
    all_times = sorted(scored["timestamp"].unique())
    i = 0
    while i < len(all_times) - BARS:
        ts = all_times[i]
        snap = scored[scored["timestamp"] == ts]
        # Long-only regime filter: only take risk when the index has positive 30m momentum.
        if snap.empty or float(snap["benchmark_ret_6"].iloc[0]) <= 0:
            i += 1
            continue
        candidate = snap.sort_values(["probability", "rs_6"], ascending=False).iloc[0]
        p = float(candidate["probability"])
        if p < threshold or float(candidate["rs_6"]) <= 0:
            i += 1
            continue
        gross = float(candidate["future_return_30m"])
        net = gross - ROUND_TRIP_COST
        returns.append(net)
        equity *= 1.0 + net
        trades.append((ts, str(candidate["symbol"]), p, net))
        i += BARS

    win_rate = sum(r > 0 for r in returns) / len(returns) if returns else 0.0
    total_return = equity - 1.0
    avg_trade = float(np.mean(returns)) if returns else 0.0
    return {
        "trades": len(returns),
        "return": total_return,
        "win_rate": win_rate,
        "avg_trade": avg_trade,
        "details": trades,
    }


def choose_threshold(val_scored: pd.DataFrame):
    candidates = []
    for threshold in THRESHOLDS:
        r = simulate(val_scored, threshold)
        if r["trades"] >= MIN_VALIDATION_TRADES:
            # Penalize low trade count; prefer positive average trade and total return.
            score_value = r["return"] + 0.5 * r["avg_trade"] * r["trades"]
            candidates.append((score_value, threshold, r))
    if not candidates:
        return 0.70, simulate(val_scored, 0.70)
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, threshold, result = candidates[0]
    return threshold, result


def main() -> None:
    print("Share-Trading-AI v3 cross-sectional research backtest")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print("Strategy: rank stocks relative to NIFTY; 30-minute long-only; untouched test set")

    panel = build_panel()
    train, val, test = split_by_time(panel)
    model = fit_model(train)
    val_scored = score(model, val)
    threshold, val_result = choose_threshold(val_scored)
    test_scored = score(model, test)
    test_result = simulate(test_scored, threshold)

    print(f"Rows: train={len(train)} validation={len(val)} test={len(test)}")
    print(
        f"Selected threshold={threshold:.2f} | validation trades={val_result['trades']} "
        f"return={val_result['return']:.2%} win={val_result['win_rate']:.2%} "
        f"avg_trade={val_result['avg_trade']:.3%}"
    )
    print(
        f"UNTOUCHED TEST | trades={test_result['trades']} return={test_result['return']:.2%} "
        f"win={test_result['win_rate']:.2%} avg_trade={test_result['avg_trade']:.3%}"
    )

    by_symbol: dict[str, list[float]] = {}
    for _, symbol, _, net in test_result["details"]:
        by_symbol.setdefault(symbol, []).append(net)
    if by_symbol:
        print("\nTest trades by symbol:")
        for symbol, vals in sorted(by_symbol.items()):
            print(
                f"  {symbol:15} trades={len(vals):3d} net={np.prod([1+v for v in vals])-1:8.2%} "
                f"win={sum(v>0 for v in vals)/len(vals):7.2%}"
            )

    ready = (
        test_result["trades"] >= 15
        and test_result["return"] > 0
        and test_result["win_rate"] >= 0.50
        and test_result["avg_trade"] > 0
    )
    print("\nResearch gate:", "PASS FOR PAPER-TRADING CANDIDATE" if ready else "FAIL - RESEARCH ONLY")
    print("Live execution remains disabled regardless of this single run.")


if __name__ == "__main__":
    main()
