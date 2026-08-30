from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from run_candle_dominant_v12 import MARKET_FEATURES, NEWS_FEATURES, build_panel
from run_candle_sequence_v14 import add_sequence_lags
from run_historical_multimodal_v10 import load_events

HORIZONS = {"60m": "future_60m", "120m": "future_120m"}
ROUND_TRIP_COST = 0.0012
W_CANDLE = 0.70
W_MARKET = 0.20
W_NEWS = 0.10


def split_time(panel: pd.DataFrame):
    times = np.array(sorted(panel["timestamp"].unique()))
    t1 = times[int(len(times) * 0.60)]
    t2 = times[int(len(times) * 0.80)]
    return (
        panel[panel.timestamp < t1].copy(),
        panel[(panel.timestamp >= t1) & (panel.timestamp < t2)].copy(),
        panel[panel.timestamp >= t2].copy(),
    )


def fit_regressor(train: pd.DataFrame, features: list[str], target: str):
    model = HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=260,
        max_leaf_nodes=15,
        min_samples_leaf=70,
        l2_regularization=3.0,
        random_state=42,
    )
    model.fit(train[features], train[target])
    return model


def predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(df[features])


def add_excess_targets(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for horizon, target in HORIZONS.items():
        median = out.groupby("timestamp")[target].transform("median")
        out[f"excess_{horizon}"] = out[target] - median
    return out


def cross_sectional_metrics(score: np.ndarray, df: pd.DataFrame, raw_target: str, excess_target: str) -> dict:
    work = df[["timestamp", "symbol", raw_target, excess_target]].copy()
    work["score"] = score

    rank_ics = []
    top_excess = []
    bottom_excess = []
    top_raw = []
    bottom_raw = []
    pair_spread = []
    timestamps = 0

    for _, g in work.groupby("timestamp", sort=False):
        if len(g) < 5:
            continue
        timestamps += 1
        if g["score"].nunique() > 1 and g[excess_target].nunique() > 1:
            ic = g["score"].corr(g[excess_target], method="spearman")
            if pd.notna(ic):
                rank_ics.append(float(ic))

        # Each stock frame retains the timestamp as its DataFrame index, so those
        # index labels are duplicated across symbols. Using idxmax()/loc therefore
        # can return multiple rows. Reset the group to a positional index and use
        # argmax/argmin so exactly one deterministic scalar row is selected.
        gg = g.reset_index(drop=True)
        scores = gg["score"].to_numpy(dtype=float)
        top_pos = int(np.argmax(scores))
        bottom_pos = int(np.argmin(scores))
        top = gg.iloc[top_pos]
        bottom = gg.iloc[bottom_pos]

        top_excess.append(float(top[excess_target]))
        bottom_excess.append(float(bottom[excess_target]))
        top_raw.append(float(top[raw_target]))
        bottom_raw.append(float(bottom[raw_target]))
        pair_spread.append(float(top[raw_target] - bottom[raw_target]))

    top_raw_arr = np.asarray(top_raw, dtype=float)
    bottom_raw_arr = np.asarray(bottom_raw, dtype=float)
    pair_arr = np.asarray(pair_spread, dtype=float)

    # Diagnostic economics only. A top-stock long incurs one assumed round trip.
    # A long-top/short-bottom pair incurs two assumed round trips in total.
    top_net = top_raw_arr - ROUND_TRIP_COST
    pair_net = pair_arr - (2 * ROUND_TRIP_COST)

    return {
        "timestamps": timestamps,
        "rank_ic": float(np.mean(rank_ics)) if rank_ics else float("nan"),
        "top_excess": float(np.mean(top_excess)) if top_excess else 0.0,
        "bottom_excess": float(np.mean(bottom_excess)) if bottom_excess else 0.0,
        "top_raw": float(np.mean(top_raw_arr)) if len(top_raw_arr) else 0.0,
        "top_net": float(np.mean(top_net)) if len(top_net) else 0.0,
        "top_win": float(np.mean(top_net > 0)) if len(top_net) else 0.0,
        "pair_gross": float(np.mean(pair_arr)) if len(pair_arr) else 0.0,
        "pair_net": float(np.mean(pair_net)) if len(pair_net) else 0.0,
        "pair_win": float(np.mean(pair_net > 0)) if len(pair_net) else 0.0,
    }


def fmt(label: str, m: dict) -> str:
    return (
        f"{label:24} timestamps={m['timestamps']:4d} rank_ic={m['rank_ic']:+.3f} "
        f"top_excess={m['top_excess']:+.3%} top_raw={m['top_raw']:+.3%} "
        f"top_net={m['top_net']:+.3%} top_win={m['top_win']:.2%} "
        f"pair_gross={m['pair_gross']:+.3%} pair_net={m['pair_net']:+.3%} pair_win={m['pair_win']:.2%}"
    )


def main() -> None:
    print("Share-Trading-AI v16 cross-sectional candle ranking audit")
    print("No trades are placed.")
    print("Target: rank which liquid NSE stock should outperform its peers over the next 60/120 minutes.")
    print(f"Architecture: {W_CANDLE:.0%} candle sequence/state + {W_MARKET:.0%} market context + {W_NEWS:.0%} news")
    print(f"Economic diagnostic cost: {ROUND_TRIP_COST:.3%} per round trip")

    panel = build_panel(load_events())
    panel, sequence_features = add_sequence_lags(panel)

    # Candle block is deliberately sequence-heavy. Current-state candle features are already
    # embedded in build_panel; the sequence lags/state descriptors are the primary rank signal.
    candle_features = list(dict.fromkeys(sequence_features + [
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
        "vwap_distance", "ema_spread_6_18", "ema_spread_18_36",
        "rsi_14", "volume_ratio_12", "volume_z_36", "position_36",
        "breakout_12", "breakdown_12",
    ]))

    panel = add_excess_targets(panel)
    needed = candle_features + MARKET_FEATURES + NEWS_FEATURES + list(HORIZONS.values()) + [
        f"excess_{h}" for h in HORIZONS
    ]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    if panel.empty:
        raise RuntimeError("No usable rows after cross-sectional target/sequence construction")

    train, val, test = split_time(panel)
    print(
        f"Rows: train={len(train)} validation={len(val)} untouched_test={len(test)} "
        f"candle_features={len(candle_features)}"
    )

    passed = 0
    for horizon, raw_target in HORIZONS.items():
        excess_target = f"excess_{horizon}"

        candle_model = fit_regressor(train, candle_features, excess_target)
        market_model = fit_regressor(train, MARKET_FEATURES, excess_target)
        news_model = fit_regressor(train, NEWS_FEATURES, excess_target)

        cv = predict(candle_model, val, candle_features)
        ct = predict(candle_model, test, candle_features)
        mv = predict(market_model, val, MARKET_FEATURES)
        mt = predict(market_model, test, MARKET_FEATURES)
        nv = predict(news_model, val, NEWS_FEATURES)
        nt = predict(news_model, test, NEWS_FEATURES)

        wv = W_CANDLE * cv + W_MARKET * mv + W_NEWS * nv
        wt = W_CANDLE * ct + W_MARKET * mt + W_NEWS * nt

        candle_val = cross_sectional_metrics(cv, val, raw_target, excess_target)
        candle_test = cross_sectional_metrics(ct, test, raw_target, excess_target)
        weighted_val = cross_sectional_metrics(wv, val, raw_target, excess_target)
        weighted_test = cross_sectional_metrics(wt, test, raw_target, excess_target)

        print(f"\n{horizon} horizon")
        print(fmt("CANDLE-RANK VAL", candle_val))
        print(fmt("CANDLE-RANK TEST", candle_test))
        print(fmt("70/20/10 RANK VAL", weighted_val))
        print(fmt("70/20/10 RANK TEST", weighted_test))
        print(
            "CONTEXT LIFT: "
            f"val_rank_ic={weighted_val['rank_ic']-candle_val['rank_ic']:+.3f} "
            f"test_rank_ic={weighted_test['rank_ic']-candle_test['rank_ic']:+.3f} "
            f"val_top_excess={weighted_val['top_excess']-candle_val['top_excess']:+.3%} "
            f"test_top_excess={weighted_test['top_excess']-candle_test['top_excess']:+.3%}"
        )

        # This is deliberately a research gate, not a live-trading gate. We require the
        # rank relationship and top-stock relative edge to repeat on validation and test.
        robust = (
            weighted_val["rank_ic"] > 0.05
            and weighted_test["rank_ic"] > 0.05
            and weighted_val["top_excess"] > 0
            and weighted_test["top_excess"] > 0
        )
        passed += int(robust)
        print("RANKING GATE:", "PASS FOR EXECUTION-RULE RESEARCH" if robust else "FAIL - REDESIGN RANK SIGNAL")

    print("\nV16 conclusion")
    if passed:
        print("At least one horizon has repeatable cross-sectional ranking value. Next step: validation-selected entry spacing/holding rules and paper simulation.")
    else:
        print("Cross-sectional ranking also failed to show repeatable edge. The next redesign should add genuinely new candle information (multi-timeframe/history) rather than more threshold tuning.")
    print("Live trading remains disabled regardless of this diagnostic.")


if __name__ == "__main__":
    main()
