from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from run_candle_dominant_v12 import (
    CANDLE_FEATURES,
    MARKET_FEATURES,
    NEWS_FEATURES,
    build_panel,
)
from run_historical_multimodal_v10 import load_events

ROUND_TRIP_COST = 0.0012
HORIZONS = {"60m": "future_60m", "120m": "future_120m"}
W_CANDLE = 0.70
W_MARKET = 0.20
W_NEWS = 0.10

# Raw recent-candle state that is explicitly unfolded into the last 12 bars.
SEQUENCE_BASE = [
    "ret_1",
    "body_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_location",
    "range_pct",
    "volume_ratio_6",
]
SEQUENCE_LAGS = 12


def add_sequence_lags(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy().sort_values(["symbol", "timestamp"])
    seq_features: list[str] = []

    for col in SEQUENCE_BASE:
        for lag in range(SEQUENCE_LAGS):
            name = f"seq_{col}_lag{lag}"
            out[name] = out.groupby("symbol", sort=False)[col].shift(lag)
            seq_features.append(name)

    # Compact state descriptors extracted from the sequence itself.
    grouped_ret = out.groupby("symbol", sort=False)["ret_1"]
    out["seq_pos_share_6"] = grouped_ret.transform(lambda s: (s > 0).rolling(6).mean())
    out["seq_pos_share_12"] = grouped_ret.transform(lambda s: (s > 0).rolling(12).mean())
    out["seq_ret_mean_6"] = grouped_ret.transform(lambda s: s.rolling(6).mean())
    out["seq_ret_mean_12"] = grouped_ret.transform(lambda s: s.rolling(12).mean())
    out["seq_ret_std_6"] = grouped_ret.transform(lambda s: s.rolling(6).std())
    out["seq_ret_std_12"] = grouped_ret.transform(lambda s: s.rolling(12).std())
    out["seq_accel_3_12"] = out.groupby("symbol", sort=False)["ret_3"].transform(lambda s: s - s.shift(9))

    state = [
        "seq_pos_share_6", "seq_pos_share_12", "seq_ret_mean_6", "seq_ret_mean_12",
        "seq_ret_std_6", "seq_ret_std_12", "seq_accel_3_12",
    ]
    seq_features.extend(state)
    return out, seq_features


def split_time(panel: pd.DataFrame):
    times = np.array(sorted(panel["timestamp"].unique()))
    t1 = times[int(len(times) * 0.60)]
    t2 = times[int(len(times) * 0.80)]
    return (
        panel[panel.timestamp < t1].copy(),
        panel[(panel.timestamp >= t1) & (panel.timestamp < t2)].copy(),
        panel[panel.timestamp >= t2].copy(),
    )


def fit_direction_models(train: pd.DataFrame, features: list[str], target: str):
    ret = train[target]
    y_long = (ret > ROUND_TRIP_COST).astype(int)
    y_short = (ret < -ROUND_TRIP_COST).astype(int)

    kwargs = dict(
        learning_rate=0.035,
        max_iter=260,
        max_leaf_nodes=15,
        min_samples_leaf=70,
        l2_regularization=3.0,
        random_state=42,
    )
    long_model = HistGradientBoostingClassifier(**kwargs).fit(train[features], y_long)
    short_model = HistGradientBoostingClassifier(**kwargs).fit(train[features], y_short)
    return long_model, short_model


def directional_score(models, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    long_model, short_model = models
    p_long = long_model.predict_proba(df[features])[:, 1]
    p_short = short_model.predict_proba(df[features])[:, 1]
    return p_long - p_short


def evaluate(score: np.ndarray, df: pd.DataFrame, target: str, tail: float = 0.05) -> dict:
    ret = df[target].to_numpy()
    hi = np.quantile(score, 1.0 - tail)
    lo = np.quantile(score, tail)

    long_mask = score >= hi
    short_mask = score <= lo

    long_gross = ret[long_mask]
    short_gross = -ret[short_mask]
    gross = np.concatenate([long_gross, short_gross])
    net = gross - ROUND_TRIP_COST

    # Directional AUC uses up-vs-down excluding small moves only for diagnostic value.
    decisive = np.abs(ret) > ROUND_TRIP_COST
    if decisive.sum() > 10 and len(np.unique((ret[decisive] > 0).astype(int))) > 1:
        auc = roc_auc_score((ret[decisive] > 0).astype(int), score[decisive])
    else:
        auc = float("nan")

    rank_ic = pd.Series(score).corr(pd.Series(ret), method="spearman")
    return {
        "auc": float(auc),
        "rank_ic": float(rank_ic),
        "trades": int(len(net)),
        "long_n": int(long_mask.sum()),
        "short_n": int(short_mask.sum()),
        "gross": float(np.mean(gross)) if len(gross) else 0.0,
        "net": float(np.mean(net)) if len(net) else 0.0,
        "win": float(np.mean(net > 0)) if len(net) else 0.0,
    }


def fmt(label: str, m: dict) -> str:
    return (
        f"{label:24} auc={m['auc']:.3f} rank_ic={m['rank_ic']:+.3f} "
        f"trades={m['trades']:4d} (L={m['long_n']:3d}/S={m['short_n']:3d}) "
        f"gross={m['gross']:+.3%} net={m['net']:+.3%} win={m['win']:.2%}"
    )


def main() -> None:
    print("Share-Trading-AI v14 candle sequence/state directional audit")
    print("No trades are placed.")
    print("Primary signal: last 12 candle sequence + candle state; market/news remain secondary.")
    print(f"Architecture: {W_CANDLE:.0%} candle-sequence + {W_MARKET:.0%} market + {W_NEWS:.0%} news")
    print(f"Primary cost assumption: {ROUND_TRIP_COST:.3%} round trip")

    panel = build_panel(load_events())
    panel, sequence_features = add_sequence_lags(panel)

    candle_block = list(dict.fromkeys(CANDLE_FEATURES + sequence_features))
    needed = candle_block + MARKET_FEATURES + NEWS_FEATURES + list(HORIZONS.values())
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)

    if panel.empty:
        raise RuntimeError("No usable rows after 12-candle sequence construction")

    train, val, test = split_time(panel)
    print(
        f"Rows: train={len(train)} validation={len(val)} untouched_test={len(test)} "
        f"candle_features={len(candle_block)}"
    )

    passed = 0
    for horizon, target in HORIZONS.items():
        candle_models = fit_direction_models(train, candle_block, target)
        market_models = fit_direction_models(train, MARKET_FEATURES, target)
        news_models = fit_direction_models(train, NEWS_FEATURES, target)

        cv = directional_score(candle_models, val, candle_block)
        ct = directional_score(candle_models, test, candle_block)
        mv = directional_score(market_models, val, MARKET_FEATURES)
        mt = directional_score(market_models, test, MARKET_FEATURES)
        nv = directional_score(news_models, val, NEWS_FEATURES)
        nt = directional_score(news_models, test, NEWS_FEATURES)

        weighted_v = W_CANDLE * cv + W_MARKET * mv + W_NEWS * nv
        weighted_t = W_CANDLE * ct + W_MARKET * mt + W_NEWS * nt

        candle_val = evaluate(cv, val, target)
        candle_test = evaluate(ct, test, target)
        weighted_val = evaluate(weighted_v, val, target)
        weighted_test = evaluate(weighted_t, test, target)

        print(f"\n{horizon} horizon - strongest 5% long + strongest 5% short")
        print(fmt("SEQUENCE_ONLY VAL", candle_val))
        print(fmt("SEQUENCE_ONLY TEST", candle_test))
        print(fmt("70/20/10 VAL", weighted_val))
        print(fmt("70/20/10 TEST", weighted_test))
        print(
            "CONTEXT LIFT: "
            f"val_net={weighted_val['net']-candle_val['net']:+.3%} "
            f"test_net={weighted_test['net']-candle_test['net']:+.3%} "
            f"val_rank_ic={weighted_val['rank_ic']-candle_val['rank_ic']:+.3f} "
            f"test_rank_ic={weighted_test['rank_ic']-candle_test['rank_ic']:+.3f}"
        )

        robust = (
            weighted_val["trades"] >= 100
            and weighted_test["trades"] >= 100
            and weighted_val["net"] > 0
            and weighted_test["net"] > 0
            and weighted_val["win"] > 0.50
            and weighted_test["win"] > 0.50
            and weighted_val["rank_ic"] > 0
            and weighted_test["rank_ic"] > 0
        )
        passed += int(robust)
        print("HORIZON GATE:", "PASS FOR WALK-FORWARD REPEAT" if robust else "FAIL - RESEARCH ONLY")

    print("\nV14 conclusion")
    if passed:
        print("At least one sequence/state horizon passed this split. Next step is repeated walk-forward testing before any paper execution.")
    else:
        print("Sequence/state modelling still did not clear the robustness gate. Do not increase news weight or enable live trading.")
        print("If gross edge is positive but net edge is negative, the next diagnostic should isolate cost/slippage sensitivity and lower-turnover holding rules.")


if __name__ == "__main__":
    main()
