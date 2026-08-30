from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from run_candle_dominant_v12 import build_panel
from run_candle_sequence_v14 import add_sequence_lags
from run_historical_multimodal_v10 import load_events

TARGET = "future_120m"
EXCESS = "excess_120m"
ROUND_TRIP_COST = 0.0012
N_FOLDS = 5
MIN_TRAIN_FRAC = 0.45


def fit_regressor(train: pd.DataFrame, features: list[str]):
    model = HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=260,
        max_leaf_nodes=15,
        min_samples_leaf=70,
        l2_regularization=3.0,
        random_state=42,
    )
    model.fit(train[features], train[EXCESS])
    return model


def add_excess_target(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    med = out.groupby("timestamp")[TARGET].transform("median")
    out[EXCESS] = out[TARGET] - med
    return out


def make_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out, sequence_features = add_sequence_lags(panel)
    base = [
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
        "vwap_distance", "ema_spread_6_18", "ema_spread_18_36",
        "rsi_14", "volume_ratio_12", "volume_z_36", "position_36",
        "breakout_12", "breakdown_12",
    ]
    features = list(dict.fromkeys(sequence_features + base))
    out = add_excess_target(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [TARGET, EXCESS])
    return out, features


def metrics(score: np.ndarray, df: pd.DataFrame) -> dict:
    work = df[["timestamp", "symbol", TARGET, EXCESS]].copy().reset_index(drop=True)
    work["score"] = score

    ics = []
    top_excess = []
    top_raw = []
    pair_spread = []
    timestamps = 0

    for _, g in work.groupby("timestamp", sort=False):
        if len(g) < 5:
            continue
        timestamps += 1
        if g["score"].nunique() > 1 and g[EXCESS].nunique() > 1:
            ic = g["score"].corr(g[EXCESS], method="spearman")
            if pd.notna(ic):
                ics.append(float(ic))

        gg = g.reset_index(drop=True)
        top = gg.iloc[int(np.argmax(gg["score"].to_numpy()))]
        bottom = gg.iloc[int(np.argmin(gg["score"].to_numpy()))]
        top_excess.append(float(top[EXCESS]))
        top_raw.append(float(top[TARGET]))
        pair_spread.append(float(top[TARGET] - bottom[TARGET]))

    top_raw_arr = np.asarray(top_raw, dtype=float)
    pair_arr = np.asarray(pair_spread, dtype=float)
    return {
        "timestamps": timestamps,
        "rank_ic": float(np.mean(ics)) if ics else float("nan"),
        "top_excess": float(np.mean(top_excess)) if top_excess else 0.0,
        "top_raw": float(np.mean(top_raw_arr)) if len(top_raw_arr) else 0.0,
        "top_net": float(np.mean(top_raw_arr - ROUND_TRIP_COST)) if len(top_raw_arr) else 0.0,
        "pair_gross": float(np.mean(pair_arr)) if len(pair_arr) else 0.0,
        "pair_net": float(np.mean(pair_arr - 2 * ROUND_TRIP_COST)) if len(pair_arr) else 0.0,
    }


def fmt(m: dict) -> str:
    return (
        f"timestamps={m['timestamps']:4d} rank_ic={m['rank_ic']:+.3f} "
        f"top_excess={m['top_excess']:+.3%} top_raw={m['top_raw']:+.3%} "
        f"top_net={m['top_net']:+.3%} pair_gross={m['pair_gross']:+.3%} pair_net={m['pair_net']:+.3%}"
    )


def main() -> None:
    print("Share-Trading-AI v18 rolling walk-forward candle-rank stability audit")
    print("No trades are placed.")
    print("Frozen architecture: v16-style 120-minute candle-sequence rank model only.")
    print("Reason: repeated redesigns have already looked at the prior holdout; this tests time stability instead of adding features.")

    panel = build_panel(load_events())
    panel, features = make_features(panel)
    times = np.array(sorted(panel["timestamp"].unique()))
    n = len(times)
    start = int(n * MIN_TRAIN_FRAC)
    remaining = n - start
    fold_size = remaining // N_FOLDS
    if fold_size < 20:
        raise RuntimeError(f"Too few timestamps for {N_FOLDS} walk-forward folds: n={n}")

    print(f"Rows={len(panel)} timestamps={n} features={len(features)} folds={N_FOLDS}")
    fold_results = []

    for i in range(N_FOLDS):
        train_end = start + i * fold_size
        test_start = train_end
        test_end = n if i == N_FOLDS - 1 else min(n, test_start + fold_size)
        train_times = times[:train_end]
        test_times = times[test_start:test_end]
        if len(test_times) == 0:
            continue

        train = panel[panel["timestamp"].isin(train_times)].copy()
        test = panel[panel["timestamp"].isin(test_times)].copy()
        model = fit_regressor(train, features)
        score = model.predict(test[features])
        m = metrics(score, test)
        fold_results.append(m)
        print(f"FOLD {i+1}: train_ts={len(train_times):4d} test_ts={len(test_times):4d} | {fmt(m)}")

    if not fold_results:
        raise RuntimeError("No walk-forward folds were evaluated")

    rank_ics = np.array([m["rank_ic"] for m in fold_results], dtype=float)
    top_excess = np.array([m["top_excess"] for m in fold_results], dtype=float)
    top_net = np.array([m["top_net"] for m in fold_results], dtype=float)
    pair_net = np.array([m["pair_net"] for m in fold_results], dtype=float)

    print("\nV18 stability summary")
    print(f"median_rank_ic={np.nanmedian(rank_ics):+.3f} positive_rank_folds={np.mean(rank_ics>0):.0%}")
    print(f"median_top_excess={np.median(top_excess):+.3%} positive_top_excess_folds={np.mean(top_excess>0):.0%}")
    print(f"median_top_net={np.median(top_net):+.3%} positive_top_net_folds={np.mean(top_net>0):.0%}")
    print(f"median_pair_net={np.median(pair_net):+.3%} positive_pair_net_folds={np.mean(pair_net>0):.0%}")

    stable_rank = np.nanmedian(rank_ics) > 0.03 and np.mean(rank_ics > 0) >= 0.80
    stable_excess = np.median(top_excess) > 0 and np.mean(top_excess > 0) >= 0.80

    if stable_rank and stable_excess:
        print("Research gate: PASS FOR FRESH-FORWARD PAPER OBSERVATION")
        print("Do not treat this as live-trading approval; the next evidence must come from data not used in any prior redesign.")
    else:
        print("Research gate: FAIL - RANK SIGNAL NOT STABLE ENOUGH ACROSS TIME")
        print("Do not add more features based on the old test set. Either collect genuinely new forward data or obtain a deeper historical intraday source.")


if __name__ == "__main__":
    main()
