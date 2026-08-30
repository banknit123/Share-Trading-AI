from __future__ import annotations

import numpy as np
import pandas as pd

from run_candle_sequence_v14 import (
    CANDLE_FEATURES,
    MARKET_FEATURES,
    NEWS_FEATURES,
    W_CANDLE,
    W_MARKET,
    W_NEWS,
    add_sequence_lags,
    directional_score,
    fit_direction_models,
    split_time,
)
from run_candle_dominant_v12 import build_panel
from run_historical_multimodal_v10 import load_events

HORIZONS = {"60m": ("future_60m", 12), "120m": ("future_120m", 24)}
TAILS = [0.01, 0.02, 0.05]
COSTS = [0.0, 0.0003, 0.0006, 0.0009, 0.0012]


def _select_tail(score: np.ndarray, tail: float) -> tuple[np.ndarray, np.ndarray]:
    hi = np.quantile(score, 1.0 - tail)
    lo = np.quantile(score, tail)
    return score >= hi, score <= lo


def _cooldown_mask(df: pd.DataFrame, long_mask: np.ndarray, short_mask: np.ndarray, cooldown_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep only non-overlapping entries per symbol.

    The source panel is 5-minute data. A 12-bar cooldown means a symbol cannot
    open another selected position for 60 minutes; 24 bars means 120 minutes.
    This reduces turnover without using any future information.
    """
    keep_long = np.zeros(len(df), dtype=bool)
    keep_short = np.zeros(len(df), dtype=bool)

    work = df[["symbol", "timestamp"]].copy().reset_index(drop=True)
    candidate = long_mask | short_mask
    work["candidate"] = candidate
    work["is_long"] = long_mask

    for _, group in work.groupby("symbol", sort=False):
        last_pos = -10**9
        for pos in group.index[group["candidate"]]:
            if pos - last_pos < cooldown_bars:
                continue
            if bool(work.at[pos, "is_long"]):
                keep_long[pos] = True
            else:
                keep_short[pos] = True
            last_pos = pos
    return keep_long, keep_short


def evaluate_selection(
    score: np.ndarray,
    df: pd.DataFrame,
    target: str,
    tail: float,
    cost: float,
    cooldown_bars: int,
) -> dict:
    long_mask, short_mask = _select_tail(score, tail)
    if cooldown_bars > 0:
        long_mask, short_mask = _cooldown_mask(df.reset_index(drop=True), long_mask, short_mask, cooldown_bars)

    ret = df[target].to_numpy()
    gross = np.concatenate([ret[long_mask], -ret[short_mask]])
    net = gross - cost
    return {
        "trades": int(len(gross)),
        "gross": float(np.mean(gross)) if len(gross) else 0.0,
        "net": float(np.mean(net)) if len(net) else 0.0,
        "win": float(np.mean(net > 0)) if len(net) else 0.0,
    }


def fmt(m: dict) -> str:
    return f"trades={m['trades']:4d} gross={m['gross']:+.3%} net={m['net']:+.3%} win={m['win']:.2%}"


def main() -> None:
    print("Share-Trading-AI v15 cost / selectivity / turnover audit")
    print("No trades are placed.")
    print("Predictor is unchanged from v14: 70% candle sequence + 20% market + 10% news.")
    print("This script only tests whether stronger selectivity and lower turnover can preserve any gross edge.\n")

    panel = build_panel(load_events())
    panel, sequence_features = add_sequence_lags(panel)
    candle_block = list(dict.fromkeys(CANDLE_FEATURES + sequence_features))
    needed = candle_block + MARKET_FEATURES + NEWS_FEATURES + [x[0] for x in HORIZONS.values()]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    train, val, test = split_time(panel)
    print(f"Rows: train={len(train)} validation={len(val)} untouched_test={len(test)}")

    overall_candidate = False

    for horizon, (target, horizon_bars) in HORIZONS.items():
        candle_models = fit_direction_models(train, candle_block, target)
        market_models = fit_direction_models(train, MARKET_FEATURES, target)
        news_models = fit_direction_models(train, NEWS_FEATURES, target)

        def combined(df: pd.DataFrame) -> np.ndarray:
            c = directional_score(candle_models, df, candle_block)
            m = directional_score(market_models, df, MARKET_FEATURES)
            n = directional_score(news_models, df, NEWS_FEATURES)
            return W_CANDLE * c + W_MARKET * m + W_NEWS * n

        sv = combined(val)
        st = combined(test)

        print(f"\n{horizon} horizon")
        print("Selectivity / cooldown gross-edge table (cost=0):")

        best = None
        for tail in TAILS:
            for cooldown in [0, horizon_bars]:
                label = "none" if cooldown == 0 else f"{horizon} non-overlap"
                vm = evaluate_selection(sv, val, target, tail, 0.0, cooldown)
                tm = evaluate_selection(st, test, target, tail, 0.0, cooldown)
                print(
                    f"  tail={tail:.0%} cooldown={label:16} | VAL {fmt(vm)} | TEST {fmt(tm)}"
                )
                # Configuration choice is based on validation only, and must have a real sample.
                if vm["trades"] >= 40 and vm["gross"] > 0:
                    score = vm["gross"]
                    if best is None or score > best[0]:
                        best = (score, tail, cooldown)

        if best is None:
            print("  No validation configuration had positive gross edge with >=40 trades.")
            continue

        _, tail, cooldown = best
        print(f"\nSelected on VALIDATION only: tail={tail:.0%}, cooldown={'none' if cooldown == 0 else horizon}")
        print("Cost sensitivity on the selected configuration:")
        for cost in COSTS:
            vm = evaluate_selection(sv, val, target, tail, cost, cooldown)
            tm = evaluate_selection(st, test, target, tail, cost, cooldown)
            print(f"  cost={cost:.3%} | VAL {fmt(vm)} | TEST {fmt(tm)}")

        zero_v = evaluate_selection(sv, val, target, tail, 0.0, cooldown)
        zero_t = evaluate_selection(st, test, target, tail, 0.0, cooldown)
        realistic_v = evaluate_selection(sv, val, target, tail, 0.0012, cooldown)
        realistic_t = evaluate_selection(st, test, target, tail, 0.0012, cooldown)

        gross_repeatable = (
            zero_v["trades"] >= 40 and zero_t["trades"] >= 40
            and zero_v["gross"] > 0 and zero_t["gross"] > 0
        )
        net_repeatable = (
            realistic_v["net"] > 0 and realistic_t["net"] > 0
            and realistic_v["win"] > 0.50 and realistic_t["win"] > 0.50
        )
        overall_candidate |= net_repeatable
        print("GROSS EDGE:", "REPEATABLE" if gross_repeatable else "NOT REPEATABLE")
        print("0.12% COST GATE:", "PASS FOR WALK-FORWARD" if net_repeatable else "FAIL - RESEARCH ONLY")

    print("\nV15 conclusion")
    if overall_candidate:
        print("At least one lower-turnover configuration survived the 0.12% cost assumption. Next step is repeated walk-forward validation, not live trading.")
    else:
        print("No configuration survived the current cost assumption robustly. If gross edge is also not repeatable, redesign the predictor rather than assuming cheaper execution will solve it.")
        print("If gross edge is repeatable but net edge fails, the economics of fees/slippage and holding period become the main bottleneck.")


if __name__ == "__main__":
    main()
