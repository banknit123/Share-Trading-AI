from __future__ import annotations

import numpy as np
import pandas as pd

from run_candle_dominant_v12 import build_panel, split_time, load_events

ROUND_TRIP_COST = 0.0012
HORIZONS = ("60m", "120m")
MIN_OBS = 30


def regimes(df: pd.DataFrame) -> dict[str, tuple[pd.Series, int]]:
    """Predefined, interpretable candle regimes.

    tuple = (mask, direction), where direction is +1 for long and -1 for short.
    These are diagnostics, not trading rules approved for execution.
    """
    return {
        "momentum_long": (
            (df.ret_6 > 0)
            & (df.ret_12 > 0)
            & (df.ema_spread_6_36 > 0)
            & (df.vwap_distance > 0)
            & (df.bullish_6 >= 2 / 3),
            +1,
        ),
        "breakout_volume_long": (
            (df.breakout_12 > 0)
            & (df.volume_ratio_6 >= 1.20)
            & (df.close_location >= 0.70),
            +1,
        ),
        "pullback_uptrend_long": (
            (df.ema_spread_6_36 > 0)
            & (df.ema18_slope_6 > 0)
            & (df.ret_3 < 0)
            & (df.ret_12 > 0)
            & (df.close_location >= 0.50),
            +1,
        ),
        "oversold_reversal_long": (
            (df.rsi_14 <= 35)
            & (df.price_z_36 <= -1.0)
            & (df.lower_wick_pct > df.upper_wick_pct)
            & (df.close_location >= 0.55),
            +1,
        ),
        "momentum_short": (
            (df.ret_6 < 0)
            & (df.ret_12 < 0)
            & (df.ema_spread_6_36 < 0)
            & (df.vwap_distance < 0)
            & (df.bullish_6 <= 1 / 3),
            -1,
        ),
        "breakdown_volume_short": (
            (df.breakdown_12 < 0)
            & (df.volume_ratio_6 >= 1.20)
            & (df.close_location <= 0.30),
            -1,
        ),
        "overbought_reversal_short": (
            (df.rsi_14 >= 65)
            & (df.price_z_36 >= 1.0)
            & (df.upper_wick_pct > df.lower_wick_pct)
            & (df.close_location <= 0.45),
            -1,
        ),
    }


def stats(df: pd.DataFrame, mask: pd.Series, direction: int, horizon: str) -> dict:
    r = direction * df.loc[mask, f"future_{horizon}"].dropna()
    if len(r) == 0:
        return {"n": 0, "gross": np.nan, "net": np.nan, "win": np.nan, "median": np.nan}
    net = r - ROUND_TRIP_COST
    return {
        "n": int(len(net)),
        "gross": float(r.mean()),
        "net": float(net.mean()),
        "win": float((net > 0).mean()),
        "median": float(net.median()),
    }


def fmt(m: dict) -> str:
    if m["n"] == 0:
        return "n=   0"
    return (
        f"n={m['n']:4d} gross={m['gross']:+.3%} net={m['net']:+.3%} "
        f"median_net={m['median']:+.3%} win={m['win']:.2%}"
    )


def main() -> None:
    print("Share-Trading-AI v13 candle-regime audit")
    print("No trades are placed. This identifies which past-candle regimes, if any, have repeatable forward-return value.")
    print(f"Assumed round-trip cost: {ROUND_TRIP_COST:.3%}")

    panel = build_panel(load_events())
    train, val, test = split_time(panel)
    print(f"Rows: train={len(train)} validation={len(val)} untouched_test={len(test)}")

    val_regimes = regimes(val)
    test_regimes = regimes(test)
    stable = []

    for horizon in HORIZONS:
        print(f"\n{horizon} horizon")
        for name in val_regimes:
            vm, direction = val_regimes[name]
            tm, _ = test_regimes[name]
            vs = stats(val, vm, direction, horizon)
            ts = stats(test, tm, direction, horizon)
            ok = (
                vs["n"] >= MIN_OBS
                and ts["n"] >= MIN_OBS
                and vs["net"] > 0
                and ts["net"] > 0
                and vs["win"] > 0.50
                and ts["win"] > 0.50
            )
            if ok:
                stable.append((horizon, name, vs, ts))
            print(f"  {name:28} VALIDATION {fmt(vs)}")
            print(f"  {'':28} TEST       {fmt(ts)} {'<-- CANDIDATE' if ok else ''}")

    print("\nV13 candle-regime conclusion")
    if stable:
        print(f"Stable positive regimes found: {len(stable)}")
        for horizon, name, vs, ts in stable:
            print(
                f"  {horizon} {name}: val_net={vs['net']:+.3%} test_net={ts['net']:+.3%} "
                f"val_win={vs['win']:.2%} test_win={ts['win']:.2%}"
            )
        print("These are research candidates for repeated walk-forward testing, not permission for live trading.")
    else:
        print("No predefined candle regime was positive and repeatable across validation + untouched test after costs.")
        print("If this occurs, the next redesign should focus on richer candle sequence/state modelling rather than increasing news weight.")


if __name__ == "__main__":
    main()
