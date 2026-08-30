from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from run_candle_dominant_v12 import MARKET_FEATURES, NEWS_FEATURES, build_panel
from run_candle_sequence_v14 import add_sequence_lags
from run_historical_multimodal_v10 import load_events

TARGET = "future_120m"
EXCESS = "excess_120m"
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


def add_history_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = panel.copy().sort_values(["symbol", "timestamp"])
    feats: list[str] = []

    def by_symbol(col: str):
        return out.groupby("symbol", sort=False)[col]

    # Multi-timeframe trend / volatility state derived only from information known at each bar.
    for bars, label in [(3, "15m"), (6, "30m"), (12, "60m"), (24, "120m"), (36, "180m"), (48, "240m"), (72, "360m")]:
        ret_name = f"hist_ret_{label}"
        vol_name = f"hist_vol_{label}"
        pos_name = f"hist_posshare_{label}"
        out[ret_name] = by_symbol("Close").pct_change(bars)
        out[vol_name] = by_symbol("ret_1").transform(lambda s, b=bars: s.rolling(b).std())
        out[pos_name] = by_symbol("ret_1").transform(lambda s, b=bars: (s > 0).rolling(b).mean())
        feats += [ret_name, vol_name, pos_name]

    # Momentum acceleration / persistence across timeframes.
    out["hist_accel_15_60"] = out["hist_ret_15m"] - out["hist_ret_60m"] / 4.0
    out["hist_accel_30_120"] = out["hist_ret_30m"] - out["hist_ret_120m"] / 4.0
    out["hist_trend_60_240"] = out["hist_ret_60m"] - out["hist_ret_240m"] / 4.0
    feats += ["hist_accel_15_60", "hist_accel_30_120", "hist_trend_60_240"]

    # Session state and previous-session context.
    local_ts = pd.to_datetime(out["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    out["session_date"] = local_ts.dt.date
    out["session_minute"] = local_ts.dt.hour * 60 + local_ts.dt.minute

    gday = out.groupby(["symbol", "session_date"], sort=False)
    session_open = gday["Open"].transform("first")
    session_high_so_far = gday["High"].cummax()
    session_low_so_far = gday["Low"].cummin()
    out["session_return"] = out["Close"] / session_open - 1.0
    out["session_range_position"] = (out["Close"] - session_low_so_far) / (session_high_so_far - session_low_so_far).replace(0, np.nan)
    out["session_minutes_from_open"] = (out["session_minute"] - (9 * 60 + 15)).clip(lower=0)
    feats += ["session_return", "session_range_position", "session_minutes_from_open"]

    # Build daily summaries, shifted so only the previous completed session is used.
    daily = out.groupby(["symbol", "session_date"], as_index=False).agg(
        day_open=("Open", "first"),
        day_close=("Close", "last"),
        day_high=("High", "max"),
        day_low=("Low", "min"),
        day_volume=("Volume", "sum"),
    )
    daily["prev_close"] = daily.groupby("symbol")["day_close"].shift(1)
    daily["prev_day_return"] = daily["day_close"] / daily["day_open"] - 1.0
    daily["prev_day_return"] = daily.groupby("symbol")["prev_day_return"].shift(1)
    daily["prev_day_range"] = (daily["day_high"] - daily["day_low"]) / daily["day_close"].replace(0, np.nan)
    daily["prev_day_range"] = daily.groupby("symbol")["prev_day_range"].shift(1)
    daily["prev_day_volume"] = daily.groupby("symbol")["day_volume"].shift(1)
    daily = daily[["symbol", "session_date", "prev_close", "prev_day_return", "prev_day_range", "prev_day_volume"]]
    out = out.merge(daily, on=["symbol", "session_date"], how="left", sort=False)
    out["open_gap"] = out["Open"] / out["prev_close"] - 1.0
    out["volume_vs_prev_day"] = out["Volume"] / out["prev_day_volume"].replace(0, np.nan)
    feats += ["open_gap", "prev_day_return", "prev_day_range", "volume_vs_prev_day"]

    return out.sort_values(["timestamp", "symbol"]), feats


def add_excess_target(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    med = out.groupby("timestamp")[TARGET].transform("median")
    out[EXCESS] = out[TARGET] - med
    return out


def fit_regressor(train: pd.DataFrame, features: list[str]):
    model = HistGradientBoostingRegressor(
        learning_rate=0.03,
        max_iter=320,
        max_leaf_nodes=15,
        min_samples_leaf=70,
        l2_regularization=4.0,
        random_state=42,
    )
    model.fit(train[features], train[EXCESS])
    return model


def metrics(score: np.ndarray, df: pd.DataFrame) -> dict:
    work = df[["timestamp", "symbol", TARGET, EXCESS]].copy()
    work["score"] = score
    ics, top_excess, top_raw, pair = [], [], [], []

    for _, g in work.groupby("timestamp", sort=False):
        if len(g) < 5:
            continue
        gg = g.reset_index(drop=True)
        if gg["score"].nunique() > 1 and gg[EXCESS].nunique() > 1:
            ic = gg["score"].corr(gg[EXCESS], method="spearman")
            if pd.notna(ic):
                ics.append(float(ic))
        top = gg.iloc[int(np.argmax(gg["score"].to_numpy()))]
        bottom = gg.iloc[int(np.argmin(gg["score"].to_numpy()))]
        top_excess.append(float(top[EXCESS]))
        top_raw.append(float(top[TARGET]))
        pair.append(float(top[TARGET] - bottom[TARGET]))

    tr = np.asarray(top_raw, dtype=float)
    pr = np.asarray(pair, dtype=float)
    return {
        "timestamps": len(top_raw),
        "rank_ic": float(np.mean(ics)) if ics else float("nan"),
        "top_excess": float(np.mean(top_excess)) if top_excess else 0.0,
        "top_raw": float(np.mean(tr)) if len(tr) else 0.0,
        "top_net": float(np.mean(tr - ROUND_TRIP_COST)) if len(tr) else 0.0,
        "pair_gross": float(np.mean(pr)) if len(pr) else 0.0,
        "pair_net": float(np.mean(pr - 2 * ROUND_TRIP_COST)) if len(pr) else 0.0,
    }


def fmt(label: str, m: dict) -> str:
    return (
        f"{label:28} timestamps={m['timestamps']:4d} rank_ic={m['rank_ic']:+.3f} "
        f"top_excess={m['top_excess']:+.3%} top_raw={m['top_raw']:+.3%} "
        f"top_net={m['top_net']:+.3%} pair_gross={m['pair_gross']:+.3%} pair_net={m['pair_net']:+.3%}"
    )


def main() -> None:
    print("Share-Trading-AI v17 multi-timeframe candle ranking audit")
    print("No trades are placed.")
    print("Focus: 120-minute relative-strength ranking because v16 showed positive validation/test rank IC there.")
    print("New information: 15m/30m/60m/2h/3h/4h/6h candle state + session/open-gap + prior-session context.")

    panel = build_panel(load_events())
    panel, seq_features = add_sequence_lags(panel)
    panel, history_features = add_history_features(panel)
    panel = add_excess_target(panel)

    base_candle = list(dict.fromkeys(seq_features + [
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "vwap_distance",
        "ema_spread_6_18", "ema_spread_18_36", "rsi_14", "volume_ratio_12",
        "volume_z_36", "position_36", "breakout_12", "breakdown_12",
    ]))
    mtf_candle = list(dict.fromkeys(base_candle + history_features))
    needed = mtf_candle + MARKET_FEATURES + NEWS_FEATURES + [TARGET, EXCESS]
    panel = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=needed)
    if panel.empty:
        raise RuntimeError("No usable rows after multi-timeframe/history feature construction")

    train, val, test = split_time(panel)
    print(f"Rows: train={len(train)} validation={len(val)} untouched_test={len(test)} base_features={len(base_candle)} mtf_features={len(mtf_candle)}")

    base_model = fit_regressor(train, base_candle)
    mtf_model = fit_regressor(train, mtf_candle)
    market_model = fit_regressor(train, MARKET_FEATURES)
    news_model = fit_regressor(train, NEWS_FEATURES)

    bv = base_model.predict(val[base_candle]); bt = base_model.predict(test[base_candle])
    mv = mtf_model.predict(val[mtf_candle]); mt = mtf_model.predict(test[mtf_candle])
    mkv = market_model.predict(val[MARKET_FEATURES]); mkt = market_model.predict(test[MARKET_FEATURES])
    nv = news_model.predict(val[NEWS_FEATURES]); nt = news_model.predict(test[NEWS_FEATURES])

    wv = W_CANDLE * mv + W_MARKET * mkv + W_NEWS * nv
    wt = W_CANDLE * mt + W_MARKET * mkt + W_NEWS * nt

    base_val = metrics(bv, val); base_test = metrics(bt, test)
    mtf_val = metrics(mv, val); mtf_test = metrics(mt, test)
    weighted_val = metrics(wv, val); weighted_test = metrics(wt, test)

    print("\n120m horizon")
    print(fmt("V16-STYLE BASE CANDLE VAL", base_val))
    print(fmt("V16-STYLE BASE CANDLE TEST", base_test))
    print(fmt("MULTI-TIMEFRAME CANDLE VAL", mtf_val))
    print(fmt("MULTI-TIMEFRAME CANDLE TEST", mtf_test))
    print(fmt("70/20/10 MULTI-TF VAL", weighted_val))
    print(fmt("70/20/10 MULTI-TF TEST", weighted_test))

    print(
        "\nMULTI-TIMEFRAME LIFT: "
        f"val_rank_ic={mtf_val['rank_ic']-base_val['rank_ic']:+.3f} "
        f"test_rank_ic={mtf_test['rank_ic']-base_test['rank_ic']:+.3f} "
        f"val_top_excess={mtf_val['top_excess']-base_val['top_excess']:+.3%} "
        f"test_top_excess={mtf_test['top_excess']-base_test['top_excess']:+.3%}"
    )

    robust = (
        mtf_val["rank_ic"] > 0.05 and mtf_test["rank_ic"] > 0.05
        and mtf_val["top_excess"] > 0 and mtf_test["top_excess"] > 0
    )
    print("\nV17 research gate:", "PASS FOR REPEATED WALK-FORWARD RANKING TEST" if robust else "FAIL - MULTI-TIMEFRAME EDGE NOT YET ROBUST")
    print("Live trading remains disabled regardless of this single diagnostic.")


if __name__ == "__main__":
    main()
