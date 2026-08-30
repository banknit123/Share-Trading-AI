from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.market_data import download_history
from trading_ai.features.intraday_v2 import add_intraday_features_v2
from trading_ai.features.news_events import headline_features

DB_PATH = Path("data/news_events.sqlite3")
ROUND_TRIP_COST = 0.0012
HORIZONS = {"60m": 12, "120m": 24}

PRICE_FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "vwap_distance",
    "ema_spread_6_18", "ema_spread_18_36", "rsi_14", "range_pct",
    "volatility_12", "volatility_36", "volume_ratio_12", "volume_z_36",
    "price_z_36", "position_36",
]
NEWS_FEATURES = [
    "news_count_60m", "news_count_240m", "news_domains_240m",
    "minutes_since_news", "event_hits_240m", "rss_share_240m",
]


def _to_utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    out.index = idx.tz_convert("UTC")
    return out


def load_events() -> dict[str, pd.DataFrame]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing {DB_PATH}. Run the v9/v9.1 collectors first.")
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT symbol, provider, title, domain, published_at FROM news_events ORDER BY published_at",
            con,
        )
    finally:
        con.close()
    if df.empty:
        raise RuntimeError("News-event database is empty")
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df["event_hits"] = df["title"].fillna("").map(lambda x: headline_features(str(x)).event_hits)
    return {s: g.sort_values("published_at").reset_index(drop=True) for s, g in df.groupby("symbol")}


def add_news_features(price: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    out = price.copy()
    n = len(out)
    if events is None or events.empty:
        for c in NEWS_FEATURES:
            out[c] = 0.0
        out["minutes_since_news"] = 9999.0
        return out

    event_times = events["published_at"].astype("int64").to_numpy()
    domains = events["domain"].fillna("").astype(str).to_numpy()
    providers = events["provider"].fillna("").astype(str).to_numpy()
    hits = events["event_hits"].fillna(0).astype(float).to_numpy()
    bar_ns = out.index.astype("int64").to_numpy()

    count60 = np.zeros(n)
    count240 = np.zeros(n)
    dom240 = np.zeros(n)
    since = np.full(n, 9999.0)
    hit240 = np.zeros(n)
    rss240 = np.zeros(n)

    sixty = int(pd.Timedelta(minutes=60).value)
    twoforty = int(pd.Timedelta(minutes=240).value)

    for i, ts in enumerate(bar_ns):
        # right boundary is strictly all news known by the bar timestamp.
        right = int(np.searchsorted(event_times, ts, side="right"))
        if right <= 0:
            continue
        left60 = int(np.searchsorted(event_times, ts - sixty, side="left"))
        left240 = int(np.searchsorted(event_times, ts - twoforty, side="left"))
        count60[i] = right - left60
        count240[i] = right - left240
        if right > left240:
            d = [x for x in domains[left240:right] if x]
            dom240[i] = len(set(d))
            hit240[i] = float(hits[left240:right].sum())
            p = providers[left240:right]
            rss240[i] = float(np.mean(p == "RSS")) if len(p) else 0.0
        since[i] = max(0.0, (ts - event_times[right - 1]) / 60_000_000_000)

    out["news_count_60m"] = count60
    out["news_count_240m"] = count240
    out["news_domains_240m"] = dom240
    out["minutes_since_news"] = np.minimum(since, 9999.0)
    out["event_hits_240m"] = hit240
    out["rss_share_240m"] = rss240
    return out


def build_panel(events_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for symbol in DEFAULT_CONFIG.universe:
        raw = _to_utc(download_history(symbol, period="60d", interval="5m"))
        f = add_intraday_features_v2(raw)
        f = add_news_features(f, events_by_symbol.get(symbol))
        f["symbol"] = symbol
        f["timestamp"] = f.index
        for name, bars in HORIZONS.items():
            f[f"future_{name}"] = f["Close"].shift(-bars) / f["Close"] - 1.0
        frames.append(f)
        print(
            f"{symbol:15} bars={len(f):5d} events={len(events_by_symbol.get(symbol, [])):4d} "
            f"bars_with_60m_news={(f['news_count_60m'] > 0).sum():5d}"
        )
    panel = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    needed = PRICE_FEATURES + NEWS_FEATURES + [f"future_{h}" for h in HORIZONS]
    return panel.dropna(subset=needed).sort_values(["timestamp", "symbol"])


def split_time(panel: pd.DataFrame):
    times = np.array(sorted(panel["timestamp"].unique()))
    t1 = times[int(len(times) * 0.60)]
    t2 = times[int(len(times) * 0.80)]
    return (
        panel[panel.timestamp < t1].copy(),
        panel[(panel.timestamp >= t1) & (panel.timestamp < t2)].copy(),
        panel[panel.timestamp >= t2].copy(),
    )


def fit(train: pd.DataFrame, features: list[str], target: str):
    y = (train[target] > ROUND_TRIP_COST).astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Training target has only one class")
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(train[features], y)
    return model


def metrics(model, df: pd.DataFrame, features: list[str], target: str):
    p = model.predict_proba(df[features])[:, 1]
    y = (df[target] > ROUND_TRIP_COST).astype(int).to_numpy()
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ret = df[target].to_numpy()
    rank_ic = pd.Series(p).corr(pd.Series(ret), method="spearman")
    cutoff = np.quantile(p, 0.90)
    top = ret[p >= cutoff]
    top_net = top - ROUND_TRIP_COST
    return {
        "auc": float(auc),
        "rank_ic": float(rank_ic),
        "top_n": int(len(top_net)),
        "top_net": float(np.mean(top_net)) if len(top_net) else 0.0,
        "top_win": float(np.mean(top_net > 0)) if len(top_net) else 0.0,
    }


def fmt(label: str, m: dict) -> str:
    return (
        f"{label:22} auc={m['auc']:.3f} rank_ic={m['rank_ic']:+.3f} "
        f"top10_n={m['top_n']:4d} top10_net={m['top_net']:+.3%} top10_win={m['top_win']:.2%}"
    )


def main() -> None:
    print("Share-Trading-AI v10 historical multimodal validation")
    print("No trades are placed. Uses persistent 60-day news history + 5-minute market data.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print("Database:", DB_PATH)

    events = load_events()
    print(f"Stored events loaded: {sum(len(x) for x in events.values())} across {len(events)} symbols\n")
    panel = build_panel(events)
    train, val, test = split_time(panel)
    print(f"\nRows: train={len(train)} validation={len(val)} untouched_test={len(test)}")

    price = PRICE_FEATURES
    multimodal = PRICE_FEATURES + NEWS_FEATURES
    passes = 0

    for horizon in HORIZONS:
        target = f"future_{horizon}"
        pm = fit(train, price, target)
        mm = fit(train, multimodal, target)
        pv = metrics(pm, val, price, target)
        pt = metrics(pm, test, price, target)
        mv = metrics(mm, val, multimodal, target)
        mt = metrics(mm, test, multimodal, target)

        print(f"\n{horizon} horizon")
        print(fmt("PRICE_ONLY VALIDATION", pv))
        print(fmt("PRICE_ONLY TEST", pt))
        print(fmt("PRICE+NEWS VALIDATION", mv))
        print(fmt("PRICE+NEWS TEST", mt))
        print(
            "NEWS LIFT: "
            f"val_auc={mv['auc']-pv['auc']:+.3f} test_auc={mt['auc']-pt['auc']:+.3f} "
            f"val_rank_ic={mv['rank_ic']-pv['rank_ic']:+.3f} test_rank_ic={mt['rank_ic']-pt['rank_ic']:+.3f} "
            f"val_top10_net={mv['top_net']-pv['top_net']:+.3%} test_top10_net={mt['top_net']-pt['top_net']:+.3%}"
        )
        robust = (
            mv["top_net"] > 0
            and mt["top_net"] > 0
            and mv["top_net"] > pv["top_net"]
            and mt["top_net"] > pt["top_net"]
            and mv["rank_ic"] > 0
            and mt["rank_ic"] > 0
        )
        passes += int(robust)
        print("HORIZON GATE:", "PASS FOR NEXT PAPER-RESEARCH STAGE" if robust else "FAIL - KEEP RESEARCH ONLY")

    print("\nV10 overall research gate:")
    if passes >= 1:
        print("AT LEAST ONE HORIZON PASSED THE MULTIMODAL ROBUSTNESS GATE. NEXT STEP: repeated walk-forward validation and paper simulation.")
    else:
        print("NO HORIZON PASSED. Do not move to paper/live execution; redesign features/model rather than tune thresholds.")
    print("Live trading remains disabled regardless of this run.")


if __name__ == "__main__":
    main()
