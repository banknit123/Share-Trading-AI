from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.gdelt_news import search_articles
from trading_ai.data.market_data import download_history
from trading_ai.data.rss_news import search_google_news_rss
from trading_ai.features.news_events import headline_features
from run_news_coverage_audit_v6_1 import QUERY_MAP

ROUND_TRIP_COST = 0.0012
HORIZON_BARS = {"60m": 12, "120m": 24}

PRICE_FEATURES = [
    "ret_3", "ret_6", "ret_12", "range_pct", "volume_ratio_12",
]
NEWS_FEATURES = [
    "news_count_60m", "news_count_240m", "source_diversity_240m",
    "minutes_since_prior_news", "event_hits", "is_gdelt",
]


def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    idx = idx.tz_convert("UTC")
    out.index = idx
    return out


def _fetch_news(query: str):
    try:
        articles = search_articles(query, timespan="7d", maxrecords=40)
        if articles:
            return "GDELT", [
                (a.title, a.domain or "", a.seen_at) for a in articles if a.seen_at is not None
            ]
    except Exception:
        pass
    time.sleep(0.2)
    articles = search_google_news_rss(query, maxrecords=60, timeout=15.0)
    return "RSS", [
        (a.title, a.domain or "", a.published_at) for a in articles if a.published_at is not None
    ]


def _align_event(price: pd.DataFrame, event_time: datetime) -> int | None:
    ts = pd.Timestamp(event_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert("UTC")
    pos = price.index.searchsorted(ts, side="left")
    if pos >= len(price):
        return None
    return int(pos)


def _price_snapshot(price: pd.DataFrame, pos: int) -> dict[str, float] | None:
    if pos < 20:
        return None
    close = price["Close"].astype(float)
    high = price["High"].astype(float)
    low = price["Low"].astype(float)
    volume = price["Volume"].astype(float)

    c = float(close.iloc[pos])
    if c == 0:
        return None
    vol_mean = float(volume.iloc[pos-11:pos+1].mean())
    return {
        "ret_3": float(c / close.iloc[pos-3] - 1.0),
        "ret_6": float(c / close.iloc[pos-6] - 1.0),
        "ret_12": float(c / close.iloc[pos-12] - 1.0),
        "range_pct": float((high.iloc[pos] - low.iloc[pos]) / c),
        "volume_ratio_12": float(volume.iloc[pos] / vol_mean) if vol_mean > 0 else 1.0,
    }


def _build_rows() -> pd.DataFrame:
    rows: list[dict] = []

    for symbol in DEFAULT_CONFIG.universe:
        try:
            price = _to_utc_index(download_history(symbol, period="60d", interval="5m"))
            provider, raw_articles = _fetch_news(QUERY_MAP[symbol])

            dedup: dict[str, tuple[str, str, datetime]] = {}
            for title, domain, published_at in raw_articles:
                if published_at is None:
                    continue
                key = (title or "").strip().lower()
                if not key:
                    continue
                dedup[key] = (title, domain, published_at)

            articles = sorted(dedup.values(), key=lambda x: x[2])
            history: list[tuple[pd.Timestamp, str]] = []
            usable = 0

            for title, domain, published_at in articles:
                event_ts = pd.Timestamp(published_at)
                if event_ts.tzinfo is None:
                    event_ts = event_ts.tz_localize("UTC")
                else:
                    event_ts = event_ts.tz_convert("UTC")

                pos = _align_event(price, published_at)
                if pos is None or pos >= len(price) - max(HORIZON_BARS.values()) - 1:
                    continue
                snap = _price_snapshot(price, pos)
                if snap is None:
                    continue

                prior_60 = [(ts, d) for ts, d in history if pd.Timedelta(0) <= event_ts - ts <= pd.Timedelta(minutes=60)]
                prior_240 = [(ts, d) for ts, d in history if pd.Timedelta(0) <= event_ts - ts <= pd.Timedelta(minutes=240)]
                if history:
                    mins_prior = max(0.0, (event_ts - history[-1][0]).total_seconds() / 60.0)
                else:
                    mins_prior = 1440.0

                hf = headline_features(title)
                record = {
                    "symbol": symbol,
                    "event_time": event_ts,
                    "provider": provider,
                    "news_count_60m": float(len(prior_60) + 1),
                    "news_count_240m": float(len(prior_240) + 1),
                    "source_diversity_240m": float(len({d for _, d in prior_240 if d} | ({domain} if domain else set()))),
                    "minutes_since_prior_news": float(min(mins_prior, 1440.0)),
                    "event_hits": float(hf.event_hits),
                    "is_gdelt": 1.0 if provider == "GDELT" else 0.0,
                    **snap,
                }
                entry = float(price["Close"].iloc[pos])
                for name, bars in HORIZON_BARS.items():
                    record[f"future_return_{name}"] = float(price["Close"].iloc[pos + bars] / entry - 1.0)
                rows.append(record)
                history.append((event_ts, domain))
                usable += 1

            print(f"{symbol:15} provider={provider:5} aligned_rows={usable:3d}")
        except Exception as exc:
            print(f"{symbol:15} ERROR: {exc}")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("event_time").reset_index(drop=True)


def _split(df: pd.DataFrame):
    n = len(df)
    a = int(n * 0.60)
    b = int(n * 0.80)
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()


def _fit(train: pd.DataFrame, features: list[str], target: str):
    y = (train[target] > ROUND_TRIP_COST).astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Training target has only one class")
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=15,
        min_samples_leaf=15,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(train[features], y)
    return model


def _evaluate(model, df: pd.DataFrame, features: list[str], target: str) -> dict[str, float]:
    y = (df[target] > ROUND_TRIP_COST).astype(int)
    p = model.predict_proba(df[features])[:, 1]
    auc = float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan")

    scored = df[[target]].copy()
    scored["p"] = p
    scored["net"] = scored[target] - ROUND_TRIP_COST
    cutoff = scored["p"].quantile(0.80)
    top = scored[scored["p"] >= cutoff]
    return {
        "auc": auc,
        "avg_net": float(scored["net"].mean()),
        "top20_net": float(top["net"].mean()) if len(top) else 0.0,
        "top20_win": float((top["net"] > 0).mean()) if len(top) else 0.0,
        "top20_count": float(len(top)),
    }


def main() -> None:
    print("Share-Trading-AI v8 multimodal event-intensity audit")
    print("No trades are placed. Polarity is excluded; only event intensity/recency/source diversity are tested.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)

    df = _build_rows()
    if len(df) < 120:
        print(f"\nInsufficient aligned sample: {len(df)} rows. Research gate: FAIL - SAMPLE TOO SMALL")
        return

    train, val, test = _split(df)
    print(f"\nRows: total={len(df)} train={len(train)} validation={len(val)} test={len(test)} stocks={df.symbol.nunique()}")

    for horizon in HORIZON_BARS:
        target = f"future_return_{horizon}"
        print(f"\n{horizon} horizon")

        price_model = _fit(train, PRICE_FEATURES, target)
        multi_features = PRICE_FEATURES + NEWS_FEATURES
        multi_model = _fit(train, multi_features, target)

        for label, model, feats in [
            ("PRICE_ONLY", price_model, PRICE_FEATURES),
            ("PRICE+NEWS", multi_model, multi_features),
        ]:
            v = _evaluate(model, val, feats, target)
            t = _evaluate(model, test, feats, target)
            print(
                f"  {label:10} VALIDATION auc={v['auc']:.3f} top20_n={int(v['top20_count']):3d} "
                f"top20_net={v['top20_net']:8.3%} top20_win={v['top20_win']:7.2%}"
            )
            print(
                f"  {label:10} TEST       auc={t['auc']:.3f} top20_n={int(t['top20_count']):3d} "
                f"top20_net={t['top20_net']:8.3%} top20_win={t['top20_win']:7.2%}"
            )

        pv = _evaluate(price_model, val, PRICE_FEATURES, target)
        pt = _evaluate(price_model, test, PRICE_FEATURES, target)
        mv = _evaluate(multi_model, val, multi_features, target)
        mt = _evaluate(multi_model, test, multi_features, target)
        print(
            f"  NEWS LIFT: validation_auc={mv['auc']-pv['auc']:+.3f} test_auc={mt['auc']-pt['auc']:+.3f} "
            f"validation_top20_net={mv['top20_net']-pv['top20_net']:+.3%} "
            f"test_top20_net={mt['top20_net']-pt['top20_net']:+.3%}"
        )

    print("\nV8 research gate:")
    print("Proceed only if PRICE+NEWS improves BOTH validation and untouched test performance; otherwise keep collecting historical event data.")
    print("Live execution remains disabled.")


if __name__ == "__main__":
    main()
