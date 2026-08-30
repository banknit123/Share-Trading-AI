from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.gdelt_news import search_articles
from trading_ai.data.market_data import download_history
from trading_ai.data.rss_news import search_google_news_rss
from trading_ai.features.news_events import headline_features
from run_news_coverage_audit_v6_1 import QUERY_MAP

HORIZONS = {"30m": 6, "60m": 12, "120m": 24}


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
        articles = search_articles(query, timespan="7d", maxrecords=30)
        if articles:
            return "GDELT", [
                (a.title, a.domain, a.seen_at) for a in articles if a.seen_at is not None
            ]
    except Exception:
        pass
    time.sleep(0.25)
    articles = search_google_news_rss(query, maxrecords=60, timeout=15.0)
    return "RSS", [
        (a.title, a.domain, a.published_at) for a in articles if a.published_at is not None
    ]


def _align_event(price: pd.DataFrame, event_time: datetime):
    ts = pd.Timestamp(event_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert("UTC")
    pos = price.index.searchsorted(ts, side="left")
    if pos >= len(price):
        return None
    return int(pos)


def main() -> None:
    print("Share-Trading-AI v7 news/event response audit")
    print("No trades are placed. This tests whether recent headlines contain measurable forward-return information.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)
    print("Headline sentiment is a simple transparent research lexicon, not an LLM sentiment model.\n")

    rows = []
    provider_counts = defaultdict(int)

    for symbol in DEFAULT_CONFIG.universe:
        try:
            price = _to_utc_index(download_history(symbol, period="60d", interval="5m"))
            provider, articles = _fetch_news(QUERY_MAP[symbol])
            provider_counts[provider] += 1
            seen_titles = set()
            usable = 0

            for title, domain, published_at in articles:
                key = (title or "").strip().lower()
                if not key or key in seen_titles or published_at is None:
                    continue
                seen_titles.add(key)
                pos = _align_event(price, published_at)
                if pos is None or pos >= len(price) - max(HORIZONS.values()) - 1:
                    continue

                hf = headline_features(title)
                entry = float(price["Close"].iloc[pos])
                record = {
                    "symbol": symbol,
                    "provider": provider,
                    "domain": domain or "",
                    "headline": title,
                    "event_time": pd.Timestamp(published_at),
                    "sentiment": hf.sentiment,
                    "event_hits": hf.event_hits,
                }
                for name, bars in HORIZONS.items():
                    record[f"ret_{name}"] = float(price["Close"].iloc[pos + bars] / entry - 1.0)
                rows.append(record)
                usable += 1

            print(f"{symbol:15} source={provider:5} fetched={len(articles):3d} aligned_events={usable:3d}")
        except Exception as exc:
            print(f"{symbol:15} ERROR: {exc}")

    if not rows:
        print("\nNo timestamp-aligned news events were available. v7 cannot proceed.")
        return

    df = pd.DataFrame(rows)
    print(f"\nAligned event sample: {len(df)} headlines across {df['symbol'].nunique()} stocks")
    print("Providers used:", ", ".join(f"{k}={v}" for k, v in sorted(provider_counts.items())))
    print(f"Unique source domains: {df['domain'].replace('', np.nan).nunique(dropna=True)}")
    print(
        f"Headline labels: positive={(df.sentiment > 0).sum()} negative={(df.sentiment < 0).sum()} "
        f"neutral={(df.sentiment == 0).sum()} event_keyword_headlines={(df.event_hits > 0).sum()}"
    )

    print("\nForward-return response after any headline:")
    for name in HORIZONS:
        s = df[f"ret_{name}"]
        print(f"  {name:4} count={len(s):3d} avg={s.mean():8.3%} median={s.median():8.3%} positive={(s > 0).mean():7.2%}")

    for label, mask in [
        ("positive headlines", df.sentiment > 0),
        ("negative headlines", df.sentiment < 0),
        ("event-keyword headlines", df.event_hits > 0),
    ]:
        subset = df[mask]
        print(f"\n{label} (n={len(subset)}):")
        if subset.empty:
            print("  insufficient observations")
            continue
        for name in HORIZONS:
            s = subset[f"ret_{name}"]
            print(f"  {name:4} avg={s.mean():8.3%} median={s.median():8.3%} positive={(s > 0).mean():7.2%}")

    pos = df[df.sentiment > 0]
    neg = df[df.sentiment < 0]
    print("\nDirectional spread (positive-headline avg minus negative-headline avg):")
    for name in HORIZONS:
        if len(pos) >= 5 and len(neg) >= 5:
            spread = pos[f"ret_{name}"].mean() - neg[f"ret_{name}"].mean()
            print(f"  {name:4} spread={spread:8.3%}")
        else:
            print(f"  {name:4} insufficient positive/negative observations")

    enough = len(df) >= 80 and df["symbol"].nunique() >= 8
    directional = False
    if len(pos) >= 5 and len(neg) >= 5:
        directional = any(
            (pos[f"ret_{name}"].mean() - neg[f"ret_{name}"].mean()) > 0.001
            for name in HORIZONS
        )

    print("\nNews-feature research gate:")
    if enough and directional:
        print("PASS FOR V8 MULTIMODAL MODEL PROTOTYPE")
    elif enough:
        print("COVERAGE PASS, DIRECTIONAL SIGNAL WEAK - KEEP AS EVENT/INTENSITY FEATURES ONLY")
    else:
        print("FAIL - SAMPLE TOO SMALL FOR A NEWS-AWARE MODEL")
    print("This is a short recent-window diagnostic, not evidence for live trading.")


if __name__ == "__main__":
    main()
