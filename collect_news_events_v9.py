from __future__ import annotations

from collections import Counter
from datetime import timezone
import time

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.event_store import StoredEvent, counts, insert_events
from trading_ai.data.gdelt_news import search_articles
from trading_ai.data.rss_news import search_google_news_rss
from run_news_coverage_audit_v6_1 import QUERY_MAP


def fetch_symbol(symbol: str, query: str) -> tuple[str, list[StoredEvent]]:
    # Prefer GDELT for structured timestamps; fall back to Google News RSS.
    for attempt in range(2):
        try:
            rows = search_articles(query, timespan="7d", maxrecords=40)
            events = [
                StoredEvent(
                    symbol=symbol,
                    provider="GDELT",
                    title=a.title,
                    url=a.url,
                    domain=a.domain or "",
                    published_at=a.seen_at,
                )
                for a in rows
                if a.title and a.seen_at is not None
            ]
            if events:
                return "GDELT", events
        except Exception:
            if attempt == 0:
                time.sleep(1.0)

    rows = search_google_news_rss(query, maxrecords=60, timeout=15.0)
    events = [
        StoredEvent(
            symbol=symbol,
            provider="RSS",
            title=a.title,
            url=a.url,
            domain=a.domain or "",
            published_at=a.published_at,
        )
        for a in rows
        if a.title and a.published_at is not None
    ]
    return "RSS", events


def main() -> None:
    print("Share-Trading-AI v9 persistent news-event collector")
    print("No trades are placed. This only grows the local historical event database.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)

    source_counts: Counter[str] = Counter()
    added_total = 0
    duplicate_total = 0

    for symbol in DEFAULT_CONFIG.universe:
        try:
            provider, events = fetch_symbol(symbol, QUERY_MAP[symbol])
            inserted, skipped = insert_events(events)
            source_counts[provider] += 1
            added_total += inserted
            duplicate_total += skipped
            latest = max((e.published_at for e in events), default=None)
            latest_text = latest.astimezone(timezone.utc).isoformat() if latest else "n/a"
            print(
                f"{symbol:15} source={provider:5} fetched={len(events):3d} "
                f"new={inserted:3d} duplicates={skipped:3d} latest={latest_text}"
            )
        except Exception as exc:
            print(f"{symbol:15} ERROR: {exc}")

    total, symbols = counts()
    print("\nCollection summary")
    print("Providers used:", ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())) or "none")
    print(f"New events this run: {added_total}")
    print(f"Duplicates skipped: {duplicate_total}")
    print(f"Database total events: {total}")
    print(f"Database symbols covered: {symbols}/{len(DEFAULT_CONFIG.universe)}")
    print("Database: data/news_events.sqlite3")
    print("Run this collector repeatedly; duplicate headlines are ignored automatically.")


if __name__ == "__main__":
    main()
