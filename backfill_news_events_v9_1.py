from __future__ import annotations

from datetime import date, timedelta
import time

from trading_ai.config import DEFAULT_CONFIG
from trading_ai.data.event_store import StoredEvent, counts, insert_events
from trading_ai.data.rss_news import search_google_news_rss
from run_news_coverage_audit_v6_1 import QUERY_MAP

DAYS_BACK = 60
WINDOW_DAYS = 7
MAX_RECORDS_PER_WINDOW = 100


def _date_windows(days_back: int = DAYS_BACK, window_days: int = WINDOW_DAYS):
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=days_back)
    cursor = start
    while cursor < end:
        nxt = min(cursor + timedelta(days=window_days), end)
        yield cursor, nxt
        cursor = nxt


def main() -> None:
    print("Share-Trading-AI v9.1 60-day news backfill")
    print("No trades are placed. This backfills historical headlines into the local event database.")
    print("Source: Google News RSS date-window queries; no API key required.")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)

    total_fetched = 0
    total_new = 0
    total_duplicates = 0
    failed_windows = 0

    for symbol in DEFAULT_CONFIG.universe:
        base_query = QUERY_MAP[symbol]
        symbol_fetched = 0
        symbol_new = 0
        symbol_dup = 0
        symbol_failed = 0

        for start, end in _date_windows():
            query = f"{base_query} after:{start.isoformat()} before:{end.isoformat()}"
            try:
                articles = search_google_news_rss(
                    query,
                    maxrecords=MAX_RECORDS_PER_WINDOW,
                    timeout=20.0,
                )
            except Exception as exc:
                symbol_failed += 1
                failed_windows += 1
                print(f"  {symbol:15} {start}..{end} ERROR: {exc}")
                time.sleep(0.5)
                continue

            events: list[StoredEvent] = []
            for article in articles:
                if article.published_at is None or not article.title.strip():
                    continue
                symbol_fetched += 1
                total_fetched += 1
                events.append(
                    StoredEvent(
                        symbol=symbol,
                        provider="RSS",
                        title=article.title.strip(),
                        url=article.url or "",
                        domain=article.domain or "",
                        published_at=article.published_at,
                    )
                )

            if events:
                inserted, skipped = insert_events(events)
                symbol_new += inserted
                symbol_dup += skipped
                total_new += inserted
                total_duplicates += skipped

            time.sleep(0.15)

        print(
            f"{symbol:15} fetched={symbol_fetched:4d} new={symbol_new:4d} "
            f"duplicates={symbol_dup:4d} failed_windows={symbol_failed}"
        )

    total_events, symbols_covered = counts()
    print("\nBackfill summary")
    print(f"Fetched candidate headlines: {total_fetched}")
    print(f"New events added: {total_new}")
    print(f"Duplicates skipped: {total_duplicates}")
    print(f"Failed date windows: {failed_windows}")
    print(f"Database total events: {total_events}")
    print(f"Database symbols covered: {symbols_covered}/{len(DEFAULT_CONFIG.universe)}")
    print("Database: data/news_events.sqlite3")
    print("Run the normal v9 collector afterward to keep the database current.")


if __name__ == "__main__":
    main()
