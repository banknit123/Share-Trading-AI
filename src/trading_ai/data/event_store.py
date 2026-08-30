from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import sqlite3


DEFAULT_DB = Path("data/news_events.sqlite3")


@dataclass(frozen=True)
class StoredEvent:
    symbol: str
    provider: str
    title: str
    url: str
    domain: str
    published_at: datetime


def _event_id(symbol: str, title: str, published_at: datetime) -> str:
    raw = f"{symbol}|{published_at.astimezone(timezone.utc).isoformat()}|{title.strip().lower()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def connect(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS news_events (
            event_id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            provider TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            domain TEXT NOT NULL,
            published_at TEXT NOT NULL,
            collected_at TEXT NOT NULL
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_news_symbol_time ON news_events(symbol, published_at)")
    con.commit()
    return con


def insert_events(events: list[StoredEvent], path: Path = DEFAULT_DB) -> tuple[int, int]:
    con = connect(path)
    inserted = 0
    skipped = 0
    now = datetime.now(timezone.utc).isoformat()
    try:
        for e in events:
            published = e.published_at.astimezone(timezone.utc)
            eid = _event_id(e.symbol, e.title, published)
            cur = con.execute(
                """
                INSERT OR IGNORE INTO news_events
                (event_id, symbol, provider, title, url, domain, published_at, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (eid, e.symbol, e.provider, e.title, e.url, e.domain, published.isoformat(), now),
            )
            if cur.rowcount == 1:
                inserted += 1
            else:
                skipped += 1
        con.commit()
    finally:
        con.close()
    return inserted, skipped


def counts(path: Path = DEFAULT_DB) -> tuple[int, int]:
    con = connect(path)
    try:
        total = int(con.execute("SELECT COUNT(*) FROM news_events").fetchone()[0])
        symbols = int(con.execute("SELECT COUNT(DISTINCT symbol) FROM news_events").fetchone()[0])
        return total, symbols
    finally:
        con.close()
