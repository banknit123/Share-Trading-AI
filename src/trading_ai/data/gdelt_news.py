from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class NewsArticle:
    title: str
    url: str
    domain: str | None
    seen_at: datetime | None
    source_country: str | None
    language: str | None
    tone: float | None


def _parse_seen(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def search_articles(query: str, timespan: str = "1d", maxrecords: int = 100) -> list[NewsArticle]:
    """Search GDELT DOC 2.0 without an API key.

    Intended for research/event-feature generation. This is a public news
    aggregation source, not a licensed low-latency market-news feed.
    """
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "timespan": timespan,
        "maxrecords": max(1, min(int(maxrecords), 250)),
        "sort": "datedesc",
    }
    resp = requests.get(GDELT_DOC_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload: dict[str, Any] = resp.json()
    out: list[NewsArticle] = []
    for item in payload.get("articles", []):
        tone = item.get("tone")
        try:
            tone_value = float(tone) if tone not in (None, "") else None
        except (TypeError, ValueError):
            tone_value = None
        out.append(
            NewsArticle(
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                domain=item.get("domain"),
                seen_at=_parse_seen(item.get("seendate")),
                source_country=item.get("sourcecountry"),
                language=item.get("language"),
                tone=tone_value,
            )
        )
    return out
