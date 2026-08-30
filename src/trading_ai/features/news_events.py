from __future__ import annotations

import re
from dataclasses import dataclass


POSITIVE_WORDS = {
    "beat", "beats", "growth", "grows", "profit", "profits", "surge", "surges",
    "rise", "rises", "rally", "rallies", "gain", "gains", "upgrade", "upgrades",
    "strong", "record", "wins", "win", "approval", "approved", "expands", "expansion",
    "deal", "contract", "order", "orders", "dividend", "buyback", "outperform",
    "positive", "boost", "boosts", "higher", "improves", "improved", "recovery",
}

NEGATIVE_WORDS = {
    "miss", "misses", "loss", "losses", "fall", "falls", "drop", "drops", "decline",
    "declines", "downgrade", "downgrades", "weak", "fraud", "probe", "investigation",
    "penalty", "fine", "cuts", "cut", "warning", "warns", "lower", "slump", "slumps",
    "negative", "default", "defaults", "lawsuit", "risk", "risks", "delay", "delays",
    "debt", "concern", "concerns", "disappoints", "disappointing",
}

EVENT_WORDS = {
    "results", "earnings", "profit", "revenue", "guidance", "dividend", "buyback",
    "acquisition", "merger", "stake", "deal", "contract", "order", "launch",
    "approval", "regulator", "investigation", "probe", "rating", "upgrade", "downgrade",
}

TOKEN_RE = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class HeadlineFeatures:
    sentiment: float
    positive_hits: int
    negative_hits: int
    event_hits: int


def headline_features(title: str) -> HeadlineFeatures:
    tokens = [t.lower() for t in TOKEN_RE.findall(title or "")]
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    events = sum(t in EVENT_WORDS for t in tokens)
    denom = max(1, pos + neg)
    sentiment = (pos - neg) / denom if (pos + neg) else 0.0
    return HeadlineFeatures(
        sentiment=float(sentiment),
        positive_hits=int(pos),
        negative_hits=int(neg),
        event_hits=int(events),
    )
