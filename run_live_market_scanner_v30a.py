from __future__ import annotations

"""Share-Trading-AI v30A live market scanner + paper execution loop.

Research/forward-observation infrastructure only.

Key safety property:
- This scanner ALWAYS uses PaperBroker, regardless of live environment variables.
- It does not call Dhan order placement.
- The strategy layer is intentionally conservative and will emit no proposals unless
  a locally supplied approved-strategy configuration is present.

The purpose is to validate the real-time scanning, feature construction, proposal,
risk, journaling, and monitoring loop before any future live deployment.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from trading_ai.data.market_data import download_history
from run_candle_dominant_v12 import _to_utc, add_candle_features
from run_intraday_portfolio_v21 import IST
from run_market_general_edge_v26b import RESEARCH_UNIVERSE
from run_live_trading_shell_v30 import (
    OrderProposal,
    PaperBroker,
    execute_proposal,
    load_state,
    KILL_SWITCH_PATH,
)

ROOT = Path("data/live_v30a")
SNAPSHOT_DIR = ROOT / "snapshots"
DECISIONS_PATH = ROOT / "decisions.csv"
APPROVED_STRATEGY_PATH = Path("data/v30/approved_strategy.json")

INTERVAL = "5m"
PERIOD = os.getenv("V30A_PERIOD", "5d")
ONE_SHOT = os.getenv("V30A_ONE_SHOT", "true").strip().lower() != "false"
LOOP_SECONDS = int(os.getenv("V30A_LOOP_SECONDS", "300"))
INITIAL_EQUITY = float(os.getenv("V30_INITIAL_EQUITY", "1000000"))
MAX_SCAN_SYMBOLS = int(os.getenv("V30A_MAX_SCAN_SYMBOLS", "0"))  # 0 = all
MIN_COMPLETE_SYMBOLS = int(os.getenv("V30A_MIN_COMPLETE_SYMBOLS", "20"))
DEFAULT_POSITION_PCT = float(os.getenv("V30A_POSITION_PCT", "0.05"))

CONTEXT = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
}


@dataclass
class ScanDecision:
    scan_ts_utc: str
    signal_ts: str
    symbol: str
    decision: str
    reason: str
    score: float | None = None
    expected_net_return: float | None = None
    est_price: float | None = None
    quantity: int | None = None


def _safe_download(symbol: str) -> pd.DataFrame | None:
    try:
        raw = download_history(symbol, period=PERIOD, interval=INTERVAL)
        if raw is None or raw.empty:
            return None
        return _to_utc(raw)
    except Exception as exc:
        print(f"  {symbol:15} SKIP {type(exc).__name__}: {exc}")
        return None


def _feature_frame(symbol: str, raw: pd.DataFrame) -> pd.DataFrame:
    x = add_candle_features(raw).sort_index().copy()
    x["symbol"] = symbol
    x["timestamp"] = x.index
    return x.replace([np.inf, -np.inf], np.nan)


def build_live_snapshot() -> pd.DataFrame:
    symbols = list(RESEARCH_UNIVERSE)
    if MAX_SCAN_SYMBOLS > 0:
        symbols = symbols[:MAX_SCAN_SYMBOLS]

    frames: list[pd.DataFrame] = []
    print(f"Scanning {len(symbols)} NSE equities...")
    for i, symbol in enumerate(symbols, start=1):
        raw = _safe_download(symbol)
        if raw is None or len(raw) < 50:
            continue
        x = _feature_frame(symbol, raw)
        frames.append(x)
        print(f"  {i:2d}/{len(symbols):2d} {symbol:15} latest={pd.Timestamp(x.index.max()).isoformat()}")

    if len(frames) < MIN_COMPLETE_SYMBOLS:
        raise RuntimeError(f"Only {len(frames)} usable equity symbols; require at least {MIN_COMPLETE_SYMBOLS}")

    panel = pd.concat(frames).sort_values(["timestamp", "symbol"])

    # Use latest timestamp with broad coverage, not simply max timestamp of any one stock.
    counts = panel.groupby("timestamp")["symbol"].nunique().sort_index()
    eligible = counts[counts >= MIN_COMPLETE_SYMBOLS]
    if eligible.empty:
        raise RuntimeError("No timestamp has enough simultaneously usable stocks")
    signal_ts = eligible.index[-1]
    snap = panel[panel["timestamp"] == signal_ts].copy()

    # Market-wide contemporaneous context from the broad liquid universe.
    snap["xs_ret_6_pct"] = snap["ret_6"].rank(pct=True, method="average")
    snap["xs_ret_12_pct"] = snap["ret_12"].rank(pct=True, method="average")
    snap["xs_volume_pct"] = snap["volume_z_36"].rank(pct=True, method="average")
    snap["xs_vwap_pct"] = snap["vwap_distance"].rank(pct=True, method="average")
    snap["breadth_up_6"] = float((snap["ret_6"] > 0).mean())
    snap["breadth_up_12"] = float((snap["ret_12"] > 0).mean())
    snap["breadth_fast_trend"] = float((snap["ema_spread_6_18"] > 0).mean())
    snap["market_ret_6"] = float(snap["ret_6"].median())
    snap["market_ret_12"] = float(snap["ret_12"].median())
    snap["relative_ret_6"] = snap["ret_6"] - snap["market_ret_6"]
    snap["relative_ret_12"] = snap["ret_12"] - snap["market_ret_12"]

    # Optional official context snapshots. These are informative/journaled, not required to trade.
    for name, ticker in CONTEXT.items():
        raw = _safe_download(ticker)
        if raw is None or len(raw) < 10:
            continue
        z = raw.sort_index()
        close = z["Close"].astype(float)
        latest = float(close.iloc[-1])
        ret6 = float(close.pct_change(6).iloc[-1]) if len(close) > 6 else np.nan
        snap[f"{name.lower()}_close"] = latest
        snap[f"{name.lower()}_ret_6"] = ret6

    cols = [
        "timestamp", "symbol", "Close", "ret_1", "ret_3", "ret_6", "ret_12",
        "vwap_distance", "ema_spread_6_18", "ema_spread_18_36", "volume_z_36",
        "volume_ratio_12", "volatility_12", "breakout_12", "breakdown_12",
        "position_36", "xs_ret_6_pct", "xs_ret_12_pct", "xs_volume_pct",
        "xs_vwap_pct", "breadth_up_6", "breadth_up_12", "breadth_fast_trend",
        "market_ret_6", "market_ret_12", "relative_ret_6", "relative_ret_12",
    ]
    optional = [c for c in snap.columns if c.startswith("nifty50_") or c.startswith("banknifty_") or c.startswith("indiavix_")]
    return snap[[c for c in cols + optional if c in snap.columns]].sort_values("symbol")


def load_approved_strategy() -> dict | None:
    """Load an explicitly approved strategy configuration.

    No file -> no proposals.  This prevents a research prototype model from being
    silently used in forward trading.  The expected future file is created only
    after research-grade validation.
    """
    if not APPROVED_STRATEGY_PATH.exists():
        return None
    try:
        cfg = json.loads(APPROVED_STRATEGY_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Approved strategy file unreadable: {exc}")
        return None
    if not bool(cfg.get("approved_for_paper", False)):
        return None
    return cfg


def generate_proposals(snapshot: pd.DataFrame, state_equity: float) -> tuple[list[tuple[OrderProposal, float]], list[ScanDecision]]:
    cfg = load_approved_strategy()
    scan_ts = datetime.now(timezone.utc).isoformat()
    signal_ts = pd.Timestamp(snapshot["timestamp"].iloc[0]).isoformat()

    if cfg is None:
        decisions = [
            ScanDecision(
                scan_ts_utc=scan_ts,
                signal_ts=signal_ts,
                symbol=str(r.symbol),
                decision="ABSTAIN",
                reason="no approved_for_paper strategy configuration",
                est_price=float(r.Close),
            )
            for r in snapshot.itertuples(index=False)
        ]
        return [], decisions

    # Future approved configurations may provide simple transferable thresholds.
    # These are NOT populated by default.  No stock names are supported as rule keys.
    horizon = str(cfg.get("horizon", "120m"))
    min_score = float(cfg.get("min_score", 1.0))
    min_expected_net = float(cfg.get("min_expected_net_return", 1.0))
    max_candidates = int(cfg.get("max_candidates", 3))
    position_pct = float(cfg.get("position_pct", DEFAULT_POSITION_PCT))

    # Generic portable score.  It is only activated by an approved strategy file.
    s = snapshot.copy()
    s["score"] = (
        0.30 * s["xs_ret_12_pct"].fillna(0.5)
        + 0.20 * s["xs_ret_6_pct"].fillna(0.5)
        + 0.15 * s["xs_volume_pct"].fillna(0.5)
        + 0.15 * s["xs_vwap_pct"].fillna(0.5)
        + 0.10 * (s["ema_spread_6_18"].fillna(0) > 0).astype(float)
        + 0.10 * (s["ema_spread_18_36"].fillna(0) > 0).astype(float)
    )
    # Placeholder expected-net mapping must be supplied/calibrated by approved config.
    intercept = float(cfg.get("expected_net_intercept", -1.0))
    slope = float(cfg.get("expected_net_score_slope", 0.0))
    s["expected_net"] = intercept + slope * s["score"]
    eligible = s[(s["score"] >= min_score) & (s["expected_net"] >= min_expected_net)].copy()
    eligible = eligible.sort_values(["expected_net", "score"], ascending=False).head(max_candidates)

    proposals: list[tuple[OrderProposal, float]] = []
    decisions: list[ScanDecision] = []
    eligible_symbols = set(eligible["symbol"].astype(str))

    for r in s.itertuples(index=False):
        symbol = str(r.symbol)
        px = float(r.Close)
        if symbol not in eligible_symbols:
            decisions.append(ScanDecision(scan_ts, signal_ts, symbol, "ABSTAIN", "did not pass approved strategy gate", float(r.score), float(r.expected_net), px))
            continue
        order_value = max(0.0, state_equity * position_pct)
        qty = int(order_value // px) if px > 0 else 0
        if qty <= 0:
            decisions.append(ScanDecision(scan_ts, signal_ts, symbol, "ABSTAIN", "calculated quantity <= 0", float(r.score), float(r.expected_net), px, qty))
            continue
        proposal = OrderProposal(
            symbol=symbol,
            side="BUY",
            quantity=qty,
            reason=f"approved transferable paper strategy; horizon={horizon}",
            model_score=float(r.score),
            expected_net_return=float(r.expected_net),
        )
        proposals.append((proposal, px))
        decisions.append(ScanDecision(scan_ts, signal_ts, symbol, "PROPOSE", "passed approved strategy gate", float(r.score), float(r.expected_net), px, qty))

    return proposals, decisions


def persist_scan(snapshot: pd.DataFrame, decisions: list[ScanDecision]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    signal_ts = pd.Timestamp(snapshot["timestamp"].iloc[0])
    tag = signal_ts.strftime("%Y%m%dT%H%M%SZ")
    snapshot.to_csv(SNAPSHOT_DIR / f"snapshot_{tag}.csv", index=False)
    if decisions:
        frame = pd.DataFrame([asdict(x) for x in decisions])
        frame.to_csv(DECISIONS_PATH, mode="a", header=not DECISIONS_PATH.exists(), index=False)


def run_cycle() -> None:
    if KILL_SWITCH_PATH.exists():
        print(f"KILL SWITCH ACTIVE: {KILL_SWITCH_PATH}. Scan/execution cycle skipped.")
        return

    state = load_state(INITIAL_EQUITY)
    broker = PaperBroker()  # hard-coded paper mode in v30A
    snapshot = build_live_snapshot()
    signal_ts = pd.Timestamp(snapshot["timestamp"].iloc[0])
    print(f"\nSnapshot: {signal_ts.isoformat()} | stocks={snapshot['symbol'].nunique()}")
    print(
        f"Market context: ret6={snapshot['market_ret_6'].iloc[0]:+.3%} "
        f"ret12={snapshot['market_ret_12'].iloc[0]:+.3%} "
        f"breadth6={snapshot['breadth_up_6'].iloc[0]:.1%}"
    )

    proposals, decisions = generate_proposals(snapshot, state.current_equity)
    persist_scan(snapshot, decisions)

    if not proposals:
        print("No approved paper proposals this cycle. CASH / HOLD.")
        return

    print(f"Paper proposals: {len(proposals)}")
    for proposal, est_price in proposals:
        result = execute_proposal(broker, state, proposal, est_price)
        print(
            f"  {proposal.side} {proposal.symbol} x{proposal.quantity} @~{est_price:.2f} "
            f"score={proposal.model_score:.3f} exp_net={proposal.expected_net_return:+.3%} -> {result.message}"
        )


def main() -> None:
    print("Share-Trading-AI v30A Live Market Scanner + Paper Execution Loop")
    print("PAPER MODE IS HARD-CODED. No real broker order can be sent by this script.")
    print(f"Cycle: {'ONE-SHOT' if ONE_SHOT else f'every {LOOP_SECONDS}s'}")
    print(f"Approved strategy config: {APPROVED_STRATEGY_PATH}")

    while True:
        started = time.time()
        try:
            run_cycle()
        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as exc:
            print(f"Cycle error: {type(exc).__name__}: {exc}")

        if ONE_SHOT:
            break
        elapsed = time.time() - started
        sleep_for = max(5, LOOP_SECONDS - int(elapsed))
        print(f"Next scan in {sleep_for}s...")
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
