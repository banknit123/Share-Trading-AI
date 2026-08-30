from __future__ import annotations

"""Share-Trading-AI v30B Forward Paper Trading Monitor.

Safety:
- PAPER ONLY. This module never instantiates DhanBroker and cannot submit real orders.
- Reuses v30A market scanning/proposal logic and v30 risk concepts.
- Persists paper portfolio state across restarts.

Purpose:
- tolerant 5-minute snapshot alignment
- persistent paper positions
- mark-to-market each cycle
- timed exits by horizon
- realised/unrealised P&L
- trade ledger + cycle summaries
- clean Ctrl+C shutdown
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_live_market_scanner_v30a as v30a
from run_intraday_portfolio_v21 import IST, order_costs
from run_live_trading_shell_v30 import KILL_SWITCH_PATH

ROOT = Path("data/live_v30b")
STATE_PATH = ROOT / "paper_portfolio_state.json"
TRADES_PATH = ROOT / "paper_trades.csv"
CYCLES_PATH = ROOT / "cycle_summary.csv"
SNAPSHOT_DIR = ROOT / "snapshots"

INITIAL_EQUITY = float(os.getenv("V30_INITIAL_EQUITY", "1000000"))
ONE_SHOT = os.getenv("V30B_ONE_SHOT", "true").strip().lower() != "false"
LOOP_SECONDS = int(os.getenv("V30B_LOOP_SECONDS", "300"))
ALIGN_TOLERANCE_MIN = int(os.getenv("V30B_ALIGN_TOLERANCE_MIN", "5"))
MAX_OPEN_POSITIONS = int(os.getenv("V30B_MAX_OPEN_POSITIONS", "3"))
MAX_DAILY_LOSS_PCT = float(os.getenv("V30B_MAX_DAILY_LOSS_PCT", "0.01"))
SLIPPAGE_BPS_PER_SIDE = float(os.getenv("V30B_SLIPPAGE_BPS_PER_SIDE", "2"))
DEFAULT_HOLD_MINUTES = int(os.getenv("V30B_DEFAULT_HOLD_MINUTES", "120"))


@dataclass
class PaperPosition:
    position_id: str
    symbol: str
    quantity: int
    entry_ts: str
    entry_price: float
    horizon_minutes: int
    reason: str
    model_score: float | None = None
    expected_net_return: float | None = None
    last_price: float | None = None
    last_ts: str | None = None


@dataclass
class PaperPortfolio:
    cash: float
    realised_pnl: float = 0.0
    positions: list[PaperPosition] = field(default_factory=list)
    trading_date: str = ""
    start_equity: float = 0.0
    orders_today: int = 0


def _today_ist() -> str:
    return pd.Timestamp.now(tz=IST).strftime("%Y-%m-%d")


def _save_state(state: PaperPortfolio) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_state() -> PaperPortfolio:
    ROOT.mkdir(parents=True, exist_ok=True)
    today = _today_ist()
    if STATE_PATH.exists():
        try:
            raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            raw["positions"] = [PaperPosition(**p) for p in raw.get("positions", [])]
            state = PaperPortfolio(**raw)
        except Exception:
            state = PaperPortfolio(cash=INITIAL_EQUITY, trading_date=today, start_equity=INITIAL_EQUITY)
    else:
        state = PaperPortfolio(cash=INITIAL_EQUITY, trading_date=today, start_equity=INITIAL_EQUITY)

    if state.trading_date != today:
        equity = _equity(state)
        state.trading_date = today
        state.start_equity = equity
        state.orders_today = 0
        _save_state(state)
    return state


def _equity(state: PaperPortfolio) -> float:
    market_value = 0.0
    for p in state.positions:
        px = p.last_price if p.last_price is not None else p.entry_price
        market_value += p.quantity * px
    return state.cash + market_value


def _costs(entry_value: float, exit_value: float) -> float:
    _, buy = order_costs(entry_value, "BUY")
    _, sell = order_costs(exit_value, "SELL")
    slip = (entry_value + exit_value) * SLIPPAGE_BPS_PER_SIDE / 10000.0
    return buy + sell + slip


def _horizon_minutes_from_reason(reason: str) -> int:
    for h in (120, 60, 30, 10, 5):
        if f"{h}m" in str(reason):
            return h
    return DEFAULT_HOLD_MINUTES


def _build_tolerant_snapshot() -> pd.DataFrame:
    symbols = list(v30a.RESEARCH_UNIVERSE)
    if v30a.MAX_SCAN_SYMBOLS > 0:
        symbols = symbols[: v30a.MAX_SCAN_SYMBOLS]

    frames: list[pd.DataFrame] = []
    print(f"Scanning {len(symbols)} NSE equities with ±{ALIGN_TOLERANCE_MIN}m alignment...")
    for i, symbol in enumerate(symbols, start=1):
        raw = v30a._safe_download(symbol)
        if raw is None or len(raw) < 50:
            continue
        x = v30a._feature_frame(symbol, raw)
        frames.append(x)
        print(f"  {i:2d}/{len(symbols):2d} {symbol:15} latest={pd.Timestamp(x.index.max()).isoformat()}")

    if len(frames) < v30a.MIN_COMPLETE_SYMBOLS:
        raise RuntimeError(f"Only {len(frames)} usable equity symbols")

    latest_each = [pd.Timestamp(x.index.max()) for x in frames]
    anchor = max(latest_each)
    tol = pd.Timedelta(minutes=ALIGN_TOLERANCE_MIN)
    rows = []
    for x in frames:
        eligible = x[(x["timestamp"] <= anchor) & (x["timestamp"] >= anchor - tol)]
        if eligible.empty:
            continue
        rows.append(eligible.iloc[-1])
    if len(rows) < v30a.MIN_COMPLETE_SYMBOLS:
        raise RuntimeError(f"Only {len(rows)} symbols within alignment tolerance")

    snap = pd.DataFrame(rows).copy()
    snap["signal_anchor"] = anchor
    snap["bar_age_minutes"] = (anchor - pd.to_datetime(snap["timestamp"], utc=True)).dt.total_seconds() / 60.0
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
    return snap.sort_values("symbol")


def _latest_prices(snapshot: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, float]]:
    out = {}
    for r in snapshot.itertuples(index=False):
        out[str(r.symbol)] = (pd.Timestamp(r.timestamp), float(r.Close))
    return out


def _mark_to_market(state: PaperPortfolio, snapshot: pd.DataFrame) -> None:
    prices = _latest_prices(snapshot)
    for p in state.positions:
        if p.symbol in prices:
            ts, px = prices[p.symbol]
            p.last_price = px
            p.last_ts = ts.isoformat()


def _append_trade(row: dict) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(TRADES_PATH, mode="a", header=not TRADES_PATH.exists(), index=False)


def _close_due_positions(state: PaperPortfolio, snapshot: pd.DataFrame) -> None:
    prices = _latest_prices(snapshot)
    survivors: list[PaperPosition] = []
    for p in state.positions:
        if p.symbol not in prices:
            survivors.append(p)
            continue
        now_ts, px = prices[p.symbol]
        entry_ts = pd.Timestamp(p.entry_ts)
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.tz_localize("UTC")
        held = (now_ts - entry_ts).total_seconds() / 60.0
        same_day = now_ts.tz_convert(IST).date() == entry_ts.tz_convert(IST).date()
        due = held >= p.horizon_minutes or not same_day
        if not due:
            survivors.append(p)
            continue

        entry_value = p.entry_price * p.quantity
        exit_value = px * p.quantity
        costs = _costs(entry_value, exit_value)
        pnl = exit_value - entry_value - costs
        state.cash += exit_value - costs
        state.realised_pnl += pnl
        _append_trade({
            "event": "EXIT",
            "position_id": p.position_id,
            "symbol": p.symbol,
            "quantity": p.quantity,
            "entry_ts": p.entry_ts,
            "exit_ts": now_ts.isoformat(),
            "entry_price": p.entry_price,
            "exit_price": px,
            "gross_pnl": exit_value - entry_value,
            "costs": costs,
            "net_pnl": pnl,
            "return_on_entry": pnl / entry_value if entry_value else np.nan,
            "reason": "HORIZON_EXIT" if same_day else "SAME_DAY_SQUAREOFF",
        })
        print(f"  EXIT {p.symbol} x{p.quantity} @ {px:.2f} net_pnl={pnl:+.2f}")
    state.positions = survivors


def _enter_proposals(state: PaperPortfolio, snapshot: pd.DataFrame) -> None:
    if KILL_SWITCH_PATH.exists():
        print("Kill switch active; no new paper entries.")
        return
    if len(state.positions) >= MAX_OPEN_POSITIONS:
        return

    proposals, _ = v30a.generate_proposals(snapshot, _equity(state))
    if not proposals:
        print("No approved paper proposals this cycle. CASH / HOLD.")
        return

    existing = {p.symbol for p in state.positions}
    anchor = pd.Timestamp(snapshot["signal_anchor"].iloc[0]) if "signal_anchor" in snapshot.columns else pd.Timestamp(snapshot["timestamp"].max())
    for proposal, est_price in proposals:
        if proposal.symbol in existing:
            continue
        if len(state.positions) >= MAX_OPEN_POSITIONS:
            break
        order_value = float(est_price) * int(proposal.quantity)
        if order_value <= 0 or order_value > state.cash:
            continue
        if state.start_equity > 0 and _equity(state) <= state.start_equity * (1.0 - MAX_DAILY_LOSS_PCT):
            print("Daily loss circuit breaker reached; no new entries.")
            break

        buy_cost, _ = order_costs(order_value, "BUY")
        slip = order_value * SLIPPAGE_BPS_PER_SIDE / 10000.0
        total_debit = order_value + buy_cost + slip
        if total_debit > state.cash:
            continue

        pid = f"P-{int(time.time())}-{proposal.symbol.replace('.NS','')}"
        p = PaperPosition(
            position_id=pid,
            symbol=proposal.symbol,
            quantity=int(proposal.quantity),
            entry_ts=anchor.isoformat(),
            entry_price=float(est_price),
            horizon_minutes=_horizon_minutes_from_reason(proposal.reason),
            reason=proposal.reason,
            model_score=proposal.model_score,
            expected_net_return=proposal.expected_net_return,
            last_price=float(est_price),
            last_ts=anchor.isoformat(),
        )
        state.cash -= total_debit
        state.positions.append(p)
        state.orders_today += 1
        existing.add(p.symbol)
        _append_trade({
            "event": "ENTRY",
            "position_id": p.position_id,
            "symbol": p.symbol,
            "quantity": p.quantity,
            "entry_ts": p.entry_ts,
            "entry_price": p.entry_price,
            "gross_pnl": 0.0,
            "costs": buy_cost + slip,
            "net_pnl": 0.0,
            "reason": p.reason,
            "model_score": p.model_score,
            "expected_net_return": p.expected_net_return,
        })
        print(f"  PAPER BUY {p.symbol} x{p.quantity} @ {p.entry_price:.2f} horizon={p.horizon_minutes}m")


def _persist_cycle(state: PaperPortfolio, snapshot: pd.DataFrame) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    anchor = pd.Timestamp(snapshot["signal_anchor"].iloc[0]) if "signal_anchor" in snapshot.columns else pd.Timestamp(snapshot["timestamp"].max())
    tag = anchor.strftime("%Y%m%dT%H%M%SZ")
    snapshot.to_csv(SNAPSHOT_DIR / f"snapshot_{tag}.csv", index=False)
    equity = _equity(state)
    unrealised = 0.0
    for p in state.positions:
        px = p.last_price if p.last_price is not None else p.entry_price
        unrealised += (px - p.entry_price) * p.quantity
    row = {
        "scan_ts_utc": datetime.now(timezone.utc).isoformat(),
        "signal_anchor": anchor.isoformat(),
        "stocks": int(snapshot["symbol"].nunique()),
        "max_bar_age_min": float(snapshot["bar_age_minutes"].max()) if "bar_age_minutes" in snapshot.columns else 0.0,
        "open_positions": len(state.positions),
        "cash": state.cash,
        "equity": equity,
        "realised_pnl": state.realised_pnl,
        "unrealised_gross_pnl": unrealised,
        "orders_today": state.orders_today,
        "market_ret_6": float(snapshot["market_ret_6"].iloc[0]),
        "market_ret_12": float(snapshot["market_ret_12"].iloc[0]),
        "breadth_up_6": float(snapshot["breadth_up_6"].iloc[0]),
    }
    pd.DataFrame([row]).to_csv(CYCLES_PATH, mode="a", header=not CYCLES_PATH.exists(), index=False)
    _save_state(state)


def run_cycle(state: PaperPortfolio) -> None:
    snapshot = _build_tolerant_snapshot()
    anchor = pd.Timestamp(snapshot["signal_anchor"].iloc[0])
    print(f"\nSnapshot anchor: {anchor.isoformat()} | stocks={snapshot['symbol'].nunique()} | max_age={snapshot['bar_age_minutes'].max():.1f}m")
    print(f"Market: ret6={snapshot['market_ret_6'].iloc[0]:+.3%} ret12={snapshot['market_ret_12'].iloc[0]:+.3%} breadth6={snapshot['breadth_up_6'].iloc[0]:.1%}")

    _mark_to_market(state, snapshot)
    _close_due_positions(state, snapshot)
    _mark_to_market(state, snapshot)
    _enter_proposals(state, snapshot)
    _mark_to_market(state, snapshot)
    _persist_cycle(state, snapshot)

    equity = _equity(state)
    unrealised = sum(((p.last_price or p.entry_price) - p.entry_price) * p.quantity for p in state.positions)
    print(f"Portfolio: equity={equity:,.2f} cash={state.cash:,.2f} realised={state.realised_pnl:+,.2f} unrealised_gross={unrealised:+,.2f} open={len(state.positions)}")


def main() -> None:
    print("Share-Trading-AI v30B Forward Paper Trading Monitor")
    print("PAPER ONLY. This script cannot place real broker orders.")
    print(f"Cycle: {'ONE-SHOT' if ONE_SHOT else f'every {LOOP_SECONDS}s'}")
    print(f"Alignment tolerance: {ALIGN_TOLERANCE_MIN} minutes")
    print(f"State: {STATE_PATH}")

    state = _load_state()
    try:
        while True:
            started = time.time()
            try:
                run_cycle(state)
            except Exception as exc:
                print(f"Cycle error: {type(exc).__name__}: {exc}")
                _save_state(state)
            if ONE_SHOT:
                break
            elapsed = time.time() - started
            sleep_for = max(5, LOOP_SECONDS - int(elapsed))
            print(f"Next scan in {sleep_for}s...")
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        _save_state(state)
        print("Scanner stopped by user. State saved.")


if __name__ == "__main__":
    main()
