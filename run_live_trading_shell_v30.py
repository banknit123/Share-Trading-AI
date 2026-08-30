from __future__ import annotations

"""Share-Trading-AI v30 live-trading shell.

IMPORTANT
---------
This module is execution infrastructure, not evidence of a profitable strategy.
It defaults to PAPER mode.  Real broker orders require an explicit multi-step
opt-in and valid broker configuration.  The signal engine is intentionally
separate so a later research-approved model can be plugged in without changing
risk/execution plumbing.
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

ROOT = Path("data/live_v30")
STATE_PATH = ROOT / "state.json"
JOURNAL_PATH = ROOT / "journal.csv"
KILL_SWITCH_PATH = ROOT / "KILL_SWITCH"
SECURITY_MAP_PATH = Path("data/v29/dhan_security_map.csv")

LIVE_ENABLE_ENV = "LIVE_TRADING_ENABLED"
LIVE_CONFIRM_ENV = "LIVE_CONFIRMATION"
LIVE_CONFIRM_VALUE = "I_ACCEPT_REAL_ORDER_RISK"

MAX_DAILY_LOSS_PCT = float(os.getenv("V30_MAX_DAILY_LOSS_PCT", "0.01"))
MAX_POSITION_PCT = float(os.getenv("V30_MAX_POSITION_PCT", "0.10"))
MAX_OPEN_POSITIONS = int(os.getenv("V30_MAX_OPEN_POSITIONS", "3"))
MAX_ORDERS_PER_DAY = int(os.getenv("V30_MAX_ORDERS_PER_DAY", "20"))
MIN_CASH_BUFFER_PCT = float(os.getenv("V30_MIN_CASH_BUFFER_PCT", "0.10"))
ORDER_COOLDOWN_SECONDS = int(os.getenv("V30_ORDER_COOLDOWN_SECONDS", "60"))

DHAN_BASE = "https://api.dhan.co/v2"


@dataclass
class OrderProposal:
    symbol: str
    side: str
    quantity: int
    order_type: str = "MARKET"
    product_type: str = "INTRADAY"
    validity: str = "DAY"
    limit_price: float = 0.0
    reason: str = ""
    model_score: float | None = None
    expected_net_return: float | None = None


@dataclass
class ExecutionResult:
    accepted: bool
    mode: str
    broker_order_id: str | None
    correlation_id: str
    message: str
    raw: dict | None = None


@dataclass
class RuntimeState:
    trading_date: str
    start_equity: float
    current_equity: float
    realised_pnl: float = 0.0
    orders_today: int = 0
    open_positions: int = 0
    last_order_epoch: float = 0.0
    halted: bool = False
    halt_reason: str = ""


class Broker(Protocol):
    mode: str

    def place_order(self, proposal: OrderProposal) -> ExecutionResult: ...

    def get_orders(self) -> list[dict]: ...

    def get_positions(self) -> list[dict]: ...


class PaperBroker:
    mode = "PAPER"

    def place_order(self, proposal: OrderProposal) -> ExecutionResult:
        cid = uuid.uuid4().hex[:20]
        return ExecutionResult(
            accepted=True,
            mode=self.mode,
            broker_order_id=f"PAPER-{cid}",
            correlation_id=cid,
            message="Paper order accepted; no broker order sent.",
            raw={"proposal": asdict(proposal)},
        )

    def get_orders(self) -> list[dict]:
        return []

    def get_positions(self) -> list[dict]:
        return []


class DhanBroker:
    mode = "LIVE_DHAN"

    def __init__(self) -> None:
        self.client_id = os.getenv("DHAN_CLIENT_ID", "").strip()
        self.access_token = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
        if not self.client_id or not self.access_token:
            raise RuntimeError("Missing DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN")
        if not SECURITY_MAP_PATH.exists():
            raise RuntimeError(f"Missing security map: {SECURITY_MAP_PATH}")
        self.map = pd.read_csv(SECURITY_MAP_PATH)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id,
        }

    def _security(self, symbol: str) -> tuple[str, str]:
        cols = {c.lower(): c for c in self.map.columns}
        symbol_col = cols.get("canonical_symbol") or cols.get("symbol")
        sec_col = cols.get("security_id")
        seg_col = cols.get("exchange_segment")
        if not symbol_col or not sec_col:
            raise RuntimeError("Security map requires canonical_symbol/symbol and security_id columns")
        hit = self.map[self.map[symbol_col].astype(str) == str(symbol)]
        if hit.empty:
            raise RuntimeError(f"No Dhan security mapping for {symbol}")
        r = hit.iloc[0]
        segment = str(r[seg_col]) if seg_col and pd.notna(r[seg_col]) else "NSE_EQ"
        return str(r[sec_col]), segment

    def place_order(self, proposal: OrderProposal) -> ExecutionResult:
        security_id, exchange_segment = self._security(proposal.symbol)
        cid = uuid.uuid4().hex[:20]
        body = {
            "dhanClientId": self.client_id,
            "correlationId": cid,
            "transactionType": proposal.side.upper(),
            "exchangeSegment": exchange_segment,
            "productType": proposal.product_type,
            "orderType": proposal.order_type,
            "validity": proposal.validity,
            "securityId": security_id,
            "quantity": int(proposal.quantity),
            "disclosedQuantity": 0,
            "price": float(proposal.limit_price or 0.0),
            "triggerPrice": 0.0,
            "afterMarketOrder": False,
            "amoTime": "",
            "boProfitValue": 0.0,
            "boStopLossValue": 0.0,
        }
        resp = requests.post(f"{DHAN_BASE}/orders", headers=self.headers, json=body, timeout=15)
        try:
            raw = resp.json()
        except Exception:
            raw = {"text": resp.text}
        if not resp.ok:
            return ExecutionResult(False, self.mode, None, cid, f"Dhan order rejected HTTP {resp.status_code}", raw)
        order_id = str(raw.get("orderId") or raw.get("order_id") or "") or None
        return ExecutionResult(True, self.mode, order_id, cid, "Dhan order request accepted by API.", raw)

    def get_orders(self) -> list[dict]:
        resp = requests.get(f"{DHAN_BASE}/orders", headers=self.headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []

    def get_positions(self) -> list[dict]:
        # Dhan position endpoint is intentionally isolated here so later adapter
        # changes do not touch strategy/risk logic.
        resp = requests.get(f"{DHAN_BASE}/positions", headers=self.headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("data", []) if isinstance(data, dict) else []


def today_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def load_state(initial_equity: float) -> RuntimeState:
    ROOT.mkdir(parents=True, exist_ok=True)
    if STATE_PATH.exists():
        try:
            raw = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            state = RuntimeState(**raw)
            if state.trading_date == today_utc():
                return state
        except Exception:
            pass
    return RuntimeState(today_utc(), initial_equity, initial_equity)


def save_state(state: RuntimeState) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def journal(proposal: OrderProposal, result: ExecutionResult, decision: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    row = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        **asdict(proposal),
        "execution_mode": result.mode,
        "accepted": result.accepted,
        "broker_order_id": result.broker_order_id,
        "correlation_id": result.correlation_id,
        "message": result.message,
    }
    frame = pd.DataFrame([row])
    frame.to_csv(JOURNAL_PATH, mode="a", header=not JOURNAL_PATH.exists(), index=False)


def live_opt_in_valid() -> tuple[bool, str]:
    if KILL_SWITCH_PATH.exists():
        return False, f"kill switch present: {KILL_SWITCH_PATH}"
    if os.getenv(LIVE_ENABLE_ENV, "").strip().lower() != "true":
        return False, f"{LIVE_ENABLE_ENV} is not true"
    if os.getenv(LIVE_CONFIRM_ENV, "").strip() != LIVE_CONFIRM_VALUE:
        return False, f"{LIVE_CONFIRM_ENV} does not match required confirmation phrase"
    if not os.getenv("DHAN_CLIENT_ID") or not os.getenv("DHAN_ACCESS_TOKEN"):
        return False, "Dhan credentials are missing"
    if not SECURITY_MAP_PATH.exists():
        return False, f"Dhan security map is missing: {SECURITY_MAP_PATH}"
    return True, "explicit live opt-in present"


def choose_broker() -> Broker:
    ok, reason = live_opt_in_valid()
    if ok:
        return DhanBroker()
    print(f"Execution mode: PAPER ({reason})")
    return PaperBroker()


def risk_check(state: RuntimeState, proposal: OrderProposal, est_order_value: float) -> tuple[bool, str]:
    if state.halted:
        return False, f"runtime halted: {state.halt_reason}"
    if KILL_SWITCH_PATH.exists():
        return False, "local KILL_SWITCH file exists"
    if proposal.side.upper() not in {"BUY", "SELL"}:
        return False, "invalid side"
    if proposal.quantity <= 0:
        return False, "quantity must be positive"
    if state.orders_today >= MAX_ORDERS_PER_DAY:
        return False, "daily order limit reached"
    if state.open_positions >= MAX_OPEN_POSITIONS and proposal.side.upper() == "BUY":
        return False, "max open positions reached"
    if state.start_equity > 0 and state.current_equity <= state.start_equity * (1.0 - MAX_DAILY_LOSS_PCT):
        return False, "daily loss circuit breaker reached"
    if est_order_value > state.current_equity * MAX_POSITION_PCT:
        return False, "proposed position exceeds max position percentage"
    if proposal.side.upper() == "BUY" and state.current_equity - est_order_value < state.current_equity * MIN_CASH_BUFFER_PCT:
        return False, "cash buffer would be breached"
    if time.time() - state.last_order_epoch < ORDER_COOLDOWN_SECONDS:
        return False, "order cooldown active"
    return True, "risk checks passed"


def execute_proposal(
    broker: Broker,
    state: RuntimeState,
    proposal: OrderProposal,
    est_price: float,
) -> ExecutionResult:
    est_order_value = max(0.0, float(est_price) * int(proposal.quantity))
    allowed, reason = risk_check(state, proposal, est_order_value)
    if not allowed:
        result = ExecutionResult(False, getattr(broker, "mode", "UNKNOWN"), None, uuid.uuid4().hex[:20], reason)
        journal(proposal, result, "BLOCKED")
        return result

    result = broker.place_order(proposal)
    journal(proposal, result, "SUBMITTED" if result.accepted else "REJECTED")
    if result.accepted:
        state.orders_today += 1
        state.last_order_epoch = time.time()
        if proposal.side.upper() == "BUY":
            state.open_positions += 1
        elif proposal.side.upper() == "SELL":
            state.open_positions = max(0, state.open_positions - 1)
        save_state(state)
    return result


def create_kill_switch(reason: str = "manual") -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    KILL_SWITCH_PATH.write_text(f"{datetime.now(timezone.utc).isoformat()} {reason}\n", encoding="utf-8")


def clear_kill_switch() -> None:
    if KILL_SWITCH_PATH.exists():
        KILL_SWITCH_PATH.unlink()


def demo_signal_engine() -> list[tuple[OrderProposal, float]]:
    """Return no trades by default.

    Replace this function only after a model has been approved on research-grade
    history.  Keeping it empty prevents v30 infrastructure from turning the
    current unproven v29B prototype into a real-money strategy.
    """
    return []


def main() -> None:
    print("Share-Trading-AI v30 Live Trading Shell")
    print("Execution infrastructure only; current strategy is NOT approved for real capital.")
    print("Default signal engine returns no trades.")

    initial_equity = float(os.getenv("V30_INITIAL_EQUITY", "1000000"))
    state = load_state(initial_equity)
    broker = choose_broker()

    print(f"Broker mode: {broker.mode}")
    print(f"State: equity={state.current_equity:,.2f} orders_today={state.orders_today} open_positions={state.open_positions}")
    print(f"Kill switch: {'ON' if KILL_SWITCH_PATH.exists() else 'OFF'}")
    print(
        "Risk limits: "
        f"max_daily_loss={MAX_DAILY_LOSS_PCT:.2%}, max_position={MAX_POSITION_PCT:.2%}, "
        f"max_open_positions={MAX_OPEN_POSITIONS}, max_orders/day={MAX_ORDERS_PER_DAY}, "
        f"cash_buffer={MIN_CASH_BUFFER_PCT:.2%}"
    )

    proposals = demo_signal_engine()
    if not proposals:
        print("No approved strategy proposals. No orders submitted.")
        print("LIVE-READY INFRASTRUCTURE STATUS: PASS (execution remains dormant)")
        return

    for proposal, est_price in proposals:
        result = execute_proposal(broker, state, proposal, est_price)
        print(f"{proposal.side} {proposal.symbol} x{proposal.quantity}: {result.message}")


if __name__ == "__main__":
    main()
