from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from run_candle_dominant_v12 import build_panel
from run_candle_sequence_v14 import add_sequence_lags
from run_historical_multimodal_v10 import load_events
from run_walkforward_candle_rank_v18 import EXCESS, TARGET, fit_regressor

DB_PATH = Path("data/paper_observations_v19.sqlite3")
HORIZON_BARS = 24  # 120 minutes at 5-minute bars


def feature_block(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out, sequence_features = add_sequence_lags(panel)
    base = [
        "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
        "vwap_distance", "ema_spread_6_18", "ema_spread_18_36",
        "rsi_14", "volume_ratio_12", "volume_z_36", "position_36",
        "breakout_12", "breakdown_12",
    ]
    features = list(dict.fromkeys(sequence_features + base))
    return out.replace([np.inf, -np.inf], np.nan), features


def add_excess_target(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    med = out.groupby("timestamp")[TARGET].transform("median")
    out[EXCESS] = out[TARGET] - med
    return out


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS observations (
            signal_ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            score REAL NOT NULL,
            predicted_rank INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            settled INTEGER NOT NULL DEFAULT 0,
            exit_ts TEXT,
            exit_price REAL,
            raw_return REAL,
            excess_return REAL,
            PRIMARY KEY (signal_ts, symbol)
        )
        """
    )
    conn.commit()


def settle_existing(conn: sqlite3.Connection, panel: pd.DataFrame) -> int:
    pending = pd.read_sql_query(
        "SELECT signal_ts, symbol, entry_price FROM observations WHERE settled=0",
        conn,
    )
    if pending.empty:
        return 0

    settled_rows = []
    raw_returns_by_signal: dict[str, list[tuple[str, float, str, float]]] = {}

    # Work from the complete current panel, ordered bar-by-bar within symbol.
    work = panel[["timestamp", "symbol", "Close"]].copy()
    work = work.sort_values(["symbol", "timestamp"])

    for row in pending.itertuples(index=False):
        sym = work[work["symbol"] == row.symbol].reset_index(drop=True)
        if sym.empty:
            continue
        signal_ts = pd.Timestamp(row.signal_ts)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")
        else:
            signal_ts = signal_ts.tz_convert("UTC")

        pos = np.where(sym["timestamp"].to_numpy() == signal_ts.to_datetime64())[0]
        if len(pos) == 0:
            # Robust fallback for pandas timezone/object representations.
            matches = sym.index[sym["timestamp"].map(pd.Timestamp) == signal_ts].to_numpy()
            if len(matches) == 0:
                continue
            start = int(matches[0])
        else:
            start = int(pos[0])

        exit_pos = start + HORIZON_BARS
        if exit_pos >= len(sym):
            continue

        exit_row = sym.iloc[exit_pos]
        exit_price = float(exit_row["Close"])
        raw_ret = exit_price / float(row.entry_price) - 1.0
        key = pd.Timestamp(signal_ts).isoformat()
        raw_returns_by_signal.setdefault(key, []).append(
            (row.symbol, raw_ret, pd.Timestamp(exit_row["timestamp"]).isoformat(), exit_price)
        )

    # Compute cross-sectional excess only when at least five universe members are settleable.
    for signal_ts, vals in raw_returns_by_signal.items():
        if len(vals) < 5:
            continue
        med = float(np.median([v[1] for v in vals]))
        for symbol, raw_ret, exit_ts, exit_price in vals:
            settled_rows.append((exit_ts, exit_price, raw_ret, raw_ret - med, signal_ts, symbol))

    conn.executemany(
        """
        UPDATE observations
        SET settled=1, exit_ts=?, exit_price=?, raw_return=?, excess_return=?
        WHERE signal_ts=? AND symbol=?
        """,
        settled_rows,
    )
    conn.commit()
    return len(settled_rows)


def latest_snapshot(panel: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    usable = panel.dropna(subset=features).copy()
    counts = usable.groupby("timestamp")["symbol"].nunique()
    eligible_times = counts[counts >= 5].index
    if len(eligible_times) == 0:
        raise RuntimeError("No recent timestamp has at least five stocks with complete frozen v18 features")
    ts = max(eligible_times)
    snap = usable[usable["timestamp"] == ts].copy()
    return snap.sort_values("symbol")


def main() -> None:
    print("Share-Trading-AI v19 fresh-forward paper observer")
    print("NO BROKER ORDERS ARE SENT. This only stores and later scores frozen v18 rankings.")
    print("Frozen model: 120-minute v16/v18-style candle-sequence cross-sectional ranker.")

    raw_panel = build_panel(load_events())
    seq_panel, features = feature_block(raw_panel)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        init_db(conn)
        settled = settle_existing(conn, seq_panel)
        print(f"Previously stored observations settled this run: {settled}")

        labeled = add_excess_target(seq_panel)
        train = labeled.dropna(subset=features + [TARGET, EXCESS]).copy()
        if train.empty:
            raise RuntimeError("No labeled history available to train the frozen v18 model")

        model = fit_regressor(train, features)
        snap = latest_snapshot(seq_panel, features)
        score = model.predict(snap[features])
        snap = snap[["timestamp", "symbol", "Close"]].copy()
        snap["score"] = score
        snap["predicted_rank"] = snap["score"].rank(method="first", ascending=False).astype(int)
        snap = snap.sort_values("predicted_rank")

        signal_ts = pd.Timestamp(snap["timestamp"].iloc[0]).isoformat()
        inserts = [
            (
                signal_ts,
                str(r.symbol),
                float(r.score),
                int(r.predicted_rank),
                float(r.Close),
            )
            for r in snap.itertuples(index=False)
        ]
        before = conn.total_changes
        conn.executemany(
            """
            INSERT OR IGNORE INTO observations
            (signal_ts, symbol, score, predicted_rank, entry_price)
            VALUES (?, ?, ?, ?, ?)
            """,
            inserts,
        )
        conn.commit()
        added = conn.total_changes - before

        print(f"\nForward signal timestamp: {signal_ts}")
        print("Current frozen-model ranking:")
        for r in snap.itertuples(index=False):
            print(
                f"  rank={int(r.predicted_rank):2d} {r.symbol:15} "
                f"score={float(r.score):+.6f} reference_price={float(r.Close):.2f}"
            )
        print(f"\nNew observation rows stored: {added}")

        stats = pd.read_sql_query(
            """
            SELECT
              COUNT(DISTINCT signal_ts) AS signal_sets,
              SUM(CASE WHEN settled=1 THEN 1 ELSE 0 END) AS settled_rows,
              COUNT(*) AS total_rows
            FROM observations
            """,
            conn,
        ).iloc[0]
        print(
            f"Database: {DB_PATH} | signal_sets={int(stats.signal_sets or 0)} "
            f"settled_rows={int(stats.settled_rows or 0)} total_rows={int(stats.total_rows or 0)}"
        )

        scored = pd.read_sql_query(
            "SELECT predicted_rank, raw_return, excess_return FROM observations WHERE settled=1",
            conn,
        )
        if not scored.empty:
            top = scored[scored["predicted_rank"] == 1]
            print("\nFresh-forward evidence accumulated so far:")
            print(f"  settled signal rows={len(scored)}")
            if not top.empty:
                print(
                    f"  rank-1 observations={len(top)} "
                    f"avg_raw={top['raw_return'].mean():+.3%} "
                    f"avg_excess={top['excess_return'].mean():+.3%}"
                )
        else:
            print("\nNo forward observations have matured yet. Re-run after new market bars arrive.")

    print("\nResearch only. Do not use this ranking as live-trading approval.")


if __name__ == "__main__":
    main()
