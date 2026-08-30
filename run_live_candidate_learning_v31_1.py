from __future__ import annotations

"""Share-Trading-AI v31.1 Forward-Integrity Candidate Learning Runner.

Wraps v31 with two mandatory safeguards:
1) Do not record a candidate when the market snapshot is stale relative to wall-clock time.
2) Never resolve an outcome unless the candidate was recorded before the executable entry bar.

OBSERVATION ONLY. No broker orders and no paper positions.
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd

import run_live_candidate_learning_v31 as v31
from run_intraday_portfolio_v21 import IST

MAX_SNAPSHOT_AGE_MIN = int(os.getenv("V31_MAX_SNAPSHOT_AGE_MIN", "20"))


def _utc(ts) -> pd.Timestamp:
    x = pd.Timestamp(ts)
    if x.tzinfo is None:
        return x.tz_localize("UTC")
    return x.tz_convert("UTC")


def snapshot_fresh(snapshot: pd.DataFrame) -> tuple[bool, str]:
    anchor = _utc(snapshot["signal_anchor"].iloc[0] if "signal_anchor" in snapshot.columns else snapshot["timestamp"].max())
    now = pd.Timestamp.now(tz="UTC")
    age_min = (now - anchor).total_seconds() / 60.0
    now_ist = now.tz_convert(IST)
    anchor_ist = anchor.tz_convert(IST)

    if age_min < -2:
        return False, f"snapshot timestamp is {abs(age_min):.1f}m in the future"
    if age_min > MAX_SNAPSHOT_AGE_MIN:
        return False, f"snapshot is stale by {age_min:.1f}m (limit {MAX_SNAPSHOT_AGE_MIN}m)"
    if anchor_ist.date() != now_ist.date():
        return False, f"snapshot trading date {anchor_ist.date()} != current IST date {now_ist.date()}"
    return True, f"fresh snapshot age={age_min:.1f}m"


def record_candidates_guarded(snapshot: pd.DataFrame) -> int:
    ok, reason = snapshot_fresh(snapshot)
    if not ok:
        print(f"FORWARD-INTEGRITY GUARD: no candidates recorded - {reason}")
        return 0
    return ORIGINAL_RECORD(snapshot)


def resolve_due_candidates_guarded() -> int:
    candidates = v31._load_candidates()
    if candidates.empty:
        return 0

    resolved = v31._resolved_keys()
    cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict] = []

    for c in candidates.itertuples(index=False):
        signal_ts = _utc(c.signal_ts)
        recorded_at = _utc(c.recorded_at_utc)
        symbol = str(c.symbol)
        needed = [h for h in v31.HORIZONS if (str(c.candidate_id), h) not in resolved]
        if not needed:
            continue

        if symbol not in cache:
            cache[symbol] = v31._download_symbol_history(symbol)
        hist = cache[symbol]
        if hist is None or hist.empty:
            continue

        entry_target = signal_ts + pd.Timedelta(minutes=5)
        entry_bar = v31._exact_resolution_bar(hist, entry_target)
        if entry_bar is None:
            continue
        entry_ts = _utc(entry_bar.name)

        # Critical forward-integrity rule: the trade entry must not have been knowable
        # before the observation was actually recorded.
        if recorded_at >= entry_ts:
            continue

        entry_px = float(entry_bar["Open"])
        for horizon, minutes in v31.HORIZONS.items():
            key = (str(c.candidate_id), horizon)
            if key in resolved:
                continue
            target = entry_ts + pd.Timedelta(minutes=minutes)
            if target.tz_convert(IST).date() != entry_ts.tz_convert(IST).date():
                continue
            exit_bar = v31._exact_resolution_bar(hist, target)
            if exit_bar is None:
                continue
            exit_ts = _utc(exit_bar.name)
            if recorded_at >= exit_ts:
                continue
            exit_px = float(exit_bar["Open"])
            gross = exit_px / entry_px - 1.0
            net = v31._round_trip_net(gross)
            rows.append({
                "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
                "candidate_id": str(c.candidate_id),
                "symbol": symbol,
                "signal_ts": signal_ts.isoformat(),
                "entry_ts": entry_ts.isoformat(),
                "entry_price": entry_px,
                "horizon": horizon,
                "target_ts": target.isoformat(),
                "exit_ts": exit_ts.isoformat(),
                "exit_price": exit_px,
                "gross_return": gross,
                "net_return": net,
                "profitable_after_cost": bool(net > 0),
                "rank": int(c.rank),
                "candidate_score": float(c.candidate_score),
                "market_regime": str(c.market_regime),
            })
            resolved.add(key)

    if rows:
        v31.ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(v31.RESOLVED_PATH, mode="a", header=not v31.RESOLVED_PATH.exists(), index=False)
    return len(rows)


def run_cycle_guarded() -> None:
    resolved = resolve_due_candidates_guarded()
    if resolved:
        print(f"Resolved {resolved} genuinely forward horizon outcomes.")

    snapshot = v31.v30b._build_tolerant_snapshot()
    anchor = _utc(snapshot["signal_anchor"].iloc[0])
    fresh, freshness_reason = snapshot_fresh(snapshot)
    added = record_candidates_guarded(snapshot)
    scored = v31._score_snapshot(snapshot).head(v31.TOP_N)

    print(f"\nV31.1 snapshot: {anchor.isoformat()} | stocks={snapshot['symbol'].nunique()} | max_age={snapshot['bar_age_minutes'].max():.1f}m")
    print(f"Forward freshness: {'PASS' if fresh else 'BLOCK'} - {freshness_reason}")
    print(f"Market regime: {v31._market_regime(snapshot)} | ret12={snapshot['market_ret_12'].iloc[0]:+.3%} | breadth6={snapshot['breadth_up_6'].iloc[0]:.1%}")
    print(f"New candidates recorded: {added}")
    print("Top observed candidates (display only when stale):")
    for i, r in enumerate(scored.itertuples(index=False), start=1):
        print(
            f"  {i}. {r.symbol:15} score={r.candidate_score:.3f} "
            f"rs12={r.xs_ret_12_pct:.1%} vol={r.xs_volume_pct:.1%} "
            f"vwap={r.xs_vwap_pct:.1%} price={r.Close:.2f}"
        )

    summary = v31.write_summary()
    print(
        f"Learning ledger: candidates={summary['candidates']} "
        f"signal_times={summary['candidate_signal_times']} resolved={summary['resolved_outcomes']}"
    )
    if not fresh:
        print("STALE DATA: rankings above are informational only and are NOT added to the forward-learning ledger.")


def main() -> None:
    print("Share-Trading-AI v31.1 Forward-Integrity Candidate Learning Engine")
    print("OBSERVATION ONLY. Freshness + recorded-before-entry integrity guards are mandatory.")
    print(f"Cycle: {'ONE-SHOT' if v31.ONE_SHOT else f'every {v31.LOOP_SECONDS}s'} | max snapshot age={MAX_SNAPSHOT_AGE_MIN}m")

    while True:
        started = time.time()
        try:
            run_cycle_guarded()
        except KeyboardInterrupt:
            print("Observer stopped by user. Learning state saved.")
            break
        except Exception as exc:
            print(f"Cycle error: {type(exc).__name__}: {exc}")

        if v31.ONE_SHOT:
            break
        sleep_for = max(5, v31.LOOP_SECONDS - int(time.time() - started))
        print(f"Next observation in {sleep_for}s...")
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("Observer stopped by user. Learning state saved.")
            break


ORIGINAL_RECORD = v31.record_candidates

if __name__ == "__main__":
    main()
