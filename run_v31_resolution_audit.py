from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path("data/live_v31")
CANDIDATES = ROOT / "candidates.csv"
OUTCOMES = ROOT / "resolved_outcomes.csv"


def main() -> None:
    print("Share-Trading-AI v31 Resolution Integrity Audit")
    print("Checks whether each resolved outcome was genuinely knowable only after the candidate was recorded.")

    if not CANDIDATES.exists():
        print(f"Missing {CANDIDATES}")
        return
    if not OUTCOMES.exists():
        print(f"Missing {OUTCOMES}; nothing resolved yet.")
        return

    c = pd.read_csv(CANDIDATES)
    o = pd.read_csv(OUTCOMES)
    if c.empty or o.empty:
        print("No candidate/outcome rows to audit.")
        return

    keep = ["candidate_id", "recorded_at_utc", "signal_anchor", "signal_ts", "symbol", "rank", "candidate_score"]
    m = o.merge(c[[x for x in keep if x in c.columns]], on=["candidate_id", "symbol"], how="left", suffixes=("_outcome", "_candidate"))

    def utc(col: str) -> pd.Series:
        return pd.to_datetime(m[col], utc=True, errors="coerce")

    recorded = utc("recorded_at_utc")
    resolved_at = utc("resolved_at_utc")
    entry = utc("entry_ts")
    exit_ts = utc("exit_ts")
    target = utc("target_ts")

    m["recorded_before_entry"] = recorded <= entry
    m["recorded_before_exit"] = recorded <= exit_ts
    m["resolved_after_exit"] = resolved_at >= exit_ts
    m["exit_at_or_after_target"] = exit_ts >= target
    m["exit_delay_min"] = (exit_ts - target).dt.total_seconds() / 60.0
    m["forward_integrity_ok"] = (
        m["recorded_before_entry"].fillna(False)
        & m["recorded_before_exit"].fillna(False)
        & m["resolved_after_exit"].fillna(False)
        & m["exit_at_or_after_target"].fillna(False)
        & m["exit_delay_min"].between(0, 5, inclusive="both").fillna(False)
    )

    cols = [
        "candidate_id", "symbol", "horizon", "recorded_at_utc", "signal_ts_outcome",
        "entry_ts", "target_ts", "exit_ts", "resolved_at_utc", "net_return",
        "exit_delay_min", "forward_integrity_ok",
    ]
    cols = [x for x in cols if x in m.columns]

    print(f"\nRows audited: {len(m)}")
    print(f"Integrity PASS: {int(m['forward_integrity_ok'].sum())}")
    print(f"Integrity FAIL: {int((~m['forward_integrity_ok']).sum())}")
    print("\nDETAIL")
    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_colwidth", 40):
        print(m[cols].to_string(index=False))

    bad = m[~m["forward_integrity_ok"]]
    if bad.empty:
        print("\nAUDIT RESULT: PASS — all resolved rows were recorded before their entry/exit and resolved only after the exit existed.")
    else:
        print("\nAUDIT RESULT: FAIL — one or more rows are not genuine forward observations.")
        print("Do not use v31 learning statistics until those rows are cleared/recollected.")


if __name__ == "__main__":
    main()
