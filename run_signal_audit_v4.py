from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

from run_cross_sectional_v3 import (
    ROUND_TRIP_COST,
    FEATURES,
    build_panel,
    fit_model,
    score,
    split_by_time,
)


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def audit(name: str, scored: pd.DataFrame) -> None:
    y = (scored["excess_future_30m"] > ROUND_TRIP_COST).astype(int)
    auc = roc_auc_score(y, scored["probability"]) if y.nunique() > 1 else float("nan")
    brier = brier_score_loss(y, scored["probability"]) if y.nunique() > 1 else float("nan")
    ic_excess = safe_spearman(scored["probability"], scored["excess_future_30m"])
    ic_raw = safe_spearman(scored["probability"], scored["future_return_30m"])

    print(f"\n{name} signal audit")
    print(f"rows={len(scored)} auc={auc:.3f} brier={brier:.4f} rank_ic_excess={ic_excess:.3f} rank_ic_raw={ic_raw:.3f}")

    work = scored.copy()
    # Quantile buckets reveal whether higher probabilities actually map to higher future returns.
    try:
        work["prob_bucket"] = pd.qcut(work["probability"], 10, labels=False, duplicates="drop")
    except ValueError:
        work["prob_bucket"] = 0

    grouped = work.groupby("prob_bucket", observed=False).agg(
        rows=("probability", "size"),
        avg_probability=("probability", "mean"),
        avg_excess_return=("excess_future_30m", "mean"),
        avg_raw_return=("future_return_30m", "mean"),
        positive_excess_rate=("excess_future_30m", lambda s: float((s > ROUND_TRIP_COST).mean())),
    )
    print("Probability-decile behaviour (0=lowest, 9=highest):")
    for bucket, row in grouped.iterrows():
        print(
            f"  bucket={int(bucket):2d} rows={int(row['rows']):5d} "
            f"p={row['avg_probability']:.3f} excess={row['avg_excess_return']:.3%} "
            f"raw={row['avg_raw_return']:.3%} hit={row['positive_excess_rate']:.2%}"
        )

    # Audit only the exact candidate pool used by v3 before the confidence threshold.
    candidates = []
    for ts, snap in work.groupby("timestamp"):
        if float(snap["benchmark_ret_6"].iloc[0]) <= 0:
            continue
        snap = snap[snap["rs_6"] > 0]
        if snap.empty:
            continue
        top = snap.sort_values(["probability", "rs_6"], ascending=False).iloc[0]
        candidates.append(top)

    if candidates:
        c = pd.DataFrame(candidates)
        gross_avg = float(c["future_return_30m"].mean())
        net_avg = gross_avg - ROUND_TRIP_COST
        win_gross = float((c["future_return_30m"] > 0).mean())
        win_net = float((c["future_return_30m"] > ROUND_TRIP_COST).mean())
        print(
            f"Top-candidate pool before confidence threshold: count={len(c)} "
            f"gross_avg={gross_avg:.3%} net_avg_after_cost={net_avg:.3%} "
            f"gross_win={win_gross:.2%} net_win={win_net:.2%}"
        )

        # Compare highest-probability vs lowest-probability candidates inside the eligible pool.
        c = c.sort_values("probability")
        q = max(1, len(c) // 5)
        low = c.head(q)
        high = c.tail(q)
        print(
            f"Eligible candidate probability spread: low20 net_avg={(low['future_return_30m'].mean()-ROUND_TRIP_COST):.3%} "
            f"vs high20 net_avg={(high['future_return_30m'].mean()-ROUND_TRIP_COST):.3%}"
        )
    else:
        print("No candidates survived regime + positive relative-strength filters.")


def main() -> None:
    print("Share-Trading-AI v4 signal audit")
    print("No trades are placed. This measures whether the model has genuine ranking value before changing strategy.")

    panel = build_panel()
    train, val, test = split_by_time(panel)
    model = fit_model(train)
    val_scored = score(model, val)
    test_scored = score(model, test)

    audit("VALIDATION", val_scored)
    audit("UNTOUCHED TEST", test_scored)

    print("\nInterpretation:")
    print("- AUC/rank IC near 0.5/0 means the model has little predictive value.")
    print("- Negative rank IC means higher probabilities are associated with worse returns; do not simply invert without repeated validation.")
    print("- Positive gross but negative net candidate returns means costs/slippage dominate the edge.")
    print("- Only redesign the strategy after identifying which of these is true.")


if __name__ == "__main__":
    main()
