from __future__ import annotations

import numpy as np

from run_cross_sectional_v3 import (
    FEATURES,
    THRESHOLDS,
    build_panel,
    fit_model,
    score,
    split_by_time,
)


def describe_split(name, scored):
    print(f"\n{name} diagnostics")
    times = sorted(scored["timestamp"].unique())
    total_times = len(times)
    positive_regime = 0
    positive_rs = 0
    threshold_counts = {t: 0 for t in THRESHOLDS}
    best_probs = []
    best_rs = []

    for ts in times:
        snap = scored[scored["timestamp"] == ts]
        if snap.empty:
            continue

        regime_ok = float(snap["benchmark_ret_6"].iloc[0]) > 0
        if regime_ok:
            positive_regime += 1

        candidate = snap.sort_values(["probability", "rs_6"], ascending=False).iloc[0]
        p = float(candidate["probability"])
        rs = float(candidate["rs_6"])
        best_probs.append(p)
        best_rs.append(rs)

        if regime_ok and rs > 0:
            positive_rs += 1
            for threshold in THRESHOLDS:
                if p >= threshold:
                    threshold_counts[threshold] += 1

    print(f"timestamps={total_times}")
    print(f"market-regime pass={positive_regime} ({positive_regime/max(total_times,1):.1%})")
    print(f"regime + positive relative-strength pass={positive_rs} ({positive_rs/max(total_times,1):.1%})")
    if best_probs:
        q = np.quantile(best_probs, [0.50, 0.75, 0.90, 0.95, 0.99])
        print(
            "top-candidate probability quantiles: "
            f"p50={q[0]:.3f} p75={q[1]:.3f} p90={q[2]:.3f} p95={q[3]:.3f} p99={q[4]:.3f}"
        )
    if best_rs:
        q = np.quantile(best_rs, [0.50, 0.75, 0.90, 0.95])
        print(
            "top-candidate rs_6 quantiles: "
            f"p50={q[0]:.3%} p75={q[1]:.3%} p90={q[2]:.3%} p95={q[3]:.3%}"
        )

    print("candidate counts by probability threshold (before non-overlap cooldown):")
    for threshold in THRESHOLDS:
        print(f"  threshold={threshold:.2f}: {threshold_counts[threshold]}")


def main() -> None:
    print("Share-Trading-AI v3 filter diagnostics")
    print("No trades are placed; this script only measures where candidates are filtered out.")

    panel = build_panel()
    train, val, test = split_by_time(panel)
    model = fit_model(train)
    val_scored = score(model, val)
    test_scored = score(model, test)

    print(f"Rows: train={len(train)} validation={len(val)} test={len(test)}")
    describe_split("VALIDATION", val_scored)
    describe_split("UNTOUCHED TEST", test_scored)

    print("\nInterpretation guide:")
    print("- If market-regime pass is low, the benchmark filter is too restrictive.")
    print("- If regime+RS pass is healthy but threshold counts are near zero, probabilities are poorly calibrated or thresholds are too high.")
    print("- If threshold counts are healthy but simulated trades remain near zero, the cooldown/non-overlap rule is the bottleneck.")
    print("- Do not loosen filters until this diagnostic identifies the bottleneck.")


if __name__ == "__main__":
    main()
