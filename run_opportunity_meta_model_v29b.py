from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path("data/v29")
CANONICAL = ROOT / "canonical" / "v29a3_canonical_yahoo_5m.csv.gz"
OUT_DIR = ROOT / "v29b"

HORIZONS = ["30m", "60m", "120m"]
MIN_CANDIDATES_TRAIN = 500
MIN_CANDIDATES_VAL = 100
MIN_CANDIDATES_TEST = 100

# Feature set deliberately excludes symbol identity. All variables are transferable.
FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    "ema_spread_6_18", "ema_spread_18_36",
    "volatility_12", "volatility_36", "volume_ratio_12",
    "overnight_gap", "from_day_open", "prev_day_return", "prev_day_range",
    "tod_relative_volume", "tod_relative_volatility",
    "nifty_ret_1", "nifty_ret_3", "nifty_ret_6", "nifty_ret_12",
    "banknifty_ret_1", "banknifty_ret_6", "banknifty_ret_12",
    "vix_ret_1", "vix_ret_6", "vix_ret_12",
    "breadth_up_1", "breadth_up_6", "breadth_up_12", "breadth_above_fast_ema",
    "xs_pct_ret_1", "xs_pct_ret_3", "xs_pct_ret_6", "xs_pct_ret_12", "xs_pct_ret_24",
    "xs_pct_tod_relative_volume", "xs_pct_volatility_12", "xs_pct_overnight_gap",
    "rel_to_nifty_6", "rel_to_nifty_12", "bank_vs_nifty_6", "vix_change_6", "market_accel",
]


def load_data() -> pd.DataFrame:
    if not CANONICAL.exists():
        raise FileNotFoundError(f"Run v29A3 first; missing {CANONICAL}")
    df = pd.read_csv(CANONICAL, parse_dates=["timestamp", "entry_ts"])
    return df.replace([np.inf, -np.inf], np.nan)


def add_candidate_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Market-general opportunity triggers. No stock name is used.
    # Strong/weak cross-sectional position, unusual time-of-day volume, or unusual gap.
    top_strength = out["xs_pct_ret_6"] >= 0.80
    top_strength_12 = out["xs_pct_ret_12"] >= 0.80
    high_rvol = out["xs_pct_tod_relative_volume"] >= 0.80
    gap_extreme = (out["xs_pct_overnight_gap"] >= 0.85) | (out["xs_pct_overnight_gap"] <= 0.15)
    breadth_confirm = out["breadth_up_6"] >= 0.50
    trend_confirm = (out["ema_spread_6_18"] > 0) & (out["ema_spread_18_36"] > 0)

    # Long-candidate architecture for v29B plumbing. Later versions may add a mirrored short model.
    out["candidate"] = (
        (top_strength & (high_rvol | trend_confirm))
        | (top_strength_12 & breadth_confirm & trend_confirm)
        | (gap_extreme & top_strength & high_rvol)
    )

    # Continuous opportunity intensity used as a feature, not a hard trade threshold.
    out["candidate_strength"] = (
        out["xs_pct_ret_6"].fillna(0.5)
        + out["xs_pct_ret_12"].fillna(0.5)
        + out["xs_pct_tod_relative_volume"].fillna(0.5)
        + out["breadth_up_6"].fillna(0.5)
    ) / 4.0
    return out


def usable_features(df: pd.DataFrame) -> list[str]:
    cols = [c for c in FEATURES + ["candidate_strength"] if c in df.columns]
    # Drop columns with no usable information in this particular provider store.
    return [c for c in cols if df[c].notna().sum() > 0 and df[c].nunique(dropna=True) > 1]


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=180,
        max_leaf_nodes=15,
        min_samples_leaf=60,
        l2_regularization=1.0,
        random_state=42,
    )


def safe_auc(y: pd.Series, p: np.ndarray) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def threshold_grid(val: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Thresholds are calibrated on validation only. Test never participates.
    for t in np.arange(0.50, 0.81, 0.02):
        z = val[val["prob"] >= t]
        if len(z) < 25:
            continue
        by_day = z.groupby("date")["net"].mean()
        rows.append({
            "threshold": float(round(t, 2)),
            "trades": int(len(z)),
            "days": int(z["date"].nunique()),
            "mean_net": float(z["net"].mean()),
            "median_net": float(z["net"].median()),
            "win_rate": float((z["net"] > 0).mean()),
            "positive_days": float((by_day > 0).mean()) if len(by_day) else np.nan,
        })
    return pd.DataFrame(rows)


def choose_threshold(grid: pd.DataFrame) -> float | None:
    if grid.empty:
        return None
    viable = grid[
        (grid["mean_net"] > 0)
        & (grid["win_rate"] > 0.50)
        & (grid["positive_days"] >= 0.50)
        & (grid["trades"] >= 25)
    ].copy()
    if viable.empty:
        return None
    # Conservative selection: maximize mean net, then prefer more support.
    viable = viable.sort_values(["mean_net", "trades"], ascending=[False, False])
    return float(viable.iloc[0]["threshold"])


def summarize_selected(z: pd.DataFrame) -> dict:
    if z.empty:
        return {
            "trades": 0, "days": 0, "stocks": 0, "mean_net": np.nan,
            "median_net": np.nan, "win_rate": np.nan, "positive_days": np.nan,
            "compounded_equal_weight": 0.0,
        }
    by_day = z.groupby("date")["net"].mean().sort_index()
    return {
        "trades": int(len(z)),
        "days": int(z["date"].nunique()),
        "stocks": int(z["symbol"].nunique()),
        "mean_net": float(z["net"].mean()),
        "median_net": float(z["net"].median()),
        "win_rate": float((z["net"] > 0).mean()),
        "positive_days": float((by_day > 0).mean()) if len(by_day) else np.nan,
        "compounded_equal_weight": float(np.prod(1.0 + by_day.values) - 1.0) if len(by_day) else 0.0,
    }


def evaluate_horizon(df: pd.DataFrame, horizon: str, features: list[str]) -> dict:
    net_col = f"net_{horizon}"
    profit_col = f"profit_{horizon}"
    base = df[df["candidate"]].dropna(subset=features + [net_col, profit_col]).copy()
    base["target"] = base[profit_col].astype(int)
    base["net"] = base[net_col].astype(float)

    train = base[base["split"] == "TRAIN"].copy()
    val = base[base["split"] == "VALIDATION"].copy()
    test = base[base["split"] == "TEST"].copy()

    result = {
        "horizon": horizon,
        "train_n": int(len(train)), "validation_n": int(len(val)), "test_n": int(len(test)),
        "status": "OK", "threshold": None,
    }
    if len(train) < MIN_CANDIDATES_TRAIN or len(val) < MIN_CANDIDATES_VAL or len(test) < MIN_CANDIDATES_TEST:
        result["status"] = "INSUFFICIENT_CANDIDATES"
        return result
    if train["target"].nunique() < 2:
        result["status"] = "ONE_CLASS_TRAIN"
        return result

    model = make_model()
    model.fit(train[features], train["target"])
    val["prob"] = model.predict_proba(val[features])[:, 1]
    test["prob"] = model.predict_proba(test[features])[:, 1]

    result["val_auc"] = safe_auc(val["target"], val["prob"])
    result["test_auc"] = safe_auc(test["target"], test["prob"])

    grid = threshold_grid(val)
    threshold = choose_threshold(grid)
    result["threshold"] = threshold

    grid_path = OUT_DIR / f"v29b_{horizon}_validation_thresholds.csv"
    grid.to_csv(grid_path, index=False)

    if threshold is None:
        result["status"] = "ABSTAIN_NO_POSITIVE_VALIDATION_THRESHOLD"
        result["validation"] = summarize_selected(pd.DataFrame())
        result["test"] = summarize_selected(pd.DataFrame())
        return result

    val_sel = val[val["prob"] >= threshold].copy()
    test_sel = test[test["prob"] >= threshold].copy()
    result["validation"] = summarize_selected(val_sel)
    result["test"] = summarize_selected(test_sel)

    # Save test candidates for audit; threshold was fixed before these rows were evaluated.
    audit_cols = ["date", "timestamp", "symbol", "prob", "net", "target"]
    test[audit_cols].sort_values(["date", "prob"], ascending=[True, False]).to_csv(
        OUT_DIR / f"v29b_{horizon}_test_candidates.csv", index=False
    )
    return result


def main() -> None:
    print("Share-Trading-AI v29B Prototype Opportunity Meta-Model")
    print("NO BROKER ORDERS ARE SENT. Research mechanics only.")
    print("IMPORTANT: Yahoo/59-session canonical data is prototype-only and cannot approve a final trading model.")
    print("Architecture: unusual opportunity -> classifier -> validation-calibrated threshold -> frozen TEST evaluation.")

    df = add_candidate_flags(load_data())
    features = usable_features(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nRows loaded: {len(df):,}")
    print(f"Stocks: {df['symbol'].nunique()} | sessions: {df['date'].nunique()}")
    print(f"Candidate rows: {int(df['candidate'].sum()):,} ({df['candidate'].mean():.1%})")
    print(f"Transferable features used: {len(features)}")

    results = []
    for horizon in HORIZONS:
        r = evaluate_horizon(df, horizon, features)
        results.append(r)

    rows = []
    for r in results:
        flat = {k: v for k, v in r.items() if k not in {"validation", "test"}}
        for prefix in ["validation", "test"]:
            for k, v in r.get(prefix, {}).items():
                flat[f"{prefix}_{k}"] = v
        rows.append(flat)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "v29b_summary.csv", index=False)
    (OUT_DIR / "v29b_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\nV29B PROTOTYPE META-MODEL SUMMARY")
    for r in results:
        print(f"\n{r['horizon']}: status={r['status']} train={r['train_n']} val={r['validation_n']} test={r['test_n']}")
        if "val_auc" in r:
            print(f"  val_auc={r['val_auc']:.3f} test_auc={r['test_auc']:.3f}")
        if r.get("threshold") is not None:
            v = r["validation"]
            t = r["test"]
            print(f"  threshold={r['threshold']:.2f}")
            print(f"  VALIDATION selected={v['trades']} mean={v['mean_net']:+.3%} win={v['win_rate']:.1%} positive_days={v['positive_days']:.1%}")
            print(f"  TEST       selected={t['trades']} mean={t['mean_net']:+.3%} win={t['win_rate']:.1%} positive_days={t['positive_days']:.1%} compounded_day_avg={t['compounded_equal_weight']:+.3%}")
        else:
            print("  no validation threshold demonstrated positive after-cost expectancy -> CASH")

    print("\nINTERPRETATION")
    print("  This run validates architecture only. A positive TEST result is interesting but NOT sufficient evidence because the provider history is only 59 Yahoo sessions.")
    print("  Final approval requires rebuilding the same canonical dataset from 1-3 years of research-grade Dhan history, then rerunning v29B without changing the test protocol.")
    print("\nFILES")
    print(f"  {OUT_DIR / 'v29b_summary.csv'}")
    print(f"  {OUT_DIR / 'v29b_summary.json'}")


if __name__ == "__main__":
    main()
