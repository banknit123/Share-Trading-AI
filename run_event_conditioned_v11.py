from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from trading_ai.config import DEFAULT_CONFIG
from run_historical_multimodal_v10 import (
    ROUND_TRIP_COST,
    PRICE_FEATURES,
    NEWS_FEATURES,
    load_events,
    build_panel,
    split_time,
)

HORIZON = "120m"
TARGET = f"future_{HORIZON}"
EVENT_WINDOWS = {
    "fresh_60m": lambda df: df["news_count_60m"] > 0,
    "fresh_240m": lambda df: df["news_count_240m"] > 0,
}


def fit_model(train: pd.DataFrame, features: list[str]):
    y = (train[TARGET] > ROUND_TRIP_COST).astype(int)
    if y.nunique() < 2:
        raise RuntimeError("Training target has only one class")
    model = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=240,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=2.0,
        random_state=42,
    )
    model.fit(train[features], y)
    return model


def metrics(model, df: pd.DataFrame, features: list[str]):
    if len(df) < 20:
        return {
            "n": len(df), "auc": float("nan"), "rank_ic": float("nan"),
            "top_n": 0, "top_net": 0.0, "top_win": 0.0,
        }
    p = model.predict_proba(df[features])[:, 1]
    y = (df[TARGET] > ROUND_TRIP_COST).astype(int).to_numpy()
    auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    ret = df[TARGET].to_numpy()
    rank_ic = pd.Series(p).corr(pd.Series(ret), method="spearman")
    cutoff = np.quantile(p, 0.90)
    top = ret[p >= cutoff] - ROUND_TRIP_COST
    return {
        "n": int(len(df)),
        "auc": float(auc),
        "rank_ic": float(rank_ic) if pd.notna(rank_ic) else float("nan"),
        "top_n": int(len(top)),
        "top_net": float(np.mean(top)) if len(top) else 0.0,
        "top_win": float(np.mean(top > 0)) if len(top) else 0.0,
    }


def fmt(label: str, m: dict) -> str:
    return (
        f"{label:24} n={m['n']:5d} auc={m['auc']:.3f} rank_ic={m['rank_ic']:+.3f} "
        f"top10_n={m['top_n']:4d} top10_net={m['top_net']:+.3%} top10_win={m['top_win']:.2%}"
    )


def main() -> None:
    print("Share-Trading-AI v11 event-conditioned multimodal audit")
    print("No trades are placed. Only bars with genuinely recent company news are evaluated.")
    print("Horizon: 120 minutes")
    print("Live trading enabled:", DEFAULT_CONFIG.live_trading_enabled)

    events = load_events()
    panel = build_panel(events)
    train, val, test = split_time(panel)
    print(f"\nFull rows: train={len(train)} validation={len(val)} untouched_test={len(test)}")

    price_features = PRICE_FEATURES
    multimodal_features = PRICE_FEATURES + NEWS_FEATURES
    passes = 0

    for window_name, selector in EVENT_WINDOWS.items():
        tr = train[selector(train)].copy()
        va = val[selector(val)].copy()
        te = test[selector(test)].copy()

        print(f"\nEvent window: {window_name}")
        print(f"Rows after conditioning: train={len(tr)} validation={len(va)} untouched_test={len(te)}")

        if len(tr) < 100 or len(va) < 30 or len(te) < 30:
            print("WINDOW GATE: FAIL - insufficient event-conditioned sample")
            continue

        pm = fit_model(tr, price_features)
        mm = fit_model(tr, multimodal_features)
        pv = metrics(pm, va, price_features)
        pt = metrics(pm, te, price_features)
        mv = metrics(mm, va, multimodal_features)
        mt = metrics(mm, te, multimodal_features)

        print(fmt("PRICE_ONLY VALIDATION", pv))
        print(fmt("PRICE_ONLY TEST", pt))
        print(fmt("PRICE+NEWS VALIDATION", mv))
        print(fmt("PRICE+NEWS TEST", mt))
        print(
            "NEWS LIFT: "
            f"val_auc={mv['auc']-pv['auc']:+.3f} test_auc={mt['auc']-pt['auc']:+.3f} "
            f"val_rank_ic={mv['rank_ic']-pv['rank_ic']:+.3f} test_rank_ic={mt['rank_ic']-pt['rank_ic']:+.3f} "
            f"val_top10_net={mv['top_net']-pv['top_net']:+.3%} test_top10_net={mt['top_net']-pt['top_net']:+.3%}"
        )

        robust = (
            mv["top_net"] > 0
            and mt["top_net"] > 0
            and mv["rank_ic"] > 0
            and mt["rank_ic"] > 0
            and mv["top_net"] > pv["top_net"]
            and mt["top_net"] > pt["top_net"]
            and mv["top_n"] >= 20
            and mt["top_n"] >= 20
        )
        passes += int(robust)
        print("WINDOW GATE:", "PASS FOR WALK-FORWARD PAPER RESEARCH" if robust else "FAIL - KEEP RESEARCH ONLY")

    print("\nV11 overall research gate:")
    if passes:
        print("AT LEAST ONE EVENT WINDOW PASSED. NEXT: repeated walk-forward validation before paper simulation.")
    else:
        print("NO EVENT WINDOW PASSED. Next redesign should add genuinely different context (market/sector/global regime), not threshold tuning.")
    print("Live trading remains disabled regardless of this run.")


if __name__ == "__main__":
    main()
