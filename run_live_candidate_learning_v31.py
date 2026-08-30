from __future__ import annotations

"""Share-Trading-AI v31 Live Candidate Observation & Learning Engine.

OBSERVATION ONLY
----------------
- Never sends broker orders.
- Never opens paper positions.
- Ranks market-general candidate hypotheses every scan.
- Persists candidate features at signal time.
- Resolves exact 30m/60m/120m outcomes only when a later 5-minute bar exists.
- Uses after-cost outcomes (Kotak-profile costs + configurable slippage).

The purpose is to collect genuinely forward, unseen evidence that can later be
used to decide what (if anything) deserves to feed the v30B paper portfolio.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import run_live_market_scanner_v30a as v30a
import run_forward_paper_monitor_v30b as v30b
from run_intraday_portfolio_v21 import IST, order_costs

ROOT = Path("data/live_v31")
CANDIDATES_PATH = ROOT / "candidates.csv"
RESOLVED_PATH = ROOT / "resolved_outcomes.csv"
SUMMARY_PATH = ROOT / "learning_summary.json"

ONE_SHOT = os.getenv("V31_ONE_SHOT", "true").strip().lower() != "false"
LOOP_SECONDS = int(os.getenv("V31_LOOP_SECONDS", "300"))
TOP_N = int(os.getenv("V31_TOP_N", "5"))
MIN_CANDIDATE_SCORE = float(os.getenv("V31_MIN_CANDIDATE_SCORE", "0.0"))
SLIPPAGE_BPS_PER_SIDE = float(os.getenv("V31_SLIPPAGE_BPS_PER_SIDE", "2"))
NOTIONAL = float(os.getenv("V31_NOTIONAL", "300000"))
RESOLUTION_TOLERANCE_MIN = int(os.getenv("V31_RESOLUTION_TOLERANCE_MIN", "5"))
HORIZONS = {"30m": 30, "60m": 60, "120m": 120}


def _round_trip_net(gross: float) -> float:
    sell_value = max(0.0, NOTIONAL * (1.0 + gross))
    _, buy_cost = order_costs(NOTIONAL, "BUY")
    _, sell_cost = order_costs(sell_value, "SELL")
    slip = (2.0 * SLIPPAGE_BPS_PER_SIDE) / 10000.0
    return gross - (buy_cost + sell_cost) / NOTIONAL - slip


def _market_regime(snapshot: pd.DataFrame) -> str:
    r12 = float(snapshot["market_ret_12"].iloc[0])
    breadth = float(snapshot["breadth_up_6"].iloc[0])
    if r12 <= -0.003 and breadth < 0.40:
        return "WEAK"
    if r12 >= 0.003 and breadth > 0.60:
        return "STRONG"
    return "MIXED"


def _score_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Create a portable observation score, not a trading signal.

    The score is intentionally based on normalized cross-sectional/state inputs,
    not stock identity.  It is used only to decide which hypotheses to observe.
    """
    s = snapshot.copy()
    for col in ["xs_ret_12_pct", "xs_ret_6_pct", "xs_volume_pct", "xs_vwap_pct"]:
        if col not in s:
            s[col] = 0.5
    s["trend_confirm"] = (
        (s["ema_spread_6_18"].fillna(0) > 0).astype(float)
        + (s["ema_spread_18_36"].fillna(0) > 0).astype(float)
    ) / 2.0
    s["breakout_confirm"] = ((s["breakout_12"].fillna(0) > 0) | (s["position_36"].fillna(0.5) >= 0.90)).astype(float)
    s["candidate_score"] = (
        0.26 * s["xs_ret_12_pct"].fillna(0.5)
        + 0.18 * s["xs_ret_6_pct"].fillna(0.5)
        + 0.18 * s["xs_volume_pct"].fillna(0.5)
        + 0.14 * s["xs_vwap_pct"].fillna(0.5)
        + 0.14 * s["trend_confirm"]
        + 0.10 * s["breakout_confirm"]
    )
    return s.sort_values(["candidate_score", "xs_ret_12_pct", "xs_volume_pct"], ascending=False)


def _load_candidates() -> pd.DataFrame:
    if not CANDIDATES_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CANDIDATES_PATH)
    except Exception:
        return pd.DataFrame()


def _append_candidates(rows: list[dict]) -> int:
    if not rows:
        return 0
    ROOT.mkdir(parents=True, exist_ok=True)
    existing = _load_candidates()
    new = pd.DataFrame(rows)
    if not existing.empty:
        keys = set(zip(existing["symbol"].astype(str), existing["signal_ts"].astype(str)))
        new = new[~new.apply(lambda r: (str(r["symbol"]), str(r["signal_ts"])) in keys, axis=1)].copy()
    if new.empty:
        return 0
    new.to_csv(CANDIDATES_PATH, mode="a", header=not CANDIDATES_PATH.exists(), index=False)
    return len(new)


def record_candidates(snapshot: pd.DataFrame) -> int:
    scored = _score_snapshot(snapshot)
    scored = scored[scored["candidate_score"] >= MIN_CANDIDATE_SCORE].head(TOP_N)
    if scored.empty:
        return 0

    anchor = pd.Timestamp(snapshot["signal_anchor"].iloc[0]) if "signal_anchor" in snapshot.columns else pd.Timestamp(snapshot["timestamp"].max())
    regime = _market_regime(snapshot)
    scan_ts = datetime.now(timezone.utc).isoformat()
    rows = []
    for rank, r in enumerate(scored.itertuples(index=False), start=1):
        signal_ts = pd.Timestamp(r.timestamp)
        rows.append({
            "candidate_id": uuid.uuid4().hex,
            "recorded_at_utc": scan_ts,
            "signal_anchor": anchor.isoformat(),
            "signal_ts": signal_ts.isoformat(),
            "symbol": str(r.symbol),
            "rank": rank,
            "signal_price": float(r.Close),
            "candidate_score": float(r.candidate_score),
            "market_regime": regime,
            "market_ret_6": float(r.market_ret_6),
            "market_ret_12": float(r.market_ret_12),
            "breadth_up_6": float(r.breadth_up_6),
            "ret_1": float(r.ret_1) if pd.notna(r.ret_1) else np.nan,
            "ret_3": float(r.ret_3) if pd.notna(r.ret_3) else np.nan,
            "ret_6": float(r.ret_6) if pd.notna(r.ret_6) else np.nan,
            "ret_12": float(r.ret_12) if pd.notna(r.ret_12) else np.nan,
            "relative_ret_6": float(r.relative_ret_6) if pd.notna(r.relative_ret_6) else np.nan,
            "relative_ret_12": float(r.relative_ret_12) if pd.notna(r.relative_ret_12) else np.nan,
            "xs_ret_6_pct": float(r.xs_ret_6_pct),
            "xs_ret_12_pct": float(r.xs_ret_12_pct),
            "xs_volume_pct": float(r.xs_volume_pct),
            "xs_vwap_pct": float(r.xs_vwap_pct),
            "vwap_distance": float(r.vwap_distance) if pd.notna(r.vwap_distance) else np.nan,
            "ema_spread_6_18": float(r.ema_spread_6_18) if pd.notna(r.ema_spread_6_18) else np.nan,
            "ema_spread_18_36": float(r.ema_spread_18_36) if pd.notna(r.ema_spread_18_36) else np.nan,
            "volume_z_36": float(r.volume_z_36) if pd.notna(r.volume_z_36) else np.nan,
            "volume_ratio_12": float(r.volume_ratio_12) if pd.notna(r.volume_ratio_12) else np.nan,
            "volatility_12": float(r.volatility_12) if pd.notna(r.volatility_12) else np.nan,
            "breakout_12": float(r.breakout_12) if pd.notna(r.breakout_12) else np.nan,
            "breakdown_12": float(r.breakdown_12) if pd.notna(r.breakdown_12) else np.nan,
            "position_36": float(r.position_36) if pd.notna(r.position_36) else np.nan,
        })
    return _append_candidates(rows)


def _resolved_keys() -> set[tuple[str, str]]:
    if not RESOLVED_PATH.exists():
        return set()
    try:
        x = pd.read_csv(RESOLVED_PATH, usecols=["candidate_id", "horizon"])
        return set(zip(x["candidate_id"].astype(str), x["horizon"].astype(str)))
    except Exception:
        return set()


def _download_symbol_history(symbol: str) -> pd.DataFrame | None:
    raw = v30a._safe_download(symbol)
    if raw is None or raw.empty:
        return None
    x = raw.sort_index().copy()
    x["timestamp"] = x.index
    return x


def _exact_resolution_bar(history: pd.DataFrame, target: pd.Timestamp) -> pd.Series | None:
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")
    ts = pd.DatetimeIndex(history.index)
    eligible = history[(ts >= target) & (ts <= target + pd.Timedelta(minutes=RESOLUTION_TOLERANCE_MIN))]
    if eligible.empty:
        return None
    return eligible.iloc[0]


def resolve_due_candidates() -> int:
    candidates = _load_candidates()
    if candidates.empty:
        return 0
    resolved = _resolved_keys()
    now_latest_cache: dict[str, pd.DataFrame | None] = {}
    rows: list[dict] = []

    for c in candidates.itertuples(index=False):
        signal_ts = pd.Timestamp(c.signal_ts)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")
        else:
            signal_ts = signal_ts.tz_convert("UTC")

        symbol = str(c.symbol)
        needed = [h for h in HORIZONS if (str(c.candidate_id), h) not in resolved]
        if not needed:
            continue
        if symbol not in now_latest_cache:
            now_latest_cache[symbol] = _download_symbol_history(symbol)
        hist = now_latest_cache[symbol]
        if hist is None or hist.empty:
            continue

        # Entry is the first real 5-minute bar after the signal bar, matching the
        # research execution convention instead of pretending the signal close is tradable.
        entry_target = signal_ts + pd.Timedelta(minutes=5)
        entry_bar = _exact_resolution_bar(hist, entry_target)
        if entry_bar is None:
            continue
        entry_ts = pd.Timestamp(entry_bar.name)
        entry_px = float(entry_bar["Open"])

        for horizon, minutes in HORIZONS.items():
            key = (str(c.candidate_id), horizon)
            if key in resolved:
                continue
            target = entry_ts + pd.Timedelta(minutes=minutes)
            # Do not create an overnight outcome for an intraday hypothesis.
            if target.tz_convert(IST).date() != entry_ts.tz_convert(IST).date():
                continue
            exit_bar = _exact_resolution_bar(hist, target)
            if exit_bar is None:
                continue
            exit_ts = pd.Timestamp(exit_bar.name)
            exit_px = float(exit_bar["Open"])
            gross = exit_px / entry_px - 1.0
            net = _round_trip_net(gross)
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
        ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(RESOLVED_PATH, mode="a", header=not RESOLVED_PATH.exists(), index=False)
    return len(rows)


def write_summary() -> dict:
    candidates = _load_candidates()
    if RESOLVED_PATH.exists():
        try:
            outcomes = pd.read_csv(RESOLVED_PATH)
        except Exception:
            outcomes = pd.DataFrame()
    else:
        outcomes = pd.DataFrame()

    summary: dict = {
        "candidates": int(len(candidates)),
        "candidate_signal_times": int(candidates["signal_ts"].nunique()) if not candidates.empty else 0,
        "candidate_symbols": int(candidates["symbol"].nunique()) if not candidates.empty else 0,
        "resolved_outcomes": int(len(outcomes)),
        "horizons": {},
        "observation_only": True,
    }
    if not outcomes.empty:
        for h, g in outcomes.groupby("horizon"):
            by_day = g.assign(date=pd.to_datetime(g["entry_ts"], utc=True).dt.tz_convert(IST).dt.date).groupby("date")["net_return"].mean()
            summary["horizons"][str(h)] = {
                "n": int(len(g)),
                "mean_net_return": float(g["net_return"].mean()),
                "median_net_return": float(g["net_return"].median()),
                "win_rate": float((g["net_return"] > 0).mean()),
                "positive_day_rate": float((by_day > 0).mean()) if len(by_day) else None,
            }
    ROOT.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_cycle() -> None:
    # First resolve old observations using any newly available bars.
    resolved = resolve_due_candidates()
    if resolved:
        print(f"Resolved {resolved} pending horizon outcomes.")

    snapshot = v30b._build_tolerant_snapshot()
    anchor = pd.Timestamp(snapshot["signal_anchor"].iloc[0])
    added = record_candidates(snapshot)
    scored = _score_snapshot(snapshot).head(TOP_N)

    print(f"\nV31 snapshot: {anchor.isoformat()} | stocks={snapshot['symbol'].nunique()} | max_age={snapshot['bar_age_minutes'].max():.1f}m")
    print(f"Market regime: {_market_regime(snapshot)} | ret12={snapshot['market_ret_12'].iloc[0]:+.3%} | breadth6={snapshot['breadth_up_6'].iloc[0]:.1%}")
    print(f"New candidates recorded: {added}")
    print("Top observed candidates:")
    for i, r in enumerate(scored.itertuples(index=False), start=1):
        print(
            f"  {i}. {r.symbol:15} score={r.candidate_score:.3f} "
            f"rs12={r.xs_ret_12_pct:.1%} vol={r.xs_volume_pct:.1%} "
            f"vwap={r.xs_vwap_pct:.1%} price={r.Close:.2f}"
        )

    summary = write_summary()
    print(
        f"Learning ledger: candidates={summary['candidates']} "
        f"signal_times={summary['candidate_signal_times']} resolved={summary['resolved_outcomes']}"
    )
    if summary["horizons"]:
        print("Resolved forward outcomes:")
        for h, st in summary["horizons"].items():
            print(
                f"  {h}: n={st['n']} mean_net={st['mean_net_return']:+.3%} "
                f"win={st['win_rate']:.1%} positive_days={st['positive_day_rate']:.1%}"
            )
    else:
        print("No forward horizons are resolvable yet. This is normal on the first fresh observation cycle.")


def main() -> None:
    print("Share-Trading-AI v31 Live Candidate Observation & Learning Engine")
    print("OBSERVATION ONLY. No broker orders and no paper positions are created.")
    print(f"Cycle: {'ONE-SHOT' if ONE_SHOT else f'every {LOOP_SECONDS}s'} | top candidates per scan={TOP_N}")

    while True:
        started = time.time()
        try:
            run_cycle()
        except KeyboardInterrupt:
            print("Observer stopped by user. Learning state saved.")
            break
        except Exception as exc:
            print(f"Cycle error: {type(exc).__name__}: {exc}")

        if ONE_SHOT:
            break
        elapsed = time.time() - started
        sleep_for = max(5, LOOP_SECONDS - int(elapsed))
        print(f"Next observation in {sleep_for}s...")
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            print("Observer stopped by user. Learning state saved.")
            break


if __name__ == "__main__":
    main()
