"""
pipeline.py — the single entry point that runs the whole spec end to end.

    from src.pipeline import run_all
    results = run_all()   # downloads/caches data, runs everything, writes output/metrics.json

This is what notebooks/results.ipynb calls, and it's also runnable as a script:
    python -m src.pipeline
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest as bt
from . import data as dat
from . import evaluate as ev
from . import signals as sig

EVAL_START = "2000-01-01"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# One full run of the model for a given (threshold, exclude) configuration
# ---------------------------------------------------------------------------
def _run_model(panel: pd.DataFrame, threshold: float = sig.VOTE_THRESHOLD,
               exclude: str | None = None, eval_start: str = EVAL_START):
    """Runs transform -> zscore -> vote -> position -> backtest for one config.
    Returns (bt_df, votes, vote_sum, positions_full) where positions_full is
    trimmed to the evaluation window (Section 3.2) before the "first position
    defaults to invested" rule is applied (Section 2.4).
    """
    transformed = sig.transform_factors(panel)
    z = sig.rolling_zscores(transformed)
    s = sig.signed_scores(z)
    votes = sig.compute_votes(s, threshold=threshold, exclude=exclude)
    vsum = sig.vote_sum(votes)

    vsum_eval = vsum[eval_start:]
    positions = sig.compute_positions(vsum_eval, start_invested=True)

    bt_df = bt.run_backtest(positions, panel["^SP500TR"])
    return bt_df, votes, vsum, positions


def _tripwire_check(bt_df: pd.DataFrame) -> dict:
    """Gate G8: full-window strategy Sharpe > 1.0 or CAGR > buy&hold + 5pp halts
    the build until a look-ahead audit is documented. We compute and flag it here;
    a human (or Fable 5 on review) still has to read and sign off on the audit."""
    strat_sharpe = ev.sharpe(bt_df["strategy_return"])
    strat_cagr = ev.cagr(bt_df["strategy_return"])
    bh_cagr = ev.cagr(bt_df["mkt_return"])
    tripped = bool(strat_sharpe > 1.0 or strat_cagr > bh_cagr + 0.05)
    return {
        "tripped": tripped,
        "strategy_sharpe": float(strat_sharpe),
        "strategy_cagr": float(strat_cagr),
        "buy_hold_cagr": float(bh_cagr),
        "note": ("TRIPWIRE HIT — halt and complete a look-ahead audit before presenting "
                 "any results (Section 7.1, gate G8)." if tripped else
                 "Tripwire not triggered; no audit required."),
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
def run_all(force_refresh: bool = False, seed: int = 20000101) -> dict:
    panel = dat.build_raw_panel(force_refresh=force_refresh)

    bt_df, votes, vsum, positions = _run_model(panel)
    n_switches, avg_holding = bt.switches_and_holding_period(bt_df["position"])

    strategy_row = ev.metrics_row(bt_df["strategy_return"], bt_df["position"],
                                   bt_df["mkt_return"], n_switches, avg_holding)
    bh_row = ev.metrics_row(bt_df["mkt_return"], pd.Series(1, index=bt_df.index),
                             bt_df["mkt_return"], 0, float(len(bt_df)))

    oracle_eq = bt.oracle_equity(panel["^SP500TR"], bt_df)
    oracle_ret = oracle_eq.pct_change().fillna(oracle_eq.iloc[0] - 1)
    oracle_pos = (bt_df["mkt_return"] > 0).astype(int)
    oracle_row = ev.metrics_row(oracle_ret, oracle_pos, bt_df["mkt_return"],
                                 int((oracle_pos != oracle_pos.shift(1)).sum()),
                                 np.nan)

    null_df = ev.random_timing_null(bt_df["mkt_return"], int(bt_df["position"].sum()),
                                     n_sims=1000, seed=seed)
    null_row = {"CAGR": null_df["CAGR"].mean(), "sharpe": null_df["sharpe"].mean()}

    metrics_table = pd.DataFrame({
        "strategy": strategy_row,
        "buy_and_hold": bh_row,
        "oracle": oracle_row,
        "random_null_mean": null_row,
    }).T

    confusion = ev.confusion_matrix(bt_df["position"], bt_df["mkt_return"])
    null_percentiles = {
        "CAGR_percentile": ev.percentile_of(strategy_row["CAGR"], null_df["CAGR"]),
        "sharpe_percentile": ev.percentile_of(strategy_row["sharpe"], null_df["sharpe"]),
    }

    # --- Robustness: threshold sensitivity ---
    threshold_rows = {}
    for th in (0.3, 0.5, 0.7):
        bdf, *_ = _run_model(panel, threshold=th)
        threshold_rows[f"threshold_{th}"] = ev.metrics_row(
            bdf["strategy_return"], bdf["position"], bdf["mkt_return"])
    threshold_table = pd.DataFrame(threshold_rows).T

    # --- Robustness: sub-periods ---
    subperiod_table = ev.sub_period_metrics(
        bt_df["strategy_return"], bt_df["position"], bt_df["mkt_return"],
        ev.DEFAULT_SUBPERIODS)

    # --- Robustness: leave-one-out ablation ---
    ablation_rows = {}
    for factor in ("growth", "inflation", "trade", "policy", "risk"):
        bdf, *_ = _run_model(panel, exclude=factor)
        ablation_rows[f"without_{factor}"] = ev.metrics_row(
            bdf["strategy_return"], bdf["position"], bdf["mkt_return"])
    ablation_table = pd.DataFrame(ablation_rows).T

    tripwire = _tripwire_check(bt_df)

    results = {
        "panel": panel,
        "backtest": bt_df,
        "votes": votes,
        "vote_sum": vsum,
        "positions": positions,
        "oracle_equity": oracle_eq,
        "metrics_table": metrics_table,
        "confusion_matrix": confusion,
        "null_distribution": null_df,
        "null_percentiles": null_percentiles,
        "threshold_table": threshold_table,
        "subperiod_table": subperiod_table,
        "ablation_table": ablation_table,
        "tripwire": tripwire,
        "n_switches": n_switches,
        "avg_holding_period_months": avg_holding,
    }

    _write_metrics_json(results)
    _make_figures(results)
    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def _jsonify(obj):
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="index"))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json(orient="index"))
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    return obj


def _write_metrics_json(results: dict) -> None:
    payload = {
        "metrics_table": _jsonify(results["metrics_table"]),
        "confusion_matrix": _jsonify(results["confusion_matrix"]),
        "null_percentiles": _jsonify(results["null_percentiles"]),
        "threshold_table": _jsonify(results["threshold_table"]),
        "subperiod_table": _jsonify(results["subperiod_table"]),
        "ablation_table": _jsonify(results["ablation_table"]),
        "tripwire": _jsonify(results["tripwire"]),
        "n_switches": results["n_switches"],
        "avg_holding_period_months": results["avg_holding_period_months"],
        "evaluation_window": {
            "start": str(results["backtest"].index.min().date()),
            "end": str(results["backtest"].index.max().date()),
            "n_months": int(len(results["backtest"])),
        },
    }
    path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _make_figures(results: dict) -> None:
    bt_df = results["backtest"]
    oracle_eq = results["oracle_equity"].reindex(bt_df.index)

    # Fig 1: growth of $1, log scale
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(bt_df.index, bt_df["strategy_equity"], label="Strategy")
    ax.plot(bt_df.index, bt_df["bh_equity"], label="Buy & Hold (TR)")
    ax.plot(oracle_eq.index, oracle_eq, label="Perfect-foresight oracle", linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Growth of $1 (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_growth_of_1.png"), dpi=140)
    plt.close(fig)

    # Fig 2: drawdown curves
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(bt_df.index, ev.drawdown_series(bt_df["strategy_return"]), label="Strategy")
    ax.plot(bt_df.index, ev.drawdown_series(bt_df["mkt_return"]), label="Buy & Hold")
    ax.set_title("Drawdowns")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_drawdowns.png"), dpi=140)
    plt.close(fig)

    # Fig 3: position ribbon over S&P
    fig, ax = plt.subplots(figsize=(10, 4))
    sp = results["panel"]["^GSPC"].reindex(bt_df.index)
    ax.plot(bt_df.index, sp, color="black", linewidth=1)
    ax.fill_between(bt_df.index, sp.min(), sp.max(),
                     where=bt_df["position"] == 1, color="tab:green", alpha=0.15,
                     label="Invested")
    ax.fill_between(bt_df.index, sp.min(), sp.max(),
                     where=bt_df["position"] == 0, color="tab:red", alpha=0.15,
                     label="Cash")
    ax.set_title("Position ribbon over S&P 500 price index")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_position_ribbon.png"), dpi=140)
    plt.close(fig)

    # Fig 4: vote-sum time series with factor-entry dates
    fig, ax = plt.subplots(figsize=(10, 4))
    vsum = results["vote_sum"][EVAL_START:]
    ax.plot(vsum.index, vsum, drawstyle="steps-post")
    ax.axhline(0, color="black", linewidth=0.8)
    entry_dates = {
        "Risk(5b) spread joins": "2002-12-01",
        "Trade factor joins": "2012-01-01",
    }
    for label, d in entry_dates.items():
        ax.axvline(pd.Timestamp(d), color="grey", linestyle=":", alpha=0.7)
        ax.text(pd.Timestamp(d), ax.get_ylim()[1]*0.9, label, rotation=90,
                fontsize=7, va="top")
    ax.set_title("Vote sum over time")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_vote_sum.png"), dpi=140)
    plt.close(fig)

    # Fig 5: histogram of random-null Sharpes with strategy marked
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(results["null_distribution"]["sharpe"], bins=40, color="tab:blue", alpha=0.7)
    strat_sharpe = results["metrics_table"].loc["strategy", "sharpe"]
    ax.axvline(strat_sharpe, color="red", linewidth=2, label="Strategy Sharpe")
    ax.set_title("Random-timing null: Sharpe distribution (n=1000)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "05_random_null_sharpe.png"), dpi=140)
    plt.close(fig)

    # Fig 6: per-factor signed-score small multiples
    s = results["votes"]  # votes, not signed scores directly reused for entry visibility
    signed = sig.signed_scores(sig.rolling_zscores(sig.transform_factors(results["panel"])))
    signed_eval = signed[EVAL_START:]
    fig, axes = plt.subplots(len(signed_eval.columns), 1, figsize=(10, 10), sharex=True)
    for ax, col in zip(axes, signed_eval.columns):
        ax.plot(signed_eval.index, signed_eval[col])
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(col, fontsize=8)
    axes[0].set_title("Per-factor signed scores over time")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "06_factor_scores.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    results = run_all()
    print(results["metrics_table"])
    print("\nTripwire:", results["tripwire"]["note"])
