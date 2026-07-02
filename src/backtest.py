"""
backtest.py — the engine itself (Section 4).

Two shifts happen in this codebase and they are NOT the same shift, which is why
each is called out explicitly in exactly one place:
  1. data.py:  CFNAI/CPI are shifted +1 month at LOAD, before any transform, because
               those releases are only known with a publication lag (Section 3.3).
  2. HERE:     the fully-formed end-of-month-t SIGNAL (i.e. the position it implies)
               is shifted +1 month before being multiplied by returns, because you
               cannot act on a signal until after you observe it (Section 3.3,
               "signal-to-return alignment").
Conflating these two would either double-lag or under-lag the strategy — kept
separate on purpose for auditability (gate G2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return_series(price_index: pd.Series) -> pd.Series:
    """Simple monthly % return from a month-end price/level series."""
    return price_index.pct_change()


def align_positions_to_returns(positions: pd.Series) -> pd.Series:
    """THE signal-to-return shift (Section 3.3): position decided at end of month t
    is the position HELD DURING month t+1, i.e. it earns month t+1's return.
    Implemented as a one-line shift(1) on the position series, so that when we later
    do `position_aligned * return`, both are indexed to the SAME return month.
    """
    return positions.shift(1)


def run_backtest(positions: pd.Series, sp500_tr_level: pd.Series) -> pd.DataFrame:
    """Section 4 engine.

    positions       : Series of {0,1} indexed by DECISION month (from signals.py),
                       already trimmed to the evaluation window.
    sp500_tr_level  : ^SP500TR month-end level series (full history is fine; will be
                       aligned/trimmed internally).

    Returns a DataFrame indexed by RETURN month with columns:
      position          - the position that was actually held during this return month
                           (i.e. already shifted — this IS month t+1's held position)
      mkt_return         - month's S&P 500 total return
      strategy_return     - position * mkt_return + (1-position) * 0   (cash=0%, Section 4.2)
      strategy_equity     - growth of $1 in the strategy
      bh_equity           - growth of $1 buy & hold (total return)
    """
    mkt_ret = total_return_series(sp500_tr_level)

    pos_aligned = align_positions_to_returns(positions)
    pos_aligned, mkt_ret_aligned = pos_aligned.align(mkt_ret, join="inner")

    strat_ret = pos_aligned * mkt_ret_aligned + (1 - pos_aligned) * 0.0  # cash = 0% (Section 4.2)

    df = pd.DataFrame({
        "position": pos_aligned,
        "mkt_return": mkt_ret_aligned,
        "strategy_return": strat_ret,
    })
    df = df.dropna(subset=["position", "mkt_return"])  # first row has no prior position -> drop

    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    df["bh_equity"] = (1 + df["mkt_return"]).cumprod()
    return df


def oracle_equity(sp500_tr_level: pd.Series, index_like: pd.Series) -> pd.Series:
    """Perfect-foresight oracle: invested in every up month, cash (0%) in every down
    month, recomputed on TOTAL-RETURN data (Section 4.4 / gate C12) — same return
    stream the strategy and buy & hold are measured on, so the three equity curves
    are directly comparable.
    """
    mkt_ret = total_return_series(sp500_tr_level)
    mkt_ret = mkt_ret.reindex(index_like.index)
    oracle_ret = np.where(mkt_ret > 0, mkt_ret, 0.0)
    oracle_ret = pd.Series(oracle_ret, index=mkt_ret.index)
    return (1 + oracle_ret).cumprod()


def switches_and_holding_period(position: pd.Series) -> tuple[int, float]:
    """Number of position switches and average holding period in months (Section 5.1).

    A "spell" is a maximal run of consecutive months at the same position. Average
    holding period = evaluation-window length in months / number of spells.
    """
    pos = position.dropna()
    changes = (pos != pos.shift(1)).sum() - 1  # first month isn't a "switch"
    changes = max(changes, 0)
    n_spells = changes + 1
    avg_holding = len(pos) / n_spells if n_spells > 0 else np.nan
    return int(changes), float(avg_holding)
