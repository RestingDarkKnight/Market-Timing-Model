"""
evaluate.py — metrics table, confusion matrix, random-timing null, robustness
suite (threshold sensitivity, sub-periods, leave-one-out ablation), and figures
(Section 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


# ---------------------------------------------------------------------------
# Core metrics (Section 5.1)
# ---------------------------------------------------------------------------
def cagr(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    n_years = len(returns) / PERIODS_PER_YEAR
    if n_years <= 0 or equity.iloc[-1] <= 0:
        return np.nan
    return equity.iloc[-1] ** (1 / n_years) - 1


def ann_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(PERIODS_PER_YEAR)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / PERIODS_PER_YEAR
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return np.nan
    return (excess.mean() / vol) * np.sqrt(PERIODS_PER_YEAR)


def drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    return equity / running_max - 1


def max_drawdown(returns: pd.Series) -> float:
    return drawdown_series(returns).min()


def longest_drawdown_duration(returns: pd.Series) -> int:
    """Longest number of consecutive months spent below a prior equity peak."""
    dd = drawdown_series(returns)
    underwater = dd < 0
    longest = cur = 0
    for flag in underwater:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return int(longest)


def calmar(cagr_value: float, maxdd_value: float) -> float:
    if maxdd_value == 0 or np.isnan(maxdd_value):
        return np.nan
    return cagr_value / abs(maxdd_value)


def hit_ratio(position: pd.Series, mkt_return: pd.Series) -> float:
    """% of months the allocation call matched the realized market direction:
    invested & market up, OR cash & market down, both count as a 'hit'."""
    matched = (position == 1) & (mkt_return > 0)
    matched |= (position == 0) & (mkt_return <= 0)
    return matched.mean()


def pct_months_in_market(position: pd.Series) -> float:
    return position.mean()


def capture_ratios(strategy_return: pd.Series, mkt_return: pd.Series) -> tuple[float, float]:
    """Upside/downside capture: compounded strategy return over benchmark-up (or
    -down) months, divided by compounded benchmark return over those same months."""
    up_mask = mkt_return > 0
    down_mask = mkt_return < 0

    def _compound(s):
        return (1 + s).prod() - 1

    up_strat, up_mkt = _compound(strategy_return[up_mask]), _compound(mkt_return[up_mask])
    down_strat, down_mkt = _compound(strategy_return[down_mask]), _compound(mkt_return[down_mask])

    upside = up_strat / up_mkt if up_mkt != 0 else np.nan
    downside = down_strat / down_mkt if down_mkt != 0 else np.nan
    return upside, downside


def metrics_row(strategy_return: pd.Series, position: pd.Series, mkt_return: pd.Series,
                 n_switches: int | None = None, avg_holding: float | None = None) -> dict:
    """One full Section 5.1 metrics row for a return series + its position series."""
    c = cagr(strategy_return)
    dd = max_drawdown(strategy_return)
    up_cap, down_cap = capture_ratios(strategy_return, mkt_return)
    row = {
        "CAGR": c,
        "ann_vol": ann_vol(strategy_return),
        "sharpe": sharpe(strategy_return, rf=0.0),
        "max_drawdown": dd,
        "calmar": calmar(c, dd),
        "longest_dd_months": longest_drawdown_duration(strategy_return),
        "hit_ratio": hit_ratio(position, mkt_return) if position is not None else np.nan,
        "pct_months_in_market": pct_months_in_market(position) if position is not None else np.nan,
        "n_switches": n_switches,
        "avg_holding_period_months": avg_holding,
        "upside_capture": up_cap,
        "downside_capture": down_cap,
    }
    return row


# ---------------------------------------------------------------------------
# Confusion matrix (Section 5.2)
# ---------------------------------------------------------------------------
def confusion_matrix(position: pd.Series, mkt_return: pd.Series) -> dict:
    up = mkt_return > 0
    down = mkt_return <= 0
    invested = position == 1
    cash = position == 0
    table = {
        "invested_and_up": int((invested & up).sum()),
        "invested_and_down": int((invested & down).sum()),
        "cash_and_up": int((cash & up).sum()),
        "cash_and_down": int((cash & down).sum()),
        "up_month_base_rate": float(up.mean()),
    }
    return table


# ---------------------------------------------------------------------------
# Random-timing null (Section 5.3)
# ---------------------------------------------------------------------------
def random_timing_null(mkt_return: pd.Series, n_invested_months: int, n_sims: int = 1000,
                        seed: int = 20000101) -> pd.DataFrame:
    """1,000 sims, each invested in the SAME NUMBER of months as the real strategy,
    months chosen uniformly at random (without replacement) from the evaluation
    window. Fixed seed for reproducibility (Section 5.3)."""
    rng = np.random.default_rng(seed)
    n_total = len(mkt_return)
    sims = []
    for _ in range(n_sims):
        invested_idx = rng.choice(n_total, size=n_invested_months, replace=False)
        pos = np.zeros(n_total)
        pos[invested_idx] = 1.0
        sim_ret = pd.Series(pos, index=mkt_return.index) * mkt_return
        sims.append({"CAGR": cagr(sim_ret), "sharpe": sharpe(sim_ret, rf=0.0)})
    return pd.DataFrame(sims)


def percentile_of(value: float, distribution: pd.Series) -> float:
    return float((distribution < value).mean() * 100)


# ---------------------------------------------------------------------------
# Robustness: threshold sensitivity, sub-periods, leave-one-out ablation (5.4)
# ---------------------------------------------------------------------------
def sub_period_metrics(strategy_return: pd.Series, position: pd.Series, mkt_return: pd.Series,
                        bounds: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows = {}
    for label, (start, end) in bounds.items():
        sr = strategy_return[start:end]
        pos = position[start:end]
        mr = mkt_return[start:end]
        if len(sr) == 0:
            continue
        rows[label] = metrics_row(sr, pos, mr)
    return pd.DataFrame(rows).T


DEFAULT_SUBPERIODS = {
    "2000-2009": ("2000-01-01", "2009-12-31"),
    "2010-2019": ("2010-01-01", "2019-12-31"),
    "2020-present": ("2020-01-01", None),
}
