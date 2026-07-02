"""
data.py — download, cache, and as-of-align the seven raw series (spec Section 3).

Design intent
-------------
This module is the ONLY place look-ahead protection is implemented (spec Section 3.3).
Every other module downstream (signals.py, backtest.py, evaluate.py) trusts that any
series leaving this module is already "as-of correct": a value dated month t is a value
that a person standing at the end of month t could actually have known.

Two kinds of series live here:
  - Market-price series (dollar index, 2Y yield, S&P levels, HY OAS): these post same-day,
    so the month-end print for month t is usable in the end-of-month-t signal with no lag.
  - Government statistical releases (CFNAI, CPI): these are published with a delay (CFNAI
    ~3 weeks after month-end; CPI ~2 weeks after month-end) and are revised after first
    release. We approximate "as of end of month t" by using the month t-1 reference value,
    via a single explicit .shift(1). This is deliberately crude — it does not model the
    exact publication calendar — but it is conservative (it never uses information from
    later than what a real-time investor could plausibly have had) and it is the one
    correction the spec requires. The residual bias (we still use the FINAL, revised value
    of CFNAI/CPI rather than the first-release "vintage" value) is disclosed in
    REPORT.md Phase 3 and cannot be cheaply removed without a real-time vintage database
    (e.g. ALFRED).
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Series registry (spec Section 3.1)
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "CFNAI": "1985-01-01",
    "CPIAUCSL": "1985-01-01",
    "DTWEXBGS": "2006-01-01",   # inception; series itself begins Jan 2006
    "DGS2": "1985-01-01",
    "BAMLH0A0HYM2": "1996-12-01",  # inception; series begins Dec 1996
}

YAHOO_SERIES = {
    "^GSPC": "1985-01-01",     # price index — used for the SIGNAL (Section 2.1, factor 5a)
    "^SP500TR": "1988-01-01",  # total return — used for PERFORMANCE (Section 4)
}

# Series that require the +1 month as-of shift before any transformation (Section 3.3)
STATISTICAL_RELEASE_SERIES = {"CFNAI", "CPIAUCSL"}


def _cache_path(name: str) -> str:
    safe = name.replace("^", "")
    return os.path.join(RAW_DIR, f"{safe}.csv")


def _fetch_fred(series_id: str, start: str) -> pd.Series:
    """Download one FRED series via pandas_datareader (no API key required).

    Requires network access to fred.stlouisfed.org — will only work in an
    environment with that egress allowed (e.g. Colab), not in a locked-down
    sandbox.
    """
    import pandas_datareader.data as web

    df = web.DataReader(series_id, "fred", start=start, end=datetime.today())
    s = df[series_id].rename(series_id)
    s.index.name = "date"
    return s


def _fetch_yahoo(ticker: str, start: str) -> pd.Series:
    """Download one Yahoo Finance monthly close series via yfinance.

    Requires network access to query1/query2.finance.yahoo.com.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, interval="1mo", progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col frame
        close = close.iloc[:, 0]
    close = close.rename(ticker)
    close.index.name = "date"
    return close


def get_series(name: str, force_refresh: bool = False) -> pd.Series:
    """Return a raw series, downloading + caching to data/raw/ on first use.

    This function does NOT apply the as-of shift — see `load_all_raw()` /
    `apply_asof_shift()`. Keeping the cache raw (unshifted) means the cache is
    an honest record of what was actually published, and the look-ahead
    correction is auditable as a single separate step.
    """
    path = _cache_path(name)
    if os.path.exists(path) and not force_refresh:
        s = pd.read_csv(path, index_col="date", parse_dates=True)[name]
        return s

    if name in FRED_SERIES:
        s = _fetch_fred(name, FRED_SERIES[name])
    elif name in YAHOO_SERIES:
        s = _fetch_yahoo(name, YAHOO_SERIES[name])
    else:
        raise KeyError(f"Unknown series: {name}")

    s.to_frame(name=name).rename_axis("date").to_csv(path)
    return s


def load_all_raw(force_refresh: bool = False) -> dict[str, pd.Series]:
    """Download/load every series in the registry. Returns raw, unshifted series."""
    out = {}
    for name in list(FRED_SERIES) + list(YAHOO_SERIES):
        out[name] = get_series(name, force_refresh=force_refresh)
    return out


def to_month_end(s: pd.Series) -> pd.Series:
    """Resample a series to month-end, taking the last observation in each month.

    Applied uniformly so every series — daily (DGS2, HY OAS), monthly (CFNAI, CPI),
    or already-monthly (Yahoo with interval='1mo') — ends up on the same monthly
    index before any as-of shift or transform is applied.
    """
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    return s.resample("ME").last()


def apply_asof_shift(series_dict: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Apply the Section 3.3 as-of availability lag.

    CFNAI and CPIAUCSL: shift forward one month at load, BEFORE any transformation,
    so the value dated end-of-month-t is actually the month t-1 reference value.
    All other (market-price) series: no shift — month-t month-end value is usable
    in the end-of-month-t signal.
    """
    out = {}
    for name, s in series_dict.items():
        me = to_month_end(s)
        if name in STATISTICAL_RELEASE_SERIES:
            out[name] = me.shift(1)   # <-- THE one-line look-ahead fix (Section 3.3, G1)
        else:
            out[name] = me
    return out


def build_raw_panel(force_refresh: bool = False) -> pd.DataFrame:
    """Full data-layer entry point: download/cache, month-end align, as-of shift.

    Returns a single monthly DataFrame with columns:
    CFNAI, CPIAUCSL, DTWEXBGS, DGS2, BAMLH0A0HYM2, ^GSPC, ^SP500TR
    already as-of-corrected. Nothing downstream needs to think about lags again.
    """
    raw = load_all_raw(force_refresh=force_refresh)
    shifted = apply_asof_shift(raw)
    panel = pd.DataFrame(shifted)
    panel.index.name = "date"
    return panel


if __name__ == "__main__":
    panel = build_raw_panel()
    print(panel.tail())
    print("\nFirst non-NaN date per column:")
    print(panel.apply(lambda c: c.first_valid_index()))
