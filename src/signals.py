"""
signals.py — factor transforms, rolling z-scores, votes, and the position rule (Section 2).

Everything here operates on the as-of-corrected monthly panel produced by data.py.
No full-sample statistic is ever computed (Section 3.3 / gate G3): every rolling
window is trailing-only via pandas .rolling(), which by construction only looks
backward from each row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_WINDOW = 120   # months (Section 2.2)
MIN_HISTORY = 60    # months (Section 2.2)
VOTE_THRESHOLD = 0.5  # Section 2.3, frozen

SIGN_PRIOR = {
    "growth": +1,
    "inflation": -1,
    "trade": +1,
    "policy": -1,
    "risk_price": +1,   # 5a
    "risk_spread": -1,  # 5b
}


# ---------------------------------------------------------------------------
# Step 1 — transformations (Section 2.1, "before z-score" column)
# ---------------------------------------------------------------------------
def transform_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Apply each factor's pre-z-score transformation. Input: as-of-corrected raw panel.

    Every transform below uses only trailing data (pct_change / rolling.mean / diff
    over a fixed backward-looking window) — none of them peek forward.
    """
    out = pd.DataFrame(index=panel.index)

    # 1. Growth: CFNAI, 3-month moving average (the Fed's own CFNAI-MA3 smoothing).
    #    Economic rationale: CFNAI is noisy month to month; the Fed's own convention
    #    for reading trend growth is the 3-month average, so we match that convention
    #    rather than inventing our own smoothing.
    out["growth"] = panel["CFNAI"].rolling(3).mean()

    # 2. Inflation: CPIAUCSL YoY % change.
    #    Economic rationale: YoY strips the strong within-year seasonality/noise of
    #    MoM CPI prints and is the number markets and the Fed actually watch.
    out["inflation"] = panel["CPIAUCSL"].pct_change(12) * 100

    # 3. Trade: DTWEXBGS 12M % change.
    #    Economic rationale: level of the dollar index is not economically meaningful
    #    on its own; the rate of change captures the directional flow of dollar
    #    strength/weakness that the sign prior is about.
    out["trade"] = panel["DTWEXBGS"].pct_change(12) * 100

    # 4. Policy: DGS2 12M change in level (percentage points, NOT % change — a yield
    #    going from 1% to 2% is a 100bp tightening, not "100%").
    #    Economic rationale: the change in the 2Y yield over the last year is the
    #    cleanest single proxy for how much tightening/easing markets have priced in.
    out["policy"] = panel["DGS2"].diff(12)

    # 5a. Risk appetite (price): S&P 500 price index trailing 12M % return.
    #    Economic rationale: momentum/trend persistence — a classic, robust equity
    #    risk-appetite signal.
    out["risk_price"] = panel["^GSPC"].pct_change(12) * 100

    # 5b. Risk appetite (credit): HY OAS 12M change in level (percentage points).
    #    Economic rationale: widening high-yield spreads are the market's own
    #    real-time read on credit stress, which historically leads equity drawdowns.
    out["risk_spread"] = panel["BAMLH0A0HYM2"].diff(12)

    return out


# ---------------------------------------------------------------------------
# Step 2 — rolling z-scores (Section 2.2)
# ---------------------------------------------------------------------------
def rolling_zscores(transformed: pd.DataFrame) -> pd.DataFrame:
    """Trailing 120-month rolling z-score per factor, NaN before 60 months of history.

    min_periods=MIN_HISTORY implements "component abstains below 60 months of history"
    at the z-score stage; the vote stage (below) turns those NaNs into explicit
    abstentions rather than accidentally treating them as 0-votes.
    """
    z = pd.DataFrame(index=transformed.index)
    for col in transformed.columns:
        mean = transformed[col].rolling(ROLL_WINDOW, min_periods=MIN_HISTORY).mean()
        std = transformed[col].rolling(ROLL_WINDOW, min_periods=MIN_HISTORY).std()
        z[col] = (transformed[col] - mean) / std
    return z


def signed_scores(z: pd.DataFrame) -> pd.DataFrame:
    """s_t = prior sign * z_t (Section 2.2)."""
    s = pd.DataFrame(index=z.index)
    for col in z.columns:
        s[col] = SIGN_PRIOR[col] * z[col]
    return s


# ---------------------------------------------------------------------------
# Step 3 — votes (Section 2.3)
# ---------------------------------------------------------------------------
def _vote_from_score(s: pd.Series, threshold: float = VOTE_THRESHOLD) -> pd.Series:
    """Map a signed score to {-1, 0, +1} with NaN (abstain) preserved as 0-vote-eligible.

    threshold is parametrized (default frozen at 0.5) solely so evaluate.py's
    Section 5.4 threshold-sensitivity check (+/-0.3 / +/-0.5 / +/-0.7) can rerun
    this function without duplicating it. The FROZEN production threshold is 0.5.
    """
    v = pd.Series(0, index=s.index, dtype=float)
    v[s > threshold] = 1
    v[s < -threshold] = -1
    v[s.isna()] = np.nan  # abstain: excluded from the vote sum entirely, not a 0-vote
    return v


def compute_votes(s: pd.DataFrame, threshold: float = VOTE_THRESHOLD,
                   exclude: str | None = None) -> pd.DataFrame:
    """One vote column per factor, plus the merged risk-appetite vote (Section 2.3).

    Risk appetite is special: it casts ONE vote from the average of the two
    signed scores (5a, 5b). Before Dec 2002 only 5a has 60 months of history,
    so per spec we fall back to 5a alone rather than abstaining entirely --
    a bull/bear-market momentum read is still informative on its own.

    exclude: factor name to drop entirely (one of "growth","inflation","trade",
    "policy","risk") -- used ONLY by the Section 5.4 leave-one-out ablation,
    never in the production run.
    """
    votes = pd.DataFrame(index=s.index)
    if exclude != "growth":
        votes["growth"] = _vote_from_score(s["growth"], threshold)
    if exclude != "inflation":
        votes["inflation"] = _vote_from_score(s["inflation"], threshold)
    if exclude != "trade":
        votes["trade"] = _vote_from_score(s["trade"], threshold)
    if exclude != "policy":
        votes["policy"] = _vote_from_score(s["policy"], threshold)

    if exclude != "risk":
        risk_price, risk_spread = s["risk_price"], s["risk_spread"]
        both = risk_price.notna() & risk_spread.notna()
        only_price = risk_price.notna() & risk_spread.isna()

        risk_signed = pd.Series(np.nan, index=s.index)
        risk_signed[both] = (risk_price[both] + risk_spread[both]) / 2
        risk_signed[only_price] = risk_price[only_price]
        # (neither available -> stays NaN -> abstain; cannot occur after 1985 given
        #  the data start dates in Section 3.1, but handled for completeness)

        votes["risk"] = _vote_from_score(risk_signed, threshold)
    return votes


# ---------------------------------------------------------------------------
# Step 4 — vote sum and position rule (Section 2.4)
# ---------------------------------------------------------------------------
def vote_sum(votes: pd.DataFrame) -> pd.Series:
    """Sum of votes, treating abstentions (NaN) as excluded from the sum, not as 0.

    A factor that hasn't reached 60 months of history contributes nothing either
    way; a factor that HAS reached 60 months but scores between -0.5 and +0.5 is a
    genuine 0 vote and IS included in the sum (it can still produce a tie).
    """
    return votes.sum(axis=1, skipna=True)


def compute_positions(vsum: pd.Series, start_invested: bool = True) -> pd.Series:
    """Sum > 0 -> invested (1); sum < 0 -> cash (0); sum == 0 -> hold previous position.

    First position of the evaluation window defaults to invested (Section 2.4).
    This function should be called AFTER the panel has been trimmed to the
    evaluation window (Jan 2000+, Section 3.2) — "first position" means the
    first month of that window, not the first month vote_sum happens to be
    non-NaN.
    """
    pos = pd.Series(index=vsum.index, dtype=float)
    prev = 1.0 if start_invested else 0.0
    for dt, v in vsum.items():
        if v > 0:
            cur = 1.0
        elif v < 0:
            cur = 0.0
        else:
            cur = prev
        pos[dt] = cur
        prev = cur
    return pos


def build_signal_pipeline(panel: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Run transform -> zscore -> signed score -> vote -> vote_sum on the full panel.

    Returns a dict of every intermediate stage (useful for the required Figure 4/6
    and for the factor-ablation robustness check, which needs to rerun this with one
    column dropped at a time).
    """
    transformed = transform_factors(panel)
    z = rolling_zscores(transformed)
    s = signed_scores(z)
    votes = compute_votes(s)
    vsum = vote_sum(votes)
    return {
        "transformed": transformed,
        "zscores": z,
        "signed_scores": s,
        "votes": votes,
        "vote_sum": vsum,
    }
