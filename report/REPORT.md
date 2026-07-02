# Macro Voting Timing Model — Report

Spec: `report/BUILD_SPEC_v3.md` (frozen, v3 — Python only, no R deliverable).

> **Status note.** Phases 1–4 below are written independent of any specific run's numbers
> — they describe the design and its known biases, which don't change. Phases 5–6 need the
> real output of `src/pipeline.run_all()` (see `notebooks/results.ipynb`), which could not
> be produced inside this build environment (no network access to FRED/Yahoo Finance — see
> `DEVIATIONS.md`). Run the notebook in Colab, then paste `output/metrics.json` and the six
> figures into the placeholders marked `[PENDING REAL RUN]` below.

---

## Phase 1 — Economic intuition

**1. Growth — CFNAI, 3-month moving average, sign +.**
The Chicago Fed National Activity Index is a diffusion-style composite of 85 real-activity
series, standardized so 0 = trend growth. We use the Fed's own CFNAI-MA3 convention (3-month
average) rather than the raw monthly print, because the raw index is noisy enough that even
the Chicago Fed itself reads the smoothed version for signal. Above-trend activity feeds
forward earnings, hence the positive prior. *Publication lag/revision:* first released
~3 weeks after month-end and revised for several months afterward as source data firms up;
we use the final, revised value, so the backtest is somewhat flattered relative to what a
real-time investor could have known (see Phase 3).

**2. Inflation — CPIAUCSL, YoY % change, sign −.**
CPIAUCSL is the seasonally-adjusted, all-items CPI. Year-over-year strips the CPI's strong
intra-year seasonality and month-to-month noise, and is the number markets and the Fed
actually quote. Higher inflation raises the discount rate applied to future cash flows and
compresses margins for firms that can't fully pass costs through — hence the negative prior.
*Revision issue:* CPIAUCSL undergoes annual seasonal-factor revisions each February/spring,
so historical CPI YoY values used in the backtest are not bit-for-bit what was "live" at the
time; the SA-revision caveat is disclosed here and in Phase 3.

**3. Trade — DTWEXBGS, 12-month % change, sign + (assignment prior).**
The broad trade-weighted dollar index. We use the sign the assignment specifies (dollar
strength read as evidence of foreign demand for US assets), but this sign is genuinely
debatable — a strong dollar also tightens financial conditions for dollar borrowers and can
hurt US multinational earnings via translation, which would argue for a negative sign
instead. We keep the assignment's prior as specified and let the Section 5.4 ablation show
how much this one factor actually moves the result (Phase 3, "economic inconsistency").

**4. Monetary policy — DGS2, 12-month change in level (pp), sign −.**
The 2-year Treasury yield is the cleanest single proxy for how much Fed tightening/easing is
priced in over the policy-relevant horizon (cleaner than Fed Funds itself, which barely moves
between meetings). A rising 2Y over the trailing year means the market has priced in more
tightening, raising the discount rate applied to equities — hence negative.

**5a. Risk appetite (price) — S&P 500 trailing 12M % return, sign +.**
Classic momentum/trend persistence: risk-taking begets risk-taking, drawdowns beget more
selling. This is also, honestly, the model's own asset voting on itself — flagged explicitly
in Phase 3 as a multicollinearity/circularity concern.

**5b. Risk appetite (credit) — BAMLH0A0HYM2 (HY OAS), 12-month change in level (pp), sign −.**
High-yield option-adjusted spreads are the credit market's real-time read on default risk and
liquidity stress. Widening spreads have historically *led* equity drawdowns (credit markets
often reprice risk before equity does), hence the negative sign on spread widening.

---

## Phase 2 — Model design

**Why voting beats a fitted model at this sample size.** The evaluation window is monthly
data from 2000, so even at 25+ years of history that's on the order of 300 observations for
5 factors. A regression (or anything fit) at that N:K ratio is prone to coefficient
instability and in-sample overfitting, especially across regime changes (2000 dot-com,
2008 GFC, 2020 pandemic, 2022 hiking cycle are four very different macro environments inside
one 25-year sample). A voting scheme with pre-committed signs acts as a strong regularizer:
the *sign* of each factor's contribution is fixed by economic theory rather than estimated,
so there's nothing for the model to overfit except the (also frozen) threshold — and even
that is stress-tested in Section 5.4 rather than tuned.

**Every frozen parameter (Section 2), and why:**
- Rolling z-score window = 120 months: long enough to span a full business cycle so the
  "normal" range each factor is judged against isn't itself regime-specific.
- Minimum history = 60 months: half the z-score window; below it a mean/std estimate is
  too noisy to trust, so the factor abstains rather than voting on thin information.
- Vote threshold = ±0.5: a genuine economic stance requires the z-score to clear half a
  standard deviation, not just tick off zero — avoids the model flipping on noise.
- Risk appetite fuses two series into one vote (rather than two independent votes) so the
  5-factor model doesn't implicitly become a 6-vote model that double-counts "risk."
- Hold-previous-position on a tied vote: a tie is genuinely "no information," and holding
  the existing position (rather than defaulting to either cash or invested) avoids injecting
  an arbitrary directional bias into every tie.
- Evaluation window starts Jan 2000, not later: needs both the 2000–02 and 2008–09 bear
  markets in-sample, since those are the regimes a timing overlay exists to help with. A
  post-2009 start window is one long bull market and can't distinguish genuine timing skill
  from simply being under-exposed.

---

## Phase 3 — Bias detection (skeptic's memo)

**Look-ahead.** Addressed in two places, kept deliberately separate for auditability
(see `src/data.py` and `src/backtest.py` docstrings): (1) CFNAI/CPI are shifted forward one
month at load, before any transform, because those releases carry a real publication lag;
(2) the fully-formed end-of-month-t signal is shifted one further month before being
multiplied against returns, since a position can't be acted on before it's observed. *Residual
bias that remains:* we use the final, revised value of CFNAI and CPI, not the first-release
"vintage" value a real-time investor would actually have seen. This flatters the backtest
somewhat, especially for CFNAI, which is revised meaningfully for months after first release.
Removing this cheaply would require a real-time vintage database (e.g. ALFRED) — out of scope
here, disclosed instead.

**Data snooping.** Every sign, transformation, window, and threshold in Section 2 was frozen
in the spec *before* any result was produced, in a separate document (`BUILD_SPEC_v3.md`)
version-controlled ahead of the build. The build instruction explicitly forbids tuning based
on results; any deviation is required to be logged in `DEVIATIONS.md`.

**Overfitting.** Zero fitted parameters — every number in Section 2 is a design choice
justified by economic reasoning (Phase 2), not by optimizing backtest performance.

**Regime dependency.** Addressed structurally by the sub-period table (Section 5.4): metrics
are reported separately for 2000–2009, 2010–2019, and 2020–present, so a reader can see
whether the model's behavior is stable or is really "one regime's story."

**Multicollinearity.** The 12-month S&P momentum factor (5a) is the strategy's own underlying
asset voting on itself — this is disclosed rather than hidden: part of what's marketed as a
"macro" model is structurally a trend-following model. High-yield spreads (5b) are also
strongly correlated with equity-market stress by construction, so risk appetite really
contributes one economically coherent signal via two correlated lenses, which is exactly why
the spec fuses them into a single vote rather than two.

**Survivorship.** Minimal here — the S&P 500 index itself already handles constituent
survivorship internally (index membership changes over time), and none of the macro series
are subject to survivorship bias (they're not asset universes). Flagged as a non-issue but
stated for completeness, since the assignment's bias checklist asks for it explicitly.

**Economic inconsistency.** The dollar factor's sign (Phase 1, factor 3) is genuinely
debatable — dollar strength could plausibly cut either way for US equities depending on the
channel (capital-flow demand vs. financial-conditions tightening vs. multinational earnings
translation). We keep the assignment-specified sign rather than choosing whichever sign
performs better in-sample (which would be data snooping), and instead let the leave-one-out
ablation (Section 5.4) show how much removing the trade factor entirely changes the result —
if the answer is "not much," the debatable sign matters less than it might seem.

---

## Phase 5 — Performance
**[PENDING REAL RUN]**

Run `notebooks/results.ipynb` in Colab, then replace this section with:
- The full metrics table (strategy / buy & hold / oracle / random-null mean) from
  `output/metrics.json`.
- The confusion matrix and up-month base rate, with the error-asymmetry note (missed
  up-months vs. avoided down-months).
- The random-timing null percentiles for Sharpe and CAGR.
- The six required figures, embedded or linked from `output/figures/`.
- The threshold-sensitivity, sub-period, and leave-one-out ablation tables.
- Switch count and average holding period.

## Phase 6 — Interpretation for senior PMs
**[PENDING REAL RUN]**

Once Phase 5's numbers exist, write here:
- Where the strategy earned its keep vs. paid rent, checked against the sub-period table
  (expected a priori: 2000–02 and 2008 are where a timing overlay should help most; long
  bull-run periods like parts of 2010–2019 are where whipsaws cost the most).
- Factor attribution from the leave-one-out ablation — which factor's removal moves the
  result most, and whether that's the momentum voter (5a) doing most of the work, which
  would corroborate the multicollinearity concern raised in Phase 3.
- Why underperforming buy-and-hold CAGR while running materially lower volatility and
  drawdown is a legitimate, even desirable, outcome for an allocation overlay — it's a risk
  budget decision, not a stock-picking one.
- Why simplicity (voting, not fitting) was the right choice at this sample size, tying back
  to Phase 2.

## Phase 7
Happens live: interview drill with Fable 5 at review, covering the Section 8 pre-committed
talking points in `BUILD_SPEC_v3.md`.
