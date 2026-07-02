# Macro Voting Timing Model — Build Spec & Review Rubric (v3, FROZEN)
## Reconciled to the SSGA ISG assignment brief

**Version note.** v3 supersedes v2. Single change from v2: the R-port requirement is dropped entirely — Python is the sole implementation and submission language, no R script is planned, built, or graded. All other v2 changes remain in force (cash return = 0%, transaction costs = 0%, CPIAUCSL (SA) instead of CPIAUCNS, DTWEXBGS as the dollar series, BAMLH0A0HYM2 (HY OAS) as the spread series, simplified inflation transformation, expanded metric set (Calmar, hit ratio, holding period, capture ratios), evaluation window extended to Jan 2000 to include two bear-market regimes).

**Roles.** Planned/frozen with Claude Fable 5 (architect/reviewer). Built by Claude Sonnet 4.6 (builder). Returned to Fable 5 for rubric grading and the Phase 7 interview drill.

**Language.** Build and validate in **Python**, and Python only. There is no R deliverable — no port, no reproduction check, no R-related gate or checklist item.

**Instruction to the builder.** Implement exactly what is specified. No tuning, no "improvements," no parameter changes — even if results look bad. Unavoidable deviations go in the Deviations Log (Section 9). Undocumented deviation = automatic rubric failure. The assignment requires heavily commented code: every logical block carries a comment stating its **economic rationale**, not just its mechanics — the comments are the primary explanation for reviewers.

---

## 1. Project charter

**Goal.** A clean, honest, institution-grade backtest of a 5-factor macro voting model timing the S&P 500 vs cash (0%), monthly, binary allocation, 2000–present. Submission-shaped for the SSGA ISG brief: robustness, explainability, and economic intuition over predictive performance.

**Non-goals.** Beating buy-and-hold. Any parameter optimization. Machine learning. Fractional allocations. Multi-asset extensions.

**Success definition (committed before results).** Volatility and max drawdown meaningfully below buy-and-hold; Sharpe ≥ buy-and-hold; random-timing percentile reported truthfully whatever it says; every result explainable in one sentence of economics. Over a window containing 2000–02 and 2008, outperforming buy-and-hold is possible but is evidence to be audited, not celebrated.

**Time-box.** 2–3 build sessions. Extras go to the Later List.

---

## 2. Model specification (FROZEN)

### 2.1 Factors, priors, transformations

| # | Factor | Series | Transformation before z-score | Sign prior | Economic one-liner |
|---|--------|--------|-------------------------------|-----------|--------------------|
| 1 | Growth | CFNAI | 3-month moving average (the Fed's own CFNAI-MA3 smoothing) | **+** | Activity above trend feeds earnings |
| 2 | Inflation | CPIAUCSL | YoY inflation rate (12M % change of the index) | **−** | Inflation raises discount rates, squeezes real margins |
| 3 | Trade | DTWEXBGS | 12M % change, month-end | **+** | Per assignment prior; dollar strength read as US-asset demand. Sign stability flagged in Phase 3 |
| 4 | Monetary policy | DGS2 | 12M change in yield (percentage points), month-end | **−** | Rising 2Y = tightening priced in; higher discount rate |
| 5a | Risk appetite (i) | S&P 500 price index | Trailing 12M % return, month-end | **+** | Trend persistence; risk-taking begets risk-taking |
| 5b | Risk appetite (ii) | BAMLH0A0HYM2 (HY OAS) | 12M change (percentage points), month-end | **−** | Widening spreads = credit stress leads equity stress |

### 2.2 Z-scores
- z_t = (x_t − rolling mean) / rolling std over a **rolling 120-month window**, trailing data only (post availability-lag, Section 3.3).
- **Minimum history 60 months**; below it the component abstains.
- Signed score s_t = prior sign × z_t.

### 2.3 Votes
- s_t > +0.5 → +1; s_t < −0.5 → −1; otherwise 0 (abstain). Threshold frozen at ±0.5.
- Risk appetite casts **one** vote: average the signed scores of 5a and 5b, then apply the threshold once. **If only one component has ≥60 months of history, that component's signed score is used alone** (relevant before Dec 2002 for the spread component).

### 2.4 Position rule
- Sum of votes > 0 → 100% S&P 500 for the next month; < 0 → 100% cash; = 0 → **hold previous position**. First position of the evaluation window defaults to invested.

---

## 3. Data specification

### 3.1 Series and sources

| Series | Source | Data begins | Pull from |
|---|---|---|---|
| CFNAI | FRED `CFNAI` | 1967 | Jan 1985 |
| CPI (SA) | FRED `CPIAUCSL` | 1947 | Jan 1985 |
| Dollar index | FRED `DTWEXBGS` | Jan 2006 | inception |
| 2Y yield | FRED `DGS2` (month-end value) | 1976 | Jan 1985 |
| S&P 500 price (signal) | yfinance `^GSPC` (month-end close) | long | Jan 1985 |
| S&P 500 total return (performance) | yfinance `^SP500TR` | Jan 1988 | inception |
| HY OAS | FRED `BAMLH0A0HYM2` (month-end value) | Dec 1996 | inception |

Cache all raw downloads as CSVs in `data/raw/` for reproducibility.

### 3.2 Evaluation window and factor entry
- **Evaluation window: Jan 2000 – latest complete month.** Justification: includes two full bear markets (2000–02, 2008–09) plus 2020 and 2022 — the regimes a timing model exists for. A 2009-start window is one long bull market and cannot distinguish signal from under-exposure.
- Factor entry schedule under the 60-month rule (document in the report):
  - Growth, inflation, policy, risk(5a): warm well before Jan 2000 — voting from the start.
  - Risk(5b) spread component: joins the risk composite ~Dec 2002.
  - Trade: first vote ~Jan 2012; abstains before (DTWEXBGS begins 2006).
- With 4 active voters pre-2012, tie sums of 0 occur; the hold-previous rule covers them.
- Sub-period splits for reporting: 2000–2009, 2010–2019, 2020–present.

### 3.3 As-of availability lags (look-ahead protection — FROZEN)
- Market-price series (dollar, 2Y, S&P, HY OAS): month-t month-end values usable in the end-of-month-t signal. No extra lag.
- **CFNAI and CPI: shift forward one month at load, before any transformation** (signal at end of month t uses reference month t−1). One explicit `.shift(1)` in the data layer.
- **Signal-to-return alignment:** the end-of-month-t signal sets the position that earns the **month t+1** return. One further explicit shift between signal and returns.
- All rolling statistics trailing-only. Full-sample statistics anywhere in the signal path are prohibited.

---

## 4. Backtest engine (per assignment constraints)

1. Portfolio return in month m = position(m) × sp500_total_return(m) + (1 − position(m)) × 0.
2. **Cash return = 0%. Transaction costs = 0%.** (Assignment constraints.)
3. Track and report switch count and average holding period anyway — they are free information about strategy behavior.
4. Equity curves (growth of $1, log scale): strategy, buy & hold (total return), perfect-foresight oracle recomputed on total-return data with 0% cash months. Always-cash is a flat line at $1 — state it, no need to plot it.
5. Signal uses the price index (`^GSPC`); performance uses total return (`^SP500TR`). One line in the report explains why (dividends compound; both legs must be measured on the same return stream).

---

## 5. Evaluation specification

### 5.1 Metrics table (strategy, buy & hold, oracle, random-null mean)
CAGR; annualized volatility; Sharpe (rf = 0 per assignment); max drawdown; **Calmar ratio** (CAGR / |maxDD|); longest drawdown duration (months); **hit ratio** (% of months the allocation matched the realized sign of the market return); % months in market; number of switches; **average holding period** (months per position spell); **upside capture** and **downside capture** (compounded strategy return over benchmark-up months ÷ compounded benchmark return over those months; same for down months).

### 5.2 Classification view
Confusion matrix of monthly calls: (in market / in cash) × (market up / market down), with the up-month base rate stated alongside. Note the error asymmetry: missed up-months cost more than avoided down-months save.

### 5.3 Random-timing null (required)
1,000 simulated strategies invested in the same number of months as the real strategy, months chosen uniformly at random. Report the distribution of Sharpe and CAGR and the real strategy's percentile in each. Fixed, recorded RNG seed.

### 5.4 Robustness suite
- Threshold sensitivity: ±0.3 / ±0.5 / ±0.7 side by side, one-sentence verdict on whether the qualitative conclusion survives.
- Sub-period metrics: 2000–2009, 2010–2019, 2020–present.
- Factor ablation: rerun with each factor removed one at a time (leave-one-out). Report which factor's removal changes results most — this is the Phase 6 "which factors contributed" evidence, obtained without any optimization.

### 5.5 Required figures
1. Log-scale growth-of-$1: strategy vs buy & hold vs oracle.
2. Drawdown curves: strategy vs buy & hold, one chart.
3. Position ribbon: in/out shading over the S&P curve — bad calls visible to the eye.
4. Vote-sum time series with the zero line and factor-entry dates marked.
5. Histogram of random-null Sharpes with the strategy marked.
6. Per-factor signed-score heatmap or small-multiples over time (feeds Phase 6 attribution).

---

## 6. Deliverables mapped to the assignment's phases

```
macro-timing/
├── data/raw/                  # cached source CSVs
├── src/
│   ├── data.py                # download, cache, as-of shifts (Section 3.3 lives here, in one place)
│   ├── signals.py             # transforms, rolling z-scores, votes, position
│   ├── backtest.py            # engine of Section 4
│   └── evaluate.py            # metrics, null sim, ablation, figures
├── notebooks/results.ipynb    # narrative walkthrough, all figures inline
├── output/metrics.json        # every Section 5.1 number, machine-readable
├── report/REPORT.md           # phase-mapped write-up (below)
├── README.md                  # how to run; model in plain language
└── DEVIATIONS.md              # Section 9 log ("No deviations." if empty)
```

REPORT.md sections must map to the assignment:
- **Phase 1 — Economic intuition:** one subsection per factor: rationale, expected relationship, bullish/bearish direction of rising values, chosen transformation and why it stationarizes, revision/publication-lag issues (CFNAI: ~3-week lag, heavy revisions, direction of bias = flatters backtest; CPI SA: annual seasonal-factor revisions; market series: clean).
- **Phase 2 — Model design:** the voting design and why crude beats fitted at n≈300 monthly observations; every frozen parameter listed with its justification.
- **Phase 3 — Bias detection (skeptic's memo):** look-ahead (how Section 3.3 addresses it, what residual remains — revised CFNAI values); data snooping (signs and thresholds pre-committed in this document before any results); overfitting (zero fitted parameters); regime dependency (sub-period table); multicollinearity (12M S&P factor is the asset's own momentum; HY spreads correlated with it — the "macro" model has a trend-following voter inside it); survivorship (index-level data, minimal, but say why); economic inconsistency (the dollar factor's debatable sign — kept per assignment prior, ablation shows its marginal effect).
- **Phase 5 — Performance:** metrics table + figures.
- **Phase 6 — Interpretation for senior PMs:** where the strategy earned its keep (expected: 2000–02, 2008) and where it paid rent (whipsaws in bull runs); factor attribution from ablation; why underperformance vs buy-and-hold with materially lower drawdown is an acceptable, even desirable, outcome for an allocation overlay; why simplicity was chosen.
- **Phase 7** happens live: interview drill with Fable 5 at review.

---

## 7. Review rubric (graded by Fable 5 on return)

### 7.1 Hard gates — any single failure rejects the build
| # | Gate | Verification |
|---|------|--------------|
| G1 | CFNAI and CPI shifted +1 month at load, before transforms | Inspect `data.py` |
| G2 | End-of-month-t signal → month t+1 return; no same-month contamination | Inspection + hand spot-check of three dates |
| G3 | All rolling stats trailing-only; no full-sample statistics in the signal path | Grep + inspection |
| G4 | CPIAUCSL YoY used per assignment; SA-revision caveat present in Phase 3 memo | Inspection |
| G5 | Cash = 0%, costs = 0%, Sharpe rf = 0, binary positions only — assignment constraints exactly honored | Inspection + metrics sanity |
| G6 | No performance statistic before Jan 2000; factor-entry/abstention rules implemented as specified | Inspection |
| G7 | Every frozen parameter matches Section 2, or the change is in DEVIATIONS.md | Diff against this spec |
| G8 | **Tripwire:** full-window strategy Sharpe > 1.0, or CAGR > buy-and-hold + 5pp → build halted and a look-ahead audit documented before presenting | Presence of audit note |

### 7.2 Graded checklist — pass requires ≥ 80%
| # | Item |
|---|------|
| C1 | Metrics table complete, all Section 5.1 metrics, all four columns |
| C2 | Confusion matrix + base rate + error-asymmetry note |
| C3 | Random null: 1,000 sims, seed recorded, percentiles reported |
| C4 | Threshold sensitivity with one-sentence verdict |
| C5 | Sub-period metrics for all three windows |
| C6 | Leave-one-out factor ablation reported |
| C7 | All six figures present and legible |
| C8 | Switch count and average holding period reported |
| C9 | REPORT.md contains all phase sections incl. the full Phase 3 skeptic's memo |
| C10 | `metrics.json` consistent with notebook figures |
| C11 | Code runs end-to-end from cached CSVs; comments state economic rationale per block |
| C12 | Oracle recomputed on total-return data with 0% cash |

### 7.3 Review protocol
Return to Fable 5 with: repo (or zip), `metrics.json`, notebook, REPORT.md, DEVIATIONS.md. Review output: gate-by-gate verdict, checklist score, suspected look-ahead paths if any, then the Phase 7 interview drill — questions asked one at a time, answers graded like a real ISG interviewer, covering economic intuition, factor construction, data alignment, look-ahead, robustness, why no ML, and interpretability trade-offs.

---

## 8. Pre-committed talking points the drill will test
- Why voting beats regression at this sample size (coefficient instability, sign priors as regularization).
- Why the tripwire exists (extraordinary results are evidence of leakage, not genius).
- The residual bias you cannot remove cheaply (revised CFNAI) and its direction.
- Why the momentum voter makes this "macro" model partly a trend model, and why that's disclosed rather than hidden.
- Why 0% cash and 0 costs are stylizations, and which direction each biases the comparison.

---

## 9. Deviations log (builder fills in)

| Date | Section | What changed | Why | Expected effect on results |
|------|---------|--------------|-----|---------------------------|
| — | — | — | — | — |
