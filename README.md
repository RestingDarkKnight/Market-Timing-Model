# Macro Voting Timing Model

A 5-factor macro voting model that switches the S&P 500 timing signal between "invested"
and "cash" monthly, built to the frozen spec in `report/BUILD_SPEC_v3.md`. Python only —
no R deliverable.

## What it does, in plain language

Five macro read-outs (growth, inflation, the dollar, the 2-year yield, and risk appetite)
each get turned into a z-score against their own trailing 10-year history, then a vote:
bullish (+1), bearish (−1), or abstain (0/no-opinion) if the z-score isn't stretched enough
or the series doesn't have enough history yet. Add the votes up: positive → be in the S&P
500 next month, negative → sit in cash, tied → keep doing whatever you were already doing.
No fitting, no optimization — every threshold and window is fixed in advance.

## Why it can't run here (this chat's sandbox) but will run in Colab

This code was built and unit-tested against synthetic data in a network-locked sandbox
(no egress to `fred.stlouisfed.org` or `finance.yahoo.com`). **It has not yet been run
against real data.** It's written to run cleanly in Google Colab or any normal internet-
connected environment.

## Running it

```bash
pip install pandas_datareader yfinance pandas numpy matplotlib scipy nbformat
```

Then either:
- Open `notebooks/results.ipynb` in Colab and run all cells, or
- From the repo root: `python -m src.pipeline`

Either path downloads and caches the 7 raw series to `data/raw/*.csv` on first run
(subsequent runs reuse the cache — pass `force_refresh=True` to `run_all()` to bypass it),
then writes `output/metrics.json` and `output/figures/*.png`.

**First thing to check after any run:** `results['tripwire']['tripped']` must be `False`.
If it's `True`, stop — Section 7.1 gate G8 requires a documented look-ahead audit before
any result is presented, no exceptions.

## Repo layout

```
macro-timing/
├── data/raw/              # cached source CSVs (populated on first run)
├── src/
│   ├── data.py            # download, cache, as-of shifts (Section 3.3)
│   ├── signals.py         # transforms, rolling z-scores, votes, position rule (Section 2)
│   ├── backtest.py        # engine (Section 4)
│   ├── evaluate.py        # metrics, confusion matrix, random null, robustness (Section 5)
│   └── pipeline.py        # wires it all together, writes metrics.json + figures
├── notebooks/results.ipynb  # narrative walkthrough, run this in Colab
├── output/metrics.json    # every Section 5.1 number, machine-readable (written on run)
├── output/figures/        # the 6 required figures (written on run)
├── report/BUILD_SPEC_v3.md  # the frozen spec this was built to
├── report/REPORT.md       # phase-mapped write-up — Phases 1-4 complete, 5-6 pending a real run
└── DEVIATIONS.md          # deviation log
```

## Where to look first

- `src/data.py` docstring — explains the two *different* look-ahead shifts and why they're
  kept in separate places (gate G1/G2).
- `src/signals.py` — every transform has a one-line economic-rationale comment above it.
- `report/REPORT.md` — Phase 1 (economic intuition per factor) and Phase 3 (skeptic's memo:
  look-ahead, snooping, overfitting, regime dependency, multicollinearity, survivorship,
  economic inconsistency) are written out in full already, independent of what the real
  numbers turn out to be.
