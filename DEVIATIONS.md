# Deviations Log

| Date | Section | What changed | Why | Expected effect on results |
|------|---------|--------------|-----|---------------------------|
| 2026-07-02 | Roles / build environment | Code was written and unit-tested against synthetic, non-meaningful data instead of the real FRED/Yahoo Finance series specified in Section 3.1. | The build sandbox has no network egress to `fred.stlouisfed.org` or `finance.yahoo.com` (confirmed via direct request: `host_not_allowed`). The user's execution environment (Google Colab) does have this access. | None on the final results, provided `src/pipeline.run_all()` is actually run against live data before any number in `report/REPORT.md` Phase 5/6 is treated as real. Numbers produced during synthetic dry-run testing were discarded and never written into REPORT.md or metrics.json in this repo. |
| — | — | — | — | No other deviations. Language deviation from the original R-inclusive brief (v2 -> v3: R port dropped) was an explicit, user-confirmed scope change to the frozen spec itself, not a build-time deviation from spec — see `BUILD_SPEC_v3.md` version note. |
