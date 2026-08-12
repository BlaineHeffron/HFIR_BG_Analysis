# HFIR Gamma-Background Analysis

Public analysis and read-only browser for released HPGe spectra and selected paper products.

## Scientific invariants

- Never write to the canonical SQLite database.
- Calibrated text spectra are measured detector counts, not unfolded incident flux.
- Published ancillary unfolded-flux CSVs reproduce released results; using them is not a new unfolding.
- Preserve run IDs, calibration, live time, normalization, detector orientation, reactor-cycle classification, and response-model identity in derived products.
- Label paper-exact reproduction, replotting, new calculation, and unavailable legacy workflows distinctly.

## Context routing

- Setup and public workflows: `README.md`.
- Figure status and output semantics: `docs/PAPER_REPRODUCIBILITY.md` and `config/paper_figures.json`.

## Change economy

- This is a public scientific repository. Track code, compact configuration, durable documentation, and tests needed for public reproduction. Do not track agent transcripts, review receipts, exploratory dumps, generated fit products, or internal process artifacts.
- Before adding more than three files or roughly 500 non-generated lines, explain why existing modules and workflows cannot carry the change and obtain maintainer approval for the larger design.
- Do not enlarge an already oversized module. Consolidate repeated setup and extract only genuinely reusable mechanisms; do not create one-use abstraction layers or compatibility wrappers.
- Every test must protect a distinct numerical, semantic, or regression failure. Prefer table-driven cases and shared fixtures. Do not enumerate cross-products of statuses, consumers, seeds, or metadata when one invariant covers them.
- Every quality gate must identify the scientific claim it protects, the failure it can detect, and the consequence of failure. No post-hoc thresholds, duplicate optimizer gates, or checks that merely certify other checks.
- Negative and exploratory results are allowed when clearly labeled and reproducible. They do not justify publication-scale infrastructure unless that infrastructure has an identified maintained consumer.

## Verification

- Run the narrow public-data or analysis checks documented for the changed workflow; do not trigger multi-gigabyte setup downloads unless required.
