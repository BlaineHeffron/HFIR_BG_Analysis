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
- File count, line count, test count, and coverage are review signals, never acceptance gates. Do not add policy or CI checks that enforce numeric growth thresholds, and do not split work to satisfy a metric.
- Before introducing a new long-lived concept--package, command, configuration family, dependency, or test framework--identify the missing existing seam and the maintained public consumer. Keep this explanation brief; do not create approval theater for ordinary maintenance.
- Do not enlarge an already oversized module. Consolidate repeated setup and extract only genuinely reusable mechanisms; do not create one-use abstraction layers or compatibility wrappers.
- Every test must protect a distinct numerical, semantic, or regression failure. Prefer table-driven cases and shared fixtures. Do not enumerate cross-products of statuses, consumers, seeds, or metadata when one invariant covers them.
- Every quality gate must identify the scientific claim it protects, the concrete failure it can detect, and the reporting or analysis action caused by failure. If failure changes no action, emit a diagnostic instead. No post-hoc thresholds, duplicate optimizer gates, meta-gates, or checks that merely certify other checks.
- Negative and exploratory results are allowed when clearly labeled and reproducible. They do not justify publication-scale infrastructure unless that infrastructure has an identified maintained consumer.

## Verification

- Run the narrow public-data or analysis checks documented for the changed workflow; do not trigger multi-gigabyte setup downloads unless required.

## Shared scientific work policy

- Read and obey [`../AGENTS.md`](../AGENTS.md), especially **Scientific work —
  mandatory**. Local rules above specialize that policy and do not weaken it.
