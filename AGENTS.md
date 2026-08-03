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

## Verification

- Run the narrow public-data or analysis checks documented for the changed workflow; do not trigger multi-gigabyte setup downloads unless required.
