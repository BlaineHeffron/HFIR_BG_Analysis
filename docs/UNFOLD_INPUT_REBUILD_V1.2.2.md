# Unfold input rebuild, local v1.2.2, interim published matrices

Question: how much do the published unfolds change if the inputs use
database calibration and v1.2.2 live times, keeping the published
migration matrices.

## Inputs

- Database: `data/HFIRBG_public_data_v1.2.2/HFIRBG.db` (`HFIR_BG_Analysis` `91bd502`)
- Comparison database: v1.2.1 (same calibration, older live times)
- Spectra: `browser.py` `load_spectrum` (database A0/A1, 1-based channel centers)
- ROOT writer: `scripts/write_ge_hist.cpp` via `scripts/build_unfold_inputs.py`
- Isolated products: `data/unfold_inputs_v1.2.1/` and `data/unfold_inputs_v1.2.2/`
  (gitignored). Frozen `HFIRBG/data`, ancillary CSVs, and `unfold_inputs_v1.2.0`
  were not overwritten.
- Matrices (interim, published geometry):
  - `scripts/private/migration_matrix.root`
    sha256 `19bc5d8f6728a63fadf00aae0676414ef0f2f33a6a245ad9b559eda4d8e4d9fb`
  - `scripts/private/migration_matrix_front.root`
    sha256 `af3b19e1474dac09e063ab5c38a3361b3c5763472d583cf48aaab7c30bdbf984`
- Unfolder: `P2x_Analyze` git `ca486313`, class `GeCollimatorUnfolder`,
  ELow 40, EHigh 12000, same `run_all_unfolds.sh` settings.

Per-file calibration groups (v1.2.1 and v1.2.2 identical):

| file_id | name | group | A0 (keV) | A1 (keV/ch) | v1.2.1 live (s) | v1.2.2 live (s) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 109 | MIF on | 43 | -0.913249 | 0.700241 | 14505.63 | 14193.39 |
| 871 | MIF off | 178 | -1.031229 | 0.700241 | 86400.00 | 86116.54 |
| 186 | Shield Center (CYCLE491) | 62 | -0.023417 | 0.699415 | 81632.60 | 81469.96 |
| 402 | HB4 | 101 | 0.016636 | 0.699393 | 74759.16 | 73726.17 |
| 655 | East 18 | 145 | -0.288956 | 0.699401 | 5765.29 | 5750.95 |
| 1484 | East 1 | 287 | 0.022791 | 0.699376 | 77551.42 | 77290.16 |
| 1561 | PROSPECT_DOWN_OVERNIGHT | 300 | -0.884685 | 0.699852 | 58610.38 | 58610.38 |
| 1078 | NE_FACING_EAST | 219 | 0.0 | 0.699042 | 59271.11 | 59271.11 |

CYCLE461 is a symlink to CYCLE491. File 186 CNF real equals live in v1.2.1;
v1.2.2 applies the equal-counter dead-time model (−0.20%). That model cannot
be checked against a measured live/real deficit for this file.

## Decomposition

(a) calibration only: published ROOT axis → database axis, v1.2.1 live time.
(b) live time only: v1.2.1 → v1.2.2, same database axis.
Band integrals of `UnfoldedEnergy` (Hz/mm², 1 keV bins) versus published CSVs.
The published iso/front pair remains the response-model bracket (not a
statistical error on Richardson–Lucy).

Shield Center (feeds the PROSPECT AD1 shape-only comparison):

| scenario | band | (a) cal | (b) live | total vs published |
| --- | --- | ---: | ---: | ---: |
| iso | 0.2–1 MeV | +1.40% | +0.20% | +1.60% |
| iso | 1–6 MeV | +0.02% | +0.20% | +0.22% |
| iso | 6–11.5 MeV | +0.05% | +0.20% | +0.25% |
| front | 0.2–1 MeV | +1.22% | +0.20% | +1.42% |
| front | 1–6 MeV | +0.02% | +0.20% | +0.22% |
| front | 6–11.5 MeV | +0.08% | +0.20% | +0.28% |

None of these approach 10% or 25%. The iso/front bracket at 0.2–1 MeV is
already a factor of ~7, much larger than 1.6%.

HB4 and East 1: (a) is 0 (ROOT already used the DB axis). (b) is +1.40% and
+0.34%, equal to the live-time rate scale.

MIF on and East 18: (a) is large in the 40 keV bin / 0.2–1 MeV front MIF
because those published axes were not the DB calibration. The physically
relevant mid/high bands stay at the percent level except MIF-on iso 0.2–1,
which is an already unphysical 8e-8 Hz/mm² dip.

Executable comparison: `scripts/compare_unfold_rebuild.py`.
Unfolded ROOT: `analysis/unfold/browser_v1.2.{1,2}_{iso,front}/` (gitignored).
Paper text is not edited. Figures not committed pending review.
