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

None of these approach 10% or 25%.

Current P2x (`ca486313`) plus the published matrices plus the published ROOT
files reproduces the published iso unfolds to 1.5e-7 relative (identical-input
rerun). HB4 and East 1 (a)=0 is the same check on DB-matching axes.

## Shield Center 0.2–1 MeV is a 200 keV band edge, not a flux change

The axis at file 186 moves by 0.054 keV near 200 keV (published low edge
0.272229 keV, width 0.699400 keV → A0+A1/2 = 0.326290 keV, width 0.699415 keV).
Channel contents are identical. A line sits on the 200 keV boundary: published
channel centres 197.85, 199.25, 199.95, 200.65 keV have 3548, 5700, 4435, 2984
counts on an ~1850 continuum (Ge-71m 198.4 keV is the plausible parent).
The 4435-count channel is 199.951 keV in the published hist and 200.009 keV
after the DB axis, so it crosses 200 keV. Input counts in 200–1000 keV rise
0.30% (1.47344e6 → 1.47787e6); 210–1000 keV is unchanged. RL then concentrates
the edge: unfolded iso 0.2–1 MeV (a)=+1.40%; at 0.21–1 MeV (a)=+1.11%. This is
not a calibration-driven continuum flux change. AD1 shape remains ≪10%.

## Uncertainty

The iso/front pair is a response-model systematic, kept as its own column, not
as a per-band statistical error.

Poisson-replica toys (`analysis/unfold/toy_stats`, RL, n=4) exist only for HB4.
Relative standard deviation of the band integral:

| scenario | 40–200 keV | 0.2–1 MeV | 1–6 MeV | 6–11.5 MeV |
| --- | ---: | ---: | ---: | ---: |
| HB4 iso | 1.38% | 1.03% | 0.026% | 0.058% |
| HB4 front | 1.49% | 0.11% | 0.040% | 0.063% |

MIF-on toys in that directory record only the 50–2000 keV fraction, not these
bands. No per-band replica set exists for Shield Center or East. Where a toy
spread is missing, do not invent one from the iso/front bracket.

## Full band table

Machine table: `docs/unfold_rebuild_bands_v1.2.2.csv` (all 8 names × iso/front ×
40–200, 0.2–1, 0.21–1, 1–6, 6–11.5, 7.6–7.7 keV × published / v1.2.1 / v1.2.2 /
(a)/(b)/total / LT factor / (b)−LT).

Max |(b) − LT ratio| on 0.2–1, 1–6 and 6–11.5 MeV, excluding the MIF-on iso
0.2–1 numerical floor (8e-8 Hz/mm²): **HB4 iso 0.2–1, +0.30%**. Other AD1-style
bands are ≤ 1e-4 except MIF-on iso 7.6–7.7 (+0.39%). The 0.30% is RL stopping
under a 1.4% rate rescaling, not a second input change.

Fe-line window 7.6–7.7 MeV, (a) only (axis files):

| location | iso (a) | front (a) |
| --- | ---: | ---: |
| MIF on | −8.96% | −6.81% |
| MIF off | −2.78% | −1.88% |
| East 18 | +0.60% | +4.44% |
| Shield Center | +1.15% | +2.10% |
| HB4, East 1 | 0 | 0 |

MIF-on iso 6–11.5 MeV (a) is +0.015%; the −9% is confined to the 100 keV Fe
window, as expected for a ~6 keV axis shift at 7.6 MeV.

Executable comparison: `scripts/compare_unfold_rebuild.py`.
Unfolded ROOT: `analysis/unfold/browser_v1.2.{1,2}_{iso,front}/` (gitignored).
Paper text is not edited.
