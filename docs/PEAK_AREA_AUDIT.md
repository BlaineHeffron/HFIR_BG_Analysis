# Python peak-area and paper-impact audit

Audit target: `PeakFit.area()` and `MultiPeakFit.area()` at main revision
`6aebba05b09b6c5b40cc600c2ae1367adb6c51fb`, with historical revision
`3888c8b8fed18b4b70e6e2c7f395dc9fb106ea57` checked separately. Paper
references are to arXiv `2607.05834v1`.

This document preserves the historical audit. The separate [phase-2
peak-statistics correction](PEAK_STATISTICS_CORRECTION.md) now provides the
declared simultaneous-Poisson candidate workflow; it does not reinterpret the
legacy columns below.

## Finding

Both methods had returned an average net spectral density while naming it a
peak area. They sum measured detector counts in a window of
`centroid +/- 3.5 sigma`, subtract the fitted background summed over the same
fractional channel-index interval, and then divided that net count by
`7 sigma_keV`. The final division is not part of an area estimator:

```text
N_window = sum_index(data counts/bin - background counts/bin)       [counts]
legacy   = N_window / (7 sigma_keV)                                 [counts/keV]
area     = N_window                                                 [counts]
```

The unit-correct default is `area(mode="net_counts")`. Exact historical values
remain available as `area(mode="legacy_window_density")` and through
`--paper-legacy` on the paper-number scripts. Re-expressing the legacy fit in
counts fixes its units; it does not independently validate that fit's
background, centroid, or sigma.

This is a fitted-resolution dependence, not a material detector-bin-width
dependence. Rebinning that preserves counts scales the counts in each bin while
reducing the number of bins in the physical window. Both `N_window` and
`N_window/(7 sigma)` therefore remain stable, apart from fractional-bin
quadrature. The focused test spans 0.25--2.0 keV bins and bounds this numerical
effect below 0.2%. At fixed injected net counts, however, the legacy result
scales as `1/sigma`; a two-times-wider fitted peak returns half the alleged
area.

## Units at each seam

| Interface | Estimand | Unit |
|---|---|---|
| Released spectrum text, count column | measured events in one detector channel | counts/bin |
| `SpectrumData.data`; fitter `ys`; fitted model ordinates | measured or modeled channel content | counts/bin |
| `get_data_x()` and fit `xs` | calibrated channel center | keV |
| `integrate_lininterp_range()` | fractional-index sum, not an energy integral | input ordinate summed over bins |
| Gross and fitted-background window sums | detector events in `+/-3.5 sigma` | counts |
| `area(mode="net_counts")` | background-subtracted window count | counts |
| `area(mode="legacy_window_density")` | mean net density across the fitted window | counts/keV |
| `get_areas(..., lt=live_time)` with net mode | live-time-normalized window count | counts/s (Hz) |
| Same helper with legacy mode | live-time-normalized window density | counts/(s keV) |
| Relative and escape-peak ratios | ratio of either like unit | dimensionless, but legacy ratios contain a sigma ratio |
| ROOT-free parent-group sensitivity fit | full fitted Gaussian component and rate | counts; counts/s |
| P2x efficiency fit | `sqrt(2 pi) * sigma[MeV] * height[Hz/MeV]` | Hz |

Calibrated spectra remain measured detector counts. Nothing in this audit
interprets them as unfolded incident flux. The parent-group calculation is a
new calculation, not a new unfolding.

## History

- Initial implementation `c2c84155` returned `(net_counts, uncertainty)`.
- Commit `5cb9e0afd362e7d3703456531594b0e1424555e3` (2022-07-14) added
  the `/ (7 sigma)` division to both classes while adding the Russian-doll
  neutron/gamma comparison.
- No later commit changed the area estimand. Relevant files have identical
  blobs at `3888c8` and current main `6aebba0`.
- Later fit-window and peak-group behavior did change. Consequently, a current
  replay of an old estimator is not automatically a paper-exact replay.

## Consumer map

| Consumer | Path | Correct unit / consequence | Paper status |
|---|---|---|---|
| `PeakFit.display`, `MultiPeakFit.display` | direct | now reports net window counts | diagnostic only |
| `utilities.util.get_areas` | direct shared helper | counts, or counts/s when live time supplied | all callers below |
| `rd_peak_fitter.py` | `fit_spectra -> get_areas -> area` | cross-energy relative counts; legacy ratios carry `sigma_ref/sigma_E` | Table 3; affected; paper-exact legacy option |
| `RD_neutron_gamma_fit.py` | `retrieve_peak_areas -> get_areas -> area` | simulated/data counts/s and ratios | Cd-mixture tuning; unavailable legacy workflow |
| `peak_ratio_compare.py` | `compare_peaks -> get_areas -> area` | FEP/SEP/DEP count ratios | Table 8; affected; unavailable exact legacy workflow |
| `plot_sim_efficiencies.py::main` | direct, now explicit mode | net peak rate is Hz; `--paper-legacy` output is Hz/keV | affected dormant extraction; not a current paper product |
| `plot_sim_efficiencies.py::compare_sim_eff` | reads prior CSVs | same-energy curve ratio can largely cancel a common width factor | no fit call; simulation inputs unavailable |
| `neut_sim_peaks.py` | direct, explicit net mode | diagnostic net counts; labels updated | no paper artifact found |
| `test_24Na_decay.py` | direct, explicit net mode | diagnostic net counts versus time; axis label updated | no paper artifact found |

Calibration uses centroids. Resolution products use sigmas. Peak labels use
centroids and line lists. Those paths do not consume returned areas.

PyROOT imports were moved from module scope to the three methods that actually
construct or read ROOT objects. This is a separate enabling refactor: it lets
the count estimator, public replay, and focused tests run in a ROOT-free
environment. ROOT-backed method behavior is otherwise unchanged.

## Immutable public replay

Use an empty temporary output directory. The browser opens SQLite with
`mode=ro` and `PRAGMA query_only`; the spectrum files are only read.

```bash
audit_output_dir=$(mktemp -d /tmp/hfir-peak-area-audit.XXXXXX)
python3 scripts/audit_peak_area_impact.py \
  --bundle /path/to/HFIRBG_public_data_v1.1.0 \
  --output-dir "$audit_output_dir"
```

The command verifies database SHA-256
`c78bc8fa6ef7dbe1a8ea5d0189e69eb555c8a488fd582ff04b965a08aa1985e9`
and writes:

- `rd_peak_area_impact.csv`: all Table 3 fits, units, sigmas, legacy density
  values, legacy-fit unit-corrected values, and historical uncertainties.
- `rd_parent_group_sensitivity.csv`: independent Poisson/covariance result for
  the 7367.9/558.5 keV anchor under two calibration constraints.
- `mif_escape_peak_ratio_impact.csv`: all 24 Table 8 data ratios from the
  current ordered fit, under both area estimands.
- `mif_parent_group_sensitivity.csv`: a separate Poisson/covariance
  sensitivity calculation for four convergent parent groups.
- `claim_impact.csv`: machine-readable affected/unaffected/not-reproducible
  findings.
- `manifest.json`: run/file IDs, calibration, live time, input hashes, peak
  order, model assumptions, and history.
- `fit_diagnostics.txt`: current fitter grouping and offset diagnostics.

### Table 3: Russian-doll lines

The legacy replay uses four public files assigned to
`Cycle493_RD_low_gain`: file IDs 1716, 1765, 334, and 1676; total live time
260013.89 s; `A0=-0.38892709072078885 keV` and
`A1=0.6995359644108052 keV/channel`. It reproduces all 34 published numerical
rows to the displayed three decimals. The paper gives a dash for the broad
478 keV feature; the fitter returns an unstable value with uncertainty much
larger than its estimate, so it is retained only as an audit diagnostic.

Selected results:

| Energy (keV) | Published / legacy density ratio | Legacy fit, unit-corrected | Unit-only change |
|---:|---:|---:|---:|
| 238.6 | 0.346224 | 0.294925 | 0.85x |
| 1293.6 | 0.564969 | 0.659510 | 1.17x |
| 5433.1 | 0.002238 | 0.008385 | 3.75x |
| 5824.6 | 0.004972 | 0.012013 | 2.42x |
| 7367.9 | 0.002911 | 0.011582 | 3.98x |
| 7916.3 | 0.001253 | 0.002474 | 1.97x |

Two provenance conflicts remain. No public v1.1.0 run is named for Cycle 498.
Every RD family after Cycle493/PreCycle494 includes `lead` in its run name;
`Cycle493_RD_low_gain` is therefore the only reactor-on low-gain family
consistent with the caption's “no additional lead / water shield” qualifier.
The script selection matches the shield description, but the caption's cycle
number does not match the public catalog. The paper method also says run-by-run
normalization followed by inverse-variance combination, while the script first
combines the four Cycle 493 spectra and fits one accumulated spectrum. These
remain unresolved caption/method-versus-data discrepancies.

The Pb-207 7367.9 keV / Cd-113 558.5 keV data ratio used for the material
ansatz is materially understated. A separate in-repo Poisson fit gives
`0.00661 +/- 0.00075`, 2.27x the legacy density result `0.00291`; the legacy
fit merely re-expressed in counts gives `0.01158`, 3.98x. Its fitted
7367.9-keV sigma is inflated, so 3.98x is an overcorrection, not a validated
paper-impact magnitude. The independent fit uses 168.9 +/- 19.2 and
25547.9 +/- 184.3 counts for the two lines. Tight
`diag(0.05^2, 1e-5^2)` and ten-times-looser calibration constraints change the
ratio by only `5.4e-6`. Its Poisson deviance/dof is 2.93, exposing remaining
Gaussian/background model error. A revised Cd fraction still cannot be
derived: required neutron/gamma simulations are absent, and a common density
factor may partly cancel between data and simulation.

### Table 8: full and escape peaks

Public data file 1042 is
`MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN`, live time 60000 s,
`A0=-1.603776 keV`, `A1=0.893146 keV/channel`. The legacy simulation ROOT
files expected by `peak_ratio_compare.py` are absent. The data fit is also
stateful: `SpectrumFitter.auto_set_offset_factor` is set by the first fitted
peak, while nearby 7631.18/7645.58 keV peaks can be grouped. Changing the peak
list or order changes results. A one-triplet fit can reproduce individual
paper rows, but the current script's ordered 24-peak pass does not reproduce
the full published data column. Tuning the list to manufacture agreement is
not a paper-exact workflow.

Therefore:

- The defect's paper impact scales with the energy separation between ratio
  numerator and denominator: Table 3 spans up to 7.4 MeV from its reference,
  while Table 8 compares peaks separated by only 511 or 1022 keV.
- The current-code legacy-density and legacy-fit unit-corrected columns are
  both new calculations, not paper-exact results.
- Their unit-only factors span 0.72x--1.72x, demonstrating fit instability;
  they are not recommended physical estimates.
- Independent Poisson ratios do not overturn the published central values.
  Ten of twelve are within 5.7%; two 9718.79-keV ratios differ by 17--22%,
  still far inside the published uncertainties.
- The qualitative data-versus-simulation claim is not reproducible. A common
  estimator may partly cancel when data and simulation widths match, but the
  missing simulation products prevent that check.

The ROOT-free parent-group sensitivity uses bin-integrated Gaussians, a
Poisson likelihood, an affine background per window, and full fitted
covariance. The release has calibration values but no calibration covariance;
the audit runs both an explicit nominal constraint and a ten-times-looser
variant, recorded in the manifest. It is a model-sensitivity calculation, not
a replacement paper result. The
8998.63 and 7724.034 keV fits have elevated deviance per degree of freedom, so
their Gaussian-only absolute areas carry model error; the ratios are the more
robust output.

## Uncertainty finding: separate from the area estimand

The historical formulas remain reachable for reproduction:

```text
PeakFit:      sqrt(gross + 0.25 * background^2)
MultiPeakFit: sqrt(gross + 1.1 * background)
```

The first expression adds a count-like Poisson term to a squared-count term
and asymptotes to 50% of the fitted background. The two classes can thus give
very different errors for the same counting problem. Neither propagates the
fit covariance of the background-subtracted window. This audit documents but
does not replace those formulas, because a justified replacement requires an
explicit statistical estimator and covariance model.

The Table 8 published `dR` scale is compatible only with the `PeakFit`
formula: ordinary Poisson candidates are one to two orders of magnitude
smaller for background-rich rows, while the hard-coded background-squared term
lands on the published scale. Thus the error-column defect is paper-facing,
even though the independent central ratios are not overturned. The new
parent-group output reports its fitted-covariance uncertainty separately and
does not rewrite the legacy column.

One independent error is corrected: the 558.5 keV reference ratio is the same
fitted variable divided by itself, so it is exactly `1 +/- 0`. Treating its
numerator and denominator as independent generated the published
`1.000 +/- 0.193`. `--paper-legacy` preserves that published behavior.

The P2x C++ efficiency path separately omits the covariance between fitted
Gaussian height and width in its error propagation. That caveat does not
connect the efficiency values to the Python area defect.

## Paper-claim impact

| Product or claim | Verdict | Reproduction class | Impact |
|---|---|---|---|
| Table 3 numerical relative areas | affected | paper-exact reproduction plus new calculation | High-energy lines understated; 7367.9/558.5 changes 2.27x independently, while the legacy fit's unit-only change is 3.98x. |
| Table 3 reference uncertainty | affected | paper-exact reproduction plus new calculation | `0.193 -> 0` for the exact self-ratio. |
| 15% natural-Cd / 2% Cd-113 ansatz | not reproducible | unavailable legacy workflow | Data tuning input changes, but required simulations are absent. |
| Table 8 measured-ratio central values | affected; central values not overturned | new calculation; exact legacy workflow unavailable | Independent ratios retain the published central-value interpretation; 10/12 within 5.7%, two within 22%. |
| Table 8 isotropic-better-than-front-face statement | not reproducible | unavailable legacy workflow | Simulation ROOT products absent; no defensible revised comparison. |
| Table 8 and Table 3 fit uncertainties | affected | legacy reproduction / new covariance sensitivity | Historical formulas are not a coherent covariance calculation. |
| Table 5 measured efficiencies and optimized detector parameters | unaffected | provenance audit | Independent Rigel workbook plus P2x C++ Gaussian integral. |
| Figure 28 collimator-effectiveness curve | unaffected | external-input-required provenance audit | Independent fixed-window ROOT histogram sum, runtime, and source-area normalization. |
| Figures 14 and 19 supported public products | unaffected | paper-exact recalculation / ancillary replot | No peak-area call path. |
| Centroids, line IDs, calibration, and resolution | unaffected | code-path audit | Consumers use fitted centroid/sigma, not returned area. |

### Efficiency provenance

Read-only evidence checked during the audit:

- `Rigel Efficiencies.xlsx`, SHA-256
  `8d12a71e85f1ad488050920fb45eecda253ddee3b07074a23fc8ad5c5f1a4ca5`;
  its exported CSV matches the measured efficiency column.
- P2x `GeEfficiencyPlugin.cc`, SHA-256
  `805241fa4d7050eeb256e35301906c1d21a9771c60aac9770ae17e6196e1dac1`,
  integrates its Gaussian analytically as
  `sqrt(2*pi) * width * maximum` after the energy histogram is normalized to
  Hz/MeV. It never calls the Python fitter.
- Figure 28 is made by `plot_collimator_effectiveness.py`, which sums ROOT
  histogram bins in `+/-max(0.0025 E, 3 keV)`, then divides by runtime and
  source area. It is a third, independent path.

## Verification

```bash
python3 -m pytest -q tests/test_peak_area_semantics.py
python3 -m pytest -q tests/test_parent_group_fit.py
python3 -m py_compile scripts/audit_peak_area_impact.py
```

No canonical database write, paper edit, large download, simulation rerun,
unfolding, publication action, or external communication is part of this
audit.
