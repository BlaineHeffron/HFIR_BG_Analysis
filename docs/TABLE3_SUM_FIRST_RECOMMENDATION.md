# Table 3 sum-first/local-window recommendation

Status: new descriptive measured-data calculation; no manuscript edit. The
estimand is relative fitted detector counts in visible local features. It is
not incident flux, source emission rate, isotope activity, cadmium abundance,
or detector-response validation.

## Inputs and exact historical replay

The input is public release `HFIRBG_public_data_v1.1.0`, database SHA-256
`c78bc8fa6ef7dbe1a8ea5d0189e69eb555c8a488fd582ff04b965a08aa1985e9`.
Runtime checks require file IDs 1716, 1765, 334, and 1676 in that order; run
319 `Cycle493_RD_low_gain`; calibration group 90; exactly identical
`A0=-0.38892709072078885` keV, `A1=0.6995359644108052` keV/channel, 16384-bin
energy and width grids; and total live time 260013.89 s. Each released value
must be a nonnegative integer. Exact int64 channel addition gives 5,845,047
counts, maximum bin 156,917, and summed-vector SHA-256
`ca030c709a3ece435df510a6e1e62b4ffe866e43fc9c181f948b50e6ae2a6d24`.
No interpolation or rebinning occurs.

Individual released-spectrum SHA-256 values are: file 1716
`c234c31dfae150018b5aab69896fd654e94d9adc6b46fdc1083fc2dc69f9bed5`,
1765 `a831cb51f6a71285b2d36457900e404c810ddb189395da95a2abe2868f685638`,
334 `ffaf793b2604bd07d8f85434ec81d527cc9a0662c396a520f59eda3d1aaaf2ce`,
and 1676 `42b98a90ae366231fb8289ef9dc76fbe6f3943c301900fb87a4b4a771de54b3f`.
The runtime assertion also freezes Cycle 493 reactor-on classification,
detector configuration 16, 180-degree orientation, RD shield identity, raw
count normalization, and absence of a response-model estimand.

The unchanged historical audit first reproduced every displayed numerical
Table 3 row at manuscript precision. Its `rd_peak_area_impact.csv` SHA-256 is
`a82ae51a9c3c1374b542418da37937402fc3f4b4b140826f603bb76f60fec3f8`.
The historical operation is raw-channel sum first, local fit second, then the
incorrect density-like `N_window/(7 sigma)` ratio and independent-error
formula. The repaired historical comparator integrates the same fitted
Gaussian-plus-left-tail signal and propagates its complete local covariance;
it changes the estimand only and does not repair poor historical fits.

## Corrected model and summation validity

The reconstruction records all 34 frozen historical native-bin windows. The
corrected canonical fit selects the 27 windows containing retained or newly
assessed isolated components: 376 unchanged native bins in one exact
accumulated spectrum. It uses authoritative line energies, unit-normalized
bin-integrated Gaussian signals, a shared HPGe square-root resolution curve,
one constrained offset/stretch pair, and one quadratic local background per
window. Its 28 fitted components cover the previously retained rows plus
isolated 609.321, 707.419, 1364.339, and 1377.669-keV candidates. The
558.456-keV self-ratio is algebraically 1 with variance 0. Full fitted
parameter, detector-count, and ratio covariance matrices are generated.

Identical-bin uniform comparisons gave:

- no-tail affine: D=832.752, 376 bins, 86 parameters;
- no-tail quadratic: D=666.839, 376 bins, 113 parameters;
- common left-tail plus quadratic: D=650.357, 115 parameters, but the tail
  fraction pins at its 0.8 upper bound.

Thus the historical quadratic background is retained uniformly. The tail is
not locally identifiable and remains sensitivity-only. This is a morphology,
identifiability, and reporting decision—not row-wise p-value selection. The
global D is descriptive and is not a blanket publication gate.

Independent-window fits use the same native bins, quadratic background, no
tail, and one free constant effective width. They are a sensitivity, never a
second selectable answer. A ratio movement above the assembled fit's
conditional 1 sigma disqualifies an unqualified quantitative row. Apart from
the boundary-active 1281-keV yield, these local Fisher matrices are full rank.
The 2398.6-keV width is boundary-active and the 2550.1-keV optimizer does not
converge; both rows remain unavailable. None of these failures affects a Q row.

Five frozen per-file anchors span 351.932--2614.511 keV. Among usable files,
centroid spreads are 0.011--0.208 keV (0.015--0.298 native channel and
0.011--0.160 fitted sigma). The 1293.640-keV spread is statistically resolved
at 4.33 combined conditional sigma, but is only 0.178 native channel and 0.105
fitted sigma. Predicted centroid-mixture broadening raises sigma by at most
0.25%, below the frozen 5% threshold. Usable-anchor width spreads range from
2.7% to 44.2%; the largest, at 2614.511 keV with only two usable files, is 1.45
combined conditional sigma. The 558.456-keV width spread is 6.2% (2.04 sigma).
Thus no coherent multi-anchor spread exceeds 0.5 channel or 0.5 fitted sigma;
the width results show local uncertainty, not evidence satisfying the frozen
summation-invalidation rule. Although some row ratios are model-sensitive,
all three invalidation conditions are not met. Exact
sum-first accumulation is therefore data-wise valid; per-file nuisance terms
are not introduced into the primary fit. The anchors stop at 2614.511 keV.
Therefore the drift verdict for sparse lines above 5 MeV is conditional on
affine gain drift; extrapolation cannot exclude higher-order calibration drift.

## Complete 35-row comparison and action

Columns are the published legacy density ratio, repaired full-line ratio from
that same historical fit, prior simultaneous phase-2 conditional ratio, and
the recommended new sum-first value. Dashes mean no quantitative value should
be reported. Codes: Q quantitative; V visible/identified but area unreliable;
B broad template required; X unresolved blend; U unsupported/unavailable.
Only the new Q uncertainties are proposed for reporting, and they are local
conditional covariance—not exact-coverage statements.

| Paper keV | Legacy | Same-fit repaired | Phase 2 | New sum-first | Code |
|---:|---:|---:|---:|---:|:---:|
| 238.6 | 0.34622 | 0.18596 | 0.24676 | — | V |
| 242 | 0.2723 | 0.18532 | 0.19268 | — | V |
| 295.2 | 0.45489 | 0.42392 | 0.44132 | 0.43185 ± 0.014 | Q |
| 351.9 | 0.75846 | 0.69736 | 0.7356 | — | V |
| 478 | — | 0.23563 | — | — | B |
| 558.5 | 1 | 1 | 1 | 1 ± 0 | Q |
| 609.3 | 0.60511 | 0.66357 | — | — | V |
| 651.3 | 0.17341 | 0.15084 | 0.18343 | — | V |
| 707.4 | 0.01771 | 0.016275 | — | 0.022066 ± 0.0049 | Q |
| 725 | 0.06028 | 0.082324 | — | — | X |
| 768.4 | 0.047144 | 0.050965 | 0.057693 | 0.048931 ± 0.0052 | Q |
| 805.9 | 0.10581 | 0.16726 | 0.081549 | 0.082001 ± 0.0054 | Q |
| 1120.3 | 0.11604 | 0.14341 | 0.1259 | — | V |
| 1209.7 | 0.049526 | 0.060503 | 0.052167 | — | V |
| 1238.1 | 0.034516 | 0.03565 | 0.047622 | — | V |
| 1281 | 0.024302 | 0.052657 | 0.013119 | — | U |
| 1293.6 | 0.56497 | 0.66554 | 0.65667 | — | V |
| 1364.3 | 0.059821 | 0.088727 | — | — | V |
| 1377.7 | 0.036894 | 0.036699 | — | — | V |
| 1399.6 | 0.034612 | 0.042824 | — | — | X |
| 1489.56 | 0.013815 | 0.0087286 | 0.025794 | — | V |
| 1660.37 | 0.031665 | 0.045999 | 0.031053 | — | V |
| 1764.5 | 0.09094 | 0.14207 | 0.099851 | — | V |
| 2204.2 | 0.02145 | 0.025642 | 0.023629 | 0.026379 ± 0.0034 | Q |
| 2223 | 0.033131 | 0.048574 | 0.038329 | — | V |
| 2398.6 | 0.0077907 | 0.01025 | 0.0065744 | — | U |
| 2455.8 | 0.017115 | 0.017684 | 0.022999 | — | V |
| 2550.1 | 0.0032174 | 0.0024608 | 0.0074959 | — | U |
| 2614.53 | 0.024334 | 0.038712 | 0.035434 | — | V |
| 2660.1 | 0.024765 | 0.046077 | 0.016699 | — | V |
| 2767.5 | 0.011104 | 0.011962 | 0.0069458 | — | U |
| 5433.1 | 0.0022381 | 0.0095069 | — | — | U |
| 5824.6 | 0.004972 | 0.012333 | — | — | U |
| 7367.9 | 0.0029106 | 0.010794 | — | — | U |
| 7916.3 | 0.0012535 | 0.0026187 | — | — | U |

The domains differ: the legacy/repaired columns use narrow historical
least-squares fits; phase 2 fits four files simultaneously in broader windows
with independent run yields; the new column fits one accumulated spectrum on
historical native windows. Their D/dof values must not be ranked across
domains. Differences reflect the density bug, historical fit instability,
window/background allocation, run-yield structure, authoritative energies,
and complete reference covariance. Notably, the isolated 707.419-keV feature
is recovered quantitatively; 805.9 agrees with phase 2 rather than its unstable
same-fit repaired value.

## Excluded windows and weak-branch diagnostic

- 478 keV remains a broad B-10 reaction feature requiring a normalized
  non-Gaussian template.
- 609.321 keV is visible, but the unmodeled asymmetric 595.85-keV intrinsic-Ge
  feature makes its area response-template-sensitive.
- 707.419 keV is quantitatively recovered. The 725.298/726.863/727.330-keV
  neighborhood remains unresolved.
- 1364.339 and 1377.669 keV are visible but free-width-sensitive. The
  1399.638--1401.515-keV neighborhood remains unresolved.
- 5433.1, 5824.6, and 7916.26-keV target yields are boundary-active or
  zero-compatible. The 7367.96-keV target is unsupported; a 7374.58-keV
  nuisance does not recur at 2 sigma in all three long files and remains
  unidentified.

The frozen candidate line data come from [NNDC CapGam/ENSDF](https://www.nndc.bnl.gov/capgam/)
and [IAEA LiveChart/ENSDF](https://nds.iaea.org/relnsd/v1/data?fields=decay_rads).
Free Ac-228/Cd/Bi/Pb/Al or unidentified amplitudes are classified only as
flexible background. No same-window ratio had sufficient RD-geometry evidence
for a branching constraint; cross-energy prediction is unavailable without a
response/geometry model. Therefore no residual cluster is called explained by
a held-line prediction. The 338/352 and 1364--1408 clusters absorb flexible
components; the others remain unidentified or unmodeled.

## Paper recommendation

Replace “relative area” with “approximate relative detected full-peak counts
in the accumulated Cycle 493 RD spectrum, normalized to the 558.456-keV
feature.” State explicitly: raw detector counts, no efficiency correction or
unfolding, conditional fit uncertainties, qualitative labels where local area
is model-sensitive, and no inference of source emission, activity, neutron
flux, cadmium abundance, or detector response. Report only the six Q rows
numerically under the current rule; retain the remaining visible identities as
qualitative inventory entries with the codes above. Do not use a global fit
p-value as a table-wide acceptance gate.

## Reproduction and generated products

```bash
historical_dir=$(mktemp -d /tmp/hfir-table3-historical.XXXXXX)
python3 scripts/audit_peak_area_impact.py \
  --bundle /path/to/HFIRBG_public_data_v1.1.0 \
  --output-dir "$historical_dir"

# Reuse the reviewed phase-2 CSV, or regenerate it with
# --table 3 --table3-workflow phase2 in a separate empty output directory.
sum_first_dir=$(mktemp -d /tmp/hfir-table3-sum-first.XXXXXX)
python3 scripts/reanalyze_paper_peak_statistics.py \
  --bundle /path/to/HFIRBG_public_data_v1.1.0 \
  --output-dir "$sum_first_dir" --table 3 \
  --table3-workflow sum-first \
  --table3-phase2-comparison-csv /path/to/prior/table3_candidate.csv \
  --bootstrap-replicates 1
```

The evaluated generated directory is
`/tmp/hfir-table3-sum-first-review3.0k2B8J`. Key SHA-256 values are:

- complete 35-row map `ecbab501a89a1596430f4d8dc3db8e52509ea4ebd8b314819cd60f33a2bce25c`;
- model comparison `1749fcca39c2647b2257a8a37d6855466eb804dc512c196f9d4b11a83f2a55cc`;
- per-window sensitivity `39409f621fbb6e4a815a4d53cc0548cb1cb2e6cdb100dfb26f21d7f12629dcaa`;
- per-file drift `3f2e7ca2759aeb07316b3368a44aaf960ff9dfad393ba4ba4eaa63ea1b3d9709`;
- weak-branch diagnostic `13664882ce0c146321a52c00d425d3724741e5ee0166d08b275c58f63fd44eef`;
- count covariance `a80c319abe14f68f3e0f1b0e291a26a34846b0277f71310d6b6f92b07321ee1c`;
- ratio covariance `2c0353aa7f8bcc9299a5eb4f64a8f5d180f0a14e305994c5cda71e2282537e12`.

The prior reviewed phase-2 comparison CSV has SHA-256
`bfddf8c5b5ff19615962d6100546da5ee45071003b0abfc33b671f728e7ff166`.
All generated CSV/JSON/gzip products remain untracked under `/tmp`.
