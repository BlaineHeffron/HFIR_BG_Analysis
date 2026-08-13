# Table 8 local escape-peak ratios and response-model bridge

## Result

This is a new measured-detector-count calculation, not a paper-exact replay,
an unfold, an incident-flux estimate, or a response-model validation. Five
declared local domains replace the rejected table-wide phase-2 absolute model.
The frozen rule classifies 12 ratios as supported quantitative (`Q`), nine as
quantitative but model-sensitive (`V`), and all three 7724.034-keV ratios as
visible but quantitatively unreliable (`R`). No ratio is classified as an
unresolved blend, boundary limit, or unavailable result.

Recommended measured-data use:

- report the 12 `Q` local values with their conditional Fisher uncertainties;
- report a `V` value only with its complete declared-variant range and the
  statement that this range is a sensitivity diagnostic, not a coverage
  interval;
- do not report the three `R` values as quantitative measurements;
- make no simulation-agreement or high-energy validation claim. No matched
  simulation was run.

The generated CSV remains numerical authority. Rounded values below are for
scientist review.

## Exact input and reconstruction boundary

The only spectrum is public file 1042,
`MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN`: run 256, 60000 s live time,
released calibration `A0=-1.603776` keV and `A1=0.893146` keV/channel, and
16384 raw channels. Nothing is summed or rebinned.

Frozen identities:

- analysis base: `bf5b936a02f940f4375e41a761489f75b5e472a7`;
- public database SHA-256:
  `c78bc8fa6ef7dbe1a8ea5d0189e69eb555c8a488fd582ff04b965a08aa1985e9`;
- spectrum text SHA-256:
  `e26419d18deb2557b69317be821cb4fbaf48fc700ce60e9849af280787498faa`;
- raw `int64` count-vector SHA-256:
  `f53a0291840a1ed20cf1e302483a063a1618f4e273dab4ba62a1ca82d7155b50`;
- released `float64` energy-grid SHA-256:
  `46e8e8d967c362e2233a2a9a1150e87e8d55d72064d5344969a5047138cea552`;
- total measured detector counts: 56,036,329.

The detector orientation is 46.5 degrees. The released shield is
`collimator30`, described as seven lead rings with a 30-degree opening-angle
cone endcap. The reactor calendar class is Cycle 490C, operating. Full
metadata and code-blob identities are frozen in
[`config/table8_historical_reconstruction.json`](../config/table8_historical_reconstruction.json).

The 15 one-based inclusive native ranges are unchanged from phase 2:

| Domain | DEP | SEP | FEP |
|:---|:---:|:---:|:---:|
| 6809.610 | 6453--6510 | 7006--7083 | 7597--7655 |
| 7631.180/7645.580/7693.398/7724.034 | 7378--7529 | 7949--8102 | 8520--8670 |
| 8998.630 | 8904--8961 | 9476--9533 | 10048--10106 |
| 9718.790 | 9710--9768 | 10283--10340 | 10855--10912 |
| 11386.500 | 11578--11635 | 12150--12207 | 12722--12779 |

Parent energies are authoritative strings. Components use exactly `E`,
`E-511.000`, and `E-1022.000` keV. Paper isotope labels are separate
metadata.

## Historical and phase-2 reproduction

Run the surviving ordered audit with:

```bash
audit_output=$(mktemp -d /tmp/hfir-table8-audit.XXXXXX)
python3 scripts/audit_peak_area_impact.py \
  --bundle data/HFIRBG_public_data_v1.1.0 \
  --output-dir "$audit_output"
```

The ordered parent sequence is 11386.5, 9718.79, 8998.63, 7724.034,
7693.398, 7645.58, 7631.18, then 6809.61 keV; each parent is traversed FEP,
SEP, DEP. The surviving fit gives `D=2089.810687` on 408 raw bins with 186
parameters and 222 descriptive degrees of freedom. Its 24-row CSV SHA-256 is
`58d5466ef653e84c12033bac588badaa17d4d22313559572435ed71812ad907a`.

This is not the paper-generating workflow. The first single-peak search
mutates a persistent expected-offset factor, greedy traversal changes
`PeakFit`/`MultiPeakFit` membership, and keyed fit state accumulates. Reversing
parent order moves a full-line ratio by as much as 77.9%; a cluster-first
traversal moves one by 117.5%. Unsafe one-triplet traversals move some crowded
ratios by more than 2000%. The published 24 central values and uncertainties
are therefore transcribed historical values, not an executable exact replay.

The repaired-audit column below changes the estimand on that same surviving
ordered fit from `N_window/(7 sigma)` to the complete normalized fitted line
integral and propagates its local fit covariance. It remains state/order
dependent and is not a candidate answer. In particular, it does not repair
the historical independent-error formula or crowded grouping.

Reproduce the reviewed phase-2 lane with its default 12 plumbing replicas:

```bash
phase2_output=$(mktemp -d /tmp/hfir-table8-phase2.XXXXXX)
python3 scripts/reanalyze_paper_peak_statistics.py \
  --bundle data/HFIRBG_public_data_v1.1.0 \
  --output-dir "$phase2_output" \
  --table 8
```

The affine-calibration baseline gives `D=3194.265925`. The reviewed curvature
fit gives `D=2708.879725/1096`, full Fisher rank 80, scaled condition
`4.31e5`, and one background endpoint bound. The three multiplet windows
contribute 54% of full deviance. Curvature is
`-0.905382 +/- 0.041499` keV on its domain-normalized basis and moves any
ratio by at most 1.75%. All 24 phase-2 values were unavailable under that
lane's global applicability gate; they are retained here only as reproduced
diagnostics. The phase-2 candidate CSV SHA-256 is
`e5c8f831476a694ac7610da3a49883c9c2058db3c839d1cc718aa78c9c70bfc0`.

## Frozen local estimator and decision rule

The predeclaration is machine-readable under `table8.local_workflow` in
[`config/paper_peak_statistics.json`](../config/paper_peak_statistics.json).
It was frozen before local ratios were examined.

Each of four isolated parent triplets is fitted jointly across its three
disjoint role windows. The four crowded parents are fitted jointly as 12
target components across the DEP, SEP, and FEP multiplet windows. Every group
uses raw integer channels, bin-integrated unit-normalized components,
nonnegative yields, and exact-cone interval-nonnegative backgrounds.

The uniform family is:

1. common left tail plus quadratic continuum: canonical;
2. no tail plus quadratic continuum: line-shape sensitivity;
3. common left tail plus affine continuum: background sensitivity.

The tail choice precedes the local values. In reviewed phase 2, removing the
interior tail raised deviance from 2708.880 to 3592.274; the fitted tail
fraction was `0.21761 +/- 0.01707`. The no-tail result is nevertheless shown
for every ratio below. No row changes base status when the no-tail model is
treated as canonical, so the binding tail-status downgrade is not activated.

Canonical neighbors are Fe-54 at 6268.9 keV in the 6809 SEP window and the
Cu-63 6616/7127/7638-keV DEP/SEP/FEP family in the cluster. Predeclared
sensitivities are target-only, Ge-70 at 6276.25 keV, equal-status Al-27 and
Ge-70 alternatives near the 7724 DEP, and a Cr/Ni steel catalog. No residual
peak discovery or post-hoc isotope assignment is allowed.

Each group has its own offset/stretch parameters with the phase-2 constraint
widths. Curvature is removed. Across the widest local window, the reviewed
curvature coefficient changes by about 0.028 keV; its non-affine remainder is
about 0.003 keV, far below one 0.893146-keV native channel. Deterministic
optimizer initialization uses the already-frozen phase-2 nuisance values;
they remain free fit parameters, not priors or fixed answers.

The copied phase-2 starting vector is exact and tracked:

| Free nuisance | Starting value |
|:---|---:|
| calibration offset | `0.5863036290709057` keV |
| fractional gain stretch | `-0.0012005358693575625` |
| resolution intercept | `0.9511201902485664` keV |
| linear resolution slope | `0.00020335418384669025` keV/keV |
| common left-tail fraction | `0.21761310678796192` |
| tail scale | `1.3383259741059537` sigma |

The default model-derived initializer sent the 11386.500-keV group to a bad
`D~6125`, rank-deficient basin. Its phase-2-initialized canonical result is
therefore accepted only after two enumerated, RNG-free
restarts multiply every copied nuisance by 0.9 and 1.1. Relative to the
canonical `D=203.11855243`, they give `D=203.11855186` and `203.11855008`;
absolute penalized-NLL differences are below `3.8e-11`, deviance differences
below `2.4e-6`, physical-parameter differences below `1.3e-6`, and ratio
differences below `2.5e-7`. Both are successful, full rank, and recover the
same two active nuisance bounds. These pass the declared `1e-8` penalized-NLL,
`1e-4` deviance/parameter, and `1e-5` ratio tolerances. The generated basin
table records both complete perturbed vectors and all acceptance fields; the
restarts do not enter the model envelope. This reviewer-required optimizer
validation followed the stage-2 basin diagnosis; it changed no predeclared
model, estimand, stability threshold, or row-status rule.

Complete parameter and line-yield covariance is retained within each group.
Every ratio uses its full `J Sigma J-transpose`, including multiplet
covariance. Cross-domain covariance is exactly zero because the native bins
and fitted nuisances are disjoint. The generated package includes full line
counts, parameter covariance, line-count covariance, ratio Jacobian, and
ratio covariance.

The material-movement threshold is
`max(one local Fisher SD, 5% of the canonical ratio)`. Every successful,
full-rank declared model and the role-window free-width/no-tail refit enters
the range. A movement exceeding one SD but admitted only by the 5% floor is
annotated explicitly below. Variant ranges are sensitivity envelopes, not
probability intervals and not uncertainty inflation.

Exactly six rows pass only through the 5% floor: three `Q` rows (6809.610
FEP/SEP, 8998.630 FEP/DEP, and 9718.790 FEP/SEP) and the three already-`R`
7724.034 rows. Every one retains its driver, movement in Fisher SD, and
relative movement below.

Weak or bound target yields below three conditional Fisher SD trigger a 95%
profile. All 24 local ratios have both target yields above the threshold, so
no profile or upper limit is triggered. Background or shape bounds are
reported but are not blanket row vetoes. `Q`, `V`, `R`, `X`, `B`, and `U`
mean supported quantitative, quantitative/model-sensitive, visible but
ratio-unreliable, unresolved blend, boundary/upper-limit, and
unsupported/unavailable, respectively.

## Complete 24-ratio map

`Published` is the paper transcription. `Repaired audit` is the still-unsafe
ordered legacy fit with full-line semantics. `Phase 2` is the rejected global
diagnostic. `Local` is the canonical new calculation. All displayed `+/-`
terms are the source column's uncertainty; the local and no-tail terms are
conditional Fisher SDs.

| Parent | Ratio | Published | Repaired audit | Phase 2 | Local | No tail | Declared range | Status and reason |
|---:|:---:|---:|---:|---:|---:|---:|---:|:---|
| 6809.610 | FEP/SEP | 1.0200 +/- 1.9300 | 1.0822 +/- 4.3853 | 1.0639 +/- 0.0218 | 1.0881 +/- 0.0257 | 1.0698 +/- 0.0242 | [1.0443, 1.1035] | Q: stable, clean target core; floor annotation: role-local refit moves 0.04384 = 1.71 SD = 4.03% |
| 6809.610 | FEP/DEP | 2.1300 +/- 6.8300 | 2.2027 +/- 8.9267 | 1.9264 +/- 0.0584 | 1.9848 +/- 0.0778 | 1.9680 +/- 0.0735 | [1.9680, 2.1077] | V: affine-background movement exceeds threshold |
| 6809.610 | SEP/DEP | 2.0800 +/- 6.8800 | 2.0355 +/- 0.1996 | 1.8106 +/- 0.0546 | 1.8241 +/- 0.0604 | 1.8396 +/- 0.0596 | [1.8168, 1.9452] | V: affine-background movement exceeds threshold |
| 7631.180 | FEP/SEP | 0.9000 +/- 1.4400 | 0.9781 +/- 3.6512 | 0.8674 +/- 0.0209 | 0.8709 +/- 0.0211 | 0.8760 +/- 0.0210 | [0.8567, 0.8777] | Q: stable, clean target core |
| 7631.180 | FEP/DEP | 1.8300 +/- 7.0900 | 0.7705 +/- 1.9604 | 1.7293 +/- 0.0674 | 1.7392 +/- 0.0683 | 1.7481 +/- 0.0679 | [1.7327, 1.7992] | Q: stable, clean target core |
| 7631.180 | SEP/DEP | 2.0400 +/- 8.3000 | 0.7878 +/- 2.1592 | 1.9937 +/- 0.0787 | 1.9969 +/- 0.0790 | 1.9954 +/- 0.0781 | [1.9904, 2.0740] | Q: stable, clean target core |
| 7645.580 | FEP/SEP | 0.8600 +/- 1.5600 | 1.0967 +/- 2.1305 | 0.8564 +/- 0.0222 | 0.8637 +/- 0.0225 | 0.8711 +/- 0.0227 | [0.8552, 0.8711] | Q: stable, clean target core |
| 7645.580 | FEP/DEP | 1.5300 +/- 6.4500 | 0.3233 +/- 3.3236 | 1.5693 +/- 0.0622 | 1.5991 +/- 0.0645 | 1.6084 +/- 0.0651 | [1.5711, 1.6876] | V: Al-27 sensitivity movement exceeds threshold |
| 7645.580 | SEP/DEP | 1.7800 +/- 7.9000 | 0.2948 +/- 3.0421 | 1.8325 +/- 0.0737 | 1.8514 +/- 0.0754 | 1.8464 +/- 0.0757 | [1.8166, 1.9649] | V: Al-27 sensitivity movement exceeds threshold |
| 7693.398 | FEP/SEP | 0.9900 +/- 2.4000 | 0.7753 +/- 2.1330 | 0.9016 +/- 0.0288 | 0.9024 +/- 0.0292 | 0.8836 +/- 0.0297 | [0.8513, 0.9257] | V: role-local movement exceeds threshold |
| 7693.398 | FEP/DEP | 1.5200 +/- 7.2800 | 2.5400 +/- 0.4817 | 1.7921 +/- 0.0978 | 1.8090 +/- 0.1011 | 1.7652 +/- 0.1017 | [1.6728, 1.9243] | V: Al-27/background movement exceeds threshold |
| 7693.398 | SEP/DEP | 1.5500 +/- 7.9800 | 3.2763 +/- 9.0299 | 1.9877 +/- 0.1109 | 2.0046 +/- 0.1132 | 1.9977 +/- 0.1164 | [1.8658, 2.1366] | V: Al-27/role-local movement exceeds threshold |
| 7724.034 | FEP/SEP | 0.8700 +/- 0.2700 | 0.8563 +/- 0.1631 | 0.8809 +/- 0.0066 | 0.8796 +/- 0.0069 | 0.8653 +/- 0.0067 | [0.8484, 0.8877] | R: target-core Pearson residual at least 4; floor annotation: role-local move 0.03117 = 4.52 SD = 3.54% |
| 7724.034 | FEP/DEP | 1.8500 +/- 1.4900 | 1.9253 +/- 0.3702 | 1.8045 +/- 0.0195 | 1.7990 +/- 0.0216 | 1.7620 +/- 0.0208 | [1.7110, 1.8430] | R: target-core residual; Al-27/Ge-70 DEP ambiguity; floor annotation: Al-27 move 0.08803 = 4.07 SD = 4.89% |
| 7724.034 | SEP/DEP | 2.1200 +/- 1.7900 | 2.2484 +/- 0.0895 | 2.0485 +/- 0.0217 | 2.0453 +/- 0.0225 | 2.0363 +/- 0.0222 | [1.9505, 2.1161] | R: target-core residual; Al-27/Ge-70 DEP ambiguity; floor annotation: Al-27 move 0.09482 = 4.21 SD = 4.64% |
| 8998.630 | FEP/SEP | 0.6700 +/- 0.2800 | 2.1049 +/- 7.1766 | 0.6994 +/- 0.0139 | 0.6812 +/- 0.0150 | 0.6766 +/- 0.0143 | [0.6435, 0.6919] | V: role-local movement exceeds threshold |
| 8998.630 | FEP/DEP | 1.3600 +/- 1.4100 | 4.0452 +/- 13.9699 | 1.3092 +/- 0.0354 | 1.2324 +/- 0.0431 | 1.2284 +/- 0.0398 | [1.1799, 1.2829] | Q: stable, clean target core; floor annotation: role-local move 0.05246 = 1.22 SD = 4.26% |
| 8998.630 | SEP/DEP | 2.0400 +/- 2.2000 | 1.9218 +/- 1.0596 | 1.8717 +/- 0.0488 | 1.8091 +/- 0.0522 | 1.8155 +/- 0.0506 | [1.8091, 1.8542] | Q: stable, clean target core |
| 9718.790 | FEP/SEP | 0.7200 +/- 0.3500 | 0.6749 +/- 0.0863 | 0.6419 +/- 0.0240 | 0.6466 +/- 0.0274 | 0.6421 +/- 0.0262 | [0.6421, 0.6786] | Q: stable, clean target core; floor annotation: affine-background move 0.03201 = 1.17 SD = 4.95% |
| 9718.790 | FEP/DEP | 1.2500 +/- 1.9600 | 0.5779 +/- 7.4020 | 1.2452 +/- 0.0704 | 1.2412 +/- 0.0912 | 1.2319 +/- 0.0851 | [1.2319, 1.3584] | V: affine-background movement exceeds threshold |
| 9718.790 | SEP/DEP | 1.7400 +/- 2.7700 | 0.8564 +/- 10.9681 | 1.9399 +/- 0.1042 | 1.9196 +/- 0.1149 | 1.9185 +/- 0.1116 | [1.9066, 2.0017] | Q: stable, clean target core |
| 11386.500 | FEP/SEP | 0.5800 +/- 0.0700 | 0.8960 +/- 0.8073 | 0.5731 +/- 0.0168 | 0.5744 +/- 0.0178 | 0.5664 +/- 0.0174 | [0.5630, 0.5907] | Q: resolution intercept and FEP high-background endpoint active; free-width/no-tail role refit moves 0.01146 = 0.65 SD, within threshold |
| 11386.500 | FEP/DEP | 1.2800 +/- 0.5000 | 3.1200 +/- 3.1057 | 1.1975 +/- 0.0468 | 1.2004 +/- 0.0554 | 1.1785 +/- 0.0520 | [1.1785, 1.2409] | Q: same two active nuisance bounds; free-width/no-tail role refit moves 0.02037 = 0.37 SD, within threshold |
| 11386.500 | SEP/DEP | 2.1900 +/- 0.9000 | 3.4822 +/- 1.5024 | 2.0896 +/- 0.0773 | 2.0897 +/- 0.0832 | 2.0807 +/- 0.0808 | [2.0807, 2.1684] | Q: same two active nuisance bounds; free-width/no-tail role refit moves 0.07872 = 0.95 SD, within threshold |

## Local morphology and identifiability

The following deviances are local morphology diagnostics only. They are not
ranked across domains and no absolute p-value veto is applied.

| Domain | DEP `D`; max Pearson | SEP `D`; max Pearson | FEP `D`; max Pearson | Fisher rank/dimension; condition | Active canonical bounds |
|:---|:---:|:---:|:---:|:---:|:---|
| 6809.610 | 92.97; 3.14 | 113.94; 3.71 | 144.35; 4.85 | 19/19; `1.24e5` | none |
| four-parent cluster | 412.89; 5.30 | 402.19; 7.16 | 428.72; 7.25 | 30/30; `3.04e5` | none |
| 8998.630 | 91.61; 4.78 | 71.18; 3.39 | 63.64; 2.63 | 18/18; `3.33e5` | resolution slope |
| 9718.790 | 60.70; 2.26 | 48.14; 2.41 | 48.04; 2.26 | 18/18; `5.24e5` | none |
| 11386.500 | 70.81; 2.73 | 56.29; 3.22 | 76.02; 2.08 | 18/18; `7.01e5` | resolution intercept; FEP high-background endpoint |

At 11.4 MeV the linear slope dominates the effective width, so the resolution
intercept bound is plausibly nuisance non-identifiability, not a target-yield
boundary. The FEP continuum endpoint is also active. As an empirical bound
sensitivity check, the independent free-width/no-tail role refit moves the
three 11386.500 ratios by only 0.37--0.95 canonical Fisher SD; every row above
retains its exact movement and both active-bound names.

The large cluster-window maxima are not blanket failures. Only the
7724.034-keV target cores cross the predeclared absolute-Pearson threshold;
those three ratios are `R`. The 8998 DEP and 6809 FEP maxima lie outside the
relevant ratio target cores and instead remain morphology diagnostics.

For the 14.400-keV 7631/7645 pair, line-yield correlations are `+0.1276`
(DEP), `+0.1260` (SEP), and `+0.1216` (FEP). The cluster is full rank; none
approaches the predeclared absolute-correlation `0.95` unresolved threshold.
The equal-status Al-27 and Ge-70 correlations with the 7724 DEP are `+0.421`
and `-0.143`. They do not meet `X`, but their reallocation and the target-core
residual prevent a quantitative 7724 ratio.

Canonical fixed-nuisance one-step centroid scores remeasure antisymmetric
role morphology without adding a shift to the fit:

| Domain | DEP shift keV (`z`) | SEP shift keV (`z`) | FEP shift keV (`z`) |
|:---|---:|---:|---:|
| 6809.610 | -0.054 (-0.84) | +0.031 (+0.77) | -0.030 (-0.77) |
| four-parent cluster | -0.035 (-1.44) | +0.030 (+2.14) | -0.019 (-1.34) |
| 8998.630 | -0.068 (-0.98) | +0.053 (+1.32) | -0.036 (-0.79) |
| 9718.790 | +0.042 (+0.28) | +0.040 (+0.56) | -0.073 (-0.82) |
| 11386.500 | -0.059 (-0.59) | +0.119 (+1.95) | -0.143 (-1.91) |

Means are DEP -0.035, SEP +0.055, and FEP -0.060 keV. These final local values
are fixed-nuisance one-step score shifts: the canonical calibration, shape,
background, and yields stay fixed while a small shared line-energy displacement
is scored within each role. Phase 2 instead reported each role's signed shift
relative to its global calibration, with means DEP +0.052, SEP +0.035, and FEP
-0.199 keV. The definitions are not numerically identical, but their signs are
directly comparable. FEP remains negative and SEP positive; DEP does not retain
its phase-2 sign.

The separate role-window refit changes calibration, width, tail, and background
together. In the calibration-correction coordinate, its role-local-minus-group
canonical means are approximately DEP +0.344, SEP +0.272, and FEP +0.423 keV.
An increasing correction moves a physical line to lower energy on the released
grid: the exact local-minus-canonical released-grid centroid field is therefore
renamed explicitly and has means DEP -0.342, SEP -0.269, and FEP -0.422 keV.
That coordinate inversion explains the apparent sign conflict; neither form is
the phase-2 global-calibration shift. The fixed-nuisance score above is the
cleaner morphology bridge.

The cluster role-local calibration offsets range from -1.935 keV for FEP to
+1.452 keV for SEP, a 3.387-keV spread: 3.79 released native channels and not
small compared with the 14.400-keV closest-parent spacing. Offset/stretch are
correlated and the role refit also changes line shape, so this is not a new
energy calibration measurement. It is prominent measured evidence of
role-dependent response/charge-loss/background asymmetry. Ratios remaining
within their frozen thresholds show area robustness only; they do not erase
that response morphology or validate a detector model. No per-role calibration
or higher polynomial is promoted.

The historical independent-Gaussian parent-group check placed the two
9718.790-keV DEP-denominator ratios 16.9% and 21.6% above the paper. The
drivers are the `N_window/(7 sigma)` density artifact plus stateful peak search
and grouping. Full normalized local semantics resolve FEP/DEP: 1.2412 versus
paper 1.25, a 0.10-local-SD difference. SEP/DEP improves but does not fully
resolve: 1.9196 remains 1.56 local SD, or 10.3%, above paper 1.74. No claim is
made that semantics repair alone recovers every historical central value. With
the paper-generating ROOT products and exact fitter unavailable, the remaining
SEP/DEP difference is explicitly an unexplained legacy-provenance residual,
not assigned post hoc to one historical mechanism.

## Measured evidence carried into the NiM bridge

Three facts require response-model tests, not calibration freedom: the repeated
fixed-nuisance FEP-negative/SEP-positive score pattern; the cluster's 3.387-keV
role-local scale spread under a free-width/no-tail shape change; and stable area
ratios for most rows despite those peak-centered/antisymmetric residuals. A
matched simulation must reproduce or explain all three on identical bins. It
cannot claim validation from ratio similarity alone.

## Committed response-model inventory

This is an interface/readiness inventory. No controlled implementation or
private numerical detector value is copied into this public repository.

### PROSPECT-G4 producer

The inspected checkout is clean at
`843a22d30b5234588e833f37a2958cb05f7f8bf2`. It contains selectable
`legacy_ccf2518` and `corrected_nominal_v1` HPGe geometry profiles. The
corrected nominal profile repairs the bore-end construction and the rounded
outer dead-layer offset. Outer and bore dead layers are configurable. Its own
provenance explicitly leaves the front-stack basis inferred/unresolved and
passive internals absent/unresolved; it is not an as-built detector model.

The checked-out legacy Ge response macro disables both dead layers, requests
record level 2, uses `LeadCollimatorThrower`, and does not explicitly freeze
the EM switch or production cuts. Hadronics defaults on. The checkout has no
charge-collection/electronics response and no FEP/SEP/DEP extractor.

Boundary-safe raw deposition (`4939062`), strict transport profiles
(`4a645fd`, hardened at `b5a9775`), and event-RNG replay (`51e2a04`, receipt
validation through `5511203`) exist on separate committed descendants of the
corrected geometry. The convergence/variance-pilot branch reaches `eed9504`.
Those capabilities are not integrated into the inspected clean checkout and
are not a producer authority for a run in this session.

### phonon-response

The inspected committed revision is
`50d4c9ee265821983d2d783c7ddec97101028872`. It has canonical deposition and
event identity, raw transport adapters, zero-deposition-primary accounting,
deterministic carrier/electronics mechanics, streamed event output, spatial
response-atlas mechanics, and generic per-primary categorical covariance,
normalized-shape covariance, linear contrasts, Wilson intervals, and paired
differences (`f47e1f8`).

Its current FEP/SEP/DEP convergence path counts exclusive fixed windows on
raw deposition sums. That is not the measured native-bin fitted estimator.
The committed detector-response chain is synthetic/study-only: no public
detector-specific field/profile binds the as-built geometry, dead layers,
charge collection, trapping, or electronics to this detector; it makes no
Lynx claim. It has no measured-spectrum ingestion or peak fitter and no
validated applicability across 6.8--11.4 MeV. Committed validation also
records a material high-energy event-atomic surface-quarantine hazard; exact
controlled values remain in the response repository.

The worktree was not clean: committed HEAD plus user-owned modified
`README.md`, `docs/architecture.md`, and
`docs/hpge-electronics-energy-w23-validation.md`, and an untracked planned
CNF-settings-adapter document. The delta is documentation-only and the adapter
is pending/unimplemented. No dirty value or claim is used here. The worktree
was not modified, staged, or cleaned.

## `table8-nim-response-comparison-v1`

No machine-readable contract is added yet: there is no committed maintained
simulation consumer. This section is the versioned specification for that
consumer.

1. **Case identity.** Simulate exactly eight monoenergetic parents:
   6809.610, 7631.180, 7645.580, 7693.398, 7724.034, 8998.630, 9718.790,
   and 11386.500 keV. Preserve a separate primary/event ledger for every
   energy. A cluster estimator test must combine the four cluster cases with a
   predeclared mixture; it must never lose the monoenergy identities.
2. **Producer identity.** Record repository commit, Geant4 version, geometry
   profile and revision, every as-built/front-stack/passive-internal choice,
   outer and bore dead layers, active-volume definition, material revision,
   overlap check, and geometry receipt hash.
3. **Source identity.** Record source surface/volume, position distribution,
   angular law, collimator/aperture relation, polarization if relevant, and
   the profile revision. Isotropic, front-face, or legacy throwers are model
   variants, not interchangeable labels.
4. **Transport identity.** Explicitly set physics list, EM and hadronic
   switches, region production cuts, step limits, transport profile, record
   level, random engine, seed-to-event mapping, and primary replay identity.
   Include all generated primaries, including zero-deposition events.
5. **Deposition boundary.** Preserve total, ionizing, active-Ge, dead-layer,
   and quarantined energy ledgers. State exactly where Geant4 geometry ends
   and charge response begins. Never apply a dead layer both geometrically and
   as a second response loss.
6. **Response identity.** Record detector configuration, field, weighting
   field, carrier velocity, pair creation/Fano treatment, charge-collection
   efficiency, trapping/transition policy, electronics, calibration,
   resolution, and every stage revision/receipt. Unbound synthetic defaults
   are prohibited.
7. **Matched estimator.** Produce reconstructed event energy, bin unit-weight
   events on the exact data native-channel edges, and run the same five-group
   components, backgrounds, line-shape variants, nonnegative yields, status
   rules, and ratio Jacobian used for data. Fixed raw-deposition windows are
   not a substitute. The four-parent mixture must test deblending and
   estimator bias on the same 12-component windows.
8. **Statistical uncertainty.** Retain full fitted covariance and
   `J Sigma J-transpose`. Estimate Monte Carlo covariance and nonlinear
   estimator bias from independent event batches or an event-level resampling
   design declared before looking at agreement. Weighted histograms may not
   enter the Poisson fitter as integer counts. Data and simulation covariance
   combine only after their independent identities are verified.
9. **Convergence.** Freeze the batch size, number of independent seeds,
   precision target, maximum event budget, and stop/fail rule before the
   eight-energy run. Derive the event count from pilot per-primary peak yields
   and the weakest required FEP/SEP/DEP contrast. A post-hoc event count chosen
   because ratios look similar is invalid.
10. **Systematics.** Predeclare bounded geometry/front-stack/dead-layer,
    source position/angular, transport/cut, charge/CCE, electronics/resolution,
    and estimator-mixture variants. Keep the variant envelope separate from
    Monte Carlo and fit covariance.
11. **Normalization.** Within one parent, a common generated-primary count,
    live time, and source amplitude cancel in a peak ratio. Energy-, angle-,
    and position-dependent efficiency; geometry; dead layers; charge loss;
    quarantine; resolution; estimator bias; and cluster deblending do not.
12. **Comparison gate.** Only local `Q` measured rows may enter a quantitative
    validation claim. `V` rows may enter a labeled sensitivity comparison over
    their declared range. `R`, `X`, `B`, and `U` rows cannot validate the
    response. Simulation must independently pass geometry, response,
    convergence, matched-estimator, bias, and uncertainty gates. Visual
    similarity is not validation.

Historical ROOT files expected by `scripts/peak_ratio_compare.py` are absent.
Any calculation satisfying this contract is a new NiM result, not a
paper-exact recreation and not a continuation of the missing legacy table.

## Readiness and exact next-session gate

No simulation campaign is authorized or scientifically ready. Hard blockers:

- raw deposition, strict transport, and replay are not integrated on one
  clean, approved corrected-geometry producer revision;
- front stack, passive internals, source profile, and other as-built geometry
  choices remain unresolved;
- no clean committed detector-specific charge/field/electronics profile binds
  the response chain to this HPGe detector;
- no committed simulation consumer implements the exact local native-bin
  fitted estimator;
- high-energy response applicability and surface-quarantine behavior are not
  validated over the eight parent energies;
- no owner/reviewer-approved convergence target and bounded run budget exists.

Next session, only after clean commits and accountable authorization:

1. pin one producer commit integrating corrected geometry, raw-step output,
   strict transport receipts, and deterministic event replay; verify record
   level, zero-deposition primaries, energy ledgers, and geometry receipt with
   a non-inferential interface smoke;
2. pin one response commit with a detector-specific profile and a committed
   adapter that emits reconstructed event energies plus all stage receipts;
3. implement one maintained `table8-nim-response-comparison-v1` consumer and
   prove exact native-bin/component/ratio-Jacobian identity against this data
   workflow with manufactured events;
4. run one bounded isolated-parent estimator pilot and one bounded four-parent
   mixture/deblending pilot, with unit event weights and independent seeds;
5. use only those pilots to calculate the eight-energy event budget under a
   reviewer-approved precision/stop rule; return that bounded plan for review;
6. only then run the staged eight-energy calculation and compare eligible
   rows. Preserve negative and mixed outcomes.

## Command, verification, and generated hashes

Run only the local lane:

The frozen implementer and independent-review executions used the installed
bundle at
`/home/blaine/projects/HFIR_BG_Analysis/data/HFIRBG_public_data_v1.1.0`.
The command below shows the portable conventional repository-relative path;
the database and spectrum hashes above define content identity.

```bash
local_output=$(mktemp -d /tmp/hfir-table8-local.XXXXXX)
python3 scripts/reanalyze_paper_peak_statistics.py \
  --bundle data/HFIRBG_public_data_v1.1.0 \
  --output-dir "$local_output" \
  --table 8 \
  --table8-workflow local \
  --bootstrap-replicates 1
```

The bootstrap argument satisfies the common CLI contract; the local lane does
not run a bootstrap. Fewer than 200 replicas cannot change a declared local
reporting action. Profiles, full covariance, bounds, residuals, and declared
model movements control status.

Narrow verification:

```bash
python3 -m py_compile \
  src/public_data/table8_local.py \
  scripts/reanalyze_paper_peak_statistics.py
PYTHONPATH=. pytest -q \
  tests/test_table8_local.py \
  tests/test_peak_phase2_config.py \
  tests/test_peak_caller_audit.py \
  tests/test_peak_area_semantics.py \
  tests/test_peak_likelihood.py
```

Result: 73 tests and 210 subtests passed. Two existing profile tests emitted
SciPy's transient SLSQP bounds-clipping warning; neither failed. No canonical
database write, setup download, simulation, manuscript edit,
response-repository mutation, push, or upload was performed.

Final review-package SHA-256 values from the command above:

| Generated product | SHA-256 |
|:---|:---|
| `table8_local_candidate.csv` | `6d90b851b43e8940344145d672dbc2e27f36537e0b8ad63a843406ee1dd67b8e` |
| `table8_local_basin_stability.csv` | `338779fa18d4eece4ba3abc44807f7743d63af6e70d9716308ce9ff442814de2` |
| `table8_local_full_line_counts.csv` | `1ed96c8156c18acdbbd232b0f04fd220fb4bb056b0962c9fae0c5e4b39944ca8` |
| `table8_local_full_line_count_covariance.csv` | `144e645f24b757defd13f26363ee148aea5bcfc4284082a0318ada5e8be1a9a6` |
| `table8_local_full_parameter_covariance.csv` | `54342a38cec2092011cdac618f8b3783a4828643f7af044bb13b548030d6625f` |
| `table8_local_ratio_jacobian.csv` | `a767cb20807dc8c98abdd5124159bbd4767c2d63715357364c6e996fc5deeb52` |
| `table8_local_ratio_covariance.csv` | `84bc86dd0f8a55b28b4129e80ab2b44267d88e3d1a1821546e044748973b57cf` |
| `table8_local_model_comparison.csv` | `6fffb41b4747889cac54c444e4978d8236ff32f48e531cecaf82f206047ec20a` |
| `table8_local_role_sensitivity.csv` | `8f6f3d5a72bffb20b3ddc6de7e385550df5b807a2e8b1e12555346ba5f9a8f96` |
| `table8_local_role_centroid_shifts.csv` | `f102f0baa22f9981a91e822c980b28f649469a2576b7068af5866a97600f15bd` |
| `table8_local_line_correlations.csv` | `bd8456f58f25245e665a083cf8b178984fccc4014b365e10bff80dc5930bb81a` |
| `table8_local_window_diagnostics.csv` | `aa312c5b76703c5c6e750e5204bd3ed59f29138ed138b644f72923a3faba171e` |
| `table8_local_profile_intervals.csv` | `70ffcfb067bbcdaffcff18b78fcd8e8e45740d1b9330b86df608401a4b85a9d8` |
| `table8_fit_diagnostics.json` | `4caee595193669eaa13ec93f0057424bb66be910585f0cbcad0597ee913032ba` |
| generated historical record | `e37e220ebdee58096d9451e50344813e2c31e4e4fc4560151686748c3e7b7081` |

Fit-bin gzip container hashes are retained in the generated manifest. They
include the existing gzip timestamp and are run receipts, not stable content
identities. Generated products remain in ignored temporary storage and are
not tracked.
