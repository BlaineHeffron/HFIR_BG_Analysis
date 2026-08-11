# Paper peak-statistics correction, phase 1

Status: **new exploratory measured-data calculation**. The outputs described
here are not approved manuscript replacements. Historical numerical replay
remains separate and explicitly named.

This workflow replaces the paper-facing uncertainty calculation for the
measured-data portions of Tables 3 and 8. It does not infer cadmium abundance,
run neutron transport, regenerate a response matrix, unfold a measured
spectrum, or construct missing simulations. Released calibrated spectra are
detector counts.

Frozen definitions: [`config/paper_peak_statistics.json`](../config/paper_peak_statistics.json).
Historical consumer inventory: [`config/peak_area_callers.json`](../config/peak_area_callers.json).
Prior audit: [Python peak-area and paper-impact audit](PEAK_AREA_AUDIT.md).

## Statistical model

### Raw observations and bin-integrated components

Let (y_{rwi}) be the integer count in raw detector channel (i), declared
window (w), spectrum (r). No fit input is rebinned. The released affine
calibration is (A_{0r},A_{1r}). A shared offset (delta_0) and fractional
gain stretch (delta_1), plus declared zero-centered per-run deviations
(d_{0r},d_{1r}), give calibrated channel edges

\[
 e^{\rm lo}_{ri}=A_{0r}+\delta_0+d_{0r}
 +A_{1r}(1+\delta_1+d_{1r})(i+1/2),
\]

and likewise with (i+3/2) for the high edge. Table 3 fixes
(d_{00}=d_{10}=0) as the relative convention. For its later spectra,
(sigma(d_0)=0.25) keV and (sigma(d_1)=10^{-4}); their bounds and diagonal
Gaussian covariance are frozen in configuration. Table 8 has no per-run term.

Every window and component is declared before fitting. There is no peak
discovery, order-dependent grouping, or persistent fitter state. Component
(k), with energy (E_k), uses the declared normalized mixture

\[
 s_k(E)=(1-f)\,\mathcal N(E;E_k,\sigma_k)
       +f\,{\rm ExG}_{\rm left}(E;E_k,\sigma_k,\tau_k),
\]

when a tail is enabled. Table 3's canonical phase-1 variant removes both tail
nuisance parameters and uses the HPGe-motivated form

\[
 \sigma_{rk}=q_r\sqrt{a^2+bE_k}.
\]

Table 3 fixes (q_0=1) and constrains every later-run relative resolution scale
to (q_r=1\pm0.12), with bounds 0.6--1.4. These run-drift constraints are
declared sensitivity assumptions: the release provides no drift covariance.

The declared comparison set also fits linear
(\(\sigma_k=a+bE_k\)) and constant-tail variants on identical raw bins. A
zero-weight tail is not represented by an unidentified free tail scale.

The second term is a Gaussian charge measurement minus a positive exponential
loss. Both terms integrate to one. The expected fraction in a raw bin is

\[
 p_{rwi,k}=\int_{e^{\rm lo}_{ri}}^{e^{\rm hi}_{ri}}s_k(E)\,dE,
\]

evaluated as CDF differences. Therefore fitted (lambda_k) has unit detector
counts/s for the full normalized line, independent of display bin width or
fit-window width.

### Simultaneous runs, background, and likelihood

For Table 3, (u_{rc}) is a signal-intensity scale for run (r) and frozen
physical-origin class (c), while (v_r) is a common background-intensity scale;
(u_{0c}=v_0=1) fixes conventions. Every component explicitly declares either
`neutron_capture` or `radioactive_decay`; H-1 is assigned to capture and Ar-41
to decay in configuration, not inferred by name at run time.
Every canonical Table 3 window has an independent nonnegative quadratic
Bernstein background-rate density with coefficients
(\(\alpha_w,\gamma_w,\beta_w\)). Its fraction (\(x_{rwi}\in[0,1]\)) is
evaluated at the calibrated bin center, not frozen from nominal calibration.
Expected counts are

\[
 \mu_{rwi}=t_r\left[
 \sum_{k\in w}u_{r,c(k)}\lambda_k p_{rwi,k}
 +v_r\Delta e_{ri}
   \left\{(1-x_{rwi})^2\alpha_w
   +2x_{rwi}(1-x_{rwi})\gamma_w+x_{rwi}^2\beta_w\right\}
 \right],
\]

where (t_r) is live time in seconds and (Delta e_{ri}) is bin width in
keV. For the single Table 8 spectrum, run scales equal one. This is a product
likelihood, not a sum of count arrays.

A single scalar (u_r) was tested and rejected: neutron-capture lines and
radioactive-decay lines change independently between the four measurement
times. The two-class construction is the smallest declared model representing
that physical distinction; it does not fit an independent scale per line.

The affine background is nested in this quadratic basis and is refit as a
declared sensitivity variant. Shape derivatives with respect to
centroid/calibration, line rate, tail fraction, and background coefficients
are analytic. Derivatives with respect to core width and tail scale use
explicit central finite differences; the workflow does not call the complete
gradient analytic.

With nuisance vector (eta), the penalized binned Poisson negative log
likelihood is

\[
 \ell(\lambda,\eta)=
 \sum_{rwi}\left[\mu_{rwi}-y_{rwi}\log\mu_{rwi}\right]
 +\frac12(\eta_G-m_G)^T V_G^{-1}(\eta_G-m_G)+C(y).
\]

(C(y)) makes the saturated Poisson data term zero computationally. It cannot
change estimates, likelihood ratios, gradients, or covariance. The release
provides (A_0,A_1), but no calibration or run-drift covariance. Explicit
(m_G,V_G) blocks constrain the common calibration, Table 3 per-run calibration
deviations, and Table 3 relative resolution scales. Assumptions are recorded in
configuration and each manifest; they are not released metrology.

The full pure count-data Poisson deviance is always reported, with
\(n_{\rm bins}-n_{\rm free}\) as its descriptive degrees-of-freedom
convention. No asymptotic chi-square p-value is attached to low-expectation
bins. Global/window chi-square-reference diagnostics and normal/binomial
outlier counts are restricted to bins with fitted expectation at least five;
eligible and excluded counts are recorded. The global reference degrees of
freedom subtract all fit parameters. Window allocation of shared parameters is
not unique, so window p-values use eligible raw-bin count and remain diagnostic.
Every Gaussian constraint and quadratic contribution is also reported under a
separate penalized-objective/pseudo-observation convention. Priors are not
silently credited to the count-data deviance.

## Yield and ratio covariance

At a regular interior maximum, expected Fisher information is

\[
 I_{ab}=\sum_{rwi}\frac{1}{\mu_{rwi}}
 \frac{\partial\mu_{rwi}}{\partial\theta_a}
 \frac{\partial\mu_{rwi}}{\partial\theta_b}
 +(V_G^{-1})_{ab},
\]

with full approximation (Sigma_\theta=I^{-1}). The reported yield
covariance is the complete (lambda)-submatrix after jointly fitting all
nuisances. Equivalently,

\[
 \Sigma_\lambda=\left(I_{\lambda\lambda}
 -I_{\lambda\eta}I_{\eta\eta}^{-1}I_{\eta\lambda}\right)^{-1}.
\]

Background, calibration, resolution, tail, and run-scale correlations are not
discarded.

For (R=A/B),

\[
 \nabla R=\left(\frac1B,-\frac{A}{B^2}\right)
\]

and

\[
 {\rm Var}(R)=\frac{{\rm Var}(A)}{B^2}
 +\frac{A^2{\rm Var}(B)}{B^4}
 -\frac{2A\,{\rm Cov}(A,B)}{B^3}.
\]

The last term is absent from historical `unc_ratio`.

For (R_i=A_i/B), (R_j=A_j/B) sharing a denominator,

\[
 \begin{split}
 {\rm Cov}(R_i,R_j)={}&\frac{{\rm Cov}(A_i,A_j)}{B^2}
 -\frac{A_j{\rm Cov}(A_i,B)}{B^3}
 -\frac{A_i{\rm Cov}(A_j,B)}{B^3}\\
 &+\frac{A_iA_j{\rm Var}(B)}{B^4}.
 \end{split}
\]

The implementation constructs the complete Jacobian (J) and writes
(J\Sigma_\lambda J^T), covering arbitrary shared numerators/denominators.

For Table 3, the reference-line window remains rejected by residual tests.
An explicitly sensitivity-based covariance is written: the
mean outer product of ratio shifts from each successful declared
shape/background alternative relative to the canonical fit. It is positive
semidefinite. The candidate CSV keeps Fisher and model-variant RMS terms
separate. A rank-one reference-denominator covariance uses the maximum
absolute run-level fractional observed-minus-expected balance in the frozen
556.8--559.6 keV core after the origin-scale/drift refit, restricted to the
three runs with at least 10000 s live time. Every non-self ratio has sensitivity
equal to its fitted value; the exact self-ratio has sensitivity zero.

A separate rank-one origin-composition covariance applies only to
radioactive-decay numerators divided by the neutron-capture reference. Its
fractional scale is the maximum fitted decay/capture composition change among
non-reference long runs: the inverse capture/decay double ratio minus one,
matching the reported ratio orientation. Capture-to-capture ratios get exactly
zero origin-composition term because their run scale cancels. The total
exploratory covariance sums Fisher, model-variant, reference-core, and
origin-composition terms. These are declared sensitivity terms, not a claim
that alternatives or residual envelopes have known probability distributions.

For a reference self-ratio, the random variable is not two independent copies:
(A/A=1). Differentiating the same variable gives

\[
 \frac{d(A/A)}{dA}=\frac1A-\frac{A}{A^2}=0.
\]

Its Jacobian row is exactly zero. Its variance and covariance with every other
ratio are exactly zero. Table 3 thus reports the 558.5 keV reference as one
with zero variance.

## When Fisher errors are not canonical

The quadratic/Fisher approximation is not trusted when:

- a yield or nuisance maximum lies on a bound;
- a weak yield has an asymmetric likelihood or includes zero;
- counts are too sparse for local normality;
- information is rank deficient or badly conditioned;
- optimization fails;
- material residual structure shows model misfit; or
- a weak nuisance produces non-quadratic correlations.

For weak or boundary ratios, the canonical construction profiles every other
yield and nuisance. A regular two-sided 95% interval solves
(Delta\ell=\chi^2_{1,0.95}/2=1.92073). When a nonnegative numerator is on
the boundary or zero remains included, a one-sided 95% upper limit uses
(chi^2_{1,0.90}/2=1.35277). A failed profile is labeled failed; it is not
silently replaced by a symmetric error.

For regular interior fits, full Fisher covariance is the canonical local
uncertainty. A deterministic parametric bootstrap is a **diagnostic**, not a
second selectable answer. It draws Poisson counts in every fitted raw bin,
refits the complete model, and records empirical spread, nominal 68%/95%
Fisher coverage, convergence, and boundary frequency. Seeds are the first 64
bits of SHA-256 of stable case identity. Replicate counts are bounded and
recorded. The default 12 replicates are explicitly labeled descriptive-only;
coverage is untested until at least the configured 200 successful replicates.

## Table 3 selection and applicability

The four inputs are fixed:

| File ID | Live time (s) | Run | A0 (keV) | A1 (keV/channel) |
|---:|---:|---|---:|---:|
| 1716 | 86400.00 | `Cycle493_RD_low_gain` | -0.3889270907 | 0.6995359644 |
| 1765 | 86400.00 | `Cycle493_RD_low_gain` | -0.3889270907 | 0.6995359644 |
| 334 | 86400.00 | `Cycle493_RD_low_gain` | -0.3889270907 | 0.6995359644 |
| 1676 | 813.89 | `Cycle493_RD_low_gain` | -0.3889270907 | 0.6995359644 |

They share calibration group, detector, orientation, shield, and run family.
Count combination would also assume identical signal and background rates.
The simultaneous likelihood is preferred: it preserves four integer spectra,
shares within-origin line rates/line shape, and fits separate capture, decay,
and background scales for runs after the first. The revised model also fits the declared relative
offset, stretch, and resolution-scale nuisances described above. These terms
permit run drift without assigning an independent line yield to every run.

There is no public Cycle 498 run. `Cycle493_RD_low_gain` is the only public
reactor-on low-gain RD family consistent with no added water and no added floor
lead. This caption/catalog conflict remains in provenance.

All photopeak groups are explicit. The 238.6/242.0, 707.4/725.0,
1209.7/1238.1, 1281.0/1293.6, 1364.3/1377.7/1399.6,
2204.2/2223.0, and 2614.533/2660.1 groups are joint. The broad 478 keV
Doppler feature is not a Gaussian photopeak. Phase 1 reports it unavailable
until a normalized physical Doppler shape is declared.

The revised public-data fit converges with full-rank covariance and no active
bounds. It reports a self-ratio exactly one with zero variance. It does **not**
pass applicability. In the one-replicate origin-class smoke run, full
count-data deviance is 7332.8 on 5518 descriptive degrees of freedom. The
high-count chi-square reference retains 4381 bins, excludes 1255 bins with
fitted expectation below five, and gives deviance 5992.6 on 4263 degrees of
freedom (p about \(9.9\times10^{-63}\)). Among eligible bins, 11 exceed four
normal-reference standard deviations and two exceed five.

The `rd_558` reference window remains the worst aggregate window: its
high-count reference is 386.7 on 168 bins (diagnostic p about
\(2.8\times10^{-19}\)). Separate capture/decay scales do cure its coherent
long-run core imbalance: in the frozen 556.8--559.6 keV core, files 1716,
1765, and 334 are 1.04%, 1.17%, and 0.39% high, all near one aggregate
standard deviation. File 1676 is 3.68% high but only +0.31 aggregate standard
deviations because the short run has about 70 expected core counts. The
long-run 1.17% envelope is retained as the reference-core covariance; the
rejected full window remains an applicability blocker.

The origin scales expose the stronger scientific result. Relative to file
1716, fitted capture/decay double ratios are exactly one by convention for
1716, then 0.8771 +/- 0.0159 for 1765, 0.8900 +/- 0.0161 for 334, and
0.8117 +/- 0.1095 for 1676. Thus the two long comparison runs differ from the
reference run by 11--12%, well beyond statistical uncertainty. A
radioactive-decay line divided by the Cd-113 capture reference is consequently
not a run-invariant observable; its value depends on which run anchors the two
origin scales. Such cross-origin candidate ratios are labeled reference-run
relative and receive a 14.02% origin-composition envelope. Capture-to-capture
ratios remain within-class observables and receive zero composition term.
Table 3 mixes both estimands. This non-invariance and the still-rejected count
model are manuscript blockers; manuscript replacement remains false.

The repeated isolated fixed-energy residual in `rd_609` has no asserted new
line identity. It remains visible in per-bin/window diagnostics and contributes
to non-applicability; inventing a contaminant without released provenance
would be stronger than the evidence.

Residual audits also reject the 5433, 5825, 7368, and 7916 keV windows without
explicit neighboring/escape components. Those four lines are removed from the
canonical likelihood and written as unavailable, not absorbed by background
or reported with false precision. The broad 478 keV feature remains
unavailable for its separate physical-shape reason.

The model-comparison CSV refits sqrt/no-tail quadratic (canonical), its nested
affine-background version, sqrt/constant-tail quadratic, and linear/no-tail
quadratic. The affine background loses about 240 in twice penalized NLL for 22
fewer parameters in the smoke run. Constant-tail and linear-resolution
variants improve likelihood but do not cure global rejection; the tail pair
remains correlated. Their ratio shifts therefore enter the declared model
sensitivity instead of selecting a paper-ready shape by AIC alone.

## Table 8 selection and applicability

Input is public file ID 1042,
`MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN`, live time 60000 s,
with released (A_0=-1.603776) keV and (A_1=0.893146) keV/channel.

Every FEP, SEP, DEP component for eight parents is explicit. Nearby parents
7631.180, 7645.580, 7693.398, and 7724.034 keV form three genuine four-line
multiplets: one joint FEP window, one SEP window, one DEP window. List order
cannot change membership.

No additional contaminant identity is asserted in phase 1; configuration has
an explicit empty contaminant list. Residual structure is not hidden through
automatic peak discovery. Current deviance/dof is about 3.35. Cluster windows
and the 11386.5-keV SEP window are among poor local fits. The high-count
chi-square reference retains 1156 bins, excludes one bin below expectation
five, and gives deviance 3684.2 on 1096 degrees of freedom (p about
\(2.4\times10^{-276}\)); rejection is unchanged by the low-count convention.
The manifest sets
manuscript-replacement applicability false. This is a concrete remaining
model/contaminant blocker, not permission to quote false precision.

Legacy monoenergetic simulation ROOT files are absent. Phase 1 writes measured
candidates only. Each future generated energy and response identity must be
fit separately. Missing simulations must not be fabricated, added, or inferred
from measured fits.

## Public command and outputs

Use a new empty directory:

```bash
peak_output_dir=$(mktemp -d /tmp/hfir-peak-statistics.XXXXXX)
python3 scripts/reanalyze_paper_peak_statistics.py \
  --bundle /path/to/HFIRBG_public_data_v1.1.0 \
  --output-dir "$peak_output_dir"
```

Default: 12 deterministic diagnostic bootstrap replicas per table; about 2--3
minutes on a typical workstation. `--table 3` and `--table 8` select one lane.
`--bootstrap-replicates N` changes the bounded diagnostic count and is recorded.

Outputs include:

- `table3_candidate.csv`, `table8_candidate.csv`;
- full long-form nuisance/yield parameter, line-rate, and ratio covariance CSVs;
- separate Table 3 Fisher, model-variant RMS, reference-core,
  origin-composition, and summed exploratory ratio covariance CSVs, plus the
  likelihood/model-variant comparison;
- per-window and Table 3 run-level reference-core/origin-scale diagnostics,
  plus gzip per-bin observed/expected files;
- deterministic bootstrap diagnostic CSVs;
- convergence, active bounds, parameter errors, rank/condition, deviance, and
  applicability diagnostics; and
- `manifest.json` with database/spectrum/config/output hashes, code revision,
  definitions, seeds, units, identities, and scientific non-scope.

Candidate CSVs are ignored by repository data policy. The command and frozen
definitions are tracked. Reviewer-selected results need separate approved data
release before promotion.

## Legacy isolation and caller audit

`PeakFit.area`, `MultiPeakFit.area`, `unc_ratio` remain for historical replay
and diagnostics. Table-facing legacy CLIs require explicit `--paper-legacy` or
`--legacy-window-counts`. The latter fixes area units only; it retains ad hoc
historical uncertainty and is not corrected.

The corrected command imports none of these estimators. Tests scan direct
callers against the machine-readable inventory. New unclassified calls or a
legacy estimator in the corrected command fail verification.

## Verification

```bash
python3 -m pytest -q tests/test_peak_likelihood.py
python3 -m pytest -q tests/test_peak_caller_audit.py
python3 -m pytest -q tests/test_peak_area_semantics.py
python3 -m pytest -q tests/test_parent_group_fit.py
python3 -m py_compile scripts/reanalyze_paper_peak_statistics.py
```

Manufactured cases cover normalized bin integration, isolated peaks, shared
background multiplets, calibrated-center quadratic backgrounds, linear and
sqrt resolution forms, tail omission, simultaneous run nuisances, correlated
ratios, independent declared origin-class run scales, per-run
calibration/width drift recovery and checked nuisance gradients, shared
denominators, exact self-ratios, raw-bin rejection, checked
analytic/finite-difference gradient components,
boundaries/profile upper limits, deterministic bootstrap behavior, and
repeated injected-yield Fisher coverage.
