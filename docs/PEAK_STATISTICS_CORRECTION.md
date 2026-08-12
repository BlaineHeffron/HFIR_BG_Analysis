# Paper peak-statistics correction, phase 2

Status: **new exploratory measured-data calculation**. The outputs described
here are not approved manuscript replacements. Historical numerical replay
remains separate and explicitly named.

Comparison limit: the new absolute fit-quality diagnostics were applied to the
phase-2 models, not to the historical paper procedure. The historical replay
checks arithmetic provenance only. Failure of a phase-2 model therefore does
not establish adequacy of the legacy model, and no replacement decision should
be made until both methods are evaluated on the same spectra, windows, and
diagnostics.

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

when a tail is enabled. Table 3's canonical phase-2 variant removes both tail
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

evaluated as CDF differences. Therefore each fitted (lambda_{rk}) has unit detector
counts/s for the full normalized line, independent of display bin width or
fit-window width.

### Simultaneous runs, background, and likelihood

For Table 3, every run and component has its own nonnegative detected
full-peak rate (lambda_{rk}). Physical-origin labels remain provenance only;
they impose no equality or scaling constraint. A background-intensity scale
(v_r), with (v_0=1), retains a shared background shape while allowing each
run's overall continuum level to change.
Every canonical Table 3 window has an independent interval-nonnegative
quadratic Bernstein background-rate density with coefficients
(\(\alpha_w,\gamma_w,\beta_w\)). Its fraction (\(x_{rwi}\in[0,1]\)) is
evaluated at the calibrated bin center, not frozen from nominal calibration.
The endpoint coefficients obey \(\alpha_w,\beta_w\geq0\). The middle
coefficient is not artificially restricted to be nonnegative; the exact
quadratic cone
\(\gamma_w+\sqrt{\alpha_w\beta_w}\geq0\) is necessary and sufficient for
nonnegativity over the full interval. The implementation first fits the
relaxed middle coefficient, records the cone margin and analytic minimum for
every quadratic window, and invokes a transformed constrained solve if the
relaxed result violates the cone.
Expected counts are

\[
 \mu_{rwi}=t_r\left[
 \sum_{k\in w}\lambda_{rk} p_{rwi,k}
 +v_r\Delta e_{ri}
   \left\{(1-x_{rwi})^2\alpha_w
   +2x_{rwi}(1-x_{rwi})\gamma_w+x_{rwi}^2\beta_w\right\}
 \right],
\]

where (t_r) is live time in seconds and (Delta e_{ri}) is bin width in
keV. For the single Table 8 spectrum there is one rate per component. This is a product
likelihood, not a sum of count arrays.

Two restrictive phase-1-style factorizations remain in one compact
yield-model comparison. The first gives every line one common run scale. The
second gives capture and decay lines separate run scales. Their nested
likelihood ratio tests distinct questions; neither selects the independent
model automatically. Independent yields are canonical because they define the
requested per-run estimands without silently imposing either factorization.
These repaired likelihood comparisons are not the historical paper procedure;
the deferred common-diagnostics comparison described above remains out of
scope.

Shape derivatives with respect to centroid/calibration, line rate, tail
fraction, and background coefficients are analytic. Derivatives with respect
to core width and tail scale use explicit central finite differences; the
workflow does not call the complete gradient analytic.
With nuisance vector (eta), the penalized binned Poisson negative log
likelihood is

\[
 \ell(\lambda,\eta)=
 \sum_{rwi}\left[\mu_{rwi}-y_{rwi}\log\mu_{rwi}\right]
 +\frac12(\eta_G-m_G)^T V_G^{-1}(\eta_G-m_G)+C(y).
\]

Optimization uses the per-bin NLL relative to the saturated-data constant.
Expected-Fisher repair steps normally require strict Armijo decrease. A step
whose stable NLL change is no larger than the frozen \(10^{-9}\) numerical
floor may instead be accepted only when it reduces the scaled projected
gradient by at least a factor of two (or reaches the \(10^{-3}\) stationarity
gate). Both thresholds and every accepted repair are recorded.

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

Let $S={\rm diag}(s_a)$ contain the declared physical parameter scales. Rank,
condition number, and numerical inversion are evaluated in scaled coordinates,

\[
 I_z=S I_\theta S, \qquad
 \Sigma_\theta=S I_z^{-1} S.
\]

This avoids inverting the poorly scaled physical-coordinate matrix after using
the scaled matrix for its validity gate. The reported yield covariance is the
complete $\lambda$-submatrix of $\Sigma_\theta$ after jointly fitting all
nuisances. Equivalently,

\[
 \Sigma_\lambda=\left(I_{\lambda\lambda}
 -I_{\lambda\eta}I_{\eta\eta}^{-1}I_{\eta\lambda}\right)^{-1}.
\]

Background, calibration, resolution, tail, and cross-run yield correlations are not
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

For Table 3, let (t_r) be run live time and stack the independently fitted
rates in run-major order. The primary aggregate estimands are

\[
 N_k=\sum_r t_r\lambda_{rk}, \qquad
 \bar\lambda_k=N_k/\sum_r t_r.
\]

These are summed fitted detector counts and their total-live-time detected
rate. If (W) maps the run-major rate vector to (N), the workflow writes
(W\Sigma W^T) and the corresponding rate covariance. No efficiency,
response, abundance, or incident-flux correction is applied. Per-run ratios
and aggregate ratios use their full Jacobians. The aggregate ratio is a ratio
of exposure-summed fitted counts, not an unweighted or inverse-variance mean
of run ratios.

A generalized constrained profile is implemented for weak or boundary
ratios of arbitrary linear combinations of run yields. It profiles every run
yield and shared nuisance. Every profile inner solve uses the stable NLL
difference from the fitted point and the same exact quadratic-background cone
coordinates as the canonical fit. A profile point is rejected if SLSQP fails,
if its scaled KKT residual exceeds (10^{-3}), or, for a linear-combination
ratio, if its normalized equality residual exceeds (10^{-8}). The profile
curve is anchored to a separately tightened solve at the fitted ratio; the
entire interval fails closed if that base differs from the fitted penalized NLL
by more than 0.005. This is less than 0.4% of the smaller one-sided 95%
profile threshold. The output records this difference, the largest inner KKT
and equality residuals, and the count of failed inner solves. The fit result
records the implementation-owned KKT, cone, and repair configuration; the
reporting configuration owns only the profile confidence level, weak-line
trigger, and fitted/profile-base consistency tolerance.

The primitive component-ratio profiler accepts only the shared-origin-scale
parameterization whose line names match the supplied specification. An
independent-run result fails immediately with an instruction to use the
generalized linear-combination profiler; it is never silently reconstructed as
a different yield model.

Every component also receives a covariance-aware GLS constant-rate test, and
every non-self within-run ratio receives the analogous heterogeneity test.
These diagnostics use only regular interior run estimates. A boundary-pinned
component yield is excluded from its rate GLS; a run is excluded from ratio GLS
when either its numerator or denominator yield is pinned. The per-run estimate
and the exposure-summed descriptive aggregate still retain every run. Output
rows record included and excluded spectrum indices/file IDs and the reduced
covariance-rank degrees of freedom. Fewer than two interior runs makes the test
unavailable rather than evidence for homogeneity. This reports temporal
incompatibility without treating a boundary Fisher approximation as regular or
hiding variation in a pooled ratio.

Official cycle metadata resolves these four observations only as consecutive
reactor-on measurements; it supplies no interval-resolved power history or
irradiation/source history. Reactor-power scaling and known/free-half-life
activation-decay models are therefore labeled non-identifiable. The declared
known-half-life case is the NNDC/ENSDF Ar-41 value (109.61 min) for the
1293.6-keV line; its numerical half-life does not supply the missing production
history. The saturated
independent-run model is descriptive, not a causal activation model.

The 558-keV audit tests the paper energy 558.5 keV against the IAEA PGAA
Cd-113 energy 558.32(3) keV and an extended-window sensitivity containing the
known Tl-208 583.187-keV decay line. NNDC/IAEA energy matches for Co-59,
In-115, and W-186 are recorded but not fitted or promoted because the release
does not establish those source materials. Ratio shifts from successful
predeclared reference-window variants form a positive-semidefinite RMS
outer-product covariance, written separately from Fisher covariance. Nuclear
energy agreement alone never promotes a component. The declared RMS
construction includes the two successful noncanonical variants,
`iaea_energy_558_32` and `paper_energy_plus_tl208_583`.

The same audit explicitly records and dismisses four detector-process
alternatives: the asymmetric intrinsic Ge-74 595.85-keV inelastic feature lies
above the fitted 540--578-keV ROI; the 511-keV annihilation feature lies below
it and can affect only the declared local-background sensitivities; no parent
supports an escape peak at the reference energy; and no parent pair, geometry,
or summing model supports a true-coincidence sum there. None is silently fitted
as a narrow 558-keV component.

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
- an exact quadratic-background cone constraint is active, making the local
  model nonregular;
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
second selectable answer. It draws Poisson counts in every fitted raw bin and,
independently, redraws the complete vector of Gaussian auxiliary measurements
as (m^*\sim N(\hat\eta,V_G)). Each replica is refit with (m^*), rather
than with the fixed released/declaration-centered measurement (m_G); holding
(m_G) fixed would omit repeated-measurement variation while comparing against
Fisher information that includes that measurement. The Poisson stream retains
the stable case-identity seed. The independent Gaussian stream uses the same
case identity plus the frozen `gaussian-pseudo-observations-v1` domain. The
manifest records both seeds, ordered constrained-parameter names, fitted
generating values, declared covariance, and the draw convention. The workflow
refits the complete model and records empirical spread, nominal 68%/95%
Fisher coverage, bound frequency, exact-cone fallback/invalid/active counts,
and the minimum replica cone margin. Seeds are the first 64
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
The simultaneous likelihood preserves four integer spectra, fits an
independent yield for every run/component, and shares only declared
line-shape, calibration/resolution, and background-shape structure. It also
fits the declared relative offset, stretch, resolution-scale, and background
scale nuisances described above.

There is no public Cycle 498 run. `Cycle493_RD_low_gain` is the only public
reactor-on low-gain RD family consistent with no added water and no added floor
lead. This caption/catalog conflict remains in provenance.

All photopeak groups are explicit. The 238.6/242.0, 707.4/725.0,
1209.7/1238.1, 1281.0/1293.6, 1364.3/1377.7/1399.6,
2204.2/2223.0, and 2614.533/2660.1 groups are joint. The broad 478 keV
Doppler feature is not a Gaussian photopeak. Phase 2 reports it unavailable
until a normalized physical Doppler shape is declared.

Current phase-2 candidates remain exploratory because the declared residual
diagnostics reject the fitted count model. Exact fit values, active-bound
records, and bootstrap summaries belong to generated CSV/JSON outputs, not
this maintained method document. The bounded default bootstrap remains
descriptive only and cannot certify coverage or bypass applicability failure.

The candidate table reports exposure-summed detected full-peak counts and
rates as primary estimands. Aggregate ratios remain explicitly
detector-response-dependent secondary summaries: cross-energy HPGe efficiency
does not cancel, so they cannot establish incident neutron flux or cadmium
abundance. Rate and ratio GLS results, including boundary exclusions and file
IDs, live once in `table3_run_heterogeneity.csv`; the candidate estimand
states that stationarity is not assumed.
The repeated isolated fixed-energy residual in `rd_609` has no asserted new
line identity. It remains visible in per-bin/window diagnostics and contributes
to non-applicability; inventing a contaminant without released provenance
would be stronger than the evidence.

Residual audits also reject the 5433, 5825, 7368, and 7916 keV windows without
explicit neighboring/escape components. Those four lines are removed from the
canonical likelihood and written as unavailable, not absorbed by background
or reported with false precision. The broad 478 keV feature remains
unavailable for its separate physical-shape reason.

The historical reconstruction record freezes the exact source revision, git
blobs, file order, spectrum hashes, calibration, live times, operation order,
and every legacy ratio. It establishes that the paper numbers came from
combining four Cycle493 count arrays before fitting and then dividing
`N_window/(7*sigma_keV)` values. They were not run-by-run inverse-variance
averages and were not integrated peak-count or flux ratios. The area-like bug
entered `Spectrum.py` in commit `5cb9e0a`; it inserts an unintended
sigma-reference/sigma-line factor. The paper caption says Cycle498, but the
reconstructed workflow and released inputs are Cycle493.

## Table 8 selection and applicability

Input is public file ID 1042,
`MIF_BOX_REACTOR_OPTIMIZED_OVERNIGHT_LOWEST_GAIN`, live time 60000 s,
with released (A_0=-1.603776) keV and (A_1=0.893146) keV/channel.

Every FEP, SEP, DEP component for eight parents is explicit. Nearby parents
7631.180, 7645.580, 7693.398, and 7724.034 keV form three genuine four-line
multiplets: one joint FEP window, one SEP window, one DEP window. List order
cannot change membership.

Phase 2 audits only authoritative NNDC CapGam/ENSDF energies with a plausible
released or intrinsic material source. The canonical component set adds the
declared Fe-54 full-energy line and Cu-63 FEP/SEP/DEP family. Their
interpretation depends on the fitted calibration nuisance and its declared
prior; it is not an independent energy measurement.

The close Al-27 and Ge-70 hypotheses remain equal-status noncanonical
sensitivities. Both can reallocate the target 7724.034-keV DEP, and neither is
promoted in the canonical result. Cr/Ni steel lines remain catalogued
sensitivity components. Residual-only automatic discovery remains prohibited.
Declared comparisons include target-only, Fe/Cu, Ge, Al, steel-catalog,
no-tail, and quadratic-background variants. The phase-2
`t8_6809_sep` window is widened identically for every variant to admit the
declared Fe-54 candidate and local sideband. Phase 1 used different raw bins,
so its absolute deviance is not a likelihood comparison with phase 2. A
runtime identity check carries the canonical component specification,
resolution, and background parameterization into every profile and bootstrap.
Model-comparison CSVs label their AIC/BIC-shaped arithmetic as **nonstandard
penalized-objective descriptive arithmetic**. The objective includes Gaussian
constraint penalties, so those columns are not standard AIC or BIC and are
never used for model promotion.

The current canonical count model fails the declared fit-quality diagnostics,
so target ratios remain exploratory. The all-declared RMS covariance includes
every successful declared variant, including rejected fits. The separate
fit-quality-acceptable covariance is unavailable, not zero, when no
noncanonical variant passes every applicability check. Generated comparison,
covariance, fit-diagnostic, and bootstrap files contain the current values;
the bounded bootstrap remains descriptive and cannot route around a failed
applicability gate.
Legacy monoenergetic simulation ROOT files are absent. Phase 2 writes measured
candidates only. Each future generated energy and response identity must be
fit separately. Missing simulations must not be fabricated, added, or inferred
from measured fits.

## Recommendation

For any future Table 3 replacement, use exposure-summed fitted detector counts
or the equivalent total-live-time detected count rates as the primary
estimands. Preserve per-run rates and GLS heterogeneity beside them. Do not
label cross-energy detector ratios as relative emission probability, neutron
flux, or cadmium abundance without a declared efficiency/response treatment.
The present fit-quality rejection means even these corrected detector
estimands remain exploratory.

Do not replace Table 8 yet. The Fe/Cu canonical audit and equal-status Al/Ge
sensitivities materially improve the model,
but the canonical count model is still rejected and several ratios are
model-systematic dominated. A defensible replacement needs a better validated
high-energy line/background response and the missing separately identified
monoenergetic simulation products. No manuscript files are changed by this
phase.

## Public command and outputs

Use a new empty directory:

```bash
peak_output_dir=$(mktemp -d /tmp/hfir-peak-statistics.XXXXXX)
python3 scripts/reanalyze_paper_peak_statistics.py \
  --bundle /path/to/HFIRBG_public_data_v1.1.0 \
  --output-dir "$peak_output_dir"
```

Default: 12 deterministic diagnostic bootstrap replicas per table.
`--table 3` and `--table 8` select one lane. `--bootstrap-replicates N`
changes the bounded diagnostic count and is recorded.

Outputs include:

- `table3_candidate.csv`, whose primary columns are exposure-summed fitted
  detector counts/rates, and `table8_candidate.csv`;
- Table 3 per-run line rates, exposure-summed estimands, per-run/aggregate
  ratios, one normalized interior-only GLS heterogeneity table with boundary
  exclusions, temporal identifiability decisions, and corresponding
  covariance matrices;
- the exact historical Table 3 reconstruction JSON, compact restrictive
  yield-model comparison, and 558-keV component/model audit;
- Table 8 component audit, explicit Al-27/Ge-70 discrimination record, model
  comparison, and separate Fisher/all-declared/acceptable-only/total
  target-ratio covariance files;
- per-window diagnostics and gzip per-bin observed/expected files;
- deterministic bootstrap diagnostic CSVs;
- convergence, active bounds, parameter errors, rank/condition, deviance, and
  applicability diagnostics; and
- `manifest.json` with database/spectrum/config/output hashes, code revision,
  reporting choices, definitions, seeds, units, identities, and scientific
  non-scope.

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
python3 -m pytest -q tests/test_run_estimands.py
python3 -m pytest -q tests/test_peak_caller_audit.py
python3 -m pytest -q tests/test_peak_area_semantics.py
python3 -m pytest -q tests/test_parent_group_fit.py
python3 -m pytest -q tests/test_peak_phase2_config.py
python3 -m py_compile scripts/reanalyze_paper_peak_statistics.py
```

Manufactured cases protect bin integration, covariance, gradients, response
drift, independent-run aggregation, exact quadratic-cone handling,
optimization failure/repair paths, profile limits, Gaussian/Poisson bootstrap
behavior, boundary-aware heterogeneity, temporal identifiability, and Fisher
coverage.
