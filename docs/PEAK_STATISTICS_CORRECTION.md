# Paper peak-statistics correction, phase 2

Status: **measured-data arithmetic correction with rejected absolute fit
diagnostics**. Historical numerical replay remains separate and explicitly
named. Selection of corrected table entries does not certify either the
historical or phase-2 count model.

Comparison limit: the audit applies one Poisson-deviance diagnostic family to
the recovered historical fits and phase 2. Their complete fitted domains
differ. A separate descriptive comparison therefore intersects their native
channels and evaluates both fitted expectations on the same 324 combined-count
bins. It gives D = 253.346 for the historical local fits and D = 963.240
for the final phase-2 expectation. The parameterizations and the other bins
that informed the phase-2 nuisance estimates still differ, so this is not a
promotion ranking. Failure or a smaller matched-bin deviance for either family
does not establish adequacy of that family.

This workflow replaces the paper-facing uncertainty calculation for the
measured-data portions of Tables 3 and 8. It does not infer cadmium abundance,
run neutron transport, regenerate a response matrix, unfold a measured
spectrum, or construct missing simulations. Released calibrated spectra are
detector counts.

Frozen definitions: [`config/paper_peak_statistics.json`](../config/paper_peak_statistics.json).
Historical consumer inventory: [`config/peak_area_callers.json`](../config/peak_area_callers.json).
Prior audit: [Python peak-area and paper-impact audit](PEAK_AREA_AUDIT.md).

## Manuscript map and correction selection

The built manuscript maps `tab:rd_lines` to Table 3 and
`table:peak_ratio_compare` to Table 8. Table 5 (`table:eff`), the peak-width
table (`tab:peak-fit`), rate tables, and the source catalogue have independent
calculation paths and are not changed by the Python peak-area defect.

The pre-repair numerical selection checkpoint is the clean analysis revision
`6ff9ff0424282ab28302567dfa0e99dfcd677b10`, schema-8 configuration SHA-256
`2f3c9fb199d88272a635950706700f36a90285b27bfd9fb5689e2d6e10d90cb8`,
and one-replica diagnostic replay manifest SHA-256
`81915772224bdfc0a296368b9f82840112943bd2fdf23e7822b213189ee50035`.
The candidate CSV SHA-256 values are
`dd7569439ede5a1c2d3aa5bb13abed2068524232d778156afd0915e1776deaea`
for Table 3 and
`7dfba8c896c689c78498761eb64b243ab0b668d72c5ff858fc341f8ffd46a870`
for Table 8. The single bootstrap replica checks deterministic plumbing only;
it is not used for the table values or for coverage claims.

No phase-2 Table 3 row is selected for manuscript replacement. The candidate
CSV retains conditional fitted values for diagnosis: 22 rows carry local
Fisher plus declared reference-model sensitivity, two require boundary-aware
treatment, and 11 are not fitted. The absolute count model remains rejected,
so none of those conditional values has demonstrated coverage. The exact
self-ratio is still one with zero local variance; that identity does not make
the surrounding model applicable.

No phase-2 Table 8 ratio is selected for manuscript replacement. All 24 ratios
are unavailable because the final absolute count model is rejected and one
background endpoint is on its physical bound. Fisher covariance and the RMS
spread over successful declared variants remain conditional diagnostics, not
uncertainties with established coverage. The unrecoverable legacy simulation
rows are withdrawn rather than mixed with corrected measured ratios. This
removes the former data-versus-simulation validation claim and requires author
ratification.

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
After the baseline residual map showed coherent antisymmetric centroid lobes
and a non-affine U-shaped energy trend, Table 8 adds exactly one curvature
coefficient,

\[
 e_{ri}\mathrel{+}=c\left(\frac{E^{\rm nominal}_{ri}-8587.055}
 {2825.445}\right)^2.
\]

The pivot and scale are fixed by the full declared Table 8 energy span. The
constraint is c = 0 +/- 0.893146 keV with fixed three-sigma bounds. One
released channel motivates that scale; it is an applicability assumption, not
released nonlinearity metrology. The same term and prior are used in every
component, tail, and background comparison. No per-line centroid freedom or
higher polynomial is allowed.

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

The 558-keV audit uses the current NNDC CapGam Cd-113 energy 558.456 keV and
tests the rounded paper energy 558.5 keV, the IAEA PGAA value 558.32(3) keV,
and an extended-window sensitivity containing the known Tl-208 583.187-keV
decay line. NNDC/IAEA energy matches for Co-59, In-115, and W-186 are recorded
but not fitted or promoted because the release does not establish those source
materials. Ratio shifts from successful predeclared reference-window variants
form a positive-semidefinite RMS outer-product covariance, written separately
from Fisher covariance. Nuclear energy agreement alone never promotes a
component. The declared RMS construction includes all three successful
noncanonical variants: `paper_energy_558_5`, `iaea_energy_558_32`, and
`capgam_energy_plus_tl208_583`.

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

### Table 3 residual diagnosis and repair

The unmodified schema-8 replay is byte-exact at the candidate CSV level. It
has D = 7243.933/5434; bins meeting the declared expectation floor give
D = 5920.104/4178. Per-file D/bin is 1.307, 1.392, and 1.401
for the three 86400-s files. The 813.89-s file gives 1.041 over all bins and
0.931 over eligible bins. Short-run padding is not the cause. Same-channel
signed-residual correlations among the three long runs are only 0.24--0.29,
while moderate excess scatter spans many windows.

The strongest baseline morphologies were peak-centered or centroid-like in
the 352, 558, 595.85, 1408, 1764, 2224, 2456, and 2617-keV neighborhoods.
A pooled low-energy-tail score was only 1.04 standard deviations with mixed
line signs. Sideband shelf scores were not recurrent, and accurately known
decay-line centroid shifts did not form one coherent quadratic trend. Global
tail, Compton-shelf, Table 3 curvature, and freed per-run background-shape
models are therefore rejected rather than added for aggregate score gain.

Every fitted line energy was checked against current IAEA LiveChart/ENSDF
decay data or NNDC CapGam capture data. Component names and the separate
`paper_row_energy_keV` field retain manuscript labels; `energy_keV` is the
value used in the likelihood. The simultaneous authoritative-energy update
uses identical native bins, adds no parameter, reduces D by 43.675, and
moves the short-run 2455.8-keV row off its zero bound. The 2223.245-keV H-1
and 2456.0-keV Cd-113 corrections reduce named centroid error, but both
windows still fail an absolute diagnostic. The CapGam 558.456-keV reference
is canonical; rounded-paper, IAEA-PGAA, and Tl-208 window-extension fits remain
declared sensitivities.

The current evaluated Bi-214 record is 2204.10(4) keV; 2204.215(40) keV is
one input measurement, not the evaluated weighted average. Current CapGam
also confirms the retained Cd-113 capture energies 2398.6(1), 2550.1(1),
2660.1(1), and 2767.5(2) keV. The 2550.1-keV centroid-score shift therefore
remains unresolved residual structure rather than a catalog-rounding repair.

The only added line is the catalogued Ac-228 338.320-keV nuisance in `rd_352`.
Its three long-run rates are 0.00627, 0.00620, and 0.00531 counts/s and remain
interior. Against the complete Ni-58 339.418-keV alternative, the Ni yield is
boundary-pinned in two long runs; the Ac interpretation changes the reported
351.9-keV ratio by only 0.00087. On 4668 identical bins, adding Ac changes
D = 5824.443 with 169 parameters to D = 5734.740 with 173 parameters;
both Fisher matrices are full rank. Ac is retained because it has established
Th-chain provenance, recurrent run support, and a concrete row-level action,
not because of a threshold crossing.

Three windows cannot support an identifiable released-data model and are
removed whole:

- `rd_609` contains the intrinsic asymmetric 595.85-keV Ge inelastic feature,
  but no released normalized response shape; a Gaussian surrogate is wrong.
- `rd_707_725` contains unresolved Cd-113 725.298, Ac-228 726.863, and
  Bi-212 727.330-keV structure. Candidate correlations exceed 0.915, bounds
  activate, and long-run 725-keV yields move by 20--31%.
- `rd_1364_1400` contains unresolved Cd/Pb/Bi/Al structure from 1399.638 to
  1408.300 keV. Candidate correlations exceed 0.977, several bounds activate,
  and long-run 1399.6-keV yields move by 17--51%.

The earlier exclusions at 478, 5433, 5825, 7368, and 7916 keV remain. Thus 11
of 35 manuscript rows are not fitted. The final 24-row fit gives
D = 5734.740/4495; eligible bins give D = 4654.708/3475, with reference
p = 6.78 x 10^-38. The 558, 1281/1294, 1764, 2204/2223, 2456, and
2615/2660 windows still fail the familywise window check. Fisher information
is full rank 173 with scaled condition 3.36 x 10^5; no nonnegative
background cone is active. Only short-run 242.0- and 2614.533-keV yields remain
on zero bounds. Broad residual discrepancy therefore remains unresolved; no
single deviance-scale multiplier is applied.

The candidate table retains exposure-summed detector counts/rates and
detector-response-dependent ratios only as conditional diagnostics. Cross-
energy HPGe efficiency does not cancel, so these values cannot establish
incident neutron flux or cadmium abundance. The 22 regular rows carry local
Fisher plus declared reference-model sensitivity; two rows require bounded
treatment; all 24 remain unavailable for manuscript replacement because the
absolute model is rejected. The default 12-replica bootstrap is descriptive;
the one-replica reproduction run tests plumbing only, and coverage remains
untested below 200 successful replicas.

The historical reconstruction record freezes the exact source revision, git
blobs, file order, spectrum hashes, calibration, live times, operation order,
and every legacy ratio. It establishes that the paper numbers came from
combining four Cycle493 count arrays before fitting and then dividing
`N_window/(7*sigma_keV)` values. They were not run-by-run inverse-variance
averages and were not integrated peak-count or flux ratios. The area-like bug
entered `Spectrum.py` in commit `5cb9e0a`; it inserts an unintended
sigma-reference/sigma-line factor. The paper caption says Cycle498, but the
reconstructed workflow and released inputs are Cycle493.

On the recovered historical fits, the complete normalized signal integral is

\[
 N_{\rm full}=\frac{H}{A_1}\left[\sqrt{2\pi}(1-R)\sigma
 +2R\beta\exp\left(-\frac{\sigma^2}{2\beta^2}\right)\right].
\]

The audit propagates the full gradient of this expression through each
historical fit covariance and retains within-multiplet covariance in ratios.
This exact same-bin/background re-expression has post-fit Poisson deviance
375.072 on 238 descriptive degrees of freedom
(chi-square-reference \(p=3.38\times10^{-8}\)). Several weak-line tail
parameters make the extrapolated full-line covariance unstable. The comparison
columns are retained for sensitivity, but this corrected-legacy estimator is
not selected table-wide. The final phase-2 fit uses different windows and has
deviance 5734.740 on 4495 descriptive degrees of freedom. Those complete-domain
values are unmatched and cannot rank the families. On the strict 324-channel
intersection, historical and phase-2 deviances are 253.346 and 963.240,
respectively; phase 2 has 12 versus two bins above four Pearson units. This is
still descriptive because the local historical fits and the simultaneous
phase-2 nuisance model were estimated from different complete domains. Both
complete models fail their absolute diagnostic.

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

All eight parent energies exactly match NNDC CapGam. The configured SEP/DEP
arithmetic retains the paper convention (E-511.000) and (E-1022.000) keV;
using the CODATA electron rest energy would change them by at most 0.002099
keV, negligible beside the nuclear-energy uncertainties and fitted resolution.
Thus catalog rounding does not explain the high-energy residual trend.

### Table 8 residual diagnosis and repair

The byte-exact schema-8 baseline gives D = 3194.266/1097. The three cluster
multiplet windows contribute 1726 deviance units, 54% of the total. Their
dominant residuals are antisymmetric and peak-centered, not smooth sideband
curvature. Independent centroid scores show a U-shaped non-affine trend: for
example +0.248 keV at 5787, -0.668 keV at 9719, +0.945 keV at 10365, and
+0.915 keV at 10876. This morphology and the completed energy audit motivate
the single constrained curvature coefficient above.

On the same 1176 native bins, the affine-calibration comparator reproduces
D = 3194.266; the one-curvature model gives D = 2708.880/1096. Its
coefficient is -0.90538 +/- 0.04150 keV, interior to the fixed bounds. The
offset/stretch/curvature correlation block has off-diagonal values -0.915,
-0.368, and +0.241; the scaled Fisher condition improves from
4.40 x 10^5 to 4.31 x 10^5, with full rank 80. Maximum movement
among the 24 reported ratios is 1.75%. The term is therefore identifiable and
addresses a named morphology, but it does not make the model adequate.

Large role-dependent centroid scores remain after curvature: mean conditional
shifts are -0.199 keV for FEPs, +0.035 keV for SEPs, and +0.052 keV for DEPs,
with individual high-energy FEP/escape shifts of opposite sign. This is not a
smooth calibration residual and no higher-order calibration term is added.
All three cluster windows still fail, as do all three 6809-keV windows, the
11387-keV DEP, the 8999-keV DEP/SEP, and the 9719-keV FEP. The final eligible
diagnostic is D = 2708.806/1095, p = 9.90 x 10^-138, with 26 bins
above four and 11 above five Pearson units. Fisher covariance is valid, but
the high endpoint of the `t8_11387_fep` nonnegative background is on its bound.

The close Al-27 and Ge-70 hypotheses remain equal-status noncanonical
sensitivities. They reduce local deviance but directly reallocate the
7724.034-keV DEP and still fail every absolute applicability check. The full
steel catalog is rejected as numerically fragile (scaled Fisher condition
1.21 x 10^11). Removing the tail raises D to 3592.274; an affine
background raises it to 3042.539. These results reject those hypotheses rather
than invite parameter stacking. Fe/Cu remains the minimal provenance-supported
component set; no claim is made that it explains the remaining unidentified
structure. Residual-only automatic discovery remains prohibited.

Every component, tail, calibration, and background comparison uses the same
native bins. The phase-2 `t8_6809_sep` window is widened identically for all of
them. Model-comparison AIC/BIC-shaped columns remain explicitly nonstandard
penalized-objective diagnostics and do not select a model. No declared variant
passes. Consequently all 24 Table 8 rows are unavailable; the acceptable-model
covariance is absent, not zero, and the all-variant RMS remains sensitivity
only.
Legacy monoenergetic simulation ROOT files are absent. Phase 2 writes measured
candidates only. Each future generated energy and response identity must be
fit separately. Missing simulations must not be fabricated, added, or inferred
from measured fits.

## Manuscript-use recommendation

Do not replace any Table 3 row with a phase-2 number. Mark the 478, 609,
707/725, 1364/1377/1399, 5433, 5825, 7368, and 7916-keV rows unavailable for
their local reasons. Mark the other 24 rows unavailable because the shared
absolute model remains rejected; conditional counts, rates, Fisher terms, and
model sensitivities may be retained only as reproducibility diagnostics. The
cadmium-tuning claim remains unavailable. Any future use as an explicitly
exploratory appendix result is an author decision and must state that coverage
is unknown and that no efficiency correction or unfolding was performed.

Do not replace any Table 8 row with a phase-2 ratio. Replacing only the
published uncertainty while retaining its unrecoverable legacy central fit
would mix estimators; retaining simulation rows would preserve defective,
unrecomputable arithmetic. Withdraw the measured-versus-simulation validation
claim. The author decision is whether to remove the table, retain the complete
historical table with an explicit erratum, or present the phase-2 values only
as rejected-model diagnostics. None is a validated high-energy response
result.

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
  yield-model comparison, matched Ac-228 repair comparison and row impacts,
  and 558-keV component/model audit;
- Table 8 component audit, explicit Al-27/Ge-70 discrimination record, model
  comparison, and separate Fisher/all-declared/acceptable-only/total
  target-ratio covariance files;
- per-window signed-deviance/cluster diagnostics and gzip per-native-bin files
  preserving channel, count, nominal/fitted energy, nearest component, and
  exact Poisson-deviance contribution;
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
