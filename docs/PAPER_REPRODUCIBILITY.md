# Paper figure reproducibility

The paper contains 28 numbered figures. The project does **not** claim that all
28 can currently be recalculated from the public HFIR spectrum bundle. At this
release checkpoint, Figure 14 is recalculated from public measurements and the
requested three-location portion of Figure 19 is replotted from published
ancillary CSVs; neither `--all` nor copying a publication artifact changes that
scope. The
machine-readable inventory at [`config/paper_figures.json`](../config/paper_figures.json)
records the status, inputs, published artifact, supported command, and known
limitations for every figure.

The separate [Python peak-area audit](PEAK_AREA_AUDIT.md) records the estimand,
consumer graph, immutable public-data replay, and paper-claim impact for the
legacy Gaussian peak-window path.

The [phase-2 peak-statistics correction](PEAK_STATISTICS_CORRECTION.md) defines
the separate exploratory simultaneous-Poisson workflow for the measured-data
parts of Tables 3 and 8. Candidate numbers remain unapproved manuscript
replacements; both measured-data models currently fail the declared
fit-quality criteria. That failure does not validate the historical method:
the common-diagnostics comparison is a separate future lane. Table 3 reports
independent per-run detected yields, aggregate detector counts/rates, one
normalized covariance-aware heterogeneity table with boundary exclusions,
temporal-model identifiability, a compact yield-model comparison, and an exact
historical record.
Table 8 adds an authoritative Fe/Cu component audit, equal-status Al-27/Ge-70
sensitivities, and separate all-declared versus fit-quality-acceptable model
sensitivity.

List the complete inventory after running the public setup:

```bash
.venv/bin/python scripts/reproduce_paper.py --list
```

Process one figure or the full inventory:

```bash
source .env
.venv/bin/python scripts/reproduce_paper.py --figure 14
.venv/bin/python scripts/reproduce_paper.py --all
```

For every figure, the command copies the publication artifact downloaded from
the official arXiv source into `analysis/paper_figures/published/`. It also runs
a supported recalculation or ancillary replot where one exists. This makes the distinction
between the paper's image and a newly calculated result visible in the output
tree rather than silently substituting one for the other.

Use `--dry-run` to inspect all planned actions without creating files:

```bash
.venv/bin/python scripts/reproduce_paper.py --all --dry-run
```

## Status meanings

- `reproducible`: regenerated from the public calibrated spectra and database.
- `published-ancillary`: numerical results are distributed with the official
  arXiv ancillary bundle, but their upstream calculation needs other inputs.
- `not-yet-ported`: required measurements are public, while the exact legacy
  selection/plotting workflow has not yet been promoted to a supported command.
- `source-artifact`: original diagram, drawing, or photograph.
- `external-input-required`: exact recreation needs non-public experiment data
  or simulation products.

Figure 14 is presently the paper-exact public-data regeneration checkpoint.
Figure 19 also has a supported ancillary-data command for the three requested
locations, while the copied paper artifact contains the full six-location view.
Figure 7 remains `not-yet-ported` for paper-exact contour styling, but its
underlying point selection, individual spectra, and binning statistics now have
a supported ROOT-free workflow in `scripts/analyze_floor_scan_statistics.py`.
The manifest should be updated whenever another legacy analysis becomes a
supported, tested entry point.
