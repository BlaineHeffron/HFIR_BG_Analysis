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
The manifest should be updated whenever another legacy analysis becomes a
supported, tested entry point.
