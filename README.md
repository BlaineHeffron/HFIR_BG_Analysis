# HFIR gamma-background analysis

This repository provides read-only access to the released HPGe gamma spectra,
run metadata, detector locations/orientations, shield configurations, and
selected paper products from *Gamma Backgrounds for Experiments at the High
Flux Isotope Reactor* ([arXiv:2607.05834](https://arxiv.org/abs/2607.05834)).

The public bundle contains 1,802 calibrated text spectra in 354 run records.
The browser and export tools do not require CERN ROOT and never write to the
canonical SQLite database.

## Start here

### Browse and export the released data

On Linux or macOS, from the repository root:

```bash
./scripts/setup_analysis.sh --browser-only
./scripts/run_data_browser.sh
```

Open <http://localhost:8501>. The browser supports:

- filtering by official HFIR cycle, operating/outage classification, shield,
  run text, and recorded location;
- a selectable HFIR location map with cart and detector orientation;
- a run timeline in the `America/New_York` timezone;
- overlays of up to six calibrated spectra with energy range, rebinning,
  normalization, logarithmic scale, and Poisson errors; and
- CSV downloads of filtered run records and displayed spectrum bins.

The setup is safe to rerun. It downloads and verifies the versioned data
bundle, creates `.env` with repository-relative defaults, removes the original
creator's absolute directory from the downloaded database, and installs the
lightweight browser environment in `.venv`.

### Run the paper-facing analyses

The full setup also installs PyROOT and downloads the official arXiv source and
ancillary files:

```bash
./scripts/setup_analysis.sh
source .env
.venv/bin/python scripts/public_analysis.py all
```

This needs Linux x86-64 for the PyPI ROOT wheel. On macOS or for a production
ROOT build, use the conda instructions printed by the setup script:

```bash
./scripts/setup_analysis.sh
conda env create -f environment.yml
conda activate hfir-bg-analysis
source .env
python scripts/check_public_data_setup.py --sanitize-database-path
python scripts/public_analysis.py all
```

Allow approximately 3 GB for the downloads, environment, and generated output.

## Does this reproduce every paper plot?

No. The repository makes the boundary explicit:

- Figure 14 is recalculated from the public calibrated spectra and database.
- The requested three-location portion of Figure 19 is replotted from the
  authoritative published unfolded-flux CSVs. It is not a new unfolding.
- Publication artifacts for all 28 numbered figures can be collected, but the
  other figures are source artwork, ancillary results, require external inputs,
  or have not yet been ported from legacy analysis code.

The complete machine-readable inventory is
[`config/paper_figures.json`](config/paper_figures.json). Inspect it with:

```bash
.venv/bin/python scripts/reproduce_paper.py --list
.venv/bin/python scripts/reproduce_paper.py --all --dry-run
```

See [Paper figure reproducibility](docs/PAPER_REPRODUCIBILITY.md) for the status
definitions and exact output behavior. The broader release remains useful even
when a paper-exact plotting workflow is unavailable: all 1,802 calibrated
spectra can be browsed, filtered, loaded, and exported independently.

## Requested analysis products

After the full setup, run:

```bash
.venv/bin/python scripts/public_analysis.py all
```

Outputs are written to `analysis/public_analysis/`:

- `unfolded_key_locations.{png,pdf}`: isotropic and front-face response-model
  results for MIF reactor-on, MIF reactor-off, and Shield Center;
- `unfolded_key_locations_manifest.csv`: the exact official ancillary input
  used for each curve;
- `shield_configuration_spectra.{png,pdf,csv}`: the Figure 14 comparison of
  no added water/lead, seven water layers, six water layers plus floor lead,
  and seven water layers plus floor lead; and
- `shield_configuration_rates_30_60_keV.csv`: reactor-on, reactor-off, and
  reactor-only rates with statistical errors.

Generate either group separately with `public_analysis.py unfolded` or
`public_analysis.py shields`.

The text spectra are measured detector counts, not unfolded incident flux.
Recalculating the unfolds requires Geant4 response matrices and the external
unfolder, which are not included in the public bundle. The full setup therefore
uses the official arXiv ancillary CSVs as the authoritative unfolded results.

## Export data without the web interface

The exporter writes ordinary, long-form CSV and accepts environment variables
or explicit `--db`/`--data-root` paths.

Export the complete run and file catalogs:

```bash
.venv/bin/python scripts/export_public_data.py catalog
.venv/bin/python scripts/export_public_data.py files
```

Filter runs using the official calendar classification:

```bash
.venv/bin/python scripts/export_public_data.py catalog \
  --cycle 491 --calendar-state operating --contains MIF \
  --output analysis/exports/cycle491_mif.csv
```

Find file IDs for a run, then export calibrated bins for one or more files:

```bash
.venv/bin/python scripts/export_public_data.py files \
  --run-id 42 --output analysis/exports/run42_files.csv

.venv/bin/python scripts/export_public_data.py spectra \
  --file-id 100 --file-id 101 \
  --normalization counts/s/keV --rebin 8 --emin 30 --emax 3000 \
  --output analysis/exports/selected_spectra.csv
```

Use `spectra --run-id 42` to export every calibrated file assigned to a run.
Each spectrum row retains the file/run identifiers, live time, canonical A0/A1
calibration, bin center/width, raw counts, normalized value, and statistical
error.

For Python or notebooks, the same ROOT-free API is available directly:

```python
from src.public_data.browser import (
    load_spectrum,
    query_file_metadata,
    rebin_by_factor,
    spectrum_dataframe,
)

files = query_file_metadata()             # one row per calibrated file
spectrum = load_spectrum(int(files.iloc[0].file_id))
spectrum = rebin_by_factor(spectrum, 8)
bins = spectrum_dataframe(spectrum, "counts/s/keV")
```

## HFIR reactor-cycle calendar

[`reference_data/hfir_cycle_calendar.csv`](reference_data/hfir_cycle_calendar.csv)
is the durable cycle-date record covering this measurement campaign. It is
transcribed from ORNL/TM-2023/3207 Appendix A, page A-3, and stores the source,
DOI, retrieval date, schedule basis, and precision with every row.

The source publishes dates rather than transition times. Classifications
therefore use day precision in `America/New_York`:

- `calendar_cycle` and `calendar_reactor_state` come from the official date
  intervals;
- `reactor_cycle` and `reactor_state` are separate labels inferred only from
  the original run name/description;
- a run spanning classifications is `mixed`; and
- malformed timestamps or dates outside the recorded interval remain
  `unknown`.

Do not infer an exact startup or shutdown time from a boundary date. Full
provenance and limitations are in
[`reference_data/README.md`](reference_data/README.md).

## Data layout and portable paths

The setup downloads the
[`data-v1.0.0` release](https://github.com/BlaineHeffron/HFIR_BG_Analysis/releases/tag/data-v1.0.0)
into this ignored layout:

```text
data/
└── HFIRBG_public_data_v1.0.0/
    ├── HFIRBG.db
    └── spectra/
        └── 1,802 calibrated .txt spectra
```

Configuration is controlled by:

- `HFIRBGDATA`: spectrum directory;
- `HFIRBG_CALDB`: canonical SQLite database;
- `HFIRBG_ANALYSIS`: generated-output root; and
- `HFIRBG_PAPER_DATA`: official paper ancillary directory.

`.env.example` supplies portable repository-relative defaults. The setup check
modifies only the downloaded SQLite copy, replacing its creator-machine path
with the bundle-relative `spectra` directory. `HFIRBGDATA` remains the
authoritative override, so the repository contains no required user-specific
absolute paths.

Validate a moved or manually downloaded bundle with:

```bash
python3 scripts/check_public_data_setup.py --sanitize-database-path
```

Do not run `db.sync_files()` against the canonical public database; all file,
run, coordinate, shield, and calibration relationships are already populated.

## Supplemental unfolding README

[`docs/UNFOLDING_RESULTS_README.md`](docs/UNFOLDING_RESULTS_README.md) is an
adapted and expanded repository guide to the ancillary file formats. After the
full setup, the byte-for-byte official supplemental README referenced by the
paper is at:

```text
data/arxiv_2607.05834/anc/UNFOLDING_RESULTS_README.md
```

The same directory contains all per-location unfolded CSVs, plots, measurement
metadata, and response-matrix validation summaries distributed with the paper.

## Advanced and legacy analysis

The historical `scripts/`, `src/database/`, and `src/utilities/` modules include
peak fitting, calibration management, ROOT output, cart scans, and specialized
paper-development workflows. Some can modify a database or write beside input
spectra. They are intentionally not the default public-data path.

See [Legacy analysis workflows](docs/LEGACY_ANALYSIS.md) before using those
interfaces with a copy of the public database.

## Troubleshooting

- **Browser says data are not configured:** run
  `./scripts/setup_analysis.sh --browser-only`, then relaunch it.
- **Port 8501 is occupied:** run
  `./scripts/run_data_browser.sh --server.port 8502`.
- **`ImportError: ROOT`:** the browser/export path does not need ROOT. For the
  full legacy/paper workflow, use the full setup or conda environment.
- **Moved data:** set `HFIRBGDATA` and `HFIRBG_CALDB` in `.env`, using quoted
  paths when they contain spaces.
