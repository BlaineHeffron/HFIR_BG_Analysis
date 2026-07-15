## HFIR_BG_Analysis – Getting Started

### Overview
HFIR_BG_Analysis contains utilities to read, process, calibrate, and plot HPGe gamma spectra from ORNL HFIR background studies. It supports:
- Reading CNF-converted .txt spectra (with A0/A1 calibration and timing metadata)
- Rebinning, combining runs, background subtraction
- Peak fitting and spectrum calibration
- Writing ROOT histograms and Canberra .spe files
- Querying files and metadata from a SQLite calibration database
- Coordinate utilities for cart scans

### Fastest complete setup

On Linux x86-64, the setup can be performed from the repository root with:

```bash
./scripts/setup_analysis.sh
source .env
.venv/bin/python scripts/public_analysis.py all
```

This downloads the calibrated public spectra/database and the official paper
ancillary results, creates an isolated environment with PyROOT, validates the
database, and generates the key unfolded-flux and Figure 14 shield-comparison
products. See the [public analysis guide](docs/PUBLIC_ANALYSIS_GUIDE.md) for
output descriptions, individual commands, conda/macOS setup, and the distinction
between measured spectra and published unfolded flux.

The [paper figure reproducibility inventory](docs/PAPER_REPRODUCIBILITY.md)
tracks all 28 numbered figures without implying that figures needing external
simulation or PROSPECT inputs can be rebuilt from this data release. List it
with `.venv/bin/python scripts/reproduce_paper.py --list`.

### Prerequisites
- Python 3.10+ recommended
- CERN ROOT with PyROOT (import ROOT must work in Python)
  - Conda: conda install -c conda-forge root
  - Homebrew (macOS): brew install root
  - System packages or binaries from root.cern
- C compiler (to build the optional write_spe.so)
  - macOS: clang (Xcode Command Line Tools)
  - Linux: gcc/clang
- OS support: Linux/macOS are supported; Windows is partially supported (ROOT and write_spe build may require changes)

### Python packages

The complete package set is recorded in [`requirements.txt`](requirements.txt)
and [`environment.yml`](environment.yml). For a manual Linux installation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The PyPI ROOT wheel is currently alpha and Linux x86-64 only. Conda is the
recommended portable alternative:

```bash
conda env create -f environment.yml
conda activate hfir-bg-analysis
```

Optional (headless plotting): set environment variable MPLBACKEND=Agg to avoid display issues on servers.

### Environment variables
Set these before running the scripts:
- HFIRBGDATA
  - Path to the directory that holds your CNF-converted .txt spectra.
  - Many scripts read this to find input files.
  - Example (bash):
    export HFIRBGDATA=/path/to/your/spectra
- HFIRBG_CALDB
  - Optional path override for the SQLite calibration/database file (contains run metadata, calibration groups, detector configurations, and run-to-calibration assignments).
  - If unset, the code looks for `db/HFIRBG.db` and then for `HFIRBG.db` beside the `HFIRBGDATA` directory (the public release layout).
  - Example (bash):
    export HFIRBG_CALDB=/path/to/calibration.db
- HFIRBG_ANALYSIS
  - Optional output root for generated plots and CSVs.
  - The portable environment defaults to the repository's ignored `analysis/` directory.
- HFIRBG_PAPER_DATA
  - Optional location of the official arXiv ancillary CSVs and plots.
  - The portable environment defaults to `data/arxiv_2607.05834/anc/`.

Paths may contain spaces; quote them when exporting, for example:

```bash
export HFIRBGDATA="$HOME/data/HFIR gamma spectra"
export HFIRBG_CALDB="$HOME/data/HFIRBG.db"  # only if not using db/HFIRBG.db
```

### Public data + canonical database setup

The versioned public bundle contains 1,802 calibrated text spectra and the pre-populated canonical SQLite database. The setup below keeps it in the repository's ignored `data/` directory.

1. Clone and enter the analysis repository:

```bash
git clone https://github.com/BlaineHeffron/HFIR_BG_Analysis.git
cd HFIR_BG_Analysis
```

2. Download, verify, and extract the bundle from the [`data-v1.0.0` GitHub Release](https://github.com/BlaineHeffron/HFIR_BG_Analysis/releases/tag/data-v1.0.0):

```bash
mkdir -p data
curl -L -o data/HFIRBG_public_data_v1.0.0.tar.gz \
  https://github.com/BlaineHeffron/HFIR_BG_Analysis/releases/download/data-v1.0.0/HFIRBG_public_data_v1.0.0.tar.gz

echo "037dfce3383a7b86d40772a45253423c0e63f1d803eba246279cccb124c9b2c4  data/HFIRBG_public_data_v1.0.0.tar.gz" | sha256sum -c -
tar -xzf data/HFIRBG_public_data_v1.0.0.tar.gz -C data
```

The extracted layout is:

```text
data/
└── HFIRBG_public_data_v1.0.0/
    ├── HFIRBG.db
    └── spectra/
        └── 1,802 calibrated .txt spectra
```

You do **not** need to build, re-index, or recalibrate this database. The v1.0.0 database contains a legacy absolute directory from the machine on which it was created. `HFIRBGDATA` always overrides stored directory values, and the setup check below can replace the legacy value with the portable, bundle-relative `spectra` path.

3. Load the repository-relative environment defaults:

```bash
cp .env.example .env
source .env
```

That is the complete configuration for the standard public bundle. If the data or database is moved elsewhere, set `HFIRBGDATA` and `HFIRBG_CALDB` before sourcing `.env`; existing values take precedence over its defaults.

Environment variables set this way last only for the current shell unless you source `.env` from `~/.bashrc`, `~/.zshrc`, or another shell startup file. `.env` is intentionally ignored by Git.

Verify the download and database mapping without installing ROOT:

```bash
python3 scripts/check_public_data_setup.py --sanitize-database-path
```

The sanitizing option is safe to run repeatedly. It removes the creator's absolute path from the extracted SQLite file; future checks can omit the option.

Then perform a database-backed smoke test (requires the dependencies listed above, including PyROOT):

```bash
.venv/bin/python - <<'PY'
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import retrieve_data

db = HFIRBG_DB()
file_id = db.retrieve_file_ids(["PROSPECT_DOWN_19"])[0]
path = db.get_file_paths_from_ids([file_id])[0]
spectrum = retrieve_data(path, db)
print(path)
print(f"live time: {spectrum.live:.2f} s")
PY
```

Do not run `db.sync_files()` for the canonical public database: its file, run, coordinate, and calibration relationships are already populated. `HFIRBGDATA` remains the authoritative path override even after the stored path is sanitized.

Repository maintainers: the uncompressed `.txt` collection is roughly 682 MB, so it is distributed as a versioned GitHub Release asset rather than committed through ordinary Git history. The current `.gitignore` intentionally ignores `data/`.

If you don’t have a database yet:
- You can still load and analyze individual .txt spectra (A0/A1 and times are read from the file).
- Some features (e.g., calibration group management, cart scan queries) require a populated SQLite DB. Obtain a prepared DB or build one internally.

### Build the .spe writer (optional)
There’s a small C helper to write .spe files used by write_spe in src/utilities/util.py.

cd src/utilities
cc -fPIC -shared -o write_spe.so write_spe.c

This creates write_spe.so next to write_spe.c.

Data format expectations
The .txt spectra are assumed to be CNF-converted text with:
- Header lines starting with #, including:
  - Start time: YYYY-MM-DD, HH:MM:SS
  - Live time ...
  - #     A0: ...
  - #     A1: ...
- Tab-separated data rows; counts read from the 3rd column (index 2).

# Quick start (Python snippets)
1) Load a spectrum from a file
from src.utilities.util import retrieve_data
from src.database.HFIRBG_DB import HFIRBG_DB

## Use DB if available (to override A0/A1 and times)
db = HFIRBG_DB()  # uses HFIRBG_CALDB
spec = retrieve_data("/full/path/to/00001234.txt", db)  # or just db=None

2) Plot one or multiple spectra
from src.utilities.util import plot_multi_spectra

## Single spectrum dict: name -> SpectrumData
fdict = {"Run_1234": spec}
plot_multi_spectra(fdict, n="run_1234_plot", emin=20, emax=3000, ylog=True)

Saves run_1234_plot.png

3) Combine runs and subtract background
from src.utilities.util import combine_runs, background_subtract

## Suppose fdict is name -> SpectrumData (or a set/list for each name, then combine)
combine_runs(fdict)  # sums lists/sets into single SpectrumData per key

## Background subtract (rebin edges must match)
bin_edges = fdict["Run_1234"].bin_edges
sub = background_subtract(fdict, subkey="Background_Run", bin_edges=bin_edges)

4) Fit peaks and write calibration back to DB
from src.utilities.util import calibrate_spectra

expected_peaks = [511.0, 661.657, 1173.228, 1332.492]
calibrate_spectra(fdict, expected_peaks, db, plot_dir="fit_plots", user_verify=False, plot_fit=True)

5) Write ROOT file for a spectrum
from src.utilities.util import write_root_with_db

write_root_with_db(spec, name="GeDataHist", db=db)  # writes alongside original .txt path as .root

6) Write .spe file
from src.utilities.util import write_spe

write_spe("/tmp/output.spe", spec.data.astype(float))

# Working with the database

## Building or updating a non-canonical database

The following indexing workflow is for maintainers or users creating their own database. It is not required for the canonical public database.

Index your current files into the DB (adds start/live times and file locations):
from src.database.HFIRBG_DB import HFIRBG_DB
db = HFIRBG_DB()  # uses HFIRBG_CALDB
db.sync_files()   # scans HFIRBGDATA and adds entries

## Load additional run/shield metadata (requires a DB with schema and project CSVs/JSON in repo/db):
db.sync_db()
**Also available: db.set_calibration_groups() to create initial cal groups from .txt headers**

## Retrieve Russian-doll shield runs, filter by RX on/off or gain, and get SpectrumData:
from src.utilities.util import populate_rd_data, combine_runs
rd_files = db.get_rd_files(min_time=100, rxon_only=True)  # returns shield/acquisition grouped paths
rd_data = populate_rd_data(rd_files, db)                  # maps to SpectrumData
for shield in rd_data:
    combine_runs(rd_data[shield], ignore_failed_add=True)

# Coordinate system (cart scans)
Cart and detector orientation are represented using a 2D room coordinate system (x, z), in inches. Two points define the cart footprint on the floor:
- Right corner: (Rx, Rz)
- Left corner: (Lx, Lz)

## Angles:
- phi (cart angle): measured in the detector plane, derived from the vector between left and right corners.
  - Reference: phi = 0° when the cart faces East (Lx = Rx and Lz < Rz).
  - phi increases counter-clockwise:
  - Computed by convert_coord_to_phi(Rx, Rz, Lx, Lz)
- theta (detector tilt/rotation): the “angle” stored in the DB row["angle"] and used in position transforms.

## Detector face location:
- convert_cart_coord_to_det_coord(Rx, Rz, Lx, Lz, angle) returns the approximate center of the detector face in room coordinates, using:
  - Axis of rotation ≈ 8.8 in West of the right corner
  - Detector face center offset ≈ 8.0 in from rotation axis
  - When phi = 90° (cart oriented North), the face center is ≈ 16 in South and 0.8 in West of the right corner
- All distances in inches; angle in degrees.

# Using cart scan helpers:
from src.database.CartScanFiles import CartScanFiles

cdb = CartScanFiles()  # inherits HFIRBG_DB
pos = cdb.retrieve_position_spectra(min_E_range=11500)
**pos is a dict keyed by "theta phi" string, each value is a list of tuples:**
    **(SpectrumData, [detector_face_x, detector_face_z])**

# Troubleshooting
- ImportError: No module named ROOT
  - Ensure ROOT is installed and your Python environment picks up PyROOT (e.g., use conda-forge root, or source ROOT’s thisroot.sh before activating your venv).
- RuntimeError: set environment variable HFIRBGDATA…
  - export HFIRBGDATA=/path/to/spectra
- Database errors (missing tables)
  - You need an initialized SQLite DB with the expected schema. Obtain it from your project, or contact maintainers. Some features don’t require the DB (pass db=None when allowed).
- write_spe.so not found or load failure
  - Rebuild it in src/utilities with cc -fPIC -shared -o write_spe.so write_spe.c
  - On Windows this would need a .dll build and code changes; not supported out-of-the-box.

# Notes
- Many functions accept either SpectrumData objects or file paths; if you pass file paths and a DB handle, calibration/time may be overridden using DB entries.
- Rebinning: Spec histograms include an underflow and overflow bin; methods that operate on normalized rates ignore those.
- Time-series plotting and relative comparisons (ratio/difference) are included in src/utilities/util.py.

# Directory recap
- src/analysis: Spectrum fitting and models (Gaussian + skew + background)
- src/utilities: IO, plotting helpers, rebinning, time-series, write_spe
- src/database: SQLite manager, HFIRBG_DB wrapper, cart scan helpers
- db/: Project CSV/JSON metadata (used by sync_db), not a DB schema generator

## If you need a minimal end-to-end test:
- Set HFIRBGDATA and HFIRBG_CALDB
- pip install numpy scipy matplotlib numba
- Ensure ROOT works in Python (import ROOT)
- Build write_spe.so (optional)
- Load a file with retrieve_data and call plot_multi_spectra to create a PNG

That should verify the environment is correctly set up.
