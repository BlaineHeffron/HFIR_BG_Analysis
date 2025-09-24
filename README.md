## HFIR_BG_Analysis – Getting Started

### Overview
HFIR_BG_Analysis contains utilities to read, process, calibrate, and plot HPGe gamma spectra from ORNL HFIR background studies. It supports:
- Reading CNF-converted .txt spectra (with A0/A1 calibration and timing metadata)
- Rebinning, combining runs, background subtraction
- Peak fitting and spectrum calibration
- Writing ROOT histograms and Canberra .spe files
- Querying files and metadata from a SQLite calibration database
- Coordinate utilities for cart scans

### Prerequisites
- Python 3.8+ recommended
- CERN ROOT with PyROOT (import ROOT must work in Python)
  - Conda: conda install -c conda-forge root
  - Homebrew (macOS): brew install root
  - System packages or binaries from root.cern
- C compiler (to build the optional write_spe.so)
  - macOS: clang (Xcode Command Line Tools)
  - Linux: gcc/clang
- OS support: Linux/macOS are supported; Windows is partially supported (ROOT and write_spe build may require changes)

### Python packages (install with pip)
- numpy
- scipy
- matplotlib
- numba

Example:
pip install numpy scipy matplotlib numba

Optional (headless plotting): set environment variable MPLBACKEND=Agg to avoid display issues on servers.

### Environment variables
Set these before running the scripts:
- HFIRBGDATA
  - Path to the directory that holds your CNF-converted .txt spectra.
  - Many scripts read this to find input files.
  - Example (bash):
    export HFIRBGDATA=/path/to/your/spectra
- HFIRBG_CALDB
  - Path to the SQLite calibration/database file (contains run metadata, calibration groups, etc.).
  - Example (bash):
    export HFIRBG_CALDB=/path/to/calibration.db

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
## Index your current files into the DB (adds start/live times and file locations):
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
