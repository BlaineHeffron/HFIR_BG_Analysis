# Legacy analysis workflows

The public browser and `scripts/export_public_data.py` are read-only and should
be the first choice for exploring the released data. The interfaces below are
retained for specialized and historical analysis; several can update a
database or write files beside spectra. Work on a copy when experimenting.

## Full environment

Legacy modules generally require CERN ROOT with PyROOT plus the packages in
`requirements.txt`. On Linux x86-64, `scripts/setup_analysis.sh` installs the
full environment. A conda-forge ROOT build is described in the root README and
`environment.yml`.

The optional Canberra `.spe` writer can be built with:

```bash
cc -fPIC -shared -o src/utilities/write_spe.so src/utilities/write_spe.c
```

## Load and plot one spectrum

```python
from src.database.HFIRBG_DB import HFIRBG_DB
from src.utilities.util import plot_multi_spectra, retrieve_data

db = HFIRBG_DB()
file_id = db.retrieve_file_ids(["PROSPECT_DOWN_19"])[0]
path = db.get_file_paths_from_ids([file_id])[0]
spectrum = retrieve_data(path, db)
plot_multi_spectra({"PROSPECT_DOWN_19": spectrum}, n="prospect_down_19")
```

The database wrapper uses `HFIRBG_CALDB`; file lookup honors `HFIRBGDATA`.

## Database-writing operations

The following operations are for maintainers constructing a new database, not
for normal use of the canonical release:

```python
from src.database.HFIRBG_DB import HFIRBG_DB

db = HFIRBG_DB()
db.sync_files()              # scans HFIRBGDATA and adds file records
db.sync_db()                 # imports project metadata
db.set_calibration_groups()  # creates calibration assignments
```

Calibration helpers can also write fitted coefficients back to the selected
database. Preserve the released database and use a working copy for these
operations.

## Coordinate conventions

Cart and detector orientation use the two-dimensional room coordinates `x` and
`z`, in inches. `(Rx, Rz)` and `(Lx, Lz)` record the cart's right and left
corners. The database `angle` is detector tilt; the cart azimuth is derived from
the corner vector. `src/database/CartScanFiles.py` contains the historical
position conversion used by cart-scan analyses.

The interactive public browser mirrors the historical azimuth calculation:
downward-facing detectors are marked with a star, while tilted detectors use
an arrow with `dz = L cos(phi)` and `dx = -L sin(phi)` from the same released
cart-corner geometry. Tilt remains available in map hover text; arrow length
is only a visual cue and does not encode tilt magnitude. The browser does not
require the legacy database wrapper.
