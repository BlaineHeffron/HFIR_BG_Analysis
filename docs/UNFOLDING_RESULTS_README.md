# Unfolding results data format

This is the repository copy of the supplemental README accompanying
*Gamma Backgrounds for Experiments at the High Flux Isotope Reactor*
([arXiv:2607.05834](https://arxiv.org/abs/2607.05834)). The official ancillary
copy and all files it describes can be downloaded with
`scripts/setup_analysis.sh` and are placed in
`data/arxiv_2607.05834/anc/`.

## Response scenarios

The paper-facing unfolding outputs contain two detector-response scenarios:

- `isotropic`: uniform flux over the outer surface of the
  collimator-detector system.
- `front`: uniform directional flux incident on the collimator front face.

Together these are limiting cases for the incident gamma-flux normalization.
The region between them is a response-model bracket, not a statistical
uncertainty interval or a full reconstruction of the room's angular field.

## Key files

Scenario-tagged unfolded spectra:

- `unfolded_spectrum_isotropic_##_<LOCATION>.csv`
- `unfolded_spectrum_front_##_<LOCATION>.csv`

Measured spectra shared by both scenarios:

- `measured_spectrum_##_<LOCATION>.csv`

Summary products:

- `all_unfolded_spectra_{isotropic,front}.{png,pdf}`
- `measured_vs_unfolded_comparison_{isotropic,front}.{png,pdf}`
- `unfolded_spectrum_bounds.{png,pdf}`
- `measurement_locations.png`

Per-location unfolded plots are supplied for MIF Rx On, MIF Rx Off, Shield
Center, HB4, PROSPECT East 1, and PROSPECT East 2 under both scenarios.

Metadata and simulation diagnostics:

- `measurement_metadata.csv`
- `measurement_case_files.csv`
- `unfolding_scenarios.csv`
- `migration_matrix_stats_comparison.{csv,pdf,png}`
- `migration_matrix_support_comparison.{txt,pdf,png}`
- `raw_sim_efficiency_comparison.{csv,pdf,png}`

## Measurement locations

| Number | Filename | Alias |
|---|---|---|
| 01 | `MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN` | MIF (Rx On) |
| 02 | `MIF_BOX_AT_REACTOR_RXOFF` | MIF (Rx Off) |
| 03 | `CYCLE461_DOWN_FACING_OVERNIGHT` | Shield Center |
| 04 | `HB4_DOWN_OVERNIGHT_1` | HB4 |
| 05 | `EAST_FACE_18` | PROSPECT East 1 |
| 06 | `EAST_FACE_1` | PROSPECT East 2 |

## CSV columns

Unfolded spectra contain:

| Column | Units | Description |
|---|---|---|
| `Energy_keV` | keV | Gamma-ray energy-bin center |
| `Flux_Hz_per_mm2_per_keV` | Hz/mm²/keV | Unfolded incident gamma flux |

Measured spectra contain:

| Column | Units | Description |
|---|---|---|
| `Energy_keV` | keV | Calibrated energy-bin center |
| `Rate_Hz_per_keV` | Hz/keV | Measured detector count rate |

The files cover 40–12000 keV at 1 keV spacing. Values use scientific notation.

## Response-matrix support

`unfolding_scenarios.csv` records the migration matrix used for each scenario
and its direct simulated-energy support. Both current matrices use 166 directly
simulated energies spanning 40–12000 keV. Detailed error/runtime comparisons
are in `migration_matrix_stats_comparison.csv`.

## Python example

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

paper_data = Path("data/arxiv_2607.05834/anc")
isotropic = pd.read_csv(
    paper_data
    / "unfolded_spectrum_isotropic_01_"
      "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.csv"
)
front = pd.read_csv(
    paper_data
    / "unfolded_spectrum_front_01_"
      "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN.csv"
)

for label, frame in (("Isotropic", isotropic), ("Front face", front)):
    plt.step(
        frame["Energy_keV"],
        frame["Flux_Hz_per_mm2_per_keV"],
        where="mid",
        label=label,
    )
plt.yscale("log")
plt.xlabel("Energy [keV]")
plt.ylabel("Flux [Hz/mm²/keV]")
plt.legend()
plt.show()
```

For ready-made MIF Rx On/Off and Shield Center plots, run:

```bash
.venv/bin/python scripts/public_analysis.py unfolded
```
