# Reproducing the public HFIR gamma analyses

This guide covers the public-data analyses most directly associated with
*Gamma Backgrounds for Experiments at the High Flux Isotope Reactor*
([arXiv:2607.05834](https://arxiv.org/abs/2607.05834)).

## One-command setup

On Linux x86-64, from the repository root:

```bash
./scripts/setup_analysis.sh
source .env
```

The setup downloads and verifies the calibrated spectrum/database bundle,
creates `.venv`, installs the analysis packages and PyROOT, downloads the
official arXiv v1 source/ancillary results, removes the legacy absolute path
from the extracted SQLite database, and runs import/database checks. Downloads,
the environment, and generated outputs are ignored by Git.

The full setup needs roughly 2 GB of disk space. The PyPI ROOT package is
currently an alpha Linux x86-64 distribution. On macOS the same setup command
downloads/configures the data and then directs you to use conda. When a
production ROOT build is preferred, conda can also be used on Linux:

```bash
./scripts/setup_analysis.sh
conda env create -f environment.yml
conda activate hfir-bg-analysis
source .env
python scripts/check_public_data_setup.py --sanitize-database-path
```

## Generate the requested products

```bash
.venv/bin/python scripts/public_analysis.py all
```

Results are written to `analysis/public_analysis/`:

- `unfolded_key_locations.{png,pdf}` compares the isotropic and front-face
  response-model results for MIF reactor-on, MIF reactor-off, and Shield Center.
- `unfolded_key_locations_manifest.csv` records the exact official ancillary
  CSV used for every curve.
- `shield_configuration_spectra.{png,pdf,csv}` reproduces the Figure 14
  30–60 keV energy-spectrum comparison from the public raw spectra/database.
- `shield_configuration_rates_30_60_keV.csv` exports reactor-on, reactor-off,
  and reactor-only rates with statistical errors.

Either analysis can be generated independently:

```bash
.venv/bin/python scripts/public_analysis.py unfolded
.venv/bin/python scripts/public_analysis.py shields
```

## Unfolded gamma flux spectra

The public calibrated text spectra are detector counts, not unfolded incident
flux. Rebuilding the unfolds requires the Geant4 response matrices and the
P2x/GeCollimatorUnfolder executable; those large simulation products are not in
the public spectrum bundle. The paper therefore publishes the authoritative
unfolded CSVs as official arXiv ancillary files. The setup script stores them at:

```text
data/arxiv_2607.05834/anc/
```

The requested inputs are:

| Location | Isotropic result | Front-face result |
|---|---|---|
| MIF (Rx On) | `unfolded_spectrum_isotropic_01_...csv` | `unfolded_spectrum_front_01_...csv` |
| MIF (Rx Off) | `unfolded_spectrum_isotropic_02_...csv` | `unfolded_spectrum_front_02_...csv` |
| Shield Center | `unfolded_spectrum_isotropic_03_...csv` | `unfolded_spectrum_front_03_...csv` |

The two response cases are limiting angular assumptions. The band between them
is a response-model bracket, not a statistical uncertainty interval.

## Figure 14 shield configurations

Figure 14 compares reactor-on high-gain spectra from 30–60 keV for:

1. Russian Doll baseline, without added water or floor lead.
2. Seven water-brick layers (60.5 water bricks).
3. Six water-brick layers plus a tiled lead layer beneath the cart.
4. Seven water-brick layers plus the floor lead (68 water bricks).

The script selects the corresponding database shield IDs and acquisition modes,
combines qualifying reactor-on/off runs, applies the canonical calibration,
normalizes by live time and energy-bin width, and exports mHz/kg/keV. The HPGe
active mass is approximately 1 kg, matching the paper normalization.

## Supplemental README and complete paper products

The repository copy is [UNFOLDING_RESULTS_README.md](UNFOLDING_RESULTS_README.md).
The setup also downloads the official arXiv copy alongside all scenario CSVs,
per-location plots, metadata, validation plots, and the paper source. The
official arXiv ancillary list is available from the
[paper page](https://arxiv.org/abs/2607.05834).
