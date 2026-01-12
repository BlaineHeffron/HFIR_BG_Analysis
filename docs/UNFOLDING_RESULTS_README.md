# Unfolding Results Data Format

This document describes the format and organization of the detector response unfolding results.

## Directory Structure

```
unfolding_results/
├── comparison_01.png              # Measured vs reconstructed spectrum plots
├── comparison_02.png
├── ...
├── unfolded_spectrum_01.csv       # Unfolded gamma flux spectra
├── unfolded_spectrum_02.csv
├── ...
├── measured_spectrum_01.csv       # Calibrated measured detector spectra
├── measured_spectrum_02.csv
├── ...
├── measurement_metadata.csv       # Measurement position metadata
└── measurement_locations.png      # Diagram showing all measurement locations
```

## CSV File Format: Unfolded Spectra

Each `unfolded_spectrum_##.csv` file contains the unfolded gamma flux spectrum for measurement number ##.

### Column Descriptions

| Column Name | Units | Description |
|------------|-------|-------------|
| `Energy_keV` | keV | Gamma-ray energy bin center |
| `Flux_Hz_per_cm2_per_keV` | Hz/cm²/keV | Unfolded incident gamma flux at this energy |

### Data Format

- **File format**: CSV (comma-separated values)
- **Header**: First row contains column names
- **Energy range**: Typically 40-11400 keV in 1 keV bin widths
- **Flux normalization**: Flux values represent incident gamma flux per unit area assuming isotropic emission from the detector's perspective
- **Scientific notation**: Flux values are written in exponential format (e.g., `1.234567e-05`)

## CSV File Format: Measured Spectra

Each `measured_spectrum_##.csv` file contains the calibrated measured detector spectrum for measurement number ##.

### Column Descriptions

| Column Name | Units | Description |
|------------|-------|-------------|
| `Energy_keV` | keV | Calibrated gamma-ray energy bin center |
| `Rate` | Hz/keV | Measured detector counts in this energy bin |

### Data Format

- **File format**: CSV (comma-separated values)
- **Header**: First row contains column names
- **Energy range**: Typically 40-11400 keV in 1 keV bin widths
- **Calibration**: Energy scale calibrated using known gamma lines
- **Scientific notation**: Count values are written in exponential format (e.g., `1.234567e+03`)

## Measurement Numbering Scheme

Measurements are numbered sequentially (01, 02, 03, ...) in the order they were processed. The numbering is arbitrary and does not indicate any temporal or spatial ordering. 

### Metadata File: `measurement_metadata.csv`

This file provides the mapping between measurement numbers and physical locations.

| Column Name | Units | Description |
|------------|-------|-------------|
| `number` | - | Sequential measurement identifier |
| `filename` | - | Original data file name (without extension) |
| `z_pos` | inches | Detector position along z-axis (HFIR coordinate system) |
| `x_pos` | inches | Detector position along x-axis (HFIR coordinate system) |
| `angle` | degrees | Detector tilt angle (0° = pointing straight down) |
| `phi_deg` | degrees | Azimuthal angle of detector pointing direction |

### HFIR Coordinate System

- **Origin**: Experiment hall NW corner
- **x-axis**: Positive direction is away from the reactor core (North to South)
- **z-axis**: Increases from West to East (i.e. left to right)
- **Angles**: 
  - `angle = 0°`: Detector pointing straight down (into the page / vertical)
  - `angle > 0°`: Detector tilted from vertical
  - `phi_deg`: Direction of tilt in x-z plane (0° = pointing in +z direction, 90° = pointing in -x direction)

## Location Diagram

The file `measurement_locations.png` shows a top-down view of the HFIR reactor area with:

- **Numbered circles**: Each measurement location, with the number corresponding to the data files
- **Arrows**: For measurements where the detector was not pointing straight down, an arrow shows the pointing direction
- **Reference features**: Reactor core, PROSPECT detector, beam lines (HB3, HB4), lead shields, pool walls

## Unfolding Method

The flux spectra were obtained using the Richardson-Lucy deconvolution algorithm applied to the measured detector response. The detector response matrix was generated using Geant4 simulations with:

- Monochromatic gamma rays from 40-11400 keV in 1 keV steps
- Isotropic flux of 1 Hz/cm²/keV at the detector surface
- Full detector geometry including dead layers
- Energy-dependent resolution: σ(E) = A + B×E, where A = 0.98 keV, B = 0.00018

## Usage Examples

### Loading Data in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load unfolded spectrum
df_unfolded = pd.read_csv('unfolded_spectrum_01.csv')

# Load measured spectrum
df_measured = pd.read_csv('measured_spectrum_01.csv')

# Plot both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot unfolded spectrum
ax1.step(df_unfolded['Energy_keV'], df_unfolded['Flux_Hz_per_cm2_per_keV'], 
         where='mid', color='blue')
ax1.set_xlabel('Energy [keV]')
ax1.set_ylabel('Flux [Hz/cm²/keV]')
ax1.set_yscale('log')
ax1.set_title('Unfolded Gamma Flux')
ax1.grid(True, alpha=0.3)

# Plot measured spectrum
ax2.step(df_measured['Energy_keV'], df_measured['Counts'], 
         where='mid', color='red')
ax2.set_xlabel('Energy [keV]')
ax2.set_ylabel('Counts')
ax2.set_yscale('log')
ax2.set_title('Measured Detector Spectrum')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Loading Metadata

```python
import pandas as pd

# Load measurement locations
metadata = pd.read_csv('measurement_metadata.csv')

# Find measurements at specific locations
mif_measurements = metadata[metadata['filename'].str.contains('MIF')]
print(mif_measurements)
```

## Contact

For questions about this data, contact:
Blaine Heffron
baheffron@gmail.com