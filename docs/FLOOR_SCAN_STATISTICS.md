# Figure 7 floor-scan point statistics

This workflow answers the supplemental-data question: how many counts are in a
typical individual floor-scan acquisition, and what energy binning is supported
by those statistics?

The repository includes a generated
[review package](../reports/floor_scan_statistics/README.md) from the current
public release. Regenerate it rather than hand-editing its tables or plots.

## Generate the review package

After either public setup mode:

```bash
./scripts/setup_analysis.sh --browser-only
.venv/bin/python scripts/analyze_floor_scan_statistics.py
```

The ROOT-free command writes to `analysis/floor_scan_statistics/`:

- `README.md` and `summary.json`: concise numerical conclusions;
- `floor_scan_points.csv`: acquisition, location, live time, counts, rate,
  calibration, and routine/extended flags for every original scan file;
- `floor_scan_spectra.csv.gz`: all 122 individual scan histograms with the
  suggested piecewise binning and Poisson errors, keyed to the point manifest;
- `floor_scan_binning_summary.csv`: occupancy and counting-statistics metrics
  for 1–200 keV candidate bin widths in four energy bands;
- `representative_point_spectrum.csv`: a typical point with the suggested
  piecewise binning;
- `floor_scan_point_statistics.{png,pdf}`: map and distributions;
- `representative_point_adaptive_binning.{png,pdf}`: the typical spectrum; and
- `floor_scan_spectrum_quantiles.{png,pdf}`: five spectra spanning the scan's
  total-count distribution.

Use a different destination with `--output-dir`. Explicit `--db` and
`--data-root` options follow the same precedence as the public browser.

To independently export the native calibrated channels for the selected
representative point identified by the current report:

```bash
.venv/bin/python scripts/export_public_data.py files \
  --run-id 102 --output analysis/floor_scan_statistics/representative_file.csv

.venv/bin/python scripts/export_public_data.py spectra \
  --file-id 1729 --normalization counts/s/keV --emin 50 --emax 11400 \
  --output analysis/floor_scan_statistics/representative_native_channels.csv
```

In the web browser, searching for `position_scan_3_HB4_DOWN_2` exposes the same
run and spectrum interactively.

## Exact selection

The canonical database contains later monitoring runs at some of the same
coordinates. A generic filter for down-facing collimator data would therefore
not reproduce the Figure 7 population.

The script selects the original `position_scan_3` through `position_scan_8`
campaign runs, the `collimator30` configuration, detector angle zero, non-track
coordinates, valid timestamps, no lead-test descriptions, and calibrations
covering 11.4 MeV. It retains every matching acquisition in the inventory.

For the word “average,” the statistics summary uses routine acquisitions with
100–400 s live time. This matches the paper's description of approximately 100
points measured for about four minutes while separating longer follow-up and
overnight exposures. The representative spectrum minimizes robust scaled
distance from the routine medians of live time, total counts, and count rate.

## Interpreting the binning study

The command evaluates each candidate width independently in 50–1000,
1000–3000, 3000–7000, and 7000–11400 keV. For each point and band it records
mean counts per bin, nonzero-bin fraction, and fractions of bins with at least
10 or 25 counts, then reports the median across routine points.

The suggested piecewise display widths are 2, 10, 50, and 200 keV in those four
bands. These are convenience products, not a replacement for native calibrated
channel counts. The highest-energy region remains sparse and should always be
shown with Poisson uncertainties.
