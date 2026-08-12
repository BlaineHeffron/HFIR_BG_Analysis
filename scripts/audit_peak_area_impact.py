#!/usr/bin/env python3
"""Replay public peak-area products with legacy and corrected estimands.

The input bundle is opened read-only.  Outputs are written only to an empty
caller-supplied directory.  The paper values below are transcribed from arXiv
2607.05834v1, tables ``tab:rd_lines`` and ``table:peak_ratio_compare``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy.stats import chi2

from scripts.rd_peak_fitter import EXPECTED_PEAKS as RD_PEAKS
from src.analysis.Spectrum import (
    PEAK_AREA_LEGACY_DENSITY,
    PEAK_AREA_NET_COUNTS,
    MultiPeakFit,
    SpectrumData,
    SpectrumFitter,
)
from src.public_data.browser import load_spectrum, query_file_metadata
from src.public_data.parent_group_fit import (
    AffineCalibrationConstraint,
    FitWindow,
    ParentGroupSpec,
    PeakComponent,
    fit_parent_group,
)
from src.utilities.util import get_areas, unc_ratio


PUBLIC_V1_1_DB_SHA256 = (
    "c78bc8fa6ef7dbe1a8ea5d0189e69eb555c8a488fd582ff04b965a08aa1985e9"
)
RD_RUN_NAME = "Cycle493_RD_low_gain"
MIF_FILE_ID = 1042
MIF_PARENTS_FIT_ORDER = (
    11386.5,
    9718.79,
    8998.63,
    7724.034,
    7693.398,
    7645.58,
    7631.18,
    6809.61,
)

PAPER_RD_VALUES = {
    238.6: (0.346, 0.049),
    242.0: (0.272, 0.040),
    295.2: (0.455, 0.793),
    351.9: (0.758, 0.547),
    478.0: (None, None),
    558.5: (1.000, 0.193),
    609.3: (0.605, 0.143),
    651.3: (0.173, 0.141),
    707.4: (0.018, 0.110),
    725.0: (0.060, 0.087),
    768.4: (0.047, 0.101),
    805.9: (0.106, 0.080),
    1120.3: (0.116, 0.058),
    1209.7: (0.050, 0.044),
    1238.1: (0.035, 0.056),
    1281.0: (0.024, 0.053),
    1293.6: (0.565, 0.087),
    1364.3: (0.060, 0.038),
    1377.7: (0.037, 0.042),
    1399.6: (0.035, 0.040),
    1489.56: (0.014, 0.048),
    1660.368: (0.032, 0.029),
    1764.5: (0.091, 0.026),
    2204.2: (0.021, 0.023),
    2223.0: (0.033, 0.020),
    2398.6: (0.008, 0.021),
    2455.8: (0.017, 0.023),
    2550.1: (0.003, 0.023),
    2614.533: (0.024, 0.016),
    2660.1: (0.025, 0.012),
    2767.5: (0.011, 0.015),
    5433.1: (0.002, 0.001),
    5824.6: (0.005, 0.001),
    7367.9: (0.003, 0.000),
    7916.3: (0.001, 0.000),
}

PAPER_MIF_VALUES = {
    6809.61: (1.02, 1.93, 2.13, 6.83, 2.08, 6.88),
    7631.18: (0.90, 1.44, 1.83, 7.09, 2.04, 8.30),
    7645.58: (0.86, 1.56, 1.53, 6.45, 1.78, 7.90),
    7693.398: (0.99, 2.40, 1.52, 7.28, 1.55, 7.98),
    7724.034: (0.87, 0.27, 1.85, 1.49, 2.12, 1.79),
    8998.63: (0.67, 0.28, 1.36, 1.41, 2.04, 2.20),
    9718.79: (0.72, 0.35, 1.25, 1.96, 1.74, 2.77),
    11386.5: (0.58, 0.07, 1.28, 0.50, 2.19, 0.90),
}

RATIO_COMPONENTS = (
    ("fep/sep", 0, 1),
    ("fep/dep", 0, 2),
    ("sep/dep", 1, 2),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_number(value: float | None) -> str:
    return "" if value is None else format(float(value), ".12g")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty audit product: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _format_number(value) if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def _fit(spec: SpectrumData, peaks: list[float] | tuple[float, ...]):
    fitter = SpectrumFitter(list(peaks))
    diagnostics = io.StringIO()
    with redirect_stdout(diagnostics), redirect_stderr(diagnostics):
        fitter.fit_peaks(spec)
    return fitter.fit_values, diagnostics.getvalue()


def _areas(fits, live_time: float, mode: str):
    values: dict[str, float] = {}
    errors: dict[str, float] = {}
    get_areas(fits, values, errors, lt=live_time, area_mode=mode)
    return values, errors


def _fitted_full_line_area(
    parameters: np.ndarray,
    covariance: np.ndarray,
    energy_bin_width_keV: float,
    peak_index: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Integrate one historical fitted signal and propagate its fit covariance.

    The historical model ordinates are counts per detector channel evaluated
    on an energy axis.  Dividing the analytic energy integral by the calibrated
    channel width therefore gives full-line detector counts.  Background
    parameters and the centroid have zero derivative for this full-line
    estimand.
    """

    values = np.asarray(parameters, dtype=np.float64)
    fit_covariance = np.asarray(covariance, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("fit parameters must be one-dimensional")
    if fit_covariance.shape != (values.size, values.size):
        raise ValueError("fit covariance must match the parameter vector")
    if not np.isfinite(values).all() or not np.isfinite(fit_covariance).all():
        raise ValueError("fit parameters and covariance must be finite")
    if not np.isfinite(energy_bin_width_keV) or energy_bin_width_keV <= 0:
        raise ValueError("energy-bin width must be finite and positive")
    start = 5 * int(peak_index)
    if peak_index < 0 or start + 5 > values.size:
        raise IndexError("peak index is outside the fitted parameter vector")

    height, tail_fraction, _, sigma, tail_scale = values[start : start + 5]
    if sigma <= 0 or tail_scale <= 0:
        raise ValueError("fitted sigma and tail scale must be positive")
    root_two_pi = np.sqrt(2.0 * np.pi)
    tail_attenuation = np.exp(-0.5 * (sigma / tail_scale) ** 2)
    integral_per_height = (
        root_two_pi * (1.0 - tail_fraction) * sigma
        + 2.0 * tail_fraction * tail_scale * tail_attenuation
    )
    area_counts = height * integral_per_height / energy_bin_width_keV

    gradient = np.zeros(values.size, dtype=np.float64)
    gradient[start] = integral_per_height / energy_bin_width_keV
    gradient[start + 1] = height * (
        -root_two_pi * sigma + 2.0 * tail_scale * tail_attenuation
    ) / energy_bin_width_keV
    gradient[start + 3] = height * (
        root_two_pi * (1.0 - tail_fraction)
        - 2.0 * tail_fraction * sigma * tail_attenuation / tail_scale
    ) / energy_bin_width_keV
    gradient[start + 4] = height * (
        2.0
        * tail_fraction
        * tail_attenuation
        * (1.0 + (sigma / tail_scale) ** 2)
    ) / energy_bin_width_keV
    variance = float(gradient @ fit_covariance @ gradient)
    if variance < -1e-10 * max(area_counts * area_counts, 1.0):
        raise ValueError("fit covariance gives a negative full-line variance")
    return float(area_counts), float(np.sqrt(max(variance, 0.0))), gradient


def _fitted_full_line_records(fits, energy_bin_width_keV: float, live_time: float):
    records: dict[str, dict[str, object]] = {}
    for key, fit in fits.items():
        energies = key.split(",") if isinstance(key, str) else [key]
        for peak_index, energy in enumerate(energies):
            area, uncertainty, gradient = _fitted_full_line_area(
                fit.parameters,
                fit.cov,
                energy_bin_width_keV,
                peak_index,
            )
            records[f"{float(energy):.2f}"] = {
                "value": area / live_time,
                "uncertainty": uncertainty / live_time,
                "gradient": gradient / live_time,
                "fit": fit,
            }
    return records


def _covariance_ratio(records, numerator: str, denominator: str):
    if numerator == denominator:
        return 1.0, 0.0
    numerator_record = records[numerator]
    denominator_record = records[denominator]
    numerator_value = float(numerator_record["value"])
    denominator_value = float(denominator_record["value"])
    numerator_uncertainty = float(numerator_record["uncertainty"])
    denominator_uncertainty = float(denominator_record["uncertainty"])
    value = numerator_value / denominator_value
    variance = (
        numerator_uncertainty**2 / denominator_value**2
        + numerator_value**2
        * denominator_uncertainty**2
        / denominator_value**4
    )
    if numerator_record["fit"] is denominator_record["fit"]:
        fit = numerator_record["fit"]
        cross_covariance = float(
            numerator_record["gradient"]
            @ fit.cov
            @ denominator_record["gradient"]
        )
        variance -= 2.0 * numerator_value * cross_covariance / denominator_value**3
    return float(value), float(np.sqrt(max(variance, 0.0)))


def _local_covariance_status(value: float, uncertainty: float, *, exact: bool = False):
    if exact:
        return "exact self-ratio"
    if uncertainty >= abs(value):
        return "unstable: linearized one-sigma covariance interval reaches zero"
    return (
        "finite local covariance sensitivity; absolute count-model fit still rejected"
    )


def _postfit_poisson_diagnostic(fits) -> dict[str, object]:
    """Apply one common absolute count-model diagnostic to recovered fits."""

    deviance = 0.0
    bin_count = 0
    parameter_count = 0
    for fit in fits.values():
        observed = np.asarray(fit.ys, dtype=np.float64)
        expected = np.asarray(fit.get_y(), dtype=np.float64)
        if np.any(expected <= 0) or not np.isfinite(expected).all():
            return {
                "valid": False,
                "reason": "fitted expected counts are nonpositive or nonfinite",
            }
        positive = observed > 0
        terms = expected - observed
        terms[positive] += observed[positive] * np.log(
            observed[positive] / expected[positive]
        )
        deviance += 2.0 * float(np.sum(terms))
        bin_count += observed.size
        parameter_count += len(fit.parameters)
    degrees_of_freedom = bin_count - parameter_count
    return {
        "valid": True,
        "poisson_deviance": deviance,
        "raw_bin_count": bin_count,
        "free_parameter_count": parameter_count,
        "descriptive_degrees_of_freedom": degrees_of_freedom,
        "chi_square_reference_p_value": float(chi2.sf(deviance, degrees_of_freedom)),
        "comparison_limit": (
            "same diagnostic family as phase 2, but the recovered historical "
            "and phase-2 fits use different window sets; do not rank models "
            "from their deviances or per-degree-of-freedom values"
        ),
    }


def _fit_line_metadata(fits) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for key, fit in fits.items():
        if isinstance(key, str):
            energies = [float(value) for value in key.split(",")]
            sigmas = fit.sigmas
        else:
            energies = [float(key)]
            sigmas = [fit.sigma]
        for energy, sigma in zip(energies, sigmas):
            result[f"{energy:.2f}"] = {
                "fit_type": type(fit).__name__,
                "fit_group": str(key),
                "sigma_keV": float(sigma),
            }
    return result


def _ratio(values, errors, numerator: str, denominator: str):
    value = values[numerator] / values[denominator]
    error = unc_ratio(
        values[numerator],
        values[denominator],
        errors[numerator],
        errors[denominator],
    )
    return float(value), float(error)


def _public_file_path(data_root: Path, file_name: str) -> Path:
    name = file_name if file_name.lower().endswith(".txt") else f"{file_name}.txt"
    path = data_root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_rd_combination(db_path: Path, data_root: Path):
    metadata = query_file_metadata(db_path)
    selected = metadata[metadata["run_name"] == RD_RUN_NAME].sort_values(
        ["start_time", "file_id"]
    )
    if selected.empty:
        raise RuntimeError(f"public bundle has no {RD_RUN_NAME!r} files")
    spectra = [
        load_spectrum(int(file_id), db_path, data_root)
        for file_id in selected["file_id"]
    ]
    reference = spectra[0]
    for spectrum in spectra[1:]:
        if (
            spectrum.counts.shape != reference.counts.shape
            or spectrum.calibration_A0 != reference.calibration_A0
            or spectrum.calibration_A1 != reference.calibration_A1
        ):
            raise RuntimeError("RD spectra do not share a count grid and calibration")
    combined = SpectrumData(
        np.sum([spectrum.counts for spectrum in spectra], axis=0),
        "",
        sum(spectrum.live_time for spectrum in spectra),
        reference.calibration_A0,
        reference.calibration_A1,
        RD_RUN_NAME,
    )
    return combined, spectra


def _rd_rows(spec: SpectrumData):
    fits, diagnostics = _fit(spec, tuple(RD_PEAKS))
    legacy, legacy_error = _areas(fits, spec.live, PEAK_AREA_LEGACY_DENSITY)
    unit_corrected, unit_corrected_error = _areas(
        fits, spec.live, PEAK_AREA_NET_COUNTS
    )
    fitted_full_line = _fitted_full_line_records(fits, spec.A1, spec.live)
    metadata = _fit_line_metadata(fits)
    reference = "558.50"
    rows = []
    for energy in sorted(RD_PEAKS):
        key = f"{energy:.2f}"
        paper_value, paper_error = PAPER_RD_VALUES[energy]
        legacy_ratio, legacy_ratio_error = _ratio(
            legacy, legacy_error, key, reference
        )
        unit_corrected_ratio, unit_corrected_ratio_error = _ratio(
            unit_corrected, unit_corrected_error, key, reference
        )
        full_line_ratio, full_line_ratio_error = _covariance_ratio(
            fitted_full_line, key, reference
        )
        if key == reference:
            unit_corrected_ratio_error = 0.0
        display_match = "not compared; paper reports '-'"
        if paper_value is not None:
            display_match = str(
                f"{legacy_ratio:.3f}" == f"{paper_value:.3f}"
                and f"{legacy_ratio_error:.3f}" == f"{paper_error:.3f}"
            ).lower()
            if display_match != "true":
                raise RuntimeError(
                    f"legacy RD replay does not match the paper at {energy:g} keV"
                )
        rows.append(
            {
                "energy_keV": energy,
                "fit_type": metadata[key]["fit_type"],
                "fit_group": metadata[key]["fit_group"],
                "fitted_sigma_keV": metadata[key]["sigma_keV"],
                "paper_ratio": paper_value,
                "paper_ratio_uncertainty": paper_error,
                "paper_display_match": display_match,
                "legacy_rate_counts_per_s_per_keV": legacy[key],
                "legacy_rate_uncertainty_counts_per_s_per_keV": legacy_error[key],
                "legacy_ratio": legacy_ratio,
                "legacy_ratio_uncertainty_historical": legacy_ratio_error,
                "legacy_fit_unit_corrected_rate_counts_per_s": unit_corrected[key],
                "legacy_fit_unit_corrected_rate_uncertainty_historical": (
                    unit_corrected_error[key]
                ),
                "legacy_fit_unit_corrected_ratio": unit_corrected_ratio,
                "legacy_fit_unit_corrected_ratio_uncertainty_historical": (
                    unit_corrected_ratio_error
                ),
                "historical_fit_full_line_rate_counts_per_s": fitted_full_line[key][
                    "value"
                ],
                "historical_fit_full_line_rate_covariance_uncertainty_counts_per_s": (
                    fitted_full_line[key]["uncertainty"]
                ),
                "historical_fit_full_line_ratio": full_line_ratio,
                "historical_fit_full_line_ratio_covariance_uncertainty": (
                    full_line_ratio_error
                ),
                "historical_fit_full_line_comparison_status": (
                    _local_covariance_status(
                        full_line_ratio,
                        full_line_ratio_error,
                        exact=key == reference,
                    )
                ),
                "unit_corrected_over_density_ratio": (
                    unit_corrected_ratio / legacy_ratio
                ),
                "result_semantics": (
                    "legacy columns: paper-exact reproduction; "
                    "unit-corrected columns: new calculation"
                    if paper_value is not None
                    else "paper omission; unstable fit retained for audit"
                ),
            }
        )
    return rows, diagnostics, _postfit_poisson_diagnostic(fits)


def _mif_window_rows(spec: SpectrumData):
    all_peaks = [
        energy
        for parent in MIF_PARENTS_FIT_ORDER
        for energy in (parent, parent - 511.0, parent - 1022.0)
    ]
    fits, diagnostics = _fit(spec, all_peaks)
    legacy, legacy_error = _areas(fits, spec.live, PEAK_AREA_LEGACY_DENSITY)
    unit_corrected, unit_corrected_error = _areas(
        fits, spec.live, PEAK_AREA_NET_COUNTS
    )
    fitted_full_line = _fitted_full_line_records(fits, spec.A1, spec.live)
    metadata = _fit_line_metadata(fits)
    rows = []
    for parent in sorted(MIF_PARENTS_FIT_ORDER):
        keys = [f"{energy:.2f}" for energy in (parent, parent - 511, parent - 1022)]
        paper = PAPER_MIF_VALUES[parent]
        for ratio_index, (label, numerator_index, denominator_index) in enumerate(
            RATIO_COMPONENTS
        ):
            numerator = keys[numerator_index]
            denominator = keys[denominator_index]
            legacy_ratio, legacy_ratio_error = _ratio(
                legacy, legacy_error, numerator, denominator
            )
            unit_corrected_ratio, unit_corrected_ratio_error = _ratio(
                unit_corrected, unit_corrected_error, numerator, denominator
            )
            full_line_ratio, full_line_ratio_error = _covariance_ratio(
                fitted_full_line, numerator, denominator
            )
            rows.append(
                {
                    "file_id": MIF_FILE_ID,
                    "live_time_s": spec.live,
                    "calibration_A0_keV": spec.A0,
                    "calibration_A1_keV_per_channel": spec.A1,
                    "parent_energy_keV": parent,
                    "ratio": label,
                    "numerator_fit_type": metadata[numerator]["fit_type"],
                    "denominator_fit_type": metadata[denominator]["fit_type"],
                    "numerator_sigma_keV": metadata[numerator]["sigma_keV"],
                    "denominator_sigma_keV": metadata[denominator]["sigma_keV"],
                    "paper_ratio": paper[2 * ratio_index],
                    "paper_ratio_uncertainty": paper[2 * ratio_index + 1],
                    "current_legacy_ratio": legacy_ratio,
                    "current_legacy_ratio_uncertainty_historical": legacy_ratio_error,
                    "legacy_fit_unit_corrected_ratio": unit_corrected_ratio,
                    "legacy_fit_unit_corrected_ratio_uncertainty_historical": (
                        unit_corrected_ratio_error
                    ),
                    "current_fit_full_line_ratio": full_line_ratio,
                    "current_fit_full_line_ratio_covariance_uncertainty": (
                        full_line_ratio_error
                    ),
                    "current_fit_full_line_comparison_status": (
                        _local_covariance_status(full_line_ratio, full_line_ratio_error)
                    ),
                    "unit_corrected_over_density_ratio": (
                        unit_corrected_ratio / legacy_ratio
                    ),
                    "result_semantics": (
                        "new calculation; exact legacy workflow unavailable"
                    ),
                    "fit_context": "current ordered 24-peak SpectrumFitter pass",
                }
            )
    return rows, diagnostics, _postfit_poisson_diagnostic(fits)


def _parent_group_spec(parent: float) -> ParentGroupSpec:
    roles = ("fep", "sep", "dep")
    energies = (parent, parent - 511.0, parent - 1022.0)
    half_window = 45.0 if parent == 7724.034 else 40.0
    windows = tuple(
        FitWindow(role, energy - half_window, energy + half_window)
        for role, energy in zip(roles, energies)
    )
    components = [
        PeakComponent(role, role, energy, role, 2.5, 18.0, 0.4, 10.0)
        for role, energy in zip(roles, energies)
    ]
    if parent == 7724.034:
        neighbor = 7693.398
        for role, energy in zip(
            roles, (neighbor, neighbor - 511.0, neighbor - 1022.0)
        ):
            components.append(
                PeakComponent(
                    f"neighbor_{role}",
                    "contaminant",
                    energy,
                    role,
                    2.5,
                    12.0,
                    0.4,
                    10.0,
                )
            )
    return ParentGroupSpec(str(parent), windows, tuple(components))


def _calibration_constraints(public_spectrum):
    # The release database has A0/A1 but no covariance.  Run both an explicit
    # nominal constraint and a ten-times-looser sensitivity variant.
    return {
        "tight": AffineCalibrationConstraint(
            public_spectrum.calibration_A0,
            public_spectrum.calibration_A1,
            np.diag([0.05**2, 1e-5**2]),
        ),
        "loose": AffineCalibrationConstraint(
            public_spectrum.calibration_A0,
            public_spectrum.calibration_A1,
            np.diag([0.5**2, 1e-4**2]),
        ),
    }


def _fit_parent_group_variants(public_spectrum, fit_spec):
    results = {
        name: fit_parent_group(public_spectrum, fit_spec, constraint)
        for name, constraint in _calibration_constraints(public_spectrum).items()
    }
    for name, result in results.items():
        if not result.success or not result.covariance_valid:
            raise RuntimeError(
                f"parent-group {name} fit failed for {fit_spec.name}"
            )
    return results


def _ratio_from_parent_group(result, numerator, denominator):
    areas = result.target_areas_counts
    covariance = result.target_area_covariance
    value = float(areas[numerator] / areas[denominator])
    gradient = np.zeros(3)
    gradient[numerator] = 1.0 / areas[denominator]
    gradient[denominator] = -areas[numerator] / areas[denominator] ** 2
    uncertainty = float(np.sqrt(gradient @ covariance @ gradient))
    return value, uncertainty


def _mif_parent_group_rows(public_spectrum, window_rows):
    window_lookup = {
        (row["parent_energy_keV"], row["ratio"]): row for row in window_rows
    }
    rows = []
    for parent in (11386.5, 9718.79, 8998.63, 7724.034):
        results = _fit_parent_group_variants(
            public_spectrum, _parent_group_spec(parent)
        )
        for label, numerator, denominator in RATIO_COMPONENTS:
            tight_value, tight_error = _ratio_from_parent_group(
                results["tight"], numerator, denominator
            )
            loose_value, loose_error = _ratio_from_parent_group(
                results["loose"], numerator, denominator
            )
            comparison = window_lookup[(parent, label)]
            rows.append(
                {
                    "file_id": MIF_FILE_ID,
                    "parent_energy_keV": parent,
                    "ratio": label,
                    "paper_ratio": comparison["paper_ratio"],
                    "legacy_fit_unit_corrected_ratio": comparison[
                        "legacy_fit_unit_corrected_ratio"
                    ],
                    "parent_group_ratio_tight_calibration": tight_value,
                    "parent_group_statistical_uncertainty_tight": tight_error,
                    "parent_group_ratio_loose_calibration": loose_value,
                    "parent_group_statistical_uncertainty_loose": loose_error,
                    "loose_minus_tight_ratio": loose_value - tight_value,
                    "poisson_deviance_per_dof_tight": (
                        results["tight"].poisson_deviance
                        / results["tight"].degrees_of_freedom
                    ),
                    "poisson_deviance_per_dof_loose": (
                        results["loose"].poisson_deviance
                        / results["loose"].degrees_of_freedom
                    ),
                    "covariance_valid": "true",
                    "result_semantics": "new calculation",
                }
            )
    return rows


def _rd_parent_group_spec():
    lines = (
        ("lead_7367", "fep", 7367.9, "lead", 2.0),
        ("reference_558", "sep", 558.5, "reference", 1.0),
        ("copper_7916", "dep", 7916.3, "copper", 2.0),
    )
    windows = tuple(
        FitWindow(window, energy - 20.0, energy + 20.0)
        for _, _, energy, window, _ in lines
    )
    components = tuple(
        PeakComponent(name, role, energy, window, sigma, 15.0, 0.3, 8.0)
        for name, role, energy, window, sigma in lines
    )
    return ParentGroupSpec("RD anchors 7367.9/558.5/7916.3", windows, components)


def _rd_parent_group_rows(public_spectrum, rd_rows):
    results = _fit_parent_group_variants(
        public_spectrum, _rd_parent_group_spec()
    )
    tight_ratio, tight_ratio_error = _ratio_from_parent_group(
        results["tight"], 0, 1
    )
    loose_ratio, loose_ratio_error = _ratio_from_parent_group(
        results["loose"], 0, 1
    )
    rd_lookup = {row["energy_keV"]: row for row in rd_rows}
    return [
        {
            "run_id": public_spectrum.run_id,
            "combined_live_time_s": public_spectrum.live_time,
            "ratio": "7367.9/558.5",
            "paper_legacy_density_ratio": rd_lookup[7367.9]["legacy_ratio"],
            "legacy_fit_unit_corrected_ratio": rd_lookup[7367.9][
                "legacy_fit_unit_corrected_ratio"
            ],
            "parent_group_ratio_tight_calibration": tight_ratio,
            "parent_group_statistical_uncertainty_tight": tight_ratio_error,
            "parent_group_ratio_loose_calibration": loose_ratio,
            "parent_group_statistical_uncertainty_loose": loose_ratio_error,
            "lead_area_counts_tight": results["tight"].target_areas_counts[0],
            "lead_area_statistical_uncertainty_tight": np.sqrt(
                results["tight"].target_area_covariance[0, 0]
            ),
            "reference_area_counts_tight": results[
                "tight"
            ].target_areas_counts[1],
            "reference_area_statistical_uncertainty_tight": np.sqrt(
                results["tight"].target_area_covariance[1, 1]
            ),
            "poisson_deviance_per_dof_tight": (
                results["tight"].poisson_deviance
                / results["tight"].degrees_of_freedom
            ),
            "poisson_deviance_per_dof_loose": (
                results["loose"].poisson_deviance
                / results["loose"].degrees_of_freedom
            ),
            "result_semantics": "new calculation",
        }
    ]


def _claim_rows(rd_rows, rd_parent_rows, mif_rows, mif_parent_rows):
    rd_lookup = {row["energy_keV"]: row for row in rd_rows}
    rd_parent = rd_parent_rows[0]
    mif_factors = [
        row["unit_corrected_over_density_ratio"] for row in mif_rows
    ]
    mif_relative_differences = [
        abs(row["parent_group_ratio_tight_calibration"] - row["paper_ratio"])
        / row["paper_ratio"]
        for row in mif_parent_rows
    ]
    return [
        {
            "paper_item": "Table 3 Russian-doll relative peak areas",
            "classification": "affected",
            "workflow_status": "paper-exact reproduction plus new calculation",
            "magnitude": (
                "7367.9/558.5: paper {:.6f}; independent Poisson {:.6f} "
                "({:.2f}x); legacy fit unit-corrected {:.6f} ({:.2f}x)"
            ).format(
                rd_lookup[7367.9]["legacy_ratio"],
                rd_parent["parent_group_ratio_tight_calibration"],
                rd_parent["parent_group_ratio_tight_calibration"]
                / rd_lookup[7367.9]["legacy_ratio"],
                rd_lookup[7367.9]["legacy_fit_unit_corrected_ratio"],
                rd_lookup[7367.9]["unit_corrected_over_density_ratio"],
            ),
            "reason": (
                "Published values are ratios of mean window density, not "
                "integrated net counts; 34 displayed rows replay exactly. "
                "The independent model confirms the direction but not the "
                "legacy fit's full 3.98x magnitude."
            ),
        },
        {
            "paper_item": "Table 3 reference-line uncertainty",
            "classification": "affected",
            "workflow_status": "paper-exact reproduction plus new calculation",
            "magnitude": "558.5 keV self-ratio uncertainty: 0.193 -> 0",
            "reason": (
                "A/A is exactly one; the legacy path propagated the same fit "
                "as two independent variables."
            ),
        },
        {
            "paper_item": "15% natural-Cd / 2% Cd-113 modeling ansatz",
            "classification": "not reproducible",
            "workflow_status": "unavailable legacy workflow",
            "magnitude": (
                "Independent data-side 7367.9/558.5 estimate changes by {:.2f}x; no "
                "revised material fraction can be computed."
            ).format(
                rd_parent["parent_group_ratio_tight_calibration"]
                / rd_lookup[7367.9]["legacy_ratio"]
            ),
            "reason": (
                "Required RD neutron/gamma simulation products are absent; "
                "a common density factor may partly cancel between data and simulation."
            ),
        },
        {
            "paper_item": "Table 8 measured-ratio central values",
            "classification": "affected; central values not overturned",
            "workflow_status": "new calculation; exact legacy workflow unavailable",
            "magnitude": (
                "Independent ratios differ from paper by {:.1f}% median and "
                "{:.1f}% maximum; legacy-fit unit factors span {:.2f}x-{:.2f}x"
            ).format(
                100.0 * float(np.median(mif_relative_differences)),
                100.0 * max(mif_relative_differences),
                min(mif_factors),
                max(mif_factors),
            ),
            "reason": (
                "The density factor nearly cancels for peaks separated by only "
                "511 or 1022 keV. Independent Poisson fits do not overturn the "
                "published central values; 10/12 are within 5.7%, while two "
                "Gaussian-only 9718.79 keV rows differ by 17-22% and remain "
                "well inside the published uncertainties."
            ),
        },
        {
            "paper_item": "Table 8 isotropic-vs-front-face agreement claim",
            "classification": "not reproducible",
            "workflow_status": "unavailable legacy workflow",
            "magnitude": "No defensible revised comparison",
            "reason": (
                "The simulation ROOT inputs are absent from public v1.1.0. "
                "The current legacy data fit is also peak-list/order dependent."
            ),
        },
        {
            "paper_item": "Peak-ratio uncertainty columns",
            "classification": "affected",
            "workflow_status": "documented legacy formula; new covariance sensitivity",
            "magnitude": "Legacy PeakFit uses sqrt(gross + 0.25 background^2)",
            "reason": (
                "The expression mixes counts and counts^2 and dominates "
                "background-rich peaks. It is preserved only for reproduction; "
                "parent-group sensitivity rows report covariance errors."
            ),
        },
        {
            "paper_item": "Table 5 efficiencies and fitted detector parameters",
            "classification": "unaffected",
            "workflow_status": "provenance audit",
            "magnitude": "No Python PeakFit.area dependency",
            "reason": (
                "Measured values come from the Rigel workbook; simulated "
                "efficiencies use the independent P2x C++ Gaussian integral "
                "sqrt(2*pi)*sigma*height."
            ),
        },
        {
            "paper_item": "Figure 28 collimator-effectiveness curve",
            "classification": "unaffected",
            "workflow_status": "provenance audit; external inputs required",
            "magnitude": "No Python PeakFit.area dependency",
            "reason": (
                "plot_collimator_effectiveness.py sums a fixed histogram window "
                "and normalizes by runtime and source area."
            ),
        },
        {
            "paper_item": "Figures 14 and 19 public workflows",
            "classification": "unaffected",
            "workflow_status": "paper-exact recalculation / published-ancillary replot",
            "magnitude": "No peak-area consumer in either call graph",
            "reason": "Measured-spectrum and unfolded-ancillary workflows are separate.",
        },
        {
            "paper_item": "Peak centroids, line IDs, and resolution fit",
            "classification": "unaffected",
            "workflow_status": "code-path audit",
            "magnitude": "Area return scaling does not alter fitted parameters",
            "reason": "Calibration and resolution consumers read centroids and sigmas.",
        },
        {
            "paper_item": "Dormant plot_sim_efficiencies.py extraction",
            "classification": "affected",
            "workflow_status": "unavailable legacy workflow; not a current paper product",
            "magnitude": "Historical output labeled Hz was actually Hz/keV",
            "reason": (
                "Its extraction function calls PeakFit.area across energies, but "
                "__main__ does not invoke it and required ROOT simulations are absent."
            ),
        },
    ]


def _input_record(spectrum, data_root: Path) -> dict[str, object]:
    path = _public_file_path(data_root, spectrum.file_name)
    return {
        "file_id": spectrum.file_id,
        "run_id": spectrum.run_id,
        "run_name": spectrum.run_name,
        "file_name": spectrum.file_name,
        "live_time_s": spectrum.live_time,
        "calibration_A0_keV": spectrum.calibration_A0,
        "calibration_A1_keV_per_channel": spectrum.calibration_A1,
        "spectrum_sha256": _sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        required=True,
        type=Path,
        help="path to the immutable HFIRBG_public_data_v1.1.0 directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="empty directory for CSV, JSON, and fit-diagnostic outputs",
    )
    args = parser.parse_args()

    bundle = args.bundle.expanduser().resolve()
    db_path = bundle / "HFIRBG.db"
    data_root = bundle / "spectra"
    if not db_path.is_file() or not data_root.is_dir():
        raise FileNotFoundError(
            "bundle must contain HFIRBG.db and the spectra directory"
        )
    db_sha256 = _sha256(db_path)
    if db_sha256 != PUBLIC_V1_1_DB_SHA256:
        raise RuntimeError(
            "database hash does not match the immutable public v1.1.0 input"
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError("output directory must be empty")

    rd_spec, rd_inputs = _load_rd_combination(db_path, data_root)
    rd_rows, rd_diagnostics, rd_postfit_diagnostic = _rd_rows(rd_spec)
    rd_public = replace(
        rd_inputs[0],
        file_id=0,
        file_name=f"{RD_RUN_NAME}_combined",
        live_time=rd_spec.live,
        counts=rd_spec.data.copy(),
        metadata={"input_file_ids": [item.file_id for item in rd_inputs]},
    )
    rd_parent_rows = _rd_parent_group_rows(rd_public, rd_rows)
    mif_public = load_spectrum(MIF_FILE_ID, db_path, data_root)
    mif_spec = SpectrumData(
        mif_public.counts.copy(),
        "",
        mif_public.live_time,
        mif_public.calibration_A0,
        mif_public.calibration_A1,
        mif_public.file_name,
    )
    mif_rows, mif_diagnostics, mif_postfit_diagnostic = _mif_window_rows(mif_spec)
    mif_parent_rows = _mif_parent_group_rows(mif_public, mif_rows)
    claim_rows = _claim_rows(
        rd_rows, rd_parent_rows, mif_rows, mif_parent_rows
    )

    _write_csv(output_dir / "rd_peak_area_impact.csv", rd_rows)
    _write_csv(
        output_dir / "rd_parent_group_sensitivity.csv", rd_parent_rows
    )
    _write_csv(output_dir / "mif_escape_peak_ratio_impact.csv", mif_rows)
    _write_csv(
        output_dir / "mif_parent_group_sensitivity.csv", mif_parent_rows
    )
    _write_csv(output_dir / "claim_impact.csv", claim_rows)
    (output_dir / "fit_diagnostics.txt").write_text(
        "[Russian-doll combined fit]\n"
        + "post-fit Poisson diagnostic: "
        + json.dumps(rd_postfit_diagnostic, sort_keys=True)
        + "\n"
        + rd_diagnostics
        + "\n[MIF ordered 24-peak fit]\n"
        + "post-fit Poisson diagnostic: "
        + json.dumps(mif_postfit_diagnostic, sort_keys=True)
        + "\n"
        + mif_diagnostics,
        encoding="utf-8",
    )

    manifest = {
        "audit": "Python peak-area paper impact",
        "paper": "arXiv:2607.05834v1",
        "input_release": "HFIRBG_public_data_v1.1.0",
        "database_sha256": db_sha256,
        "database_access": "SQLite mode=ro with PRAGMA query_only",
        "spectrum_access": "read-only text input",
        "area_modes": {
            PEAK_AREA_LEGACY_DENSITY: "net counts / (7 sigma_keV)",
            PEAK_AREA_NET_COUNTS: "background-subtracted counts in +/-3.5 sigma",
            "historical_fit_full_line_counts": (
                "analytic integral of the complete fitted Gaussian-plus-left-tail "
                "signal divided by calibrated channel width; fit covariance "
                "propagated with the complete component gradient"
            ),
        },
        "history": {
            "division_introduced": "5cb9e0afd362e7d3703456531594b0e1424555e3",
            "historical_revision_checked": (
                "3888c8b8fed18b4b70e6e2c7f395dc9fb106ea57"
            ),
            "main_revision_checked": "6aebba05b09b6c5b40cc600c2ae1367adb6c51fb",
        },
        "rd": {
            "selected_run": RD_RUN_NAME,
            "input_files": [
                _input_record(spectrum, data_root) for spectrum in rd_inputs
            ],
            "combined_live_time_s": rd_spec.live,
            "peak_fit_order_keV": list(RD_PEAKS),
            "paper_provenance_note": (
                "The paper says Cycle 498 and run-by-run weighting; the "
                "paper-number-reproducing script combines Cycle 493 files first."
            ),
            "historical_fit_postfit_poisson_diagnostic": rd_postfit_diagnostic,
            "parent_group_sensitivity": {
                "lines_keV": [7367.9, 558.5, 7916.3],
                "fitter_role_mapping": {
                    "fep_slot": "7367.9 keV Pb-207 line",
                    "sep_slot": "558.5 keV Cd-113 reference line",
                    "dep_slot": "7916.3 keV Cu-63 line",
                },
                "window_half_width_keV": 20.0,
                "model": (
                    "bin-integrated Gaussian components, Poisson likelihood, "
                    "affine background per window"
                ),
            },
        },
        "mif": {
            "input_file": _input_record(mif_public, data_root),
            "parent_fit_order_keV": list(MIF_PARENTS_FIT_ORDER),
            "peak_fit_order_keV": [
                round(energy, 6)
                for parent in MIF_PARENTS_FIT_ORDER
                for energy in (parent, parent - 511.0, parent - 1022.0)
            ],
            "legacy_status": (
                "unavailable: simulation ROOT files absent and data fit is "
                "peak-list/order dependent"
            ),
            "current_ordered_fit_postfit_poisson_diagnostic": (
                mif_postfit_diagnostic
            ),
            "parent_group_sensitivity": {
                "parents_keV": [11386.5, 9718.79, 8998.63, 7724.034],
                "model": (
                    "bin-integrated Gaussian components, Poisson likelihood, "
                    "affine background per window"
                ),
            },
        },
        "calibration_covariance_sensitivity": {
            "warning": (
                "The release has A0/A1 but no calibration covariance; both "
                "explicit assumptions below are new-calculation sensitivities."
            ),
            "tight": [[0.0025, 0.0], [0.0, 1e-10]],
            "loose": [[0.25, 0.0], [0.0, 1e-8]],
            "order": ["A0_keV", "A1_keV_per_channel"],
        },
        "uncertainty": {
            "legacy_peakfit": "sqrt(gross + 0.25 * background^2)",
            "legacy_multipeakfit": "sqrt(gross + 1.1 * background)",
            "status": (
                "historical formulas retained; reference self-ratio fixed; "
                "historical full-line comparison and parent-group sensitivity "
                "use fitted covariance"
            ),
        },
        "outputs": [
            "rd_peak_area_impact.csv",
            "rd_parent_group_sensitivity.csv",
            "mif_escape_peak_ratio_impact.csv",
            "mif_parent_group_sensitivity.csv",
            "claim_impact.csv",
            "fit_diagnostics.txt",
            "manifest.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
