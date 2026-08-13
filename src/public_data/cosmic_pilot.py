"""Neutral, read-only helpers for the bounded HPGe cosmic-shape pilot.

The simulation input is treated as a generic HDF5 event table.  Source labels
describe separately configured runs; the format has no track-parent ancestry.
Measured spectra remain calibrated detector counts.  Nothing here estimates an
absolute cosmic rate or writes the canonical SQLite database.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .browser import PublicSpectrum, load_spectrum, query_file_metadata


REQUIRED_HDF_FIELDS = {
    "evt": {"evt", "t", "ct", "flg"},
    "prim": {"PID", "x", "p", "E", "t", "evt", "vol"},
    "ioni": {"E", "t", "x", "EdEdx", "Eq", "vol", "PID", "evt"},
}
NONDETERMINISTIC_HDF_FIELDS = {"evt": {"ct"}}


@dataclass(frozen=True)
class MeasuredCombination:
    """Count sum on an exactly shared calibrated channel grid."""

    run_id: int
    run_name: str
    file_ids: tuple[int, ...]
    file_names: tuple[str, ...]
    spectrum_sha256: tuple[str, ...]
    live_time_s: float
    calibration_A0: float
    calibration_A1: float
    energy_keV: np.ndarray
    bin_width_keV: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class SimulationBatch:
    """Validated event accounting plus raw event-summed Ge deposition."""

    label: str
    path: Path
    event_ids: np.ndarray
    event_time_ns: np.ndarray
    primary_pids: np.ndarray
    deposition_keV: np.ndarray
    depositing_pids: np.ndarray
    depositing_energy_keV: np.ndarray
    runtime_s: float
    nprim: int
    semantic_sha256: str

    @property
    def deposited_event_count(self) -> int:
        return int(np.count_nonzero(self.deposition_keV > 0))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported HPGe cosmic-pilot config schema")
    return config


def _exact_float(actual: Any, expected: float, name: str) -> None:
    value = float(actual)
    if not math.isfinite(value) or value != float(expected):
        raise ValueError(f"{name} mismatch: expected {expected!r}, found {value!r}")


def _validate_run_metadata(db_path: Path, selection: Mapping[str, Any]) -> int:
    """Resolve one run through SQLite URI read-only mode plus query_only."""

    sql = """
    SELECT r.id, r.detector_configuration, dc.acquisition_settings, dc.shield,
           shield.name, coord.angle, coord.Rx, coord.Rz, coord.Lx, coord.Lz,
           acq.coarse_gain, acq.PUR_guard, acq.offset, acq.fine_gain, acq.LLD,
           acq.LTC_mode, acq.memory_group
    FROM runs AS r
    JOIN detector_configuration AS dc ON dc.id = r.detector_configuration
    JOIN acquisition_settings AS acq ON acq.id = dc.acquisition_settings
    LEFT JOIN shield_configuration AS shield ON shield.id = dc.shield
    LEFT JOIN detector_coordinates AS coord ON coord.id = r.detector_coordinates
    WHERE r.name = ?
    """
    uri = f"{db_path.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(sql, (selection["run_name"],)).fetchall()
    if len(rows) != 1:
        raise ValueError(
            f"expected exactly one run named {selection['run_name']!r}; found {len(rows)}"
        )
    row = rows[0]
    integer_expectations = (
        (
            row["detector_configuration"],
            selection["detector_configuration_id"],
            "detector configuration",
        ),
        (
            row["acquisition_settings"],
            selection["acquisition_settings_id"],
            "acquisition settings",
        ),
        (row["shield"], selection["shield_id"], "shield"),
    )
    for actual, expected, name in integer_expectations:
        if int(actual) != int(expected):
            raise ValueError(f"{name} mismatch: expected {expected}, found {actual}")
    if str(row["name"]) != str(selection["shield_name"]):
        raise ValueError("shield name mismatch")
    _exact_float(row["angle"], selection["orientation_angle_deg"], "orientation")
    for key in ("Rx", "Rz", "Lx", "Lz"):
        _exact_float(row[key], selection["coordinates"][key], f"coordinate {key}")
    for key, expected in selection["acquisition_settings"].items():
        _exact_float(row[key], expected, f"acquisition {key}")
    return int(row["id"])


def select_measured_spectra(
    config: Mapping[str, Any], bundle_root: Path
) -> tuple[PublicSpectrum, ...]:
    """Select the predeclared reactor-off family without database mutation."""

    release = config["public_release"]
    selection = config["measured_selection"]
    bundle_root = bundle_root.resolve()
    db_path = bundle_root / "HFIRBG.db"
    spectra_root = bundle_root / "spectra"
    if sha256_file(db_path) != release["database_sha256"]:
        raise ValueError("public database SHA256 mismatch")
    run_id = _validate_run_metadata(db_path, selection)

    metadata = query_file_metadata(db_path, run_id=run_id)
    expected_names = tuple(selection["file_names_chronological"])
    indexed = {str(row.file_name): row for row in metadata.itertuples(index=False)}
    if set(indexed) != set(expected_names) or len(indexed) != len(metadata):
        raise ValueError("predeclared measured file family does not match database")

    spectra: list[PublicSpectrum] = []
    for name in expected_names:
        row = indexed[name]
        _exact_float(
            row.calibration_A0,
            selection["calibration_A0_keV"],
            f"{name} calibration A0",
        )
        _exact_float(
            row.calibration_A1,
            selection["calibration_A1_keV_per_channel"],
            f"{name} calibration A1",
        )
        spectra.append(load_spectrum(int(row.file_id), db_path, spectra_root))
    return tuple(spectra)


def combine_identical_spectra(
    spectra: Sequence[PublicSpectrum], spectra_root: Path
) -> MeasuredCombination:
    """Sum only spectra sharing exact grid, calibration, and run metadata."""

    if not spectra:
        raise ValueError("no measured spectra supplied")
    first = spectra[0]
    shared_metadata = ("detector_configuration_id", "shield_id", "coordinate_id")
    for spectrum in spectra:
        if not np.array_equal(spectrum.counts, np.floor(spectrum.counts)):
            raise ValueError("measured spectra must contain integer detector counts")
    for spectrum in spectra[1:]:
        if spectrum.run_id != first.run_id or spectrum.run_name != first.run_name:
            raise ValueError("measured spectra span multiple run identities")
        if (
            spectrum.calibration_A0 != first.calibration_A0
            or spectrum.calibration_A1 != first.calibration_A1
            or not np.array_equal(spectrum.energy_keV, first.energy_keV)
            or not np.array_equal(spectrum.bin_width_keV, first.bin_width_keV)
        ):
            raise ValueError("measured spectra do not have an identical calibrated grid")
        for key in shared_metadata:
            if spectrum.metadata.get(key) != first.metadata.get(key):
                raise ValueError(f"measured metadata mismatch: {key}")

    names = tuple(spectrum.file_name for spectrum in spectra)
    hashes = tuple(
        sha256_file(spectra_root / f"{name}.txt") for name in names
    )
    return MeasuredCombination(
        run_id=first.run_id,
        run_name=first.run_name,
        file_ids=tuple(spectrum.file_id for spectrum in spectra),
        file_names=names,
        spectrum_sha256=hashes,
        live_time_s=float(sum(spectrum.live_time for spectrum in spectra)),
        calibration_A0=first.calibration_A0,
        calibration_A1=first.calibration_A1,
        energy_keV=first.energy_keV.copy(),
        bin_width_keV=first.bin_width_keV.copy(),
        counts=np.sum([spectrum.counts for spectrum in spectra], axis=0),
    )


def _scalar_attribute(dataset: Any, name: str) -> float:
    if name not in dataset.attrs:
        raise ValueError(f"dataset {dataset.name} lacks {name!r} attribute")
    values = np.asarray(dataset.attrs[name]).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise ValueError(f"dataset {dataset.name} has invalid {name!r} attribute")
    return float(values[0])


def _semantic_hdf_sha256(datasets: Mapping[str, Any]) -> str:
    """Hash physical fields and deterministic attributes, independent of layout.

    ``evt.ct`` is measured per-event CPU time.  It is retained and checked for
    finiteness by Geant4, but cannot be part of deterministic physics replay.
    """

    digest = hashlib.sha256()
    for dataset_name in ("evt", "prim", "ioni"):
        dataset = datasets[dataset_name]
        values = dataset[()]
        digest.update(dataset_name.encode("ascii") + b"\0")
        digest.update(str(values.shape).encode("ascii") + b"\0")
        for field in values.dtype.names or ():
            if field in NONDETERMINISTIC_HDF_FIELDS.get(dataset_name, set()):
                continue
            column = np.ascontiguousarray(values[field])
            digest.update(field.encode("ascii") + b"\0")
            digest.update(column.dtype.str.encode("ascii") + b"\0")
            digest.update(column.tobytes())
        for attribute in sorted(dataset.attrs):
            value = np.asarray(dataset.attrs[attribute])
            digest.update(attribute.encode("utf-8") + b"\0")
            digest.update(value.dtype.str.encode("ascii") + b"\0")
            digest.update(str(value.shape).encode("ascii") + b"\0")
            if value.dtype.kind in {"O", "S", "U"}:
                digest.update(repr(value.tolist()).encode("utf-8"))
            else:
                digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def load_simulation_batch(
    path: Path, label: str, expected_primary_pids: Iterable[int]
) -> SimulationBatch:
    """Load one simulation batch and enforce event-accounting closure."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("h5py is required to inspect simulation HDF5") from error

    path = path.resolve()
    allowed_pids = np.asarray(sorted(set(expected_primary_pids)), dtype=np.int64)
    if allowed_pids.size == 0:
        raise ValueError("expected primary PID set may not be empty")
    with h5py.File(path, "r") as handle:
        for name, required_fields in REQUIRED_HDF_FIELDS.items():
            if name not in handle:
                raise ValueError(f"simulation lacks required dataset {name!r}")
            fields = set(handle[name].dtype.names or ())
            if fields != required_fields:
                missing = sorted(required_fields - fields)
                extra = sorted(fields - required_fields)
                raise ValueError(
                    f"dataset {name!r} field mismatch: missing={missing}, extra={extra}"
                )
        datasets = {name: handle[name] for name in REQUIRED_HDF_FIELDS}
        evt = datasets["evt"][()]
        prim = datasets["prim"][()]
        ioni = datasets["ioni"][()]
        attributes = {
            name: {
                key: _scalar_attribute(dataset, key) for key in ("nprim", "runtime")
            }
            for name, dataset in datasets.items()
        }
        semantic_sha = _semantic_hdf_sha256(datasets)

    event_ids = np.asarray(evt["evt"], dtype=np.int64)
    if event_ids.size == 0 or np.unique(event_ids).size != event_ids.size:
        raise ValueError("evt identifiers must be nonempty and unique")
    if event_ids.size > 1 and not np.all(np.diff(event_ids) == 1):
        raise ValueError("evt identifiers must be ordered and consecutive")
    raw_nprim = [values["nprim"] for values in attributes.values()]
    if any(not value.is_integer() for value in raw_nprim):
        raise ValueError("nprim attributes must be integer-valued")
    nprim_values = {int(value) for value in raw_nprim}
    runtime_values = {values["runtime"] for values in attributes.values()}
    if len(nprim_values) != 1 or nprim_values.pop() != event_ids.size:
        raise ValueError("nprim attributes do not equal the processed-event count")
    if len(runtime_values) != 1:
        raise ValueError("runtime attributes disagree across event tables")
    runtime_s = runtime_values.pop()
    if runtime_s < 0:
        raise ValueError("generator runtime must be nonnegative")

    primary_events = np.asarray(prim["evt"], dtype=np.int64)
    primary_pids = np.asarray(prim["PID"], dtype=np.int64)
    if not np.array_equal(np.unique(primary_events), event_ids):
        raise ValueError("every event must have at least one primary and no foreign primary")
    unexpected = np.setdiff1d(np.unique(primary_pids), allowed_pids)
    if unexpected.size:
        raise ValueError(f"primary PID mismatch for {label}: {unexpected.tolist()}")

    event_time_ns = np.asarray(evt["t"], dtype=np.float64)
    if not np.isfinite(event_time_ns).all() or np.any(np.diff(event_time_ns) < 0):
        raise ValueError("event generator times must be finite and monotonic")
    event_cpu_time_s = np.asarray(evt["ct"], dtype=np.float64)
    if not np.isfinite(event_cpu_time_s).all() or np.any(event_cpu_time_s < 0):
        raise ValueError("per-event CPU times must be finite and nonnegative")
    if not math.isclose(
        event_time_ns[-1], runtime_s * 1.0e9, rel_tol=1.0e-12, abs_tol=1.0e-6
    ):
        raise ValueError("final event time does not match the recorded generator runtime")

    ionization_events = np.asarray(ioni["evt"], dtype=np.int64)
    ionization_mev = np.asarray(ioni["E"], dtype=np.float64)
    if not np.isfinite(ionization_mev).all() or np.any(ionization_mev < 0):
        raise ValueError("ionization energies must be finite and nonnegative")
    if np.setdiff1d(np.unique(ionization_events), event_ids).size:
        raise ValueError("ionization table refers to a foreign event")
    deposition_keV = np.zeros(event_ids.size, dtype=np.float64)
    if ionization_events.size:
        indices = ionization_events - event_ids[0]
        np.add.at(deposition_keV, indices, ionization_mev * 1000.0)

    return SimulationBatch(
        label=label,
        path=path,
        event_ids=event_ids,
        event_time_ns=event_time_ns,
        primary_pids=primary_pids,
        deposition_keV=deposition_keV,
        depositing_pids=np.asarray(ioni["PID"], dtype=np.int64),
        depositing_energy_keV=ionization_mev * 1000.0,
        runtime_s=runtime_s,
        nprim=event_ids.size,
        semantic_sha256=semantic_sha,
    )


def concatenate_batches(label: str, batches: Sequence[SimulationBatch]) -> SimulationBatch:
    """Join batches after validation without pretending their event IDs are global."""

    if not batches or any(batch.label != label for batch in batches):
        raise ValueError("batch labels must be nonempty and identical")
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch.semantic_sha256.encode("ascii"))
    elapsed_ns = 0.0
    time_parts: list[np.ndarray] = []
    for batch in batches:
        time_parts.append(batch.event_time_ns + elapsed_ns)
        elapsed_ns += batch.runtime_s * 1.0e9
    return SimulationBatch(
        label=label,
        path=Path("<aggregate>"),
        event_ids=np.arange(sum(batch.nprim for batch in batches), dtype=np.int64),
        event_time_ns=np.concatenate(time_parts),
        primary_pids=np.concatenate([batch.primary_pids for batch in batches]),
        deposition_keV=np.concatenate([batch.deposition_keV for batch in batches]),
        depositing_pids=np.concatenate([batch.depositing_pids for batch in batches]),
        depositing_energy_keV=np.concatenate(
            [batch.depositing_energy_keV for batch in batches]
        ),
        runtime_s=float(sum(batch.runtime_s for batch in batches)),
        nprim=sum(batch.nprim for batch in batches),
        semantic_sha256=digest.hexdigest(),
    )


def semantic_replay_equal(first: SimulationBatch, second: SimulationBatch) -> bool:
    return first.semantic_sha256 == second.semantic_sha256


_VALUE_UNIT = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$"
)


def _value_unit(text: str) -> tuple[float, str]:
    match = _VALUE_UNIT.match(text)
    if not match:
        raise ValueError(f"could not parse Geant4 value/unit string {text!r}")
    return float(match.group(1)), match.group(2).replace(" ", "")


def _area_cm2(text: str) -> float:
    value, unit = _value_unit(text)
    factors = {
        "mm2": 0.01,
        "mm^2": 0.01,
        "cm2": 1.0,
        "cm^2": 1.0,
        "m2": 10_000.0,
        "m^2": 10_000.0,
    }
    if unit not in factors:
        raise ValueError(f"unsupported surface unit {unit!r}")
    return value * factors[unit]


def _flux_per_cm2_s(text: str) -> float:
    value, unit = _value_unit(text)
    normalized = unit.replace("cm^2", "cm2")
    if not normalized.endswith("/cm2"):
        raise ValueError(f"unsupported flux area unit {unit!r}")
    frequency = normalized[: -len("/cm2")]
    factors = {
        "Hz": 1.0,
        "kHz": 1.0e3,
        "MHz": 1.0e6,
        "mHz": 1.0e-3,
        "microHz": 1.0e-6,
        "/s": 1.0,
        "1/s": 1.0,
        "s^-1": 1.0,
    }
    if frequency not in factors:
        raise ValueError(f"unsupported frequency unit {frequency!r}")
    return value * factors[frequency]


def check_sato_runtime(
    batch: SimulationBatch, xml_path: Path, relative_tolerance: float = 2.0e-4
) -> dict[str, float | int | str]:
    """Recompute Sato exposure from XML attempts/(surface area * flux)."""

    root = ET.parse(xml_path).getroot()
    elements = list(root.iter())
    module = next(
        (element for element in elements if element.tag.split("}")[-1] == "CosmicNeutron"),
        None,
    )
    thrower = next(
        (
            element
            for element in elements
            if element.tag.split("}")[-1] in {"CosineThrower", "SurfaceThrower"}
            and "s_area" in element.attrib
        ),
        None,
    )
    if module is None or thrower is None:
        raise ValueError("Sato XML lacks CosmicNeutron/CosineThrower provenance")
    flux = _flux_per_cm2_s(module.attrib["flux"])
    area = _area_cm2(thrower.attrib["s_area"])
    attempts = int(thrower.attrib["nAttempts"])
    expected_runtime = attempts / (area * flux)
    relative_error = abs(expected_runtime - batch.runtime_s) / expected_runtime
    if not math.isfinite(relative_error) or relative_error > relative_tolerance:
        raise ValueError(
            f"Sato runtime closure failed: relative error {relative_error:.6g}"
        )
    expected_scales = None
    if batch.label.startswith("sato_nonthermal"):
        expected_scales = (1.0, 0.0)
    elif batch.label.startswith("sato_thermal"):
        expected_scales = (0.0, 1.0)
    if expected_scales is not None:
        scale_s = float(module.attrib.get("scale_S", "1"))
        scale_t = float(module.attrib.get("scale_T", "1"))
        if (scale_s, scale_t) != expected_scales:
            raise ValueError("Sato XML scale factors disagree with the source-run label")
    return {
        "status": "pass",
        "nAttempts": attempts,
        "surface_area_cm2": area,
        "flux_per_cm2_s": flux,
        "xml_runtime_s": expected_runtime,
        "hdf_runtime_s": batch.runtime_s,
        "relative_error": relative_error,
    }


def fixed_edges(lower: float, upper: float, width: float) -> np.ndarray:
    if not (math.isfinite(lower) and math.isfinite(upper) and math.isfinite(width)):
        raise ValueError("histogram limits must be finite")
    if lower >= upper or width <= 0:
        raise ValueError("invalid histogram limits")
    count = int(math.floor((upper - lower) / width + 1.0e-12))
    edges = lower + np.arange(count + 1, dtype=np.float64) * width
    if edges[-1] < upper - width * 1.0e-10:
        edges = np.append(edges, upper)
    else:
        edges[-1] = upper
    return edges


def values_outside_masks(
    values: np.ndarray, masks: Sequence[Mapping[str, float]]
) -> np.ndarray:
    keep = np.ones(values.size, dtype=bool)
    for mask in masks:
        keep &= np.abs(values - float(mask["center"])) > float(mask["half_width"])
    return values[keep]


def unit_area_histogram(
    values: np.ndarray,
    edges: np.ndarray,
    masks: Sequence[Mapping[str, float]] = (),
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != values.shape:
            raise ValueError("histogram weights must match values")
    keep = (values >= edges[0]) & (values < edges[-1])
    for mask in masks:
        keep &= np.abs(values - float(mask["center"])) > float(mask["half_width"])
    counts, _ = np.histogram(
        values[keep], bins=edges, weights=None if weights is None else weights[keep]
    )
    total = float(np.sum(counts))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("shape histogram has no positive in-window weight")
    return np.asarray(counts, dtype=np.float64) / total, total


def shape_distances(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    if first.shape != second.shape:
        raise ValueError("shape vectors must have the same bins")
    if np.any(first < 0) or np.any(second < 0):
        raise ValueError("shape vectors must be nonnegative")
    if not np.isclose(first.sum(), 1.0) or not np.isclose(second.sum(), 1.0):
        raise ValueError("shape vectors must be unit normalized")
    total_variation = 0.5 * float(np.abs(first - second).sum())
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    cosine = float(np.dot(first, second) / denominator) if denominator else math.nan
    return {"total_variation": total_variation, "cosine_similarity": cosine}


def binomial_clopper_pearson(
    successes: int, trials: int, confidence: float = 0.95
) -> list[float]:
    """Exact central interval for an event fraction."""

    from scipy.stats import beta

    if not 0 <= successes <= trials or trials <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid binomial interval inputs")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if successes == 0 else float(
        beta.ppf(tail, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - tail, successes + 1, trials - successes)
    )
    return [lower, upper]


def area_convergence(
    first_values: np.ndarray,
    second_values: np.ndarray,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Two-sided KS plus deterministic nonparametric bootstrap on KS D."""

    from scipy.stats import ks_2samp

    first_values = np.asarray(first_values, dtype=np.float64)
    second_values = np.asarray(second_values, dtype=np.float64)
    minimum = int(settings["minimum_in_window_depositions_per_area"])
    result: dict[str, Any] = {
        "n_first": int(first_values.size),
        "n_second": int(second_values.size),
        "minimum_required": minimum,
    }
    if first_values.size < minimum or second_values.size < minimum:
        result["status"] = "underpowered"
        return result
    observed = ks_2samp(first_values, second_values, alternative="two-sided", method="auto")
    rng = np.random.default_rng(int(settings["bootstrap_seed"]))
    boot = np.empty(int(settings["bootstrap_replicates"]), dtype=np.float64)
    for index in range(boot.size):
        sample_a = rng.choice(first_values, size=first_values.size, replace=True)
        sample_b = rng.choice(second_values, size=second_values.size, replace=True)
        boot[index] = ks_2samp(sample_a, sample_b, method="auto").statistic
    threshold = float(settings["two_sided_ks_p_threshold"])
    result.update(
        {
            "status": "not_rejected" if observed.pvalue >= threshold else "rejected",
            "ks_D": float(observed.statistic),
            "ks_pvalue": float(observed.pvalue),
            "p_threshold": threshold,
            "ks_D_bootstrap_95pct": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
        }
    )
    return result


def sideband_diagnostic(
    values: np.ndarray,
    center_keV: float,
    signal_half_width_keV: float,
    sidebands_keV: Sequence[float],
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Return a descriptive local excess; no peak attribution or fit."""

    values = np.asarray(values, dtype=np.float64)
    weights_array = (
        np.ones(values.size, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if weights_array.shape != values.shape or np.any(weights_array < 0):
        raise ValueError("sideband weights must be nonnegative and match values")
    inner, outer = map(float, sidebands_keV)
    if not (0 < signal_half_width_keV < inner < outer):
        raise ValueError("invalid signal/sideband geometry")
    signal = np.abs(values - center_keV) <= signal_half_width_keV
    sideband = (np.abs(values - center_keV) >= inner) & (
        np.abs(values - center_keV) <= outer
    )
    signal_sum = float(weights_array[signal].sum())
    sideband_sum = float(weights_array[sideband].sum())
    scale = (2.0 * signal_half_width_keV) / (2.0 * (outer - inner))
    background = scale * sideband_sum
    # Poisson diagnostic for measured counts; also reported for unweighted MC.
    uncertainty = math.sqrt(signal_sum + scale * scale * sideband_sum)
    return {
        "center_keV": float(center_keV),
        "signal": signal_sum,
        "sidebands": sideband_sum,
        "background_estimate": background,
        "excess": signal_sum - background,
        "poisson_uncertainty": uncertainty,
    }


def depositing_pid_summary(batch: SimulationBatch) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for pid in np.unique(batch.depositing_pids):
        selected = batch.depositing_pids == pid
        rows.append(
            {
                "depositing_pid": int(pid),
                "hit_records": int(np.count_nonzero(selected)),
                "deposited_energy_keV": float(batch.depositing_energy_keV[selected].sum()),
            }
        )
    return rows
