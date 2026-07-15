"""ROOT-free, read-only access to public HFIR spectrum files.

This module is intentionally independent of :mod:`src.analysis.Spectrum` and
the legacy database wrapper because importing either pulls in PyROOT.  It is
the shared data layer for lightweight browsers, notebooks, and CSV exports.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STANDARD_PUBLIC_BUNDLE = (
    REPOSITORY_ROOT / "data" / "HFIRBG_public_data_v1.0.0"
)
STANDARD_DB_CANDIDATES = (
    REPOSITORY_ROOT / "db" / "HFIRBG.db",
    STANDARD_PUBLIC_BUNDLE / "HFIRBG.db",
)

Normalization = Literal["counts", "counts/s", "counts/s/keV"]


@dataclass(frozen=True)
class BrowserPaths:
    """Resolved locations used by the public-data browser.

    ``data_root`` may be ``None``.  In that case each spectrum's declared
    directory is resolved relative to the database, which is the portable
    convention used by the public bundle.
    """

    db_path: Path
    data_root: Path | None


@dataclass(frozen=True)
class PublicSpectrum:
    """A calibrated spectrum and the canonical metadata used to load it."""

    file_id: int
    run_id: int
    file_name: str
    run_name: str
    live_time: float
    calibration_A0: float
    calibration_A1: float
    counts: np.ndarray
    energy_keV: np.ndarray
    bin_width_keV: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        arrays = (self.counts, self.energy_keV, self.bin_width_keV)
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("spectrum arrays must be one-dimensional")
        if len({array.size for array in arrays}) != 1:
            raise ValueError("spectrum arrays must have equal lengths")
        if self.counts.size == 0:
            raise ValueError("spectrum must contain at least one bin")
        if not np.isfinite(self.counts).all() or (self.counts < 0).any():
            raise ValueError("spectrum counts must be finite and non-negative")
        if not np.isfinite(self.energy_keV).all():
            raise ValueError("spectrum energies must be finite")
        if (
            not np.isfinite(self.bin_width_keV).all()
            or (self.bin_width_keV <= 0).any()
        ):
            raise ValueError("spectrum bin widths must be finite and positive")


_FILE_METADATA_SQL = """
SELECT
    r.id AS run_id,
    r.name AS run_name,
    r.description AS run_description,
    df.id AS file_id,
    df.name AS file_name,
    df.start_time AS start_time,
    df.live_time AS live_time,
    df.run_number AS run_number,
    directory.id AS directory_id,
    directory.path AS stored_directory,
    cg.id AS calibration_group_id,
    cg.name AS calibration_group_name,
    cg.A0 AS calibration_A0,
    cg.A1 AS calibration_A1,
    coord.id AS coordinate_id,
    coord.Rx AS coordinate_Rx,
    coord.Rz AS coordinate_Rz,
    coord.Lx AS coordinate_Lx,
    coord.Lz AS coordinate_Lz,
    coord.angle AS coordinate_angle,
    coord.track AS coordinate_track,
    dc.id AS detector_configuration_id,
    detector.id AS detector_id,
    detector.type AS detector_type,
    dc.shield AS shield_id,
    shield.name AS shield_name,
    shield.description AS shield_description
FROM datafile AS df
JOIN directory ON directory.id = df.directory_id
JOIN run_file_list AS rfl ON rfl.file_id = df.id
JOIN runs AS r ON r.id = rfl.run_id
LEFT JOIN detector_configuration AS dc ON dc.id = r.detector_configuration
LEFT JOIN detector ON detector.id = dc.detector
JOIN file_calibration_group AS fcg
    ON fcg.file_id = df.id
    AND (dc.id IS NULL OR fcg.det = dc.detector)
JOIN calibration_group AS cg ON cg.id = fcg.group_id
LEFT JOIN detector_coordinates AS coord ON coord.id = r.detector_coordinates
LEFT JOIN shield_configuration AS shield ON shield.id = dc.shield
"""


def _expanded_path(value: os.PathLike[str] | str) -> Path:
    return Path(os.path.expandvars(os.fspath(value))).expanduser().resolve()


def _resolve_db_path(db_path: os.PathLike[str] | str | None) -> Path:
    """Resolve only the database path, without consulting spectrum settings."""

    db_value = db_path if db_path is not None else os.environ.get("HFIRBG_CALDB")
    if db_value is not None:
        resolved_db = _expanded_path(db_value)
        if not resolved_db.is_file():
            raise FileNotFoundError(f"HFIR background database not found: {resolved_db}")
        return resolved_db

    resolved_db = next(
        (candidate.resolve() for candidate in STANDARD_DB_CANDIDATES if candidate.is_file()),
        None,
    )
    if resolved_db is None:
        candidates = ", ".join(str(path) for path in STANDARD_DB_CANDIDATES)
        raise RuntimeError(
            "Pass db_path, set HFIRBG_CALDB, or download the public bundle; "
            f"checked: {candidates}"
        )
    return resolved_db


def resolve_browser_paths(
    db_path: os.PathLike[str] | str | None = None,
    data_root: os.PathLike[str] | str | None = None,
) -> BrowserPaths:
    """Resolve explicit paths, environment variables, then public defaults.

    The explicit arguments take precedence over ``HFIRBG_CALDB`` and
    ``HFIRBGDATA``.  If no database is configured, the standard downloaded
    bundle and then the historical repository database location are checked.
    """

    resolved_db = _resolve_db_path(db_path)

    data_value = data_root if data_root is not None else os.environ.get("HFIRBGDATA")
    if data_value is not None:
        resolved_data = _expanded_path(data_value)
        if not resolved_data.is_dir():
            raise FileNotFoundError(f"HFIR spectrum directory not found: {resolved_data}")
    else:
        # Prefer a directory beside the selected canonical database.  This is
        # the standard release layout and does not depend on repository paths.
        adjacent = resolved_db.parent / "spectra"
        resolved_data = adjacent.resolve() if adjacent.is_dir() else None

    return BrowserPaths(db_path=resolved_db, data_root=resolved_data)


def _read_only_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only = ON")
    return connection


def query_file_metadata(
    db_path: os.PathLike[str] | str | None = None,
    *,
    file_id: int | None = None,
    run_id: int | None = None,
) -> pd.DataFrame:
    """Return canonical metadata for public files through declared joins.

    ``file_id`` and ``run_id`` are optional, parameterized filters.  Missing
    calibration assignments are intentionally excluded: without the canonical
    A0/A1 pair a file cannot be exported as a calibrated spectrum. Detector
    configuration is optional because two public source-calibration runs do
    not declare one; their per-file calibration link remains unambiguous.
    """

    resolved = _resolve_db_path(db_path)
    conditions: list[str] = []
    parameters: list[int] = []
    for column, value in (("df.id", file_id), ("r.id", run_id)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{column} filter must be a positive integer")
            conditions.append(f"{column} = ?")
            parameters.append(value)
    suffix = ""
    if conditions:
        suffix = " WHERE " + " AND ".join(conditions)
    suffix += " ORDER BY r.id, df.start_time, df.id"

    try:
        with _read_only_connection(resolved) as connection:
            return pd.read_sql_query(_FILE_METADATA_SQL + suffix, connection, params=parameters)
    except (sqlite3.Error, pd.errors.DatabaseError) as error:
        raise RuntimeError(f"could not query canonical HFIR database: {error}") from error


def _spectrum_path(
    metadata: Mapping[str, Any],
    paths: BrowserPaths,
) -> Path:
    name = str(metadata["file_name"])
    filename = name if name.lower().endswith(".txt") else f"{name}.txt"
    if paths.data_root is not None:
        directory = paths.data_root
    else:
        stored = str(metadata["stored_directory"]).strip()
        if not stored:
            raise ValueError(f"file {metadata['file_id']} has no declared directory")
        directory = _expanded_path(stored) if Path(stored).is_absolute() else (
            paths.db_path.parent / stored
        ).resolve()
    path = directory / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"spectrum file {metadata['file_id']} was not found at {path}"
        )
    return path


def _read_public_counts(path: Path) -> np.ndarray:
    channels: list[int] = []
    counts: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"malformed spectrum row at {path}:{line_number}: expected at least 3 columns"
                )
            try:
                channel_value = float(fields[0])
                count = float(fields[2])
            except ValueError as error:
                raise ValueError(
                    f"malformed numeric value at {path}:{line_number}"
                ) from error
            if not isfinite(channel_value) or not channel_value.is_integer():
                raise ValueError(f"invalid channel at {path}:{line_number}")
            if not isfinite(count) or count < 0:
                raise ValueError(f"invalid count at {path}:{line_number}")
            channels.append(int(channel_value))
            counts.append(count)

    if not channels:
        raise ValueError(f"spectrum contains no channel rows: {path}")
    expected = list(range(1, len(channels) + 1))
    if channels != expected:
        raise ValueError(
            f"spectrum channels must be contiguous and 1-based: {path}"
        )
    return np.asarray(counts, dtype=np.float64)


def load_spectrum(
    file_id: int,
    db_path: os.PathLike[str] | str | None = None,
    data_root: os.PathLike[str] | str | None = None,
) -> PublicSpectrum:
    """Load one public spectrum using its canonical database calibration."""

    paths = resolve_browser_paths(db_path=db_path, data_root=data_root)
    metadata_table = query_file_metadata(paths.db_path, file_id=file_id)
    if metadata_table.empty:
        raise KeyError(f"no calibrated public spectrum has file_id {file_id}")
    if len(metadata_table) != 1:
        raise ValueError(f"file_id {file_id} has multiple canonical metadata rows")
    metadata = metadata_table.iloc[0].to_dict()

    try:
        live_time = float(metadata["live_time"])
        A0 = float(metadata["calibration_A0"])
        A1 = float(metadata["calibration_A1"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"file_id {file_id} has incomplete numeric metadata") from error
    if not isfinite(live_time) or live_time <= 0:
        raise ValueError(f"file_id {file_id} has invalid live time: {live_time}")
    if not isfinite(A0) or not isfinite(A1) or A1 <= 0:
        raise ValueError(f"file_id {file_id} has invalid linear calibration")

    counts = _read_public_counts(_spectrum_path(metadata, paths))
    channels = np.arange(1, counts.size + 1, dtype=np.float64)
    energy = A0 + A1 * channels
    widths = np.full(counts.size, A1, dtype=np.float64)
    return PublicSpectrum(
        file_id=int(metadata["file_id"]),
        run_id=int(metadata["run_id"]),
        file_name=str(metadata["file_name"]),
        run_name=str(metadata["run_name"]),
        live_time=live_time,
        calibration_A0=A0,
        calibration_A1=A1,
        counts=counts,
        energy_keV=energy,
        bin_width_keV=widths,
        metadata=metadata,
    )


def rebin_by_factor(spectrum: PublicSpectrum, factor: int) -> PublicSpectrum:
    """Combine adjacent bins, retaining a final partial bin when necessary."""

    if isinstance(factor, bool) or not isinstance(factor, int) or factor <= 0:
        raise ValueError("rebin factor must be a positive integer")
    if factor == 1:
        return spectrum

    starts = np.arange(0, spectrum.counts.size, factor)
    counts = np.add.reduceat(spectrum.counts, starts)
    widths = np.add.reduceat(spectrum.bin_width_keV, starts)
    # Bin centers follow from the outer edges, including for a partial last bin.
    low_edges = spectrum.energy_keV[starts] - spectrum.bin_width_keV[starts] / 2
    end_indices = np.minimum(starts + factor, spectrum.counts.size) - 1
    high_edges = (
        spectrum.energy_keV[end_indices]
        + spectrum.bin_width_keV[end_indices] / 2
    )
    energy = (low_edges + high_edges) / 2
    return replace(
        spectrum,
        counts=counts,
        energy_keV=energy,
        bin_width_keV=widths,
    )


def spectrum_dataframe(
    spectrum: PublicSpectrum,
    normalization: Normalization = "counts",
) -> pd.DataFrame:
    """Export calibrated bins with Poisson statistical uncertainties.

    The generic ``value`` and ``statistical_error`` columns use the requested
    normalization.  Raw counts and bin widths remain present so the exported
    values are independently reproducible.
    """

    if normalization not in {"counts", "counts/s", "counts/s/keV"}:
        raise ValueError(
            "normalization must be 'counts', 'counts/s', or 'counts/s/keV'"
        )
    scale: float | np.ndarray
    if normalization == "counts":
        scale = 1.0
    elif normalization == "counts/s":
        scale = spectrum.live_time
    else:
        scale = spectrum.live_time * spectrum.bin_width_keV

    return pd.DataFrame(
        {
            "energy_keV": spectrum.energy_keV,
            "bin_width_keV": spectrum.bin_width_keV,
            "counts": spectrum.counts,
            "value": spectrum.counts / scale,
            "statistical_error": np.sqrt(spectrum.counts) / scale,
            "normalization": normalization,
        }
    )
