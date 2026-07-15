"""Build a normalized, read-only catalog of public HFIR runs.

The catalog contains one row per ``runs`` record.  Data files are summarized
through ``run_file_list``; the other columns come only from foreign-key joins
declared by the canonical database schema.
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import date, datetime, timezone
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


UNKNOWN = "unknown"
CALENDAR_TIMEZONE = "America/New_York"
DEFAULT_CYCLE_CALENDAR_PATH = (
    Path(__file__).resolve().parents[2] / "reference_data" / "hfir_cycle_calendar.csv"
)

_CALENDAR_REQUIRED_COLUMNS = {
    "cycle",
    "start_date",
    "end_date",
    "date_precision",
    "schedule_basis",
    "record_status",
    "source_title",
    "source_url",
    "source_doi",
    "source_publication_date",
    "retrieved_date",
}
_CALENDAR_PROVENANCE_COLUMNS = (
    "source_title",
    "source_url",
    "source_doi",
    "retrieved_date",
)

_CYCLE_RE = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:pre|post)?cycle[\s_-]*(\d{3})([abc])?"
)
_ON_RE = re.compile(r"(?i)\b(?:reactor|rxr?|rx)[\s_-]*on\b")
_OFF_RE = re.compile(r"(?i)\b(?:reactor|rxr?|rx)[\s_-]*off\b")
_PRE_POST_RE = re.compile(r"(?i)\b(?:pre|post)[\s_-]*cycle")
_OUTAGE_RE = re.compile(r"(?i)outage")
_TRANSITION_RE = re.compile(
    r"(?i)(?:start[\s_-]*up|shut[\s_-]*down|ramp(?:ing|ed)?|"
    r"\d+\s*(?:%|pwr|power)?\s*to\s*\d+)"
)
_POSITIVE_POWER_RE = re.compile(
    r"(?i)(?<![\d.])(?:100|[1-9]\d?)(?:\.\d+)?\s*%?\s*(?:pwr|power)\b"
)


def _text(value: object) -> str:
    return "" if value is None else str(value)


def derive_reactor_cycle(name: object, description: object = None) -> str:
    """Return an explicitly named reactor cycle, or ``"unknown"``.

    The run name is authoritative when it contains exactly one cycle token.
    The description is used only when the name has none.  This avoids assigning
    a cycle to descriptions that mention multiple adjacent cycles.
    """

    for value in (name, description):
        cycles = {
            number + suffix.upper()
            for number, suffix in _CYCLE_RE.findall(_text(value))
        }
        if len(cycles) == 1:
            return cycles.pop()
        if len(cycles) > 1:
            return UNKNOWN
    return UNKNOWN


def derive_reactor_state(name: object, description: object = None) -> str:
    """Return ``on``, ``off``, ``transition``, or ``unknown``.

    Only explicit state language is used.  In particular, an otherwise
    unqualified ``Cycle###`` name is not assumed to mean reactor-on.
    Conflicting state indicators return ``unknown``.
    """

    text = " ".join(part for part in (_text(name), _text(description)) if part)
    if not text:
        return UNKNOWN
    # SQLite names commonly use underscores, while descriptions use spaces.
    # Treat both forms identically for state phrases without changing raw data.
    state_text = re.sub(r"[_-]+", " ", text)

    is_on = bool(_ON_RE.search(state_text))
    is_off = bool(
        _OFF_RE.search(state_text)
        or _PRE_POST_RE.search(state_text)
        or _OUTAGE_RE.search(state_text)
    )
    is_transition = bool(_TRANSITION_RE.search(state_text))

    # A run whose metadata explicitly describes more than one state should not
    # be forced into a single category.
    if sum((is_on, is_off, is_transition)) > 1:
        return UNKNOWN
    if is_transition:
        return "transition"
    if is_off:
        return "off"
    if is_on:
        return "on"
    if _POSITIVE_POWER_RE.search(state_text):
        return "on"
    return UNKNOWN


def load_cycle_calendar(
    path: os.PathLike[str] | str | None = None,
) -> pd.DataFrame:
    """Load and validate the authoritative, day-precision HFIR calendar.

    The returned ``start_date`` and ``end_date`` columns contain
    :class:`datetime.date` objects. The source gives no transition times, so
    callers must not interpret a boundary date as a precise startup or
    shutdown instant.
    """

    calendar_path = (
        DEFAULT_CYCLE_CALENDAR_PATH
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not calendar_path.is_file():
        raise FileNotFoundError(f"HFIR cycle calendar not found: {calendar_path}")

    calendar = pd.read_csv(calendar_path, dtype=str, keep_default_na=False)
    missing = _CALENDAR_REQUIRED_COLUMNS.difference(calendar.columns)
    if missing:
        raise ValueError(
            "HFIR cycle calendar is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if calendar.empty:
        raise ValueError("HFIR cycle calendar is empty")

    required = calendar[list(sorted(_CALENDAR_REQUIRED_COLUMNS))]
    if required.apply(lambda column: column.str.strip().eq("").any()).any():
        raise ValueError("HFIR cycle calendar contains empty required values")
    if calendar["cycle"].duplicated().any():
        raise ValueError("HFIR cycle calendar contains duplicate cycle identifiers")
    if not calendar["cycle"].str.fullmatch(r"\d{3}[A-C]?").all():
        raise ValueError("HFIR cycle calendar contains an invalid cycle identifier")
    if not calendar["date_precision"].eq("day").all():
        raise ValueError("HFIR cycle calendar must use day precision")
    if not calendar["schedule_basis"].eq("actual").all():
        raise ValueError("HFIR cycle calendar must contain actual dates")
    if not calendar["record_status"].eq("complete").all():
        raise ValueError("HFIR cycle calendar must contain complete records")

    for column in ("start_date", "end_date"):
        parsed = pd.to_datetime(calendar[column], format="%Y-%m-%d", errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"HFIR cycle calendar contains an invalid {column}")
        calendar[column] = parsed.dt.date

    calendar = calendar.sort_values(["start_date", "end_date"]).reset_index(drop=True)
    if (calendar["start_date"] > calendar["end_date"]).any():
        raise ValueError("HFIR cycle calendar contains a start date after its end date")
    for previous, current in zip(
        calendar.itertuples(index=False), calendar.iloc[1:].itertuples(index=False)
    ):
        if current.start_date <= previous.end_date:
            raise ValueError("HFIR cycle calendar contains overlapping cycle intervals")

    for column in _CALENDAR_PROVENANCE_COLUMNS:
        if calendar[column].nunique(dropna=False) != 1:
            raise ValueError(f"HFIR cycle calendar has inconsistent {column} values")
    return calendar


def _period_for_date(value: date, calendar: pd.DataFrame) -> tuple[str, str, str] | None:
    if value < calendar.iloc[0]["start_date"] or value > calendar.iloc[-1]["end_date"]:
        return None

    rows = list(calendar.itertuples(index=False))
    for index, row in enumerate(rows):
        if row.start_date <= value <= row.end_date:
            return ("operating", row.cycle, f"Cycle {row.cycle}")
        if index + 1 < len(rows):
            next_row = rows[index + 1]
            if row.end_date < value < next_row.start_date:
                return (
                    "outage",
                    UNKNOWN,
                    f"EOC {row.cycle} / pre-cycle {next_row.cycle}",
                )
    return None


def _local_date(timestamp: float) -> date:
    return datetime.fromtimestamp(timestamp, timezone.utc).astimezone(
        ZoneInfo(CALENDAR_TIMEZONE)
    ).date()


def classify_run_interval(
    start_time: object,
    end_time: object,
    calendar: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Classify a Unix-time run interval against official cycle dates.

    Intervals use the conventional half-open form ``[start_time, end_time)``.
    A run wholly inside one official cycle is ``operating``; a run wholly in
    one between-cycle gap is ``outage``; and a run crossing those day-level
    classifications is ``mixed``. Invalid timestamps and intervals outside
    the calendar's covered date range remain ``unknown``.
    """

    if calendar is None:
        calendar = load_cycle_calendar()
    result = {
        "calendar_cycle": UNKNOWN,
        "calendar_reactor_state": UNKNOWN,
        "calendar_period": UNKNOWN,
        "calendar_date_precision": str(calendar.iloc[0]["date_precision"]),
        "calendar_source_title": str(calendar.iloc[0]["source_title"]),
        "calendar_source_url": str(calendar.iloc[0]["source_url"]),
        "calendar_source_doi": str(calendar.iloc[0]["source_doi"]),
        "calendar_source_retrieved_date": str(calendar.iloc[0]["retrieved_date"]),
    }

    try:
        start = float(start_time)
        end = float(end_time)
    except (TypeError, ValueError):
        return result
    if not isfinite(start) or not isfinite(end) or start < 0 or end < start:
        return result

    # The aggregate end_time is start + live time and is therefore exclusive.
    # Probe just inside that endpoint so midnight does not spuriously assign a
    # completed run to the following calendar date.
    end_probe = start if end == start else max(start, end - min(1e-6, (end - start) / 2))
    try:
        start_period = _period_for_date(_local_date(start), calendar)
        end_period = _period_for_date(_local_date(end_probe), calendar)
    except (OverflowError, OSError, ValueError):
        return result
    if start_period is None or end_period is None:
        return result
    if start_period != end_period:
        result["calendar_reactor_state"] = "mixed"
        result["calendar_period"] = "mixed"
        return result

    state, cycle, period = start_period
    result["calendar_cycle"] = cycle
    result["calendar_reactor_state"] = state
    result["calendar_period"] = period
    return result


_CATALOG_SQL = """
WITH file_summary AS (
    SELECT
        rfl.run_id,
        COUNT(df.id) AS file_count,
        COUNT(df.live_time) AS files_with_live_time,
        MIN(df.start_time) AS start_time,
        MAX(CASE
            WHEN df.start_time IS NOT NULL AND df.live_time IS NOT NULL
            THEN df.start_time + df.live_time
        END) AS end_time,
        SUM(df.live_time) AS total_live_time,
        MIN(df.run_number) AS first_run_number,
        MAX(df.run_number) AS last_run_number
    FROM run_file_list AS rfl
    JOIN datafile AS df ON df.id = rfl.file_id
    GROUP BY rfl.run_id
)
SELECT
    r.id AS run_id,
    r.name AS run_name,
    r.description AS run_description,
    r.detector_configuration AS detector_configuration_id,
    r.detector_coordinates AS detector_coordinates_id,

    COALESCE(fs.file_count, 0) AS file_count,
    COALESCE(fs.files_with_live_time, 0) AS files_with_live_time,
    fs.start_time AS start_time,
    fs.end_time AS end_time,
    COALESCE(fs.total_live_time, 0.0) AS total_live_time,
    fs.first_run_number AS first_run_number,
    fs.last_run_number AS last_run_number,

    coord.Rx AS coordinate_Rx,
    coord.Rz AS coordinate_Rz,
    coord.Lx AS coordinate_Lx,
    coord.Lz AS coordinate_Lz,
    coord.angle AS orientation_angle,
    coord.track AS coordinate_track,

    dc.detector AS detector_id,
    det.type AS detector_type,
    det.description AS detector_description,
    dc.detector_settings AS detector_settings_id,
    ds.bias AS detector_bias,
    dc.acquisition_settings AS acquisition_settings_id,
    acq.coarse_gain AS acquisition_coarse_gain,
    acq.PUR_guard AS acquisition_PUR_guard,
    acq.offset AS acquisition_offset,
    acq.fine_gain AS acquisition_fine_gain,
    acq.LLD AS acquisition_LLD,
    acq.LTC_mode AS acquisition_LTC_mode,
    acq.memory_group AS acquisition_memory_group,
    dc.shield AS shield_id,
    shield.name AS shield_name,
    shield.description AS shield_description
FROM runs AS r
LEFT JOIN file_summary AS fs ON fs.run_id = r.id
LEFT JOIN detector_coordinates AS coord ON coord.id = r.detector_coordinates
LEFT JOIN detector_configuration AS dc ON dc.id = r.detector_configuration
LEFT JOIN detector AS det ON det.id = dc.detector
LEFT JOIN detector_settings AS ds ON ds.id = dc.detector_settings
LEFT JOIN acquisition_settings AS acq ON acq.id = dc.acquisition_settings
LEFT JOIN shield_configuration AS shield ON shield.id = dc.shield
ORDER BY r.id
"""


def _resolve_db_path(db_path: os.PathLike[str] | str | None) -> Path:
    value = db_path if db_path is not None else os.environ.get("HFIRBG_CALDB")
    if not value:
        raise RuntimeError("Pass db_path or set HFIRBG_CALDB to the canonical SQLite database")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HFIR background database not found: {path}")
    return path


def build_run_catalog(
    db_path: os.PathLike[str] | str | None = None,
    cycle_calendar_path: os.PathLike[str] | str | None = None,
) -> pd.DataFrame:
    """Query the canonical SQLite database and return one row per public run.

    SQLite is opened with ``mode=ro`` so this function cannot alter the source
    database.  ``start_time`` and ``end_time`` retain the database's Unix-time
    representation; the original run and joined metadata fields are likewise
    preserved without recoding.
    """

    path = _resolve_db_path(db_path)
    calendar = load_cycle_calendar(cycle_calendar_path)
    uri = f"{path.as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        catalog = pd.read_sql_query(_CATALOG_SQL, connection)

    catalog.insert(
        3,
        "reactor_cycle",
        [
            derive_reactor_cycle(name, description)
            for name, description in zip(
                catalog["run_name"], catalog["run_description"]
            )
        ],
    )
    catalog.insert(
        4,
        "reactor_state",
        [
            derive_reactor_state(name, description)
            for name, description in zip(
                catalog["run_name"], catalog["run_description"]
            )
        ],
    )
    calendar_metadata = pd.DataFrame(
        [
            classify_run_interval(start, end, calendar)
            for start, end in zip(catalog["start_time"], catalog["end_time"])
        ],
        index=catalog.index,
    )
    insert_at = 5
    for column in calendar_metadata.columns:
        catalog.insert(insert_at, column, calendar_metadata[column])
        insert_at += 1
    return catalog


# A discoverable verb for callers that think of the operation as loading data.
load_run_catalog = build_run_catalog
