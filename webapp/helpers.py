"""Pure helpers shared by the Streamlit browser and its tests."""

from __future__ import annotations

from math import atan2, cos, degrees, pi, sin

import pandas as pd


DEFAULT_MAP_Z_RANGE = (-25.0, 425.0)
DEFAULT_MAP_X_RANGE = (180.0, -5.0)
ORIENTATION_ARROW_LENGTH_IN = 12.0
DOWNWARD_TILT_TOLERANCE_DEGREES = 1e-6


def filter_catalog(
    catalog: pd.DataFrame,
    *,
    cycles: list[str] | None = None,
    states: list[str] | None = None,
    shields: list[str] | None = None,
    text: str = "",
    coordinate_id: int | None = None,
) -> pd.DataFrame:
    """Apply public-browser filters without modifying the input catalog."""

    result = catalog.copy()
    for column, values in (
        ("calendar_cycle", cycles),
        ("calendar_reactor_state", states),
        ("shield_name", shields),
    ):
        if values:
            result = result[result[column].fillna("unknown").astype(str).isin(values)]
    query = text.strip()
    if query:
        haystack = (
            result["run_name"].fillna("").astype(str)
            + " "
            + result["run_description"].fillna("").astype(str)
        )
        result = result[haystack.str.contains(query, case=False, regex=False)]
    if coordinate_id is not None:
        result = result[result["detector_coordinates_id"] == coordinate_id]
    return result.reset_index(drop=True)


def cart_azimuth_degrees(rx: float, rz: float, lx: float, lz: float) -> float:
    """Return the released cart azimuth convention (0 degrees faces west)."""

    return degrees(atan2(lx - rx, rz - lz)) % 360.0


def detector_face_position(
    rx: float, rz: float, lx: float, lz: float, tilt_degrees: float
) -> tuple[float, float]:
    """Convert released cart corners to detector-face ``(z, x)`` coordinates."""

    phi = atan2(lx - rx, rz - lz)
    axis_x = rx + sin(phi) * 8.8 + cos(phi) * 16.0
    axis_z = rz - cos(phi) * 8.8 + sin(phi) * 16.0
    face_length = sin(pi * tilt_degrees / 180.0) * 8.0
    return (
        axis_z + face_length * cos(phi),
        axis_x - face_length * sin(phi),
    )


def orientation_arrow_vector(
    rx: float,
    rz: float,
    lx: float,
    lz: float,
    *,
    length_in: float = ORIENTATION_ARROW_LENGTH_IN,
) -> tuple[float, float]:
    """Return the plotted ``(dz, dx)`` direction for a tilted detector.

    This is the same cart-azimuth geometry used by the paper-diagram scripts:
    ``dz = L cos(phi)`` and ``dx = -L sin(phi)``, where ``phi`` comes from
    the released right/left cart-corner coordinates.  The caller places the
    tail at the calculated detector-face center.
    """

    phi = atan2(lx - rx, rz - lz)
    return length_in * cos(phi), -length_in * sin(phi)


def is_downward_facing(tilt_degrees: float) -> bool:
    """Return whether a detector is vertically downward-facing on the map."""

    return abs(float(tilt_degrees)) <= DOWNWARD_TILT_TOLERANCE_DEGREES


def location_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return one map row per coordinate record represented by filtered runs."""

    columns = [
        "detector_coordinates_id", "coordinate_Rx", "coordinate_Rz",
        "coordinate_Lx", "coordinate_Lz", "orientation_angle",
    ]
    rows = catalog.dropna(subset=columns).groupby(columns, dropna=False).agg(
        run_count=("run_id", "nunique"), file_count=("file_count", "sum")
    ).reset_index()
    if rows.empty:
        return rows
    rows["cart_azimuth"] = rows.apply(
        lambda row: cart_azimuth_degrees(
            row.coordinate_Rx, row.coordinate_Rz, row.coordinate_Lx, row.coordinate_Lz
        ), axis=1
    )
    positions = rows.apply(
        lambda row: detector_face_position(
            row.coordinate_Rx, row.coordinate_Rz, row.coordinate_Lx,
            row.coordinate_Lz, row.orientation_angle
        ), axis=1
    )
    rows["map_z"] = [position[0] for position in positions]
    rows["map_x"] = [position[1] for position in positions]
    directions = rows.apply(
        lambda row: orientation_arrow_vector(
            row.coordinate_Rx, row.coordinate_Rz,
            row.coordinate_Lx, row.coordinate_Lz,
        ), axis=1
    )
    rows["map_direction_dz"] = [direction[0] for direction in directions]
    rows["map_direction_dx"] = [direction[1] for direction in directions]
    rows["map_direction_end_z"] = rows["map_z"] + rows["map_direction_dz"]
    rows["map_direction_end_x"] = rows["map_x"] + rows["map_direction_dx"]
    rows["downward_facing"] = rows["orientation_angle"].map(is_downward_facing)
    return rows


def measurement_map_ranges(
    locations: pd.DataFrame,
    *,
    minimum_z_span: float = 100.0,
    minimum_x_span: float = 80.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return padded Plotly ranges focused on the currently mapped locations.

    Detector-face positions and the ends of the orientation arrows are
    included so visual orientation cues remain visible. The x range is
    reversed to preserve the orientation of the published hall schematic.
    """

    required = {
        "map_z", "map_x", "map_direction_end_z", "map_direction_end_x",
    }
    if locations.empty or not required.issubset(locations.columns):
        return DEFAULT_MAP_Z_RANGE, DEFAULT_MAP_X_RANGE

    def limits(columns: tuple[str, ...], minimum_span: float) -> tuple[float, float]:
        values = pd.concat(
            [pd.to_numeric(locations[column], errors="coerce") for column in columns],
            ignore_index=True,
        ).dropna()
        if values.empty:
            raise ValueError("No finite map coordinates")
        low, high = float(values.min()), float(values.max())
        span = high - low
        center = (low + high) / 2.0
        half_span = max(minimum_span / 2.0, span * 0.56)
        return center - half_span, center + half_span

    try:
        z_low, z_high = limits(("map_z", "map_direction_end_z"), minimum_z_span)
        x_low, x_high = limits(("map_x", "map_direction_end_x"), minimum_x_span)
    except ValueError:
        return DEFAULT_MAP_Z_RANGE, DEFAULT_MAP_X_RANGE
    return (z_low, z_high), (x_high, x_low)
