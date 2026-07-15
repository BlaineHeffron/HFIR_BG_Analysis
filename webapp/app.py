"""Interactive, read-only browser for the public HFIR background data."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.public_data.browser import (  # noqa: E402
    load_spectrum, query_file_metadata, rebin_by_factor, resolve_browser_paths,
    spectrum_dataframe,
)
from src.public_data.catalog import build_run_catalog, load_cycle_calendar  # noqa: E402
from webapp.helpers import (  # noqa: E402
    filter_catalog, location_catalog, measurement_map_ranges,
)


st.set_page_config(page_title="HFIR Background Data", page_icon="📈", layout="wide")


@st.cache_data(show_spinner="Loading run catalog…")
def cached_catalog(db_path: str) -> pd.DataFrame:
    return build_run_catalog(db_path)


@st.cache_data(show_spinner="Loading spectrum metadata…")
def cached_files(db_path: str) -> pd.DataFrame:
    return query_file_metadata(db_path)


@st.cache_data(show_spinner=False)
def cached_spectrum(file_id: int, db_path: str, data_root: str | None, factor: int, norm: str) -> pd.DataFrame:
    spectrum = load_spectrum(file_id, db_path=db_path, data_root=data_root)
    frame = spectrum_dataframe(rebin_by_factor(spectrum, factor), norm)
    frame.insert(0, "file_id", file_id)
    frame.insert(1, "run_id", spectrum.run_id)
    frame.insert(2, "run_name", spectrum.run_name)
    frame.insert(3, "file_name", spectrum.file_name)
    frame.insert(4, "live_time_s", spectrum.live_time)
    frame.insert(5, "calibration_A0_keV", spectrum.calibration_A0)
    frame.insert(6, "calibration_A1_keV_per_channel", spectrum.calibration_A1)
    return frame


def schematic(locations: pd.DataFrame) -> go.Figure:
    """Top-down view cropped to the region containing released measurements."""
    fig = go.Figure()
    shapes = [
        dict(type="line", x0=0, x1=420, y0=0, y1=0, line=dict(color="black", width=2)),
        dict(type="rect", x0=165, x1=211.25, y0=44.6, y1=128, line=dict(color="cornflowerblue", width=2)),
        dict(type="rect", x0=125, x1=256.5, y0=7.5, y1=21.5, line=dict(color="gray"), fillcolor="lightgray"),
        dict(type="rect", x0=10, x1=64, y0=7.5, y1=21.5, line=dict(color="gray"), fillcolor="lightgray"),
        dict(type="rect", x0=70, x1=100, y0=0, y1=10, line=dict(color="red")),
    ]
    fig.update_layout(shapes=shapes)
    if not locations.empty:
        # Released right/left cart references show azimuth independently of tilt.
        line_x, line_y = [], []
        for row in locations.itertuples():
            line_x += [row.coordinate_Rz, row.coordinate_Lz, None]
            line_y += [row.coordinate_Rx, row.coordinate_Lx, None]
        fig.add_trace(go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Right↔left cart reference baseline",
            line=dict(color="#587384", width=2),
            hoverinfo="skip",
            showlegend=True,
        ))
        custom = locations[["detector_coordinates_id", "orientation_angle", "cart_azimuth", "run_count", "file_count"]].to_numpy()
        fig.add_trace(go.Scatter(
            x=locations["map_z"], y=locations["map_x"], mode="markers",
            name="Calculated detector-face center",
            marker=dict(size=14, color="#a51c30", line=dict(color="white", width=1)),
            customdata=custom,
            hovertemplate="<b>Detector-face center</b><br>Coordinate %{customdata[0]}<br>z=%{x:.1f} in, x=%{y:.1f} in<br>detector tilt=%{customdata[1]:.1f}°<br>cart azimuth from R↔L baseline=%{customdata[2]:.1f}°<br>%{customdata[3]} runs, %{customdata[4]} files<extra></extra>",
            showlegend=True,
        ))
    fig.add_annotation(x=188, y=100, text="PROSPECT", showarrow=False, font=dict(color="cornflowerblue"))
    fig.add_annotation(x=85, y=5, text="MIF", showarrow=False, font=dict(color="red"))
    z_range, x_range = measurement_map_ranges(locations)
    fig.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=30, b=20),
        dragmode="select",
        clickmode="event+select",
        legend=dict(
            orientation="h",
            x=0.0,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#c8d2d9",
            borderwidth=1,
            font=dict(size=14),
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=15,
            font_family="DejaVu Sans",
            align="left",
        ),
    )
    fig.update_xaxes(title="z [in]", range=list(z_range), constrain="domain")
    fig.update_yaxes(
        title="x [in]", range=list(x_range), scaleanchor="x", scaleratio=1,
        constrain="domain",
    )
    return fig


st.title("HFIR Gamma Background Data Browser")
st.caption("Read-only exploration of released calibrated spectra and metadata; no ROOT installation is used by this app.")

try:
    paths = resolve_browser_paths()
    catalog = cached_catalog(str(paths.db_path))
    files = cached_files(str(paths.db_path))
except Exception as error:
    st.error(f"Public data setup is incomplete: {error}")
    st.code("./scripts/setup_analysis.sh\n./scripts/run_data_browser.sh", language="bash")
    st.info("Download/setup details are also in the repository README. The app never writes to the database.")
    st.stop()

with st.sidebar:
    st.header("Filter runs")
    cycle_options = sorted(catalog["calendar_cycle"].fillna("unknown").astype(str).unique())
    state_options = sorted(catalog["calendar_reactor_state"].fillna("unknown").astype(str).unique())
    shield_options = sorted(catalog["shield_name"].fillna("unknown").astype(str).unique())
    cycles = st.multiselect("Official calendar cycle", cycle_options)
    states = st.multiselect("Official calendar state", state_options)
    shields = st.multiselect("Shield", shield_options)
    search = st.text_input("Run name or description")

filtered = filter_catalog(catalog, cycles=cycles, states=states, shields=shields, text=search)
locations = location_catalog(filtered)

explore, timeline, paper, about = st.tabs(["Explore", "Cycle timeline", "Paper figures", "About / provenance"])

with explore:
    st.warning("Official cycle boundaries have day precision; do not interpret a boundary date as an exact startup or shutdown time. The map is cropped to the released measurement region and omits the reactor area, where this release has no measurements.")
    st.markdown(
        "**How to read the map:** **red circles** are calculated detector-face "
        "centers. Each **gray-blue segment** connects the recorded right and left "
        "cart reference points and determines cart azimuth—it is **not identified "
        "as the cart's front edge**. Detector tilt is a separate value shown on hover."
    )
    selected_coordinate = None
    event = st.plotly_chart(schematic(locations), width="stretch", on_select="rerun", selection_mode="points", key="location_map")
    try:
        points = event.selection.points
        if points:
            selected_coordinate = int(points[0]["customdata"][0])
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        pass
    selector_column, runs_column, locations_column, files_column = st.columns([2, 1, 1, 1])
    with selector_column:
        coordinate_options = [int(value) for value in locations.get("detector_coordinates_id", pd.Series(dtype=int)).tolist()]
        fallback = st.selectbox("Location fallback selector", [None] + coordinate_options, format_func=lambda value: "All mapped locations" if value is None else f"Coordinate {value}")
        if selected_coordinate is None:
            selected_coordinate = fallback
    with runs_column:
        st.metric("Filtered runs", len(filtered))
    with locations_column:
        st.metric("Mapped locations", len(locations))
    with files_column:
        st.metric("Spectrum files", int(filtered["file_count"].sum()))

    selected_runs = filter_catalog(filtered, coordinate_id=selected_coordinate)
    run_columns = ["run_id", "run_name", "run_description", "calendar_cycle", "calendar_reactor_state", "shield_name", "file_count", "start_time", "total_live_time", "detector_coordinates_id"]
    st.subheader("Runs")
    st.dataframe(selected_runs[run_columns], width="stretch", hide_index=True)
    st.download_button("Download filtered run catalog (CSV)", selected_runs.to_csv(index=False), "hfir_filtered_runs.csv", "text/csv")

    selected_files = files[files["run_id"].isin(selected_runs["run_id"])].copy()
    file_columns = ["file_id", "file_name", "run_id", "run_name", "start_time", "live_time", "shield_name", "coordinate_id"]
    st.subheader("Spectrum files")
    st.dataframe(selected_files[file_columns], width="stretch", hide_index=True)
    labels = {int(row.file_id): f"{int(row.file_id)} — {row.run_name} / {row.file_name}" for row in selected_files.itertuples()}
    chosen = st.multiselect("Overlay up to six spectra", list(labels), max_selections=6, format_func=lambda value: labels[value])
    if chosen:
        c1, c2, c3, c4 = st.columns(4)
        normalization = c1.selectbox("Normalization", ["counts/s/keV", "counts/s", "counts"])
        factor = c2.number_input("Rebin factor", 1, 512, 8, 1)
        energy = c3.slider("Energy range [keV]", 0, 12000, (0, 12000), 10)
        log_y = c4.checkbox("Log y", True)
        show_errors = c4.checkbox("Statistical errors", False)
        frames, fig = [], go.Figure()
        for file_id in chosen:
            try:
                frame = cached_spectrum(file_id, str(paths.db_path), str(paths.data_root) if paths.data_root else None, int(factor), normalization)
            except Exception as error:
                st.warning(f"Could not load file {file_id}: {error}")
                continue
            frame = frame[frame["energy_keV"].between(*energy)].copy()
            frames.append(frame)
            fig.add_trace(go.Scatter(x=frame.energy_keV, y=frame.value, mode="lines", name=labels[file_id], error_y=dict(type="data", array=frame.statistical_error, visible=show_errors)))
        if frames:
            fig.update_layout(xaxis_title="Energy [keV]", yaxis_title=normalization, yaxis_type="log" if log_y else "linear", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
            long_form = pd.concat(frames, ignore_index=True)
            st.download_button("Download displayed spectra (long-form CSV)", long_form.to_csv(index=False), "hfir_displayed_spectra.csv", "text/csv")

with timeline:
    dated = filtered[filtered["start_time"].notna()].copy()
    # Five legacy/malformed timestamps are intentionally retained in the
    # canonical catalog. They remain visible in exports but cannot be plotted.
    dated["start"] = pd.to_datetime(
        dated["start_time"], unit="s", utc=True, errors="coerce"
    ).dt.tz_convert("America/New_York")
    dated = dated[dated["start"].notna()]
    fig = go.Figure(go.Scatter(x=dated["start"], y=dated["calendar_period"], mode="markers", customdata=dated[["run_id", "run_name", "file_count"]], hovertemplate="%{customdata[1]}<br>run %{customdata[0]} · %{customdata[2]} files<extra></extra>"))
    fig.update_layout(xaxis_title="Run start (America/New_York)", yaxis_title="Official calendar period", height=600)
    st.plotly_chart(fig, width="stretch")
    st.caption("Calendar assignments are a durable, cited snapshot in reference_data/hfir_cycle_calendar.csv and use day-precision public dates.")

with paper:
    manifest = json.loads((ROOT / "config" / "paper_figures.json").read_text(encoding="utf-8"))
    figure_rows = pd.DataFrame(manifest["figures"])
    st.dataframe(figure_rows[["number", "title", "kind", "status", "notes"]], width="stretch", hide_index=True)
    st.caption("Status meanings")
    st.json(manifest["statuses"], expanded=False)

with about:
    calendar = load_cycle_calendar()
    source = calendar.iloc[0]
    st.markdown(f"""
This browser reads the canonical SQLite database in read-only mode and resolves paths from `HFIRBG_CALDB` and `HFIRBGDATA`; it does not use historical absolute paths.

**Cycle calendar:** [{source['source_title']}]({source['source_url']}) · DOI `{source['source_doi']}` · retrieved `{source['retrieved_date']}`. Dates have **day precision**.

The top-down map reproduces only geometry and coordinate conventions already released with this repository. Red circles are calculated detector-face centers. Gray-blue segments join the recorded right and left cart reference points; they determine cart azimuth and are not identified as a front edge. Detector tilt is a separate quantity shown in point hover text.
""")
