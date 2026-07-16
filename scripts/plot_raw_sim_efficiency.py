#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    run = root.find(".//Run")
    generator = root.find(".//PrimaryGenerator")
    if run is None or generator is None:
        raise RuntimeError(f"Missing Run or PrimaryGenerator in {xml_path}")

    dt = float(run.attrib.get("dt", 0.0))
    throws = int(generator.attrib.get("throws", 0))
    direction = generator.find(".//LeadCollimatorThrower")
    direction_name = direction.attrib.get("direction") if direction is not None else "unknown"
    energy_mev = float(direction.attrib.get("energy", "0 MeV").split()[0]) if direction is not None else np.nan
    return {
        "wall_time_s": dt,
        "throws": throws,
        "direction": direction_name,
        "energy_mev": energy_mev,
    }


def count_detected_events(h5_path):
    with h5py.File(h5_path, "r") as handle:
        evt = handle["ioni"]["evt"][:]
        return int(np.unique(evt).size), int(len(evt))


def collect_rows(directory):
    rows = []
    directory = Path(directory)
    for xml_path in sorted(directory.glob("*.h5.xml")):
        h5_path = xml_path.with_suffix("")
        if not h5_path.exists():
            continue
        meta = parse_xml(xml_path)
        detected_events, deposit_rows = count_detected_events(h5_path)
        rows.append(
            {
                "file": xml_path.stem,
                "energy_mev": meta["energy_mev"],
                "direction": meta["direction"],
                "wall_time_s": meta["wall_time_s"],
                "throws": meta["throws"],
                "detected_events": detected_events,
                "deposit_rows": deposit_rows,
                "detected_per_million_throws": detected_events / meta["throws"] * 1e6 if meta["throws"] else np.nan,
                "wall_s_per_1k_detected": meta["wall_time_s"] / detected_events * 1000 if detected_events else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_plot(df, out_path):
    df = df.sort_values(["energy_mev", "direction"]).copy()
    df["label"] = df.apply(lambda r: f"{int(r.energy_mev)} MeV\n{r.direction}", axis=1)
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)

    axes[0].bar(x, df["detected_per_million_throws"], color=["#1f77b4" if d == "none" else "#d62728" for d in df["direction"]])
    axes[0].set_ylabel("Detected events\nper 1e6 throws")
    axes[0].set_title("Raw simulation comparison from pre-analysis HDF5 outputs")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, df["wall_s_per_1k_detected"], color=["#1f77b4" if d == "none" else "#d62728" for d in df["direction"]])
    axes[1].set_ylabel("Wall seconds\nper 1000 detected events")
    axes[1].set_xticks(x, df["label"])
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input-dir", default="analysis/response/energy_response")
    parser.add_argument("--out-csv", default="analysis/unfold/paper_files/raw_sim_efficiency_comparison.csv")
    parser.add_argument("--out-plot", default="analysis/unfold/paper_files/raw_sim_efficiency_comparison.png")
    args = parser.parse_args()

    df = collect_rows(args.input_dir)
    if df.empty:
        raise RuntimeError(f"No matching raw sim files found in {args.input_dir}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    save_plot(df, Path(args.out_plot))
    print(f"Saved {out_csv}")
    print(f"Saved {Path(args.out_plot)}")


if __name__ == "__main__":
    main()
