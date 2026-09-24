#!/usr/bin/env python3
"""Write GeCollimatorUnfolder inputs from browser.py (DB calibration and live time)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.public_data.browser import load_spectrum  # noqa: E402

FILES = {
    109: "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN",
    871: "MIF_BOX_AT_REACTOR_RXOFF",
    186: "CYCLE491_DOWN_FACING_OVERNIGHT",
    402: "HB4_DOWN_OVERNIGHT_1",
    655: "EAST_FACE_18",
    1484: "EAST_FACE_1",
    1561: "PROSPECT_DOWN_OVERNIGHT",
    1078: "NE_FACING_EAST",
}
ALIASES = {"CYCLE491_DOWN_FACING_OVERNIGHT": "CYCLE461_DOWN_FACING_OVERNIGHT"}


def compile_writer(src: Path, dest: Path) -> Path:
    if dest.exists():
        return dest
    subprocess.run(
        ["g++", "-O2", "-o", str(dest), str(src)]
        + subprocess.check_output(["root-config", "--cflags", "--libs"], text=True).split(),
        check=True,
    )
    return dest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = compile_writer(
        REPO / "scripts" / "write_ge_hist.cpp", out / "write_ge_hist"
    )
    for file_id, name in FILES.items():
        spec = load_spectrum(file_id, db_path=args.db)
        payload = (
            f"{name} {spec.live_time:.16g} {spec.counts.size} "
            f"{spec.calibration_A0:.16g} {spec.calibration_A1:.16g}\n"
        )
        payload += " ".join(f"{c:.16g}" for c in spec.counts) + "\n"
        dest = out / f"{name}.root"
        subprocess.run([str(writer), str(dest)], input=payload.encode(), check=True)
        print(
            file_id,
            spec.file_name,
            spec.live_time,
            spec.calibration_A0,
            spec.calibration_A1,
            spec.counts.size,
        )
    for src, alias in ALIASES.items():
        link = out / f"{alias}.root"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(src + ".root")


if __name__ == "__main__":
    main()
