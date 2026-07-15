#!/usr/bin/env python3
"""Inspect or run the supported paper-figure reproduction workflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "config" / "paper_figures.json"
DEFAULT_PAPER_SOURCE = REPO_ROOT / "data" / "arxiv_2607.05834"
DEFAULT_OUTPUT = Path(os.environ.get("HFIRBG_ANALYSIS", REPO_ROOT / "analysis")) / "paper_figures"
RUNNABLE_STATUSES = {"reproducible", "published-ancillary"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="list all figure statuses")
    selection.add_argument("--figure", type=int, help="process one paper figure")
    selection.add_argument("--all", action="store_true", help="process all paper figures")
    parser.add_argument("--dry-run", action="store_true", help="print actions without changing files")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--paper-source", type=Path, default=DEFAULT_PAPER_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / ".venv" / "bin" / "python"),
        help="Python executable for reproduction commands (default: .venv/bin/python)",
    )
    return parser.parse_args(argv)


def load_manifest(path: Path):
    with path.expanduser().open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    figures = manifest.get("figures", [])
    expected = list(range(1, manifest["paper"]["figure_count"] + 1))
    actual = [figure.get("number") for figure in figures]
    if actual != expected:
        raise ValueError(f"manifest figure numbers must be {expected}; found {actual}")
    labels = [figure.get("label") for figure in figures]
    if len(labels) != len(set(labels)):
        raise ValueError("manifest figure labels must be unique")
    known_statuses = set(manifest["statuses"])
    unknown = {figure.get("status") for figure in figures} - known_statuses
    if unknown:
        raise ValueError(f"manifest contains unknown statuses: {sorted(unknown)}")
    return manifest


def print_listing(manifest):
    for figure in manifest["figures"]:
        command_marker = "run" if figure.get("command") else "copy"
        print(
            f"{figure['number']:>2}  {figure['status']:<24} "
            f"{command_marker:<4}  {figure['title']}"
        )


def formatted_command(figure, args, output_dir: Path):
    command = figure.get("command")
    if not command:
        return None
    replacements = {
        "python": str(Path(args.python).expanduser()),
        "output_dir": str(output_dir),
        "repo_root": str(REPO_ROOT),
        "paper_source": str(args.paper_source.expanduser().resolve()),
    }
    return [part.format(**replacements) for part in command]


def copy_paper_artifact(figure, paper_source: Path, output_dir: Path, dry_run=False):
    relative = figure.get("paper_artifact")
    if not relative:
        return None
    source = paper_source / relative
    suffix = source.suffix or ".dat"
    destination = output_dir / "published" / f"figure_{figure['number']:02d}_{figure['label']}{suffix}"
    if dry_run:
        print(f"  COPY {source} -> {destination}")
        return destination
    if not source.is_file():
        raise FileNotFoundError(
            f"paper artifact not found: {source}\nRun scripts/setup_analysis.sh first."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"  Copied published artifact: {destination}")
    return destination


def process_figure(figure, args, output_dir: Path):
    print(f"Figure {figure['number']}: {figure['title']} [{figure['status']}]")
    copy_paper_artifact(
        figure,
        args.paper_source.expanduser().resolve(),
        output_dir,
        dry_run=args.dry_run,
    )
    command = formatted_command(figure, args, output_dir)
    if not command:
        print(f"  No supported recalculation/replot command. {figure['notes']}")
        return "copied"
    if figure["status"] not in RUNNABLE_STATUSES:
        raise ValueError(
            f"figure {figure['number']} has a command but non-runnable status {figure['status']}"
        )
    print("  RUN " + " ".join(command))
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return "ran"


def main(argv=None):
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.list:
        print_listing(manifest)
        return 0

    if args.figure is not None:
        if not 1 <= args.figure <= manifest["paper"]["figure_count"]:
            raise SystemExit(
                f"--figure must be between 1 and {manifest['paper']['figure_count']}"
            )
        figures = [manifest["figures"][args.figure - 1]]
    else:
        figures = manifest["figures"]

    output_dir = args.output_dir.expanduser().resolve()
    results = {"ran": 0, "copied": 0}
    for figure in figures:
        result = process_figure(figure, args, output_dir)
        results[result] += 1
    print(
        f"Processed {len(figures)} figure(s): {results['ran']} recalculation/replot command(s), "
        f"{results['copied']} published/source artifact(s) only."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
