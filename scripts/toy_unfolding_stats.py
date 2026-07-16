#!/usr/bin/env python3
import argparse
import csv
import math
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ROOT
import uproot

ROOT.gROOT.SetBatch(True)

MEASUREMENTS = {
    "MIF_BOX_REACTOR_OPTIMIZED_DAYCOUNT_OPTIMAL_GAIN": "MIF (Rx On)",
    "MIF_BOX_AT_REACTOR_RXOFF": "MIF (Rx Off)",
    "HB4_DOWN_OVERNIGHT_1": "HB4",
}

CASES = {
    "rl_isotropic": {
        "algorithm": "Richardson-Lucy",
        "response": "isotropic",
        "class": "GeCollimatorUnfolder",
        "migration": "/home/blaine/projects/HFIR_BG_Analysis/scripts/private/migration_matrix.root",
    },
    "rl_front": {
        "algorithm": "Richardson-Lucy",
        "response": "front",
        "class": "GeCollimatorUnfolder",
        "migration": "/home/blaine/projects/HFIR_BG_Analysis/scripts/private/migration_matrix_front.root",
    },
    "pgd_isotropic": {
        "algorithm": "Poisson-PGD",
        "response": "isotropic",
        "class": "GeCollimatorUnfolderAlt",
        "migration": "/home/blaine/projects/HFIR_BG_Analysis/scripts/private/migration_matrix.root",
        "deltaChiSqr": 0.005,
        "regularization": 1e-4,
        "stepSize": 1.0,
        "maxIterations": 4000,
    },
    "pgd_front": {
        "algorithm": "Poisson-PGD",
        "response": "front",
        "class": "GeCollimatorUnfolderAlt",
        "migration": "/home/blaine/projects/HFIR_BG_Analysis/scripts/private/migration_matrix_front.root",
        "deltaChiSqr": 0.005,
        "regularization": 1e-4,
        "stepSize": 1.0,
        "maxIterations": 4000,
    },
}

P2X_BIN = "/home/blaine/src/P2x/bin/P2x_Analyze"
DATA_DIR = Path('/home/blaine/projects/HFIRBG/data')
OUTDIR = Path('/home/blaine/projects/HFIR_BG_Analysis/analysis/unfold/toy_stats')


def low_fraction(result_path: Path) -> tuple[float, float, float]:
    with uproot.open(result_path) as f:
        vals, edges = f['UnfoldedEnergy'].to_numpy()
        meas, _ = f['Measured'].to_numpy()
        refold, _ = f['FoldedBack'].to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    low_mask = (centers >= 50.0) & (centers < 2000.0)
    tot_mask = (centers >= 50.0) & (centers < 11500.0)
    low = float(np.sum(vals[low_mask] * widths[low_mask]))
    total = float(np.sum(vals[tot_mask] * widths[tot_mask]))
    denom = np.where(refold > 0.0, refold, 1.0)
    mask = refold > 0.0
    chi2 = float(np.sum(((meas[mask] - refold[mask]) ** 2) / denom[mask])) if np.any(mask) else float('nan')
    return low, total, low / total if total > 0.0 else float('nan'), chi2


def make_toy_input(source_path: Path, toy_path: Path, rng: np.random.Generator) -> None:
    src = ROOT.TFile.Open(str(source_path), 'READ')
    if not src or src.IsZombie():
        raise RuntimeError(f'failed to open {source_path}')
    hist = src.Get('GeDataHist')
    live_time = src.Get('LiveTime')
    if not hist or not live_time:
        raise RuntimeError(f'missing GeDataHist or LiveTime in {source_path}')

    toy_path.parent.mkdir(parents=True, exist_ok=True)
    out = ROOT.TFile(str(toy_path), 'RECREATE')
    toy_hist = hist.Clone('GeDataHist')
    toy_hist.SetDirectory(out)
    for i in range(0, toy_hist.GetNbinsX() + 2):
        mean = hist.GetBinContent(i)
        count = int(rng.poisson(max(mean, 0.0)))
        toy_hist.SetBinContent(i, count)
        toy_hist.SetBinError(i, math.sqrt(count))
    toy_hist.Write('GeDataHist')
    live_time.Clone('LiveTime').Write('LiveTime')
    out.Close()
    src.Close()


@dataclass(frozen=True)
class Job:
    measurement: str
    toy_index: int
    case_key: str
    source_input_path: Path
    work_dir: Path


def run_job(job: Job) -> dict:
    cfg = CASES[job.case_key]
    cfg_path = job.work_dir / f'{job.measurement}_{job.toy_index:03d}_{job.case_key}.cfg'
    job_dir = job.work_dir / job.case_key
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / job.source_input_path.name
    if not input_path.exists():
        shutil.copy2(job.source_input_path, input_path)
    output_path = input_path.with_name(input_path.stem + '_unfold_results.root')
    if output_path.exists():
        output_path.unlink()

    lines = [
        f'class: "{cfg["class"]}"',
        f'migrationFile: "{cfg["migration"]}"',
        f'inputHist: "{input_path}"',
        'EHigh: 12000',
        'ELow: 40',
    ]
    if cfg['class'] == 'GeCollimatorUnfolderAlt':
        lines.extend([
            f'deltaChiSqr: {cfg["deltaChiSqr"]}',
            f'maxIterations: {cfg["maxIterations"]}',
            f'regularization: {cfg["regularization"]}',
            f'stepSize: {cfg["stepSize"]}',
        ])
    cfg_path.write_text('\n'.join(lines) + '\n')

    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '1'
    proc = subprocess.run([P2X_BIN, str(cfg_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f'job failed for {job}:\n{proc.stdout}')
    if not output_path.exists():
        raise RuntimeError(f'missing output root file for {job}:\n{proc.stdout}')

    low, total, frac, chi2 = low_fraction(output_path)
    return {
        'measurement': job.measurement,
        'measurement_label': MEASUREMENTS[job.measurement],
        'toy_index': job.toy_index,
        'case_key': job.case_key,
        'algorithm': cfg['algorithm'],
        'response': cfg['response'],
        'low_flux': low,
        'total_flux': total,
        'low_fraction': frac,
        'refold_chi2': chi2,
        'root_file': str(output_path),
    }


def summarize(values: list[float]) -> tuple[float, float, float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, float(np.quantile(arr, 0.16)), float(np.quantile(arr, 0.84))


def paired_summary(rows: list[dict], measurement: str, lhs_alg: str, lhs_resp: str, rhs_alg: str, rhs_resp: str, metric: str) -> dict:
    lhs = {(r['toy_index']): float(r[metric]) for r in rows if r['measurement'] == measurement and r['algorithm'] == lhs_alg and r['response'] == lhs_resp}
    rhs = {(r['toy_index']): float(r[metric]) for r in rows if r['measurement'] == measurement and r['algorithm'] == rhs_alg and r['response'] == rhs_resp}
    common = sorted(set(lhs) & set(rhs))
    diffs = [lhs[i] - rhs[i] for i in common]
    mean, std, q16, q84 = summarize(diffs)
    return {
        'measurement': MEASUREMENTS[measurement],
        'lhs': f'{lhs_alg} {lhs_resp}',
        'rhs': f'{rhs_alg} {rhs_resp}',
        'metric': metric,
        'n_toys': len(common),
        'mean_diff': mean,
        'std_diff': std,
        'q16_diff': q16,
        'q84_diff': q84,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Toy-MC statistical comparison for Ge unfolding algorithms.')
    parser.add_argument('--n-toys', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260416)
    parser.add_argument('--measurements', nargs='*', default=list(MEASUREMENTS.keys()))
    parser.add_argument('--keep-work', action='store_true')
    args = parser.parse_args()

    outdir = OUTDIR
    workdir = outdir / 'work'
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    toy_inputs: list[Job] = []
    selected_measurements = []
    for measurement in args.measurements:
        if measurement not in MEASUREMENTS:
            raise SystemExit(f'unknown measurement: {measurement}')
        selected_measurements.append(measurement)

    for measurement in selected_measurements:
        source_path = DATA_DIR / f'{measurement}.root'
        for toy_index in range(args.n_toys):
            toy_dir = workdir / measurement / f'toy_{toy_index:03d}'
            toy_input = toy_dir / f'{measurement}_toy.root'
            make_toy_input(source_path, toy_input, rng)
            for case_key in CASES:
                toy_inputs.append(Job(measurement, toy_index, case_key, toy_input, toy_dir))

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_job, job): job for job in toy_inputs}
        for idx, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            row = future.result()
            rows.append(row)
            print(f'[{idx}/{len(futures)}] {job.measurement} toy {job.toy_index:03d} {job.case_key} low_fraction={row["low_fraction"]:.4f} chi2={row["refold_chi2"]:.2f}')

    rows.sort(key=lambda r: (r['measurement'], r['toy_index'], r['algorithm'], r['response']))
    csv_path = outdir / 'toy_unfold_results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    paired_rows = []
    for measurement in selected_measurements:
        paired_rows.append(paired_summary(rows, measurement, 'Poisson-PGD', 'isotropic', 'Richardson-Lucy', 'isotropic', 'low_fraction'))
        paired_rows.append(paired_summary(rows, measurement, 'Poisson-PGD', 'front', 'Richardson-Lucy', 'front', 'low_fraction'))
        paired_rows.append(paired_summary(rows, measurement, 'Richardson-Lucy', 'front', 'Richardson-Lucy', 'isotropic', 'low_fraction'))
        paired_rows.append(paired_summary(rows, measurement, 'Poisson-PGD', 'front', 'Poisson-PGD', 'isotropic', 'low_fraction'))

    paired_csv = outdir / 'toy_paired_low_fraction_summary.csv'
    with open(paired_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    md_path = outdir / 'toy_unfold_summary.md'
    with open(md_path, 'w') as f:
        f.write('# Toy-MC unfolding comparison\n\n')
        f.write(f'- Number of toys per measurement: {args.n_toys}\n')
        f.write(f'- Measurements: {", ".join(MEASUREMENTS[m] for m in selected_measurements)}\n')
        f.write('- Each toy Poisson-fluctuates the original `GeDataHist` counts and reruns RL plus Poisson-PGD with both migration matrices.\n')
        f.write('- Summary metric below: paired toy difference in low-energy fraction (50-2000 keV / 50-11500 keV).\n\n')
        f.write('| Measurement | Comparison | Mean diff | Std diff | 16-84% interval |\n')
        f.write('| --- | --- | ---: | ---: | ---: |\n')
        for row in paired_rows:
            f.write(f"| {row['measurement']} | {row['lhs']} - {row['rhs']} | {row['mean_diff']:.4f} | {row['std_diff']:.4f} | [{row['q16_diff']:.4f}, {row['q84_diff']:.4f}] |\n")

    print(f'Wrote {csv_path}')
    print(f'Wrote {paired_csv}')
    print(f'Wrote {md_path}')
    if not args.keep_work:
        shutil.rmtree(workdir)


if __name__ == '__main__':
    main()
