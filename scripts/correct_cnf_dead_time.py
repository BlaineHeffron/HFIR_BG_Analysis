#!/usr/bin/env python3
"""Make local v1.2.2 from v1.2.1 using the CNF aggregate timing survey."""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics as stats
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.correct_cnf_timing import digest
from src.spectrum_names import SPECTRUM_RENAMES


def fit_normal(measured):
    # Observed tau=(real-live)/counts; non-paralyzable dead fraction is tau*r.
    x = [1 / r for r, tau in measured]
    y = [tau for r, tau in measured]
    mx, my = stats.mean(x), stats.mean(y)
    slope = sum((a-mx)*(b-my) for a, b in zip(x, y)) / sum((a-mx)**2 for a in x)
    return my - slope*mx, slope


def fit_dead_fraction(measured):
    x = [r for r, tau in measured]
    y = [r*tau for r, tau in measured]
    mx, my = stats.mean(x), stats.mean(y)
    slope = sum((a-mx)*(b-my) for a, b in zip(x, y)) / sum((a-mx)**2 for a in x)
    return my-slope*mx, slope


def build(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('destination exists; refusing to replace a database version')
    evidence = [json.loads(line) for line in (source/'cnf.jsonl').open()]
    with sqlite3.connect(f'{(source/"HFIRBG.db").as_uri()}?mode=ro', uri=True) as db:
        by_name = {name.removesuffix('.txt'): (fid, live, real) for fid, name, live, real
                   in db.execute('SELECT id,name,live_time,real_time FROM datafile')}
        settings = {fid: (acq, ltc) for fid, acq, ltc in db.execute('''
            SELECT DISTINCT rf.file_id, dc.acquisition_settings, a.LTC_mode
            FROM run_file_list rf JOIN runs r ON r.id=rf.run_id
            JOIN detector_configuration dc ON dc.id=r.detector_configuration
            JOIN acquisition_settings a ON a.id=dc.acquisition_settings''')}
    normal, max_source = [], []
    records = []
    for item in evidence:
        slot = item['chosen_record']
        if not slot:
            continue
        t = item['timing'][slot-1]
        if not t['valid'] or not item.get('total_counts'):
            continue
        name = SPECTRUM_RENAMES.get(Path(item['path']).stem, Path(item['path']).stem)
        file_id = by_name[name][0] if name in by_name else None
        setting, ltc = settings.get(file_id, (None, None))
        fields = {f['name']: f['native_value'] for f in item['fields']}
        gain = fields['gain_product_term'] * fields['fine_gain']
        counts, real = item['total_counts'], t['real_s']
        if real <= 0:
            continue
        rate = counts/real
        records.append((name, setting, ltc, gain, rate, real, t['live_s']))
        if t['live_s'] < real - 1e-5:
            if setting == 5:
                normal.append((rate, (real-t['live_s'])/counts))
            elif gain > 300 and 'STRONG_AM_SOURCE_MAXGAIN' in name:
                max_source.append(1-t['live_s']/real)
    tau0, baseline = fit_normal(normal)
    alternate_baseline, alternate_tau = fit_dead_fraction(normal)
    normal_sigma = stats.stdev([tau-(tau0+baseline/r) for r, tau in normal])
    maximum_measured_rate = max(r for r, tau in normal)
    max_dead = stats.mean(max_source)
    max_sigma = stats.stdev(max_source)
    source_digest = digest(source/'HFIRBG.db')
    changes = []
    for name, setting, ltc, gain, rate, real, native_live in records:
        if name not in by_name:
            # The seven CNFs absent from the released database remain survey evidence.
            continue
        fid, old, db_real = by_name[name]
        if db_real is None or abs(db_real-real) > 1e-4 or abs(old-native_live) > 1e-4:
            raise ValueError(f'{name}: source database disagrees with selected CNF')
        if native_live < real - 1e-5:
            continue  # Directly measured live time is authoritative.
        flags = []
        if setting == 5:
            model, tau = 'setting-5-measured', tau0 + baseline/rate
            dead = rate*tau
            sigma_dead = math.hypot(rate*normal_sigma,
                                    alternate_baseline+alternate_tau*rate-dead, 0.1*dead)
            if rate > maximum_measured_rate:
                flags.append('extrapolated-rate')
        else:
            # No transferable tau was measured at these acquisition settings.
            # The measured rate law is the central estimate; uncertainty covers
            # its entire correction (and the disputed max-gain Am fraction).
            model = f'setting-{setting}-transfer' if setting is not None else 'unassigned-setting-transfer'
            tau = tau0 + baseline/rate
            dead = rate*tau
            sigma_dead = max(dead, abs(max_dead-dead)) if gain > 300 else dead
            flags.append('unmeasured-contested-gain' if gain > 300 else 'unmeasured-setting')
            if rate > maximum_measured_rate:
                flags.append('extrapolated-rate')
        if not 0 < dead < 1:
            raise ValueError(f'{name}: invalid modeled dead fraction {dead}')
        if dead > 0.05:
            flags.append('correction-over-5-percent')
        new = real*(1-dead)
        changes.append(dict(file_id=fid, file_name=name, old_live_s=old, new_live_s=new,
                            real_s=real, recorded_rate_hz=rate, acquisition_settings_id=setting,
                            ltc_mode=ltc, gain_product=gain, tau_s=tau,
                            model_class=model, model_method='setting-5 rate-law fit' if setting == 5 else 'flagged setting-5 transfer',
                            live_sigma_s=real*sigma_dead,
                            dead_time_rate_change=old/new-1,
                            flags=';'.join(flags)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cnf-dead-time-', dir=destination.parent) as tmp:
        out = Path(tmp)
        shutil.copytree(source, out, dirs_exist_ok=True, symlinks=True)
        with sqlite3.connect(out/'HFIRBG.db') as db:
            db.executemany('UPDATE datafile SET live_time=? WHERE id=?',
                           [(r['new_live_s'], r['file_id']) for r in changes])
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('corrected database integrity failure')
        with (out/'corrections.csv').open('w', newline='') as stream:
            old_rows = list(csv.DictReader((source/'corrections.csv').open()))
            old_fields = list(old_rows[0])
            new_fields = [k for k in changes[0] if k not in old_fields]
            writer = csv.DictWriter(stream, fieldnames=old_fields+new_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerows(changes)
        summary = json.loads((source/'summary.json').read_text())
        summary.update(source_database=str(source/'HFIRBG.db'), source_sha256=source_digest,
                       corrected_sha256=digest(out/'HFIRBG.db'),
                       dead_time_model=dict(normal_tau0_s=tau0, normal_baseline_fraction=baseline,
                                            normal_tau_sigma_s=normal_sigma, normal_measured=len(normal),
                                            normal_alternate_baseline_fraction=alternate_baseline,
                                            normal_alternate_tau_s=alternate_tau,
                                            normal_max_measured_rate_hz=maximum_measured_rate,
                                            max_gain_strong_Am_fraction=max_dead,
                                            max_gain_strong_Am_sigma=max_sigma,
                                            max_gain_strong_Am_measured=len(max_source),
                                            unmeasured_central='normal-gain rate law',
                                            unmeasured_minimum_fractional_sigma=1),
                       dead_time_corrected=len(changes), dead_time_flagged=sum(bool(r['flags']) for r in changes),
                       dead_time_flags=[{'file_id': r['file_id'], 'flags': r['flags']}
                                        for r in changes if r['flags']])
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        if digest(source/'HFIRBG.db') != source_digest:
            raise ValueError('source database changed during correction')
        out.rename(destination)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/HFIRBG_public_data_v1.2.1'))
    parser.add_argument('--destination', type=Path, default=Path('data/HFIRBG_public_data_v1.2.2'))
    args = parser.parse_args()
    result = build(args.source, args.destination)
    print(json.dumps({k: result[k] for k in ('dead_time_corrected', 'dead_time_flagged')}))
