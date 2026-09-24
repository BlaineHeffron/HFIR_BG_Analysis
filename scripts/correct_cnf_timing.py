#!/usr/bin/env python3
"""Build an opt-in local database copy from a compiled CNF survey; never edit source."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# CNF start ticks are Eastern local wall time; this reproduces 1694/1694 v1.1.0 start times from intact records.
ACQUISITION_ZONE = ZoneInfo('America/New_York')


def unix_start(ticks):
    local = datetime(1858, 11, 17) + timedelta(microseconds=ticks//10)
    return local.replace(tzinfo=ACQUISITION_ZONE).timestamp()


def digest(path):
    with path.open('rb') as stream:
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            sha.update(block)
        return sha.hexdigest()


def build(source, survey, destination):
    source, survey, destination = map(lambda p: Path(p).resolve(), (source, survey, destination))
    if destination.exists():
        raise ValueError('destination exists; refusing to replace a database version')
    source_digest = digest(source)
    with sqlite3.connect(f'{source.as_uri()}?mode=ro', uri=True) as original:
        original.row_factory = sqlite3.Row
        files = {r['name'].removesuffix('.txt'): dict(r) for r in original.execute('SELECT * FROM datafile')}
        if len(files) != original.execute('SELECT count(*) FROM datafile').fetchone()[0]:
            raise ValueError('nonunique database filenames')
        rows = [json.loads(line) for line in survey.read_text().splitlines()]
        matched = set()
        table, changes = [], []
        for r in rows:
            name = Path(r['path']).stem
            db = files.get(name)
            slot = r.get('chosen_record', 0)
            if slot not in (0, 1, 2):
                raise ValueError(f'invalid source slot: {name}')
            selected = r['timing'][slot-1] if slot else None
            if selected and (not selected['valid'] or not all(math.isfinite(selected[k]) and selected[k] > 0 for k in ('live_s', 'real_s'))
                             or selected['live_s'] > selected['real_s']):
                raise ValueError(f'invalid selected timing: {name}')
            row = dict(file_id=db['id'] if db else None, file_name=name,
                       db_live_s=db['live_time'] if db else None, db_real_s=None,
                       cnf_record_1=json.dumps(r.get('timing', [None, None])[0]),
                       cnf_record_2=json.dumps(r.get('timing', [None, None])[1]),
                       chosen_record=slot, status=r['status'], rule=r.get('rule', r.get('error')),
                       cnf_live_s=selected['live_s'] if selected else None,
                       cnf_real_s=selected['real_s'] if selected else None,
                       fractional_live_change=None, rate_fractional_change=None,
                       source_sha256=r.get('sha256'), parser_revision=r.get('parser_revision'),
                       db_start=db['start_time'] if db else None,
                       cnf_start=unix_start(selected['start_ticks']) if selected else None,
                       start_corrected=False, correction='none', database_status='unmatched' if not db else 'ambiguous',
                       action='unmatched' if not db else 'ambiguous')
            if db:
                if db['id'] in matched:
                    raise ValueError(f'multiple CNFs for database file {db["id"]}')
                matched.add(db['id'])
                if selected:
                    old, new = db['live_time'], selected['live_s']
                    if old and old > 0:
                        row['fractional_live_change'] = new/old-1
                        row['rate_fractional_change'] = old/new-1
                    row['action'] = 'correct-live' if old is None or abs(old-new)>1e-5 else 'retain-live'
                    row['start_corrected'] = db['start_time'] is None or abs(db['start_time']-row['cnf_start']) > 1.5
                    row['database_status'] = 'consistent' if row['action'] == 'retain-live' else row['action']
                    if row['action'] == 'correct-live':
                        other = r['timing'][2-slot]
                        row['correction'] = ('placeholder' if old is not None and old <= 1 else 'stale-record' if old is not None and any(
                            other.get(k) is not None and abs(old-other[k]) < 1e-5 for k in ('live_s','real_s'))
                            else 'real-stored-as-live' if old is not None and abs(old-selected['real_s']) < 1e-5
                            else 'unexplained')
                    if row['action'] == 'correct-live' or row['start_corrected']:
                        changes.append(row)
            table.append(row)
        if any(r['status'] == 'ambiguous' for r in table):
            raise ValueError('ambiguous CNFs: inspect survey before making any database copy')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.cnf-timing-', dir=destination.parent) as tmp:
            output = Path(tmp)
            with sqlite3.connect(output/'HFIRBG.db') as corrected:
                original.backup(corrected)
                corrected.execute('ALTER TABLE datafile ADD COLUMN real_time REAL')
                for r in table:
                    if r['action'] in ('correct-live', 'retain-live'):
                        corrected.execute('UPDATE datafile SET live_time=?, real_time=?, start_time=? WHERE id=?',
                                          (r['cnf_live_s'] if r['action']=='correct-live' else r['db_live_s'],
                                           r['cnf_real_s'], r['cnf_start'] if r['start_corrected'] else r['db_start'],
                                           r['file_id']))
                corrected.commit()
                if corrected.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('corrected database integrity failure')
            for name, content in [('survey.csv', table), ('corrections.csv', changes)]:
                with (output/name).open('w', newline='') as stream:
                    writer=csv.DictWriter(stream, fieldnames=list(table[0]))
                    writer.writeheader(); writer.writerows(content)
            shutil.copyfile(survey, output/'cnf.jsonl')
            # Preserve relative bundle paths without copying the spectra or changing calibrations.
            for name in ('spectra', 'migration_matrices'):
                if (source.parent/name).exists():
                    (output/name).symlink_to(source.parent/name, target_is_directory=True)
            summary = dict(source_database=str(source), source_sha256=source_digest,
                           corrected_sha256=digest(output/'HFIRBG.db'), survey_sha256=digest(survey),
                           files=len(table), matched=len(matched),
                           corrected_live=sum(r['action']=='correct-live' for r in changes),
                           corrected_start=sum(r['start_corrected'] for r in changes),
                           start_time_zone=str(ACQUISITION_ZONE),
                           rule='later start; equal start longer real; exact real tie prefer live < real',
                           real_time_semantics='CNF native selected real counter; equal real/live does not establish absence of dead time',
                           over_half_percent=[r for r in changes if r['action']=='correct-live' and (r['fractional_live_change'] is None or abs(r['fractional_live_change'])>.005)])
            (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            if digest(source) != source_digest:
                raise ValueError('source database changed during correction')
            output.rename(destination)
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--reader', required=True, type=Path)
    parser.add_argument('--cnf-root', required=True, type=Path)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        survey=Path(tmp)/'cnf.jsonl'
        with survey.open('w') as stream:
            subprocess.run([str(args.reader.resolve()), str(args.cnf_root.resolve())], stdout=stream, check=True)
        result=build(args.source, survey, args.destination)
    print(json.dumps({k:result[k] for k in ('files','matched','corrected_live','corrected_start')}))
