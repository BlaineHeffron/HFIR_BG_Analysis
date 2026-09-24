"""The local dead-time build must preserve measured timing and flag unsupported transfers."""
import csv
from pathlib import Path
import sqlite3
import tempfile
import unittest

from scripts.correct_cnf_dead_time import build


class DeadTimeTests(unittest.TestCase):
    def test_local_database_copy(self):
        source = Path(__file__).parents[1] / 'data/HFIRBG_public_data_v1.2.1'
        if not (source/'HFIRBG.db').is_file():
            self.skipTest('optional local v1.2.1 data are absent')
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)/'v1.2.2'
            summary = build(source, dest)
            with sqlite3.connect(source/'HFIRBG.db') as old, sqlite3.connect(dest/'HFIRBG.db') as new:
                prior = {r[0]: r for r in old.execute('SELECT * FROM datafile')}
                current = {r[0]: r for r in new.execute('SELECT * FROM datafile')}
                columns = [r[1] for r in old.execute('PRAGMA table_info(datafile)')]
                live_index = columns.index('live_time')
                changed = [fid for fid in prior if prior[fid][live_index] != current[fid][live_index]]
                self.assertEqual(len(changed), summary['dead_time_corrected'])
                self.assertTrue(all(prior[fid][:live_index]+prior[fid][live_index+1:] ==
                                    current[fid][:live_index]+current[fid][live_index+1:]
                                    for fid in prior))
                self.assertEqual(prior[967][live_index], current[967][live_index])
            with (dest/'corrections.csv').open() as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(len(reader.fieldnames), len(set(reader.fieldnames)))
                rows = {r['file_id']: r for r in reader if r['new_live_s']}
            self.assertIn('unmeasured-contested-gain', rows['1331']['flags'])
            self.assertLess(float(rows['1331']['dead_time_rate_change']), 0.02)
            self.assertEqual(rows['1331']['acquisition_settings_id'], '11')


if __name__ == '__main__':
    unittest.main()
