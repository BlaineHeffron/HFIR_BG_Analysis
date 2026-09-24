"""A correction must preserve the source and calibration and replace placeholder timing."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('correction', Path(__file__).parents[1]/'scripts/correct_cnf_timing.py')
correction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(correction)


class CorrectionTests(unittest.TestCase):
    def test_copy_and_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'original.db'; survey=root/'survey.jsonl'; dest=root/'vnew'
            with sqlite3.connect(source) as db:
                db.executescript("CREATE TABLE datafile(id INTEGER PRIMARY KEY,name TEXT,start_time INTEGER,live_time REAL);"
                                 "INSERT INTO datafile VALUES(1,'one',NULL,10),(1078,'CYCLE461_DOWN_FACING_OVERNIGHT',NULL,0);"
                                 "CREATE TABLE calibration_group(A0 REAL,A1 REAL);"
                                 "INSERT INTO calibration_group VALUES(1,2);")
            original=source.read_bytes()
            rows=[dict(path=name+'.CNF',chosen_record=2,status='stale-copy detected',rule='later-start',
                       timing=[dict(start_ticks=1,live_s=10,real_s=10,valid=True),
                               dict(start_ticks=2,live_s=20,real_s=21,valid=True)],
                       sha256='synthetic',parser_revision='synthetic') for name in ('one','CYCLE461_DOWN_FACING_OVERNIGHT')]
            rows[1]['timing'][1]['start_ticks']=51270243427170000
            (root/'spectra').mkdir()
            (root/'spectra'/'CYCLE461_DOWN_FACING_OVERNIGHT.txt').write_text('synthetic')
            survey.write_text('\n'.join(map(json.dumps,rows)))
            result=correction.build(source,survey,dest)
            self.assertEqual((result['corrected_live'],result['corrected_start']),(2,2))
            self.assertEqual(source.read_bytes(),original)
            with sqlite3.connect(dest/'HFIRBG.db') as db:
                self.assertEqual(db.execute('select live_time,real_time from datafile order by id').fetchall(),[(20,21),(20,21)])
                self.assertEqual(db.execute('select start_time from datafile').fetchone()[0],correction.unix_start(2))
                self.assertEqual(db.execute('select * from calibration_group').fetchall(),[(1,2)])
                self.assertEqual(db.execute('select name from datafile where id=1078').fetchone()[0],'CYCLE491_DOWN_FACING_OVERNIGHT')
            self.assertEqual(result['corrected_names'],1)
            self.assertEqual([p.name for p in (dest/'spectra').iterdir()],['CYCLE491_DOWN_FACING_OVERNIGHT.txt'])
            self.assertEqual((dest/'spectra'/'CYCLE491_DOWN_FACING_OVERNIGHT.txt').read_text(),'synthetic')
            with self.assertRaises(ValueError):correction.build(source,survey,dest)
            rows[1]['timing'][1]['start_ticks']=2
            survey.write_text('\n'.join(map(json.dumps,rows)))
            with self.assertRaisesRegex(ValueError,'outside verified Cycle'):correction.build(source,survey,root/'wrong-cycle')
            rows[1]['timing'][1]['start_ticks']=51270243427170000
            rows[0]['chosen_record']=0;rows[0]['status']='ambiguous'
            survey.write_text('\n'.join(map(json.dumps,rows)))
            with self.assertRaises(ValueError):correction.build(source,survey,root/'ambiguous')
            self.assertFalse((root/'ambiguous').exists())
            self.assertEqual(source.read_bytes(),original)


if __name__=='__main__': unittest.main()
