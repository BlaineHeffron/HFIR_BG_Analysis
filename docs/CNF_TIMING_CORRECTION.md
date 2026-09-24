# Local CNF timing correction

Local v1.2.1 adds the verified file-186 name correction
`CYCLE461_DOWN_FACING_OVERNIGHT` → `CYCLE491_DOWN_FACING_OVERNIGHT` and is
preferred for new analyses. Its start (6 May 2021) and run-295 description
agree with Cycle 491 in `reference_data/hfir_cycle_calendar.csv`. No other
`CYCLE461_*` entry exists. Timing, calibration and counts are identical to
v1.2.0. Both old versions remain untouched. `src/spectrum_names.py` is the
shared bidirectional compatibility mapping. The new spectrum directory has
one link per acquisition using the canonical name; raw/released filenames
stay unchanged. `corrections.csv` includes old/new names, `name_corrected`,
and the rename evidence alongside timing corrections.


The local `data/HFIRBG_public_data_v1.2.0/HFIRBG.db` corrects 45 live times and
108 start times identified by a survey of 1,809 Canberra CNFs (1,802 database
matches). The browser and `setup_analysis.sh` prefer v1.2.1, then v1.2.0, with
v1.1.0 as the published fallback. It is not a public release; v1.1.0 is unchanged.

Select a database explicitly with the existing browser interface:

```python
from src.public_data.browser import load_spectrum
spectrum = load_spectrum(444, db_path="data/HFIRBG_public_data_v1.2.1/HFIRBG.db")
```

`HFIRBG_CALDB` also selects an explicit database; an existing `.env` that still
points at v1.1.0 is updated by rerunning `scripts/setup_analysis.sh`. Frozen
studies must continue passing their bound v1.1.0 path. Do not rerun setup
in a frozen study environment: setup upgrades the standard `.env` database
path; an explicit `db_path` binding remains authoritative. The local copy links to
the original spectra; counts, calibration assignments and run mappings are unchanged.
The added `datafile.real_time` is the selected native CNF real counter in
seconds. Equal real/live values do **not** demonstrate zero physical dead time;
no dead-time fraction is imputed. The browser still normalizes by live time.

Reproduce with the compiled reader from the private phonon-response workspace:

```sh
python3 scripts/correct_cnf_timing.py \
  --source data/HFIRBG_public_data_v1.1.0/HFIRBG.db \
  --destination data/HFIRBG_public_data_v1.2.1 \
  --reader /path/to/phonon-response/build/detresp_cnf \
  --cnf-root /path/to/HFIRBG/data
```

The destination must not exist. The generator refuses ambiguous records and
creates a new SQLite copy; it never updates the source. All bulk binary parsing
and channel summation run in C++; Python handles only small per-file metadata.
Generated `survey.csv` contains both timing records, database live time
(database real time was absent), chosen slot, rule, source digest and fractional
live/rate changes, and database and CNF start times. `corrections.csv` lists
changed live or start times and the live-time class: 19 stale-record values,
23 real-time substitutions, two placeholders (files 1078 `NE_FACING_EAST` at 0 s
and 1561 `PROSPECT_DOWN_OVERNIGHT` at 1 s) and one rule-unexplained value
(file 548). CNF start ticks are America/New_York wall time and reproduce all
1,694 intact database start times; differences above 1.5 s are replaced.
`database_status` separately records agreement with the selected live time;
the reader status describes the two CNF snapshots. `summary.json`
identifies database and survey digests, and `cnf.jsonl` retains field-level byte
evidence. These products stay in ignored storage.

Timing selection: reject missing/invalid records, then prefer later start;
with equal starts prefer longer real duration; with equal start and real,
prefer the record with live below real. Conflicting nonidentical live counters
at an otherwise tied record remain ambiguous. This is an empirically validated
acquisition rule, not vendor-certified CNF semantics. Validation includes
Am-241 peak rates and independent operator position-scan logs.

File 548 (`00002927`) is intact: the last of the 900 s preset runs
`00002918`–`00002927`, stopped at 533.68 s real, 523.13 s live. Its first record
is blank, so the text export shows an 1858 date; the database stored the preset
as live time and a negative start (2021-06-16 14:36:38 UTC in the CNF).
Negative start times made `CartScanFiles` skip 548, 1078 and 1561 as corrupted;
all three now enter the down-facing scan.

The paper's down-facing and east-face scan figures were regenerated from this
version by `HFIRBG/paper/scripts/render_position_scans.py`. The paper's rate and
RD line tables, unfolding inputs, RD reactor-on/off fits and collimator file 186
retain their live times. No frozen result or public release was modified.

`scripts/run_all_unfolds.sh` now includes `PROSPECT_DOWN_OVERNIGHT` and
`NE_FACING_EAST`. The ROOT inputs in `HFIRBG/data` carry the old live times
(1561 = 1 s, 1078 = 0 s, and 43 others), so inputs for these two were written
from this database to `data/unfold_inputs_v1.2.0`, alongside links to the six
unchanged inputs. `P2x_Analyze` must load the ROOT it was built against:
`LD_LIBRARY_PATH=/usr/local/lib/root:$P2X/lib`; a mismatched `libCore` aborts
with a double free.
