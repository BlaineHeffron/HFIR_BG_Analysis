# Local CNF timing correction

The opt-in local `data/HFIRBG_public_data_v1.2.0/HFIRBG.db` corrects 43 live
times identified by a survey of 1,809 Canberra CNFs (1,802 database matches).
It is not a public release. Version 1.1.0 and the browser default are unchanged.
Two apparent exclusion sentinels, file IDs 1078 and 1561, remain unchanged.

Select the new copy with the existing browser interface:

```python
from src.public_data.browser import load_spectrum
spectrum = load_spectrum(444, db_path="data/HFIRBG_public_data_v1.2.0/HFIRBG.db")
```

`HFIRBG_CALDB` also selects an explicit database. Frozen studies must continue
passing their bound v1.1.0 path. The local copy links to the original spectra;
counts, calibration assignments, run mappings and start times are unchanged.
The added `datafile.real_time` is the selected native CNF real counter in
seconds. Equal real/live values do **not** demonstrate zero physical dead time;
no dead-time fraction is imputed. The browser still normalizes by live time.

Reproduce with the compiled reader from the private phonon-response workspace:

```sh
python3 scripts/correct_cnf_timing.py \
  --source data/HFIRBG_public_data_v1.1.0/HFIRBG.db \
  --destination data/HFIRBG_public_data_v1.2.0 \
  --reader /path/to/phonon-response/build/detresp_cnf \
  --cnf-root /path/to/HFIRBG/data
```

The destination must not exist. The generator refuses ambiguous records and
creates a new SQLite copy; it never updates the source. All bulk binary parsing
and channel summation run in C++; Python handles only small per-file metadata.
Generated `survey.csv` contains both timing records, database live time
(database real time was absent), chosen slot, rule, source digest and fractional
live/rate changes. `corrections.csv` lists changed live times and their class: 19 stale-record
values, 23 real-time substitutions, and one unexplained value (file 548).
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

The paper's down-facing scan figure requires regeneration if this version is
adopted: twelve plotted points change, with rates from about -27% to +30%.
The east scan has sub-percent corrections. The paper's rate and RD line tables,
RD reactor-on/off fits, and collimator file 186 retain their database live times.
No frozen result or public release has been regenerated or modified. Reinstating
the two sentinel files and rerunning published figures require a user decision.
