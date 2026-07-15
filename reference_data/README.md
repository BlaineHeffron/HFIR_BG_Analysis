# Reference data

`hfir_cycle_calendar.csv` is a durable, machine-readable record of the actual
HFIR operating-cycle dates that overlap the public background-data campaign.
It was transcribed from the cycle-history table in Appendix A, printed page
A-3, of **ORNL/TM-2023/3207**, *Post-Irradiation Examination on Absorber
Material Specimens Irradiated in the High Flux Isotope Reactor*, published
August 2024:

- Report: <https://info.ornl.gov/sites/publications/Files/Pub206395.pdf>
- DOI: <https://doi.org/10.2172/2439884>
- Report PDF SHA-256: `c56411b22c869029fc0d522bd8e1e9a4f0a130bdb79c9f3cc3ac700797c66cba`
- Retrieved: 2026-07-15

The source reports dates, not transition times. The calendar therefore has
`day` precision. Code may use a date to classify a measurement as operating or
between cycles, but must not infer an exact reactor startup or shutdown time
within a boundary date. `schedule_basis=actual` distinguishes these historical
dates from projected schedules, and `record_status=complete` means every cycle
row in the cited table from 490A through 499 has been recorded here; it is not
a claim that the calendar covers dates before 490A or after 499.

Between-cycle periods are derived from the gaps between consecutive rows. They
are labeled as outages/end-of-cycle periods in the run catalog. Dates outside
the first cycle start and last cycle end remain `unknown`.
