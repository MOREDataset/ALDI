# Multi-source ALT dataset builder

These scripts build a circumarctic active layer thickness (ALT) dataset from many public
sources, not only the CALM summary table. Each source is turned into one end-of-season
value per site and year (cm). The same physical site found in several sources becomes one
series. Only sites with at least 8 valid annual values are kept. The output file
`alt_multisource_for_ALDI.csv` is the input for `Scripts/ALDI_Calculation_final.ipynb`.

## Files

| File | What it does |
|---|---|
| `alt_core.py` | All settings in one `CONFIG` dictionary, with the reason next to each value. Also the two shared rules: end of season for probing surveys, and the 0 °C crossing for ground-temperature profiles. |
| `alt_sources.py` | One reader per source. A reader only reshapes the source files into the common record layout. The list of readers is `ADAPTERS` at the bottom. |
| `build_alt_multisource.py` | Runs the readers, keeps the circumarctic sites, joins duplicates, adds flags, and writes the outputs. |

## Run

```
python build_alt_multisource.py --raw <raw folder> [<more raw folders>] --out <output folder>
```

Optional: `--min-years 8` (the default), and `--only gtnp calm_summary` to run some
readers only. A full run takes about 2 minutes. The raw files are not in this repository,
because they belong to their data owners; `source_inventory.csv` in the output lists every
source with its DOI or URL and licence.

## How a measurement becomes one annual value

- **Probing grids and transects** (CALM grids, LTER plots, Russian stations): the mean of
  all points of one survey. Dates no more than 3 days apart are one survey. The season's
  value is the deepest survey between 1 August and 17 October. Seasons surveyed only
  before 1 August are dropped, and so are partial surveys (fewer than half the points of
  the season's fullest survey).
- **Thaw tubes** (GSC Mackenzie Valley): the maximum thaw depth read from the tube.
- **Ground temperatures** (boreholes, soil stations, temperature cables): each sensor's
  maximum temperature of the year, then the depth where that profile first crosses 0 °C
  going down, by linear interpolation between the two sensors around it. A sensor is used
  in a year only if it has data on at least 80 % of the days from 1 August to 30 September.
  The value is not used when:
  - every sensor thawed (the thaw went deeper than the deepest sensor; flag `lower_bound`),
  - the two sensors around the crossing are more than 50 cm apart (or 50 % of the ALT
    when that is larger), or more than 1 m apart in any case (flag
    `wide_sensor_spacing`), or
  - the deepest thawed sensor did not clearly refreeze the following winter: its
    minimum from October to June stayed above -0.1 °C, the usual sensor accuracy
    (flag `no_refreeze`), or
  - the last thawed sensor, the one the crossing is interpolated from, reads more than
    1 °C warmer than the sensor above it. The annual maximum falls with depth, so its
    depth label or the sensor is wrong (flag `profile_inversion`).

## Flags

Values with these flags stay in the files but are not used for the 8-year count or in
the ALDI input: `lower_bound`, `wide_sensor_spacing`, `no_refreeze`, `talik`,
`possible_talik`, `sensor_cap`, `calm_footnote`, `early_measurement`, `treatment`,
`borehole_equilibration`, `profile_inversion`, `disturbed_site`, `sensor_problem`.

- `possible_talik`: Bonanza Creek LTER probing (interior Alaska). The probe, with
  extensions, is pushed to the first hard frozen ground. Once the winter frost no longer
  reaches the permafrost (a talik), that is the permafrost table or leftover seasonal
  frost, not ALT; the series then jump between about 70 and 300 cm from one year to the
  next. From the first year a series is deeper than 150 cm, that year and all later ones
  are flagged (CONFIG `talik_onset_cm`). CALM flags its two Fairbanks-area sites, U18
  and U19, as talik from about 1 m.
- `borehole_equilibration`: the first year of each GSC Mackenzie cable, which reads about
  6 % shallower than the second (CONFIG `equilibration_years`), and the first year of a
  Canada-collection cable whose record starts in May to August, i.e. that was installed
  that summer (CONFIG `install_months`).
- `disturbed_site`: Canada-collection sites that its compilers mark as disturbed
  (highway embankments, mine sites, burned or cleared ground).
- `sensor_problem`: Canada-collection years whose sensors are not comparable with the
  rest of the record (Clearwater Lake LECCT02 and LECTOUR before 2006, LECTOUR from 2017,
  Bylot Island BYLOTPD from 2013). CONFIG `sensor_problem_years` gives the evidence.

Flags belong to the physical site, not to one source. After the series of a site are
joined, `talik` and `possible_talik` apply to every later year of the site, `no_refreeze`
to every record of that year, and the survey and borehole flags (`calm_footnote`,
`sensor_cap`, `early_measurement`, `treatment`, `lower_bound`, `borehole_equilibration`,
`disturbed_site`) to the copies of the same value in other sources. Without this, a year that CALM flags came back through an unflagged copy in
GTN-P or BNZ LTER (Bonanza Creek U18 2011-2025, Pearl Creek U19 2015-2018, and 12
early surveys at Russian and Canadian sites).

Informational only: `nonmonotonic_profile`, `burned` and `burned_site`, `small_sample`,
`unusual_year_regional`, `unusual_year_isolated`, `deep_thaw_check_talik`.

## Canada ground temperature collection

`read_canada_gt` reads the Canadian permafrost ground temperature collection 2025
(Meier-Legault, Brown, Gruber et al., ESSD 2026; Zenodo doi:10.5281/zenodo.21042525):
daily mean temperatures from about 430 boreholes and cables. Only ground-temperature
sites are read. Two series within 300 m whose ALT agrees within 1 cm on at least 3
uncensored years are copies of one borehole from two contributors and are merged.
Series that the metadata calls the same borehole with new or other sensors are not
merged, because a sensor change shifts the ALT. Cables also read from their originals
(GSC Open File 9340, Umiujaq) are joined to those, and the originals stay preferred.

## Outputs

| File | Content |
|---|---|
| `alt_multisource_all_records.csv` | Every annual record from every source, before merging. |
| `alt_multisource_annual.csv` | One row per site and year for the kept sites, with all flags and the value from each source. |
| `alt_multisource_sites.csv` | One row per kept site: position, sources, method, years, SD of the valid years, and other kept sites within 1 km (`colocated_with`). |
| `alt_multisource_for_ALDI.csv` | `site_id, Location, Lat, Long, Year, ALT_cm, source, method, obs_doy, flag`, valid years only. `obs_doy` is the survey day where the source gives a date (later surveys read deeper, so it can be used as a control); `flag` holds the information flags. |
| `source_inventory.csv` | Every source checked: what was read and used, or why it was not used. |
| `data_credits.md` | Dataset citations, the GTN-P data authors to credit (GTN-P data policy), and the creator and reference of each Canada-collection series used. |
| `build_log.txt` | Everything the run printed. |
