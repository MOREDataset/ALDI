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
    (flag `no_refreeze`).

## Flags

Values with these flags stay in the files but are not used for the 8-year count or in
the ALDI input: `lower_bound`, `wide_sensor_spacing`, `no_refreeze`, `talik`,
`sensor_cap`, `calm_footnote`, `early_measurement`, `treatment`.

Informational only: `nonmonotonic_profile`, `burned` and `burned_site`, `small_sample`,
`unusual_year_regional`, `unusual_year_isolated`, `deep_thaw_check_talik`.

## Outputs

| File | Content |
|---|---|
| `alt_multisource_all_records.csv` | Every annual record from every source, before merging. |
| `alt_multisource_annual.csv` | One row per site and year for the kept sites, with all flags and the value from each source. |
| `alt_multisource_sites.csv` | One row per kept site: position, sources, method, years, and other kept sites within 1 km (`colocated_with`). |
| `alt_multisource_for_ALDI.csv` | `site_id, Location, Lat, Long, Year, ALT_cm, source, method`, valid years only. |
| `source_inventory.csv` | Every source checked: what was read and used, or why it was not used. |
| `data_credits.md` | Dataset citations, and the GTN-P data authors to credit (GTN-P data policy). |
| `build_log.txt` | Everything the run printed. |
