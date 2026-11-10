# Step 1: the ALT target (`1_alt_target.py`)

`1_alt_target.py` makes the circumarctic active layer thickness (ALT) target used by ALDI (`3_aldi.py`) and by the causal analysis. It gives one end-of-season value per site and year (cm), from probing grids, thaw tubes and ground-temperature profiles of about 35 sources. When the same physical site is found in several sources, it becomes one series. Only sites with at least 8 valid annual values are kept.

It replaces four scripts that each did part of this job:
- `alt_core.py`: the rules.
- `alt_sources.py`: the readers.
- `build_alt_multisource.py`: the merge.
- `fetch_new_sources.py`: the downloads for the new sources, with its own copy of the ALT step.

Before, the files were downloaded by hand, and the new sources were handled outside the build. Now every source is downloaded, read and checked by the same code with the same rules.

On the same raw files, the combined script gives the same four data outputs as the old build, byte for byte (checked on 2 November 2026).

## Run

See the top of `1_alt_target.py` (and `README.md` in the main folder). Every setting is at the top of the script, with the reason for it.

Raw files: `data/raw/alt_sources/` (downloads land there). Outputs: `data/1_alt_target/`. NEON needs your own token for the session (PowerShell `$env:NEON_TOKEN="..."`) and MET Norway's Frost API your client ID (`$env:FROST_CLIENT_ID="..."`, free at frost.met.no; the client secret is not needed); neither is ever written to a file.


## The three stages

**1. Download**

- **Direct links.** The 204 files with a link in `SOURCE_FILES` (at the end of `1_alt_target.py`) are downloaded. They come from PANGAEA, the Arctic Data Center, EDI/LTER, ESS-DIVE, ScienceBase, open.canada.ca, Borealis, Zenodo and NRCan, plus 50 UAF files from the Wayback Machine.
- **APIs.** These sources are queried through their APIs:
  - the CALM summary table, found from the CALM page
  - the GTN-P data platform (boreholes and active-layer grids)
  - the Arctic Data Center permafrost-observatory packages
  - the UAF site pages
  - NEON
  - Syrdakh (Zenodo)
  - NSIDC G02189
  - MET Norway (THREDDS and the Frost API)
- **By hand.** Two sources need a personal login: GEM Greenland (8 zips) and ABoVE (NASA Earthdata). The script never asks for or stores a login. The run lists where to get these files, and you put them in the raw folder yourself.
- **Report.** `download_report.csv` lists every file as downloaded, already there, failed, or by hand.

**2. Read.** One reader per source turns its files into records with the common columns. The readers only reshape and convert. The rules that make one annual value are shared by all of them (next section).

**3. Build.** The script keeps the circumarctic sites, joins duplicates, applies the flags, and keeps the sites with enough valid years. Then it writes the outputs and the checks.

## Sources added in November 2026

These sources were in `fetch_new_sources.py` before. They now go through the same readers, rules and checks as the others.

| Reader | Source | Rule for this source |
|---|---|---|
| `gtnp_platform` | GTN-P data platform boreholes (data.gtn-p.org) | Downloaded when the dataset spans at least 8 seasons from 1990. Read when at least two sensors are in the top 1.5 m. Boreholes that the platform marks as disturbed are flagged `disturbed_site`. |
| `gtnp_platform_alt` | GTN-P data platform grids and transects | Same survey rules as the other grids. They join the CALM tables by CALM code. `check_source_agreement.csv` holds their yearly differences to PANGAEA 972777. |
| `tsp_alaska` | Arctic Data Center permafrost-observatory packages: doi:10.18739/A2J34P (newest version) and the current Romanovsky packages | The package metadata give only a region, so each data file is placed from the station lists inside the packages (the yearly SitesRoster files, `site_and_file_metadata.csv`, `Site_Information.csv`, the Russian `metadata.csv`, `cadis_list.xls`, `network_sampling_sites.csv`). A file is matched by its name, else by the longest station code its name starts with, else by station name (equal, or one name containing the other when only one place fits). Long tables with a station code column (`pf_borehole_daily_temps_nirpo_*.csv`) give one series per code. A file that cannot be placed is not used: the Goldstream stations (`US_GS__*`) are listed with empty coordinates, so they stay unread. Single-day profiles (one survey date) cannot give ALT and are skipped. When one station is published twice in the collection (a station file and the network table), the two series are joined only if they agree within 5 cm (or 10 %) on at least 5 shared years. A station placed by its code keeps the code as its id (e.g. `tsp_alaska:US_A09_012`), so the id stays the same when new years are published. |
| `uaf_sites` | UAF Galbraith Lake, West Dock, Happy Valley and Chandalar Shelf | The position comes from the file's own `Location:` line, else from the station code in the Arctic Data Center station lists, else from the site page. When a file's header position is more than 5 km from the station lists, or from the median header of the same station's other files in its folder, that position wins (WD1_06.xls gives N 70.166, the latitude of Deadhorse, for West Dock at 70.38 N). Single-date borehole logs (CADIS, ArcDC, NSIDC files) cannot give ALT and are listed as such. Two series of one station in these files are joined under the same strict rules as in `tsp_alaska`. Most of these stations are also in the Arctic Data Center packages and the GTN-P platform; the build joins the copies, and ranks this source after them so the site ids stay as they were. |
| `neon` | NEON soil temperature at BARR, TOOL and HEAL | A plot-year runs from 1 January to 1 July of the next year. It is flagged `sensor_moved` and not used when a sensor of the plot changed depth by more than 2 cm (NEON writes a new position row at every re-survey, mostly with the same depth; that is not a move), or when a heave entry in the product's change log covers it. A sensor heaved above the ground surface (zOffset ≥ 0) is left out. |
| `syrdakh` | Syrdakh, Yakutia (Zenodo 14619854) | Logger temperatures (good-quality values only) give ALT by the 0 °C rule; the late-September thaw-depth surveys are used as probe or tube values, and a depth beyond the tool is a lower bound. The files give no coordinates, so every series gets the centre of the study area (62.56 N, 130.9 E, from the data paper) with a note in `site_note`. |
| `metno` | MET Norway permafrost stations (THREDDS) | Most stations start in 2019, so they reach 8 years only from about 2027. A station that the Frost API also gives is read from Frost only. |
| `metno_frost` | MET Norway Frost API (frost.met.no, added 10 November 2026) | Frost holds soil temperature at 235 stations (`METNO_frost/available_series.csv`); only the 30 on permafrost are downloaded: every station in Svalbard (north of 74 N) and the mainland boreholes with a sensor deeper than 2 m (Juvvasshøe, Juvflye, Dugurdskampen, Snøheim, Dovrefjell DB1, Iskoras). The other mainland stations stand on seasonal frost. Daily mean per depth, the daily-mean element where Frost has one; ALT by the 0 °C rule, method `borehole_temp` for the boreholes and `soil_temp` for the shallow stations. Frost Juvvasshøe (SN15270) is the CALM N2 borehole about 10 m away but reads 25-30 cm deeper every year, so `CONFIG["frost_calm_code"]` joins it to N2 by its CALM code; CALM's values stay first. |
| `tarfala_pace` | Tarfala PACE boreholes, Tarfalaryggen, northern Sweden (Stockholm University, figshare doi:10.17045/sthlmuni.3171646, 2000-2014; added 10 November 2026) | Two boreholes about 18 m apart (100 m and 15 m deep), 6-hourly thermistor readings; logger error codes (-273, -231, -113, -106 °C) removed, daily means of at least 2 readings, ALT by the 0 °C rule. The two holes are one site, so only the one with more usable years is kept (the log names it). The newer logger data after 2014 are not public. |
| `uio_boreholes` | University of Oslo mountain permafrost boreholes in Norway and Iceland (Etzelmüller et al. 2023, NIRD doi:10.11582/2023.00128, CC BY 4.0; added 10 November 2026) | 14 boreholes with daily ground temperatures, 1999-2023: Iskoras BH1/BH2, Guolasjávri, Jetta, Juvflye BH1/BH3/BH4/BH5, Juvvasshøe PACE, Tronfjell, and the Icelandic Hágöngur, Sauðafell, Vopnafjörður and Gagnheiði. The files carry no coordinates: positions from the GTN-P database (`UIO_BOREHOLES`). Four are the same loggers as Frost stations (isk2 = SN97710, juv1 = SN15265, juv3 = SN15290, juvP = SN15270); they rank after Frost and join it by position and agreement, and juvP joins CALM N2 by its CALM code (`CONFIG["uio_calm_code"]`). The `.7z` archive is unpacked by `download_uio_boreholes` (py7zr, or the `tar.exe` of Windows). |
| `g02189` | NSIDC G02189 (Russian soil temperatures to 2000) | The files have only January and July monthly means, so the value is the July thaw depth: a lower limit of that year's ALT. Used only as a check: every value is flagged `check_only` and never used. `check_pre2000_g02189.csv` compares it with the Russian ALT series of the same station (ALT should be at least the July thaw depth). |

If a reader cannot read a table, or cannot place it on the map, the table goes into `unread_tables.csv` with the reason, and never enters the data.

The first full run (2 November 2026) gave 341 sites. It could not read most of the Arctic Data Center, Syrdakh and G02189 files: their layouts were not known when the readers were written. Those three readers were rewritten from samples of the real files. The same run found that the GTN-P platform gave borehole depths in metres, which the general reader took as cm in boreholes deeper than 30 m; that is fixed.

The third run (2 November 2026) gave 358 sites and 6,528 valid site-years. The Arctic Data Center boreholes added 15 sites, Syrdakh 1. NEON gave nothing because neonutilities now names the plot position `locationReferenceLatitude`/`-Longitude`; the reader now reads both names. The UAF site files gave nothing because their positions were not found; they now use the same station reader and station lists as `tsp_alaska`.

The fourth run gave 359 sites and 6,565 valid site-years. NEON and the UAF site files mostly repeat stations already in the target. It also showed that NEON writes a new position row at every re-survey, so the move check now looks at depth changes only. Files that cannot give ALT are listed in `unread_tables.csv` with that reason: single-date borehole logs, files with one sensor, and a community questionnaire.

All temperature sources are reduced to daily means before the 0 °C rule, so every source is treated the same way.

The Frost API (10 November 2026) gave 361 sites and 6,585 valid site-years (pre-Frost build: 359 and 6,567). New sites: Juvflye BH1 (14 valid years 2009-2025) and Dovrefjell DB1 (10, 2002-2018). Frost Janssonhaugen (SN99875, 1998-2025) and Snøheim (SN9380) join CALM N1 and N5, Bayelva (SN99917) the AWI Bayelva site. Most Svalbard stations start in 2019-2025, and four measure only 5-10 cm (every year a lower bound). Snøheim (CALM N5) loses 2015-2025: the Frost profile shows the deepest thawed sensor not refreezing in those winters, and that flag covers every source of the site in the year, as for the talik years at Pearl Creek and Bonanza Creek (decision of 10 November 2026). It drops from 18 to 9 valid years.

One data request is still open, by email: the Kilpisjärvi palsas (Verdonen, 2007-2021). Janssonhaugen 1998-2019 (Isaksen, met.no) is no longer needed: the Frost API serves it.

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
`borehole_equilibration`, `profile_inversion`, `disturbed_site`, `sensor_problem`,
`sensor_moved` (NEON), `check_only` (G02189).

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
| `alt_multisource_annual.csv` | One row per site and year for the kept sites, with all flags and the `use` column. ALDI v5.1 reads this file. |
| `alt_multisource_sites.csv` | One row per kept site: position, sources, method, years, the SD of the valid years, and the other kept sites within 1 km (`colocated_with`). |
| `alt_multisource_for_ALDI.csv` | Valid years only, with the columns `site_id, Location, Lat, Long, Year, ALT_cm, source, method, obs_doy, flag`. The ALDI v4 notebook and the covariate script read this file. |
| `check_source_agreement.csv` | For sites read from more than one source: each source's yearly value against the preferred source's value. |
| `check_pre2000_g02189.csv` | G02189 against the Russian ALT series (only when G02189 was read). |
| `unread_tables.csv` | Tables of the new sources that could not be read or placed. |
| `source_inventory.csv` | Every source, with what was read and used, or why it was not. A reader that crashed shows as `reader FAILED`. |
| `download_report.csv` | Every file from stage 1 and what happened to it. |
| `data_credits.md` | The dataset citations, and the data authors to credit (GTN-P, Canada collection). |
| `build_log.txt` | Everything the run printed. |

The output file names have not changed, so ALDI, the covariate script and the notebooks need no change.
