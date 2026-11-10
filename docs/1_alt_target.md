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
| `tarfala_pace` | Tarfala PACE boreholes, Tarfalaryggen, northern Sweden (Stockholm University, figshare doi:10.17045/sthlmuni.3171646, 2000-2014; added 10 November 2026) | Two boreholes about 18 m apart (100 m and 15 m deep), 6-hourly thermistor readings; logger error codes (-273, -231, -113, -106 °C) removed, daily means of at least 2 readings, ALT by the 0 °C rule. The two holes are one site, so one is kept: the 15 m control hole. The 100 m main borehole reads ALT 5-19 cm shallower in the same years, the pattern of the lidded main borehole at Janssonhaugen (the log gives the usable years of both). The newer logger data after 2014 are not public. |
| `uio_boreholes` | University of Oslo mountain permafrost boreholes in Norway and Iceland (Etzelmüller et al. 2023, NIRD doi:10.11582/2023.00128, CC BY 4.0; added 10 November 2026) | 14 boreholes with daily ground temperatures, 1999-2023: Iskoras BH1/BH2, Guolasjávri, Jetta, Juvflye BH1/BH3/BH4/BH5, Juvvasshøe PACE, Tronfjell, and the Icelandic Hágöngur, Sauðafell, Vopnafjörður and Gagnheiði. The files carry no coordinates: positions from the GTN-P database (`UIO_BOREHOLES`). Four are the same loggers as Frost stations (isk2 = SN97710, juv1 = SN15265, juv3 = SN15290, juvP = SN15270); they rank after Frost, `CONFIG["force_join"]` joins isk2, juv1 and juv3 to the Frost site of the same hole, and juvP joins CALM N2 by its CALM code (`CONFIG["uio_calm_code"]`). Juvflye BH1 is the reason for the forced join: the UiO and Frost files agree within 0-3 cm in 2009-2012 but UiO reads 40-100 cm deeper from 2014, so the agreement rules had made one borehole two sites. The cause is in the UiO file: from August 2013 its depth labels sit one sensor too deep (two from May 2018), each column then equalling the Frost sensor one label higher, so UiO juv1 from 2013 on is flagged `sensor_problem` (`CONFIG["sensor_problem_years"]`) and the site keeps Frost's values. The `.7z` archive is unpacked by `download_uio_boreholes` (py7zr, or the `tar.exe` of Windows). |
| `roshydromet_soil` | Roshydromet daily soil temperature at depths to 320 cm, version 3, 1963-2024 (RIHMI-WDC, http://meteo.ru/data/soil-temperature/; added 10 November 2026) | Only served after a personal AISORI-M login, so the user exports the stations and saves each ZIP unchanged in `Roshydromet_soil/`; every ZIP there is read and a station-day in two ZIPs counts once. The export gives only the WMO index: positions come from the dataset's station catalogue (264 stations, embedded as `RUSOIL_STATIONS`). Only values with quality flag 0 (reliable) are used, and only the exhaust thermometers at 20-320 cm (2-15 cm are bare-soil thermometers read in the warm season). ALT by the 0 °C rule. Most station-years are lower bounds, because the cleared, snow-covered station plot thaws below the deepest sensor in use or there is no permafrost; stations that read only 80 and 160 cm (Verkhoyansk) are flagged `wide_sensor_spacing`. The ten stations that are also in the 1930-1990 Russian ALT product carry its site key (`CONFIG["rusoil_ruhms"]`), so the two are one site, and the daily values rank first. |
| `metno_janssonhaugen` | Janssonhaugen, Svalbard: shallow (15 m) borehole P11, 0.2-5 m, April 1999 to August 2025 (MET Norway, K. Isaksen; sent by email on 10 November 2026; cite Isaksen et al. 2000, 2001, 2022) | 6-hourly readings, daily means of at least 2, ALT by the 0 °C rule. This is the borehole behind CALM N1: its ALT matches CALM N1 within 0-4 cm in all 26 years (median +1.5 cm), so it joins CALM N1 by its CALM code and CALM's values stay first. The Frost series of Janssonhaugen (SN99875) is the 102 m borehole under a concrete lid: its upper metres show a damped annual cycle (about 2 °C cooler at 20 cm in summer, 1 °C warmer in winter) and its ALT reads a median 17 cm shallower than P11. It is kept as `check_only` (`CONFIG["check_only_series"]`), so it never fills a year of CALM N1. |
| `g02189` | NSIDC G02189 (Russian soil temperatures to 2000) | The files have only January and July monthly means, so the value is the July thaw depth: a lower limit of that year's ALT. Used only as a check: every value is flagged `check_only` and never used. `check_pre2000_g02189.csv` compares it with the Russian ALT series of the same station (ALT should be at least the July thaw depth). |

If a reader cannot read a table, or cannot place it on the map, the table goes into `unread_tables.csv` with the reason, and never enters the data.

The first full run (2 November 2026) gave 341 sites. It could not read most of the Arctic Data Center, Syrdakh and G02189 files: their layouts were not known when the readers were written. Those three readers were rewritten from samples of the real files. The same run found that the GTN-P platform gave borehole depths in metres, which the general reader took as cm in boreholes deeper than 30 m; that is fixed.

The third run (2 November 2026) gave 358 sites and 6,528 valid site-years. The Arctic Data Center boreholes added 15 sites, Syrdakh 1. NEON gave nothing because neonutilities now names the plot position `locationReferenceLatitude`/`-Longitude`; the reader now reads both names. The UAF site files gave nothing because their positions were not found; they now use the same station reader and station lists as `tsp_alaska`.

The fourth run gave 359 sites and 6,565 valid site-years. NEON and the UAF site files mostly repeat stations already in the target. It also showed that NEON writes a new position row at every re-survey, so the move check now looks at depth changes only. Files that cannot give ALT are listed in `unread_tables.csv` with that reason: single-date borehole logs, files with one sensor, and a community questionnaire.

All temperature sources are reduced to daily means before the 0 °C rule, so every source is treated the same way.

The Frost API (10 November 2026) gave 361 sites and 6,585 valid site-years (pre-Frost build: 359 and 6,567). New sites: Juvflye BH1 (14 valid years 2009-2025) and Dovrefjell DB1 (10, 2002-2018). Frost Janssonhaugen (SN99875, 1998-2025) and Snøheim (SN9380) join CALM N1 and N5, Bayelva (SN99917) the AWI Bayelva site. Most Svalbard stations start in 2019-2025, and four measure only 5-10 cm (every year a lower bound). Snøheim (CALM N5) loses 2015-2025: the Frost profile shows the deepest thawed sensor not refreezing in those winters, and that flag covers every source of the site in the year, as for the talik years at Pearl Creek and Bonanza Creek (decision of 10 November 2026). It drops from 18 to 9 valid years.

Tarfala PACE and the UiO boreholes (10 November 2026) gave 363 sites and 6,605 valid site-years (Frost build: 361 and 6,585). New sites: Sauðafell, Iceland (the GTN-P borehole, lengthened by the UiO file to 12 valid years, 2005-2021; it also enters ALDI) and Tarfala PACE (the 15 m hole, 8 valid years 2000-2010; gaps in the logger record leave 2011-2014 without a full August-September, so not in ALDI). Most UiO mountain boreholes give no valid year: the thaw reaches 7-11 m in warm permafrost where the sensors are metres apart or the ground no longer refreezes (`wide_sensor_spacing`, `no_refreeze`). Juvvasshøe PACE joins CALM N2 (CALM's values first), Gagnheiði joins its GTN-P copy (still fewer than 8 valid years), and the Frost copies of Juvflye BH1, BH3 and Iskoras BH2 keep their Frost values.

The Roshydromet soil temperatures (10 November 2026; the user's AISORI-M export of 110 stations, 95 of them at or north of 50 N) gave 367 sites and 6,707 valid site-years (before: 363 and 6,605). Of 5,264 station-years only 157, at 8 stations, are valid; the rest are lower bounds (thaw below the deepest sensor in use, or no permafrost). New sites: Dikson (14 valid years, 1989-2017) and Ust-Maya (35, 1975-2018), both in ALDI; Suntar (9, 1964-1972) and Chara (56.9 N) are not. Yakutsk, Isit', Oymyakon and Verkhoyansk join their 1930-1990 Russian ALT series (`ru_hms_alt`), which brings Yakutsk and Isit' into ALDI as well (ALDI-eligible sites 220 to 224). In the 92 shared years the Russian ALT product reads a median 9 cm shallower than the daily values (median |difference| 10 cm, largest 36 cm), as expected from ALT taken from monthly means. The AISORI-M site serves only these 110 of the 264 stations of the dataset's catalogue.

The Janssonhaugen shallow borehole and the change of Frost SN99875 to `check_only` (10 November 2026) left the totals unchanged (367 sites, 6,707 valid site-years, 224 ALDI-eligible): CALM N1 keeps CALM's values in all 28 years, and no record of any other site changed its flag. P11 matches CALM N1 within 0-4 cm (median +1.5 cm, 26 years); Frost SN99875 reads a median 16.6 cm shallower (largest 24 cm, 28 years). The site lies at 78°10′46″N, 16°28′01″E, 270 m a.s.l. (Isaksen et al. 2001, which gives no separate position for P11).

Site joining was corrected on 10 November 2026: when several sites pass the agreement rules, a series now joins the one it agrees with best, not the nearest that passes. Two series moved: Kytalyk Allaiha, a copy of CALM R31, had been joined to CALM R31A (1 km away from R31), which gave R31A five years that are not its own; and the Canada collection's KP313-T4 had been joined to the GSC cable KP313T2 instead of its copy KP313T4 (four cables share one coordinate). CALM R31A then has 5 valid years and drops out: 366 sites, 6,697 valid site-years, ALDI-eligible sites unchanged at 224.

Known limitation, kept on purpose (decision of 10 November 2026): ALT pinned to a sensor depth. Seven temperature series have valid ALT that varies by less than 2 cm (SD) over 8 or more years: the GSC cables KP313T5 (99.5 ± 0.2 cm, 12 years), 12TC01 (99.0 ± 0.4), T6UPLAND (163.6 ± 0.7), CAL02 (98.3 ± 1.2), NCL02 (97.2 ± 1.3) and T5UPLAND (76.1 ± 1.8), and the GTN-P borehole 432 at Bolvansky (297.8 ± 0.2). In each, the sensor below the 0 °C crossing stays just under 0 °C (a maximum of -0.04 to -0.6 °C) while the sensor above is warm, so the interpolated crossing lands a few cm above the lower sensor every year, wherever the thaw front is in between. At 12TC01, CAL02 and KP313T5 the lower sensor sits in the zero curtain (-0.2 to 0 °C for most of July-October), so the thaw front reaches it or nearly does; at NCL02, T5UPLAND and T6UPLAND it stays at -0.3 to -0.6 °C, so the true thaw depth lies somewhere in the 50 cm gap and the value is probably too deep. Either way the year-to-year change is set by the sensor spacing, not by the thaw. The series stay in the data unflagged; four of them are ALDI-eligible (KP313T5, CAL02, NCL02, T5UPLAND), and their near-zero variance can pull ALDI trends toward zero. A rule that would flag them (SD below 2 cm and below 3 % of the median over at least 8 valid years) was written (branch `pinned-sensor`) but not adopted.

One data request is still open, by email: the Kilpisjärvi palsas (Verdonen, 2007-2021). Isaksen (MET Norway) answered the Janssonhaugen request on 10 November 2026 with the shallow borehole P11 (`metno_janssonhaugen`); the data after the August 2025 logger upgrade are to follow.

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
`sensor_moved` (NEON), `check_only` (G02189, Frost SN99875), `pinned_to_sensor`. A `check_only` series never
passes its flags to the other sources of its site.

- `pinned_to_sensor`: a temperature series whose valid ALT varies by less than 2 cm (SD, and
  less than 3 % of its median) over at least 8 years. The sensor below the 0 °C crossing
  stays just under 0 °C while the one above is warm, so the interpolated crossing lands a
  few cm above the lower sensor every year, wherever the thaw front is in between (GSC
  cable KP313T5: 99.5 ± 0.2 cm over 12 years, sensors at 50 and 100 cm). The
  year-to-year change is sensor spacing, not thaw, so every year of the series is flagged,
  and copies of its values in other sources too.

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
