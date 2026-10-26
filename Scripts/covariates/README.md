# Covariates for the ALT causal analysis (Google Earth Engine)

`gee_causal_features.py` builds the treatments, confounders and mediators of the manuscript
DAG tables for every site of the multi-source ALT dataset, one row per site and year, with
the column names that `Scripts/ALDI_causality.ipynb` uses. It then joins them with the ALT
values and the ALDI classes and writes `ALDI_withYearlyData_augmented*.csv`, the files the
causal notebook reads. `GEE_causal_features_colab.ipynb` runs it in Colab.

## What you need

| Input | Where it comes from |
|---|---|
| `site_footprints.csv` (optional) | probing grid size of the CALM sites, from `build_site_footprints()`; in the project's `alt_multisource/` folder. Put it next to `alt_multisource_for_ALDI.csv`. Without it, the footprint comes from the measurement method only |
| `alt_multisource_for_ALDI.csv` | the multi-source dataset (`Scripts/multisource/`); in Drive at `UND/Index/alt_multisource/`, where the ALDI notebook reads it |
| `aldi_multisource_results.csv` | `ALDI_Calculation_final.ipynb` (run with `min_year=1990`); in Drive at `UND/Index/` |
| `permaice.shp` (+ .shx, .dbf, .prj) | IPA Circum-Arctic Map of Permafrost and Ground-Ice Conditions (Brown et al. 2002, NSIDC GGD318), the same file the MATLAB script used; put it in `UND/Index/ggd318_map_circumarctic/` |
| An Earth Engine Cloud project | <https://code.earthengine.google.com/register> |

Put `gee_causal_features.py` in `/content/drive/MyDrive/UND/Index/scripts/`. The notebook
sets all paths in its second cell.

## Run (Colab)

1. `quick_test(sites)`: three sites, one year, all variables, computed directly (a few minutes; the MODIS snow-off step is the slowest). Check that the values look sensible.
2. `export_all(sites, drive_dir=EXPORTS)`: starts about 210 Earth Engine export tasks (one per data group and year). They write CSV files to the Drive folder `ALDI_GEE_exports`. They run on Google's side; Colab can be closed. Running it again only starts the missing ones.
3. `task_status()`: how many tasks are done or failed.
4. `assemble_features(...)`: reads the exports and adds the elevation-corrected temperatures, distance to coast, continentality, the AVHRR-to-MODIS adjustment, the sunlight ratio and the permafrost map classes. It prints a check that the temperature indices computed from the daily values match those from Earth Engine. Writes `Index/multisource/ALDI_causal_features_complete.csv`.
5. `build_augmented(...)`: writes `ALDI_withYearlyData_augmented.csv` and the class and permafrost-zone subsets to `Index/multisource/`. `ALDI_causality.ipynb` now reads from that folder, so the v4 files are not overwritten.

Steps 4 and 5 need no Earth Engine and can be re-run at any time.

## Spatial resolution: matching the covariates to point ALT

ALT is measured at a point (borehole, thaw tube) or over a probing grid of 10 m to 1 km.
ERA5-Land, the source of the climate variables, is a 0.1° grid (about 9 km north-south and
3 to 4 km east-west at these latitudes; ERA5 itself is 0.25°, about 31 km). What the
script does about the gap:

1. **Each site gets its own footprint (`buffer_m`).** The 10 m to 1 km products (Landsat,
   MODIS, terrain, soil, land cover) are averaged over a circle that matches the
   measurement: half the side of the probing grid for CALM grids (500 m for a 1 km grid, 50 m
   for a 100 m grid; sizes from the PANGAEA CALM collection, `site_footprints.csv`), 15 m for
   a borehole or thaw tube, 50 m for other probing. The earlier script used 500 m for every
   site. The circle is never smaller than 0.75 pixel of the product, so a borehole still
   gets the 250 m or 1 km pixel it is in.
2. **The coordinates limit how fine we can go.** 48 of the 329 sites have coordinates
   rounded to 0.1° (±5.6 km north-south; all 29 Roshydromet stations and 19 CALM sites)
   and 33 more are given to the minute (±0.9 km). The buffer is widened to the coordinate
   uncertainty (at most 1 km) and `coarse_location = 1` marks the sites where even 1 km is
   too small. For those, Landsat and terrain values may describe ground next to the site;
   leave them out in a sensitivity run, or look up better coordinates.
3. **Air temperature is corrected to the site's elevation.** 88 sites are more than 50 m,
   58 more than 100 m and 27 more than 200 m above or below the mean elevation of their
   ERA5-Land cell (checked with the Copernicus 90 m DEM; e.g. CALM N2 in Norway 660 m above,
   the Bonanza Creek fire plots 310 m below). Each day's temperature is shifted by
   lapse rate × (site elevation − cell elevation) before TDD, FDD and heat-wave days are
   computed: −6.5 °C per km from May to September, none from October to April, because
   winter inversions make valley floors colder than the cell mean, so a standard lapse rate
   would move them the wrong way. `lapse_rate_k_per_km` in `CONFIG` changes this, and
   `TDD_grid` etc. keep the uncorrected values. The daily values are exported once
   (group `t2m_daily`) and the indices are computed in Python
   (`temperature_from_daily`), so thresholds and lapse rates can be changed without
   running Earth Engine again.
4. **Finer products where they exist.** NDVI from Landsat at 30 m (one sensor family from
   1985, with TM and ETM+ converted to OLI) instead of MODIS 250 m after 2000 and AVHRR 5 km
   before. Snow-off date from MODIS daily snow cover at 500 m (from 2000), land surface
   temperature from MODIS at 1 km (from 2000), and drainage (`twi`, `hand_m`) from
   MERIT Hydro at 90 m.

**What this changes in the causal estimates.** The panel models use site fixed effects
(or first differences), so they only use each site's year-to-year changes. A constant
offset, such as a site being 300 m above its cell, cancels out. Year-to-year anomalies of
air temperature are similar over hundreds of km, so the 9 km grid already captures them
well, and the elevation correction changes the within-site estimates only a little (through
days that cross 0 °C or 20 °C). It matters more for the site-level variables: the effect
modifiers and strata (permafrost zone, soil, landform, the TDD level of a site), comparisons
between sites, and any model without site fixed effects. It does not create new information:
sites in the same ERA5-Land cell still share the same year-to-year temperatures, so keep
clustering standard errors by `era5_cell` or `region_id` in a robustness run.

**Snow and vegetation are where local detail matters most.** Snow depth and melt are set by
wind, terrain and shrubs at the scale of metres, which no 9 km model resolves. The MODIS
snow-off date describes the 500 m around the site, but only from 2000; mixing it with
ERA5-Land before 2000 would put a step in every series (as AVHRR to MODIS did for NDVI). So
`snow_off_doy` stays ERA5-Land by default (1990 onward), and `snow_off_for_causal =
"modis"` gives a 2000-onward robustness run. Snow depth and SWE have no satellite product
at this scale for these years and stay ERA5-Land. Soil moisture also stays ERA5-Land
(SMAP starts in 2015 at 9 km); its year-to-year changes are a model translation of the
same precipitation and temperature, not a measurement at the site.

**Not used, and why.** ArcticDEM (2 m) resolves hummocks and polygon rims, which is finer
than the footprints here, and its strip mosaics have voids and mixed dates; GLO-30 matches
the 15 m to 1 km buffers. SoilGrids 2.0 has the same 250 m grid as OpenLandMap and the
same lack of permafrost soil profiles. There is no gridded organic-layer thickness, which
is one of the strongest local controls on ALT; site descriptions are the only source.

## The ALT year and the seasons

Year Y runs from 1 October of Y-1 to 30 September of Y (`year_start_month = 10` in
`CONFIG`). Annual values (FDD, MAAT, P_annual, mean soil moisture, fire days) then end with
the thaw season they are paired with, and the winter and snow season before that thaw
belong to the same year. A calendar year would add October to December, which come after
the end-of-season thaw depth, and would split the winter before it between two years.
`year_start_month = 1` gives calendar years.

| Window | Months | Used for |
|---|---|---|
| Thaw season | May to Sep | TDD, T_thaw_mean |
| Summer | Jun to Aug | HWD, P_summer, NDVI, NDWI, LAI, T_JJA_mean, T_summer_max |
| Winter | Nov (Y-1) to Apr | ROS, T_winter_mean, wind, soil T winter, snow depth winter |
| Snow season | Oct (Y-1) to Jun | SWE_max, snow_depth_max |
| Snow-off search | Mar to Jun | SDO |

## Treatments (manuscript table)

| Column | Definition | Source |
|---|---|---|
| `TDD` | Sum of daily mean air temperature above 0 °C, May to Sep, corrected to the site's elevation (`TDD_grid` keeps the ERA5-Land cell value) | ERA5-Land daily + Copernicus GLO-30 |
| `FDD` | Sum of \|daily mean air temperature\| below 0 °C over the ALT year | ERA5-Land daily |
| `heat_wave_days` | Jun to Aug days with daily maximum air temperature above 20 °C | ERA5-Land daily |
| `SWE_max` | Largest daily snow water equivalent of the snow season (m) | ERA5-Land daily |
| `snow_off_doy` | First day of Mar to Jun with snow depth below 2 mm (`snow_off_for_causal` can switch it to the 500 m MODIS date, from 2000) | ERA5-Land daily |
| `P_summer` | Jun to Aug precipitation (mm) | ERA5-Land daily |
| `soil_moisture` | Mean volumetric soil water 0 to 7 cm over the ALT year | ERA5-Land daily |
| `NDWI_annual_mean` | Jun to Aug mean Gao NDWI, clear snow-free land pixels within 500 m (name kept for the notebook) | Landsat 5/7/8/9 C2 surface reflectance |
| `NDVI_summer` | Jun to Aug mean NDVI in the site buffer, Landsat 30 m (`ndvi_for_causal = "modis_avhrr"` gives the earlier MODIS 250 m / AVHRR 5 km series, kept as `NDVI_summer_modis_avhrr`) | Landsat 5/7/8/9 C2 surface reflectance |
| `FIRMS_fire_days` | Days with a FIRMS detection (confidence at least 30 %) within 1 km, from 2001 | FIRMS |
| `ROS_days` | Nov to Apr days with snow depth above 5 mm, precipitation above 1 mm and daily mean air temperature above 0 °C | ERA5-Land daily |

`feature_dictionary.csv` (from `write_dictionary()`) lists every column, including the
confounders, mediators and the extra variables, with its definition, source and units.

## Processing choices

- **ERA5-Land at coastal sites.** ERA5-Land has no values over sea cells. When a site's 0.1° cell is sea, the mean of the land cells around it is used: 3 × 3 cells, else 5 × 5, else 11 × 11 (about ±0.5°, like the earlier 50 km fill). `era5_filled` is 0 for the site's own cell and 1, 2 or 3 for the fill used, so filled sites can be left out in a sensitivity run. A land cell is never filled from its neighbours. The same is done for AVHRR (`avhrr_filled`).
- **Snow-off date.** When the snow depth stays at 2 mm or more all of March to June (late melt in the high Arctic), `snow_off_doy` is the day after the window (1 July) and `snow_off_after_window = 1`. It is then a lower bound; leaving these years out instead would drop the latest-melt years only.
- **ERA5-Land is a model.** Snow, soil moisture and soil temperature are land-model fields at about 9 km, driven by the same reanalysis as the air temperature and precipitation. Sites in the same cell get the same values, so the number of independent climate series is the number of cells, not of sites (see the review notes on clustering of standard errors).
- **NDVI and LAI across the sensor change.** AVHRR (before 2000) and MODIS do not read the same level. Before 2000 the AVHRR value is shifted by the site's median MODIS minus AVHRR difference over the years both exist (2000 to 2013), or by the median over all sites when a site has fewer than 5 such years. Without this, the change of sensor in 2000 would be a step in every site's series, which site fixed effects do not remove. `NDVI_summer_source` says which sensor each value comes from.
- **Landsat.** Pixels flagged as fill, cloud, dilated cloud, cirrus, cloud shadow, snow or water are removed. TM and ETM+ reflectance is converted to OLI with Roy et al. (2016) before the indices. Landsat 7 is not used after 2016 (orbit drift). `NDWI_n_obs` is the mean number of clear summer observations per pixel; few observations are common in the 1990s.
- **Fire.** `fire_persistent_source = 1` marks sites with detections in 5 or more different years, which in tundra is more likely a gas flare or industrial heat source than wildfire. These are worth checking before using FIRMS_fire_days.
- **Wind.** Daily wind speed is computed from the daily mean wind components, so it is lower than the mean of hourly speeds. `wind_direction_winter` is the direction the mean winter wind comes from; `u10_winter` and `v10_winter` express the same in a form usable in regressions (a direction in degrees jumps from 359 to 0).
- **Terrain.** Copernicus GLO-30 DEM (SRTM stops at 60° N). Slope and aspect are computed per DEM tile, in the tile's own grid, and averaged over the site buffer. `northness` and `eastness` are weighted by sin(slope), so flat ground has no aspect. `tpi` is the mean elevation in the site buffer minus the mean within 3 km; `tri` is the standard deviation of elevation within 1 km (both describe the landscape around the site, so their radii stay fixed). `insolation_ratio_thaw` is the clear-sky direct sunlight on the site's slope over May to Sep relative to flat ground (sun geometry only, no shading by the hills around). `twi` (topographic wetness index) and `hand_m` (height above the nearest drainage) come from MERIT Hydro at 90 m and say whether water collects at the site.
- **Distance to coast.** Great-circle distance to the Natural Earth 1:10m coastline (downloaded in Colab). `continentality_dist = log(1 + dist_to_coast_km)`.
- **Permafrost classes.** A site inside an IPA map polygon takes its extent, ground ice and landform classes. A site in a lake or on the coast takes the nearest permafrost polygon within 20 km, otherwise "Unknown".
- **ALT column.** The augmented files name the ALT column `Max` so the notebook runs unchanged. It holds `ALT_cm` of the multi-source dataset (site-mean end-of-season thaw depth), not the v4 grid maximum.

## Changes from the earlier script

October 2026 (finer resolution): per-site buffers from the measurement footprint and
coordinate precision; elevation-corrected air temperature from daily ERA5-Land
(`TDD`, `FDD`, `MAAT`, `T_*`, `heat_wave_days`; `*_grid` keep the cell values);
`heat_wave_days_p90`; Landsat NDVI as `NDVI_summer`; MODIS snow-off date, land surface
temperature; `twi`, `hand_m`, `insolation_ratio_thaw`; Landsat summer means from fewer than
2 clear observations per pixel are left empty (NDWI too).

Earlier:

Compared with `Data_acquisition_causality_analysis.py` (the version in the project files):

- Annual and snow variables use the ALT year instead of the calendar year; winter is Nov to Apr instead of Dec to Feb.
- Terrain comes from Copernicus GLO-30 instead of SRTM, which has no data north of 60° N.
- Distance to coast uses the full Natural Earth coastline instead of 25 hand-picked coastal points.
- AVHRR NDVI is cloud, shadow, water and night masked, and adjusted to MODIS.
- `T_annual_range` is warmest minus coldest monthly mean (was Jun to Aug mean minus Dec to Feb mean).
- NDWI follows the manuscript table (Gao, summer mean). The earlier values most likely came from the Earth Engine Landsat annual NDWI composite, which is McFeeters NDWI (green and NIR) and keeps the most recent pixel of the year. `NDWI_mcfeeters_summer` is also given; `CONFIG["ndwi_for_causal"]` selects which one fills `NDWI_annual_mean`.
- Coastal ERA5-Land cells are filled instead of left empty.
- The permafrost classes are computed here (Python version of `assignEnvironmentalCategories` in the MATLAB script). The MATLAB version copied classes from the nearest classified site, but `permafrostType(k)` used `k` as a position among the classified sites to index all sites, so it copied from the wrong site.

## Data sources

ERA5-Land (Muñoz-Sabater et al. 2021, doi:10.5194/essd-13-4349-2021); MODIS MOD13Q1/MYD13Q1,
MOD15A2H, MCD43A3, MCD64A1, MCD12Q1, MOD11A2/MYD11A2 (NASA LP DAAC, Collection 6.1) and
MOD10A1/MYD10A1 (NSIDC, Collection 6.1); MERIT Hydro (Yamazaki et al. 2019,
doi:10.1029/2019WR024873); PANGAEA CALM collection for the grid sizes (Streletskiy et al. 2025,
doi:10.1594/PANGAEA.972777); NOAA CDR AVHRR NDVI
and LAI/FAPAR version 5; Landsat Collection 2 Level 2 (USGS); FIRMS (NASA LANCE); Copernicus
DEM GLO-30 (ESA); ESA WorldCover 2020 v100 (Zanaga et al. 2021); OpenLandMap soil layers
(Hengl et al.); ALOS landforms and mTPI (Theobald et al. 2015); Natural Earth coastline;
IPA permafrost map (Brown et al. 2002); Landsat harmonization coefficients from Roy et al.
(2016, doi:10.1016/j.rse.2015.12.024).
