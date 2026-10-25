# Covariates for the ALT causal analysis (Google Earth Engine)

`gee_causal_features.py` builds the treatments, confounders and mediators of the manuscript
DAG tables for every site of the multi-source ALT dataset, one row per site and year, with
the column names that `Scripts/ALDI_causality.ipynb` uses. It then joins them with the ALT
values and the ALDI classes and writes `ALDI_withYearlyData_augmented*.csv`, the files the
causal notebook reads. `GEE_causal_features_colab.ipynb` runs it in Colab.

## What you need

| Input | Where it comes from |
|---|---|
| `alt_multisource_for_ALDI.csv` | the multi-source dataset (`Scripts/multisource/`); in Drive at `UND/Index/alt_multisource/`, where the ALDI notebook reads it |
| `aldi_multisource_results.csv` | `ALDI_Calculation_final.ipynb` (run with `min_year=1990`); in Drive at `UND/Index/` |
| `permaice.shp` (+ .shx, .dbf, .prj) | IPA Circum-Arctic Map of Permafrost and Ground-Ice Conditions (Brown et al. 2002, NSIDC GGD318), the same file the MATLAB script used; put it in `UND/Index/ggd318_map_circumarctic/` |
| An Earth Engine Cloud project | <https://code.earthengine.google.com/register> |

Put `gee_causal_features.py` in `/content/drive/MyDrive/UND/Index/scripts/`. The notebook
sets all paths in its second cell.

## Run (Colab)

1. `quick_test(sites)`: three sites, one year, all variables, computed directly (about a minute). Check that the values look sensible.
2. `export_all(sites, drive_dir=EXPORTS)`: starts about 150 Earth Engine export tasks (one per data group and year). They write CSV files to the Drive folder `ALDI_GEE_exports`. They run on Google's side; Colab can be closed. Running it again only starts the missing ones.
3. `task_status()`: how many tasks are done or failed.
4. `assemble_features(...)`: reads the exports and adds distance to coast, continentality, the AVHRR-to-MODIS adjustment and the permafrost map classes. Writes `Index/multisource/ALDI_causal_features_complete.csv`.
5. `build_augmented(...)`: writes `ALDI_withYearlyData_augmented.csv` and the class and permafrost-zone subsets to `Index/multisource/`. `ALDI_causality.ipynb` now reads from that folder, so the v4 files are not overwritten.

Steps 4 and 5 need no Earth Engine and can be re-run at any time.

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
| `TDD` | Sum of daily mean air temperature above 0 °C, May to Sep | ERA5-Land daily |
| `FDD` | Sum of \|daily mean air temperature\| below 0 °C over the ALT year | ERA5-Land daily |
| `heat_wave_days` | Jun to Aug days with daily maximum air temperature above 20 °C | ERA5-Land daily |
| `SWE_max` | Largest daily snow water equivalent of the snow season (m) | ERA5-Land daily |
| `snow_off_doy` | First day of Mar to Jun with snow depth below 2 mm | ERA5-Land daily |
| `P_summer` | Jun to Aug precipitation (mm) | ERA5-Land daily |
| `soil_moisture` | Mean volumetric soil water 0 to 7 cm over the ALT year | ERA5-Land daily |
| `NDWI_annual_mean` | Jun to Aug mean Gao NDWI, clear snow-free land pixels within 500 m (name kept for the notebook) | Landsat 5/7/8/9 C2 surface reflectance |
| `NDVI_summer` | Jun to Aug mean NDVI within 500 m; MODIS from 2000, AVHRR adjusted to MODIS before | MOD13Q1 + MYD13Q1, NOAA CDR AVHRR V5 |
| `FIRMS_fire_days` | Days with a FIRMS detection (confidence at least 30 %) within 1 km, from 2001 | FIRMS |
| `ROS_days` | Nov to Apr days with snow depth above 5 mm, precipitation above 1 mm and daily mean air temperature above 0 °C | ERA5-Land daily |

`feature_dictionary.csv` (from `write_dictionary()`) lists every column, including the
confounders, mediators and the extra variables, with its definition, source and units.

## Processing choices

- **ERA5-Land at coastal sites.** ERA5-Land has no values over sea cells. When a site's 0.1° cell is sea, the mean of the land cells around it is used (3 × 3, then 5 × 5 cells) and `era5_filled = 1`. The same is done for AVHRR (`avhrr_filled`).
- **NDVI and LAI across the sensor change.** AVHRR (before 2000) and MODIS do not read the same level. Before 2000 the AVHRR value is shifted by the site's median MODIS minus AVHRR difference over the years both exist (2000 to 2013), or by the median over all sites when a site has fewer than 5 such years. Without this, the change of sensor in 2000 would be a step in every site's series, which site fixed effects do not remove. `NDVI_summer_source` says which sensor each value comes from.
- **Landsat.** Pixels flagged as fill, cloud, dilated cloud, cirrus, cloud shadow, snow or water are removed. TM and ETM+ reflectance is converted to OLI with Roy et al. (2016) before the indices. Landsat 7 is not used after 2016 (orbit drift). `NDWI_n_obs` is the mean number of clear summer observations per pixel; few observations are common in the 1990s.
- **Fire.** `fire_persistent_source = 1` marks sites with detections in 5 or more different years, which in tundra is more likely a gas flare or industrial heat source than wildfire. These are worth checking before using FIRMS_fire_days.
- **Wind.** Daily wind speed is computed from the daily mean wind components, so it is lower than the mean of hourly speeds. `wind_direction_winter` is the direction the mean winter wind comes from; `u10_winter` and `v10_winter` express the same in a form usable in regressions (a direction in degrees jumps from 359 to 0).
- **Terrain.** Copernicus GLO-30 DEM (SRTM stops at 60° N). Slope and aspect are computed per DEM tile, in the tile's own grid. `northness` and `eastness` are weighted by sin(slope), so flat ground has no aspect. `tpi` is the mean elevation within 500 m minus the mean within 3 km; `tri` is the standard deviation of elevation within 1 km.
- **Distance to coast.** Great-circle distance to the Natural Earth 1:10m coastline (downloaded in Colab). `continentality_dist = log(1 + dist_to_coast_km)`.
- **Permafrost classes.** A site inside an IPA map polygon takes its extent, ground ice and landform classes. A site in a lake or on the coast takes the nearest permafrost polygon within 20 km, otherwise "Unknown".
- **ALT column.** The augmented files name the ALT column `Max` so the notebook runs unchanged. It holds `ALT_cm` of the multi-source dataset (site-mean end-of-season thaw depth), not the v4 grid maximum.

## Changes from the earlier script

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
MOD15A2H, MCD43A3, MCD64A1 and MCD12Q1 (NASA LP DAAC, Collection 6.1); NOAA CDR AVHRR NDVI
and LAI/FAPAR version 5; Landsat Collection 2 Level 2 (USGS); FIRMS (NASA LANCE); Copernicus
DEM GLO-30 (ESA); ESA WorldCover 2020 v100 (Zanaga et al. 2021); OpenLandMap soil layers
(Hengl et al.); ALOS landforms and mTPI (Theobald et al. 2015); Natural Earth coastline;
IPA permafrost map (Brown et al. 2002); Landsat harmonization coefficients from Roy et al.
(2016, doi:10.1016/j.rse.2015.12.024).
