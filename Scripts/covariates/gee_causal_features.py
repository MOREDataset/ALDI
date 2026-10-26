"""
Covariates for the ALT causal analysis (Scripts/ALDI_causality.ipynb), from Google Earth Engine.

What it does
------------
For every site of the multi-source ALT dataset and every year (1990 onward by default) it
extracts the treatments, confounders and mediators of the manuscript DAG tables, with the
column names that ALDI_causality.ipynb uses. Then it joins them with the ALT values and the
ALDI classes and writes ALDI_withYearlyData_augmented.csv and the subset files that the
causal notebook reads.

Spatial scale
-------------
Each site's values come from the finest product available for that variable, averaged
over the site's own footprint (buffer_m: the probing grid, or 15 m for a borehole or tube,
widened when the coordinates are rounded). Air temperature from ERA5-Land (0.1 degree, about
9 km) is corrected to the site's elevation with a lapse rate (temperature_from_daily).
NDVI comes from Landsat (30 m), and snow-off date and surface temperature are also given
from MODIS (500 m, 1 km; from 2000). See README.md, "Spatial resolution".

Run it in Colab (see README.md in this folder, or GEE_causal_features_colab.ipynb):
    1. quick_test(sites)        a few sites, one year, interactive (about a minute)
    2. export_all(sites)        starts Earth Engine export tasks to Google Drive
    3. task_status()            shows how many tasks are done
    4. assemble_features(...)   reads the exports, adds the offline features
    5. build_augmented(...)     writes the files for ALDI_causality.ipynb

Steps 4 and 5 use only pandas (plus geopandas and scikit-learn for distance to coast and
the permafrost map), so they can be re-run without Earth Engine.

The "ALT year"
--------------
Year Y runs from 1 October of Y-1 to 30 September of Y (CONFIG["year_start_month"] = 10).
Annual sums and means (FDD, MAAT, P_annual, soil moisture, fire days...) therefore end with
the thaw season they are paired with, and the winter and snow season before the thaw belong
to the same year. Seasonal windows (May-Sep, JJA, Nov-Apr...) are in CONFIG. Set
year_start_month to 1 to get calendar years, as in the earlier script.
"""

import glob
import math
import os

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:  # steps 4 and 5 do not need Earth Engine
    ee = None


# =============================================================================
# SETTINGS (each value with the reason for it)
# =============================================================================
CONFIG = {
    # --- years -------------------------------------------------------------
    "first_year": 1990,            # ALDI input starts in 1990 (min_year=1990)
    "last_year": 2025,             # last complete thaw season
    "year_start_month": 10,        # ALT year = Oct of Y-1 to Sep of Y (see module docstring)

    # --- seasons: (first month, last month); a range that wraps over the new
    #     year starts in Y-1 (Nov-Apr = Nov of Y-1 to Apr of Y) ---------------
    "thaw_months": (5, 9),         # TDD, T_thaw_mean: May-Sep (manuscript)
    "summer_months": (6, 8),       # JJA: HWD, P_summer, NDVI, NDWI, LAI (manuscript)
    "winter_months": (11, 4),      # Nov-Apr: ROS (manuscript), winter means, wind, soil T winter
    "snow_season_months": (10, 6), # Oct-Jun: SWE_max, snow_depth_max (whole snow season before the thaw)
    "snow_off_months": (3, 6),     # SDO searched in Mar-Jun (manuscript)

    # --- thresholds from the manuscript table ---------------------------------
    "heat_wave_tmax_c": 20.0,      # HWD: JJA days with Tmax > 20 C
    "snow_off_depth_m": 0.002,     # SDO: snow depth < 2 mm
    "ros_snow_depth_m": 0.005,     # ROS: snow > 5 mm
    "ros_precip_m": 0.001,         # ROS: precipitation > 1 mm
    "ros_tmean_c": 0.0,            # ROS: daily mean air temperature > 0 C

    # --- site footprint: the area each site's 30 m - 1 km values are averaged over --
    # radius = max(footprint of the measurement, uncertainty of the coordinates), within
    # [min_buffer_m, max_buffer_m]; see site_buffers(). The earlier script used 500 m for all.
    "footprint_radius_m": {        # when the grid size is not known (site_footprints.csv gives
        "probe": 50,               # it for CALM grids: half the grid side, e.g. 500 m for 1 km)
        "thaw_tube": 15,           # a tube, a borehole or a sensor profile is a point
        "borehole_temp": 15,
        "soil_temp": 15,
        "soil_temp_monthly": 15,
    },
    "default_footprint_m": 50,     # other methods
    "min_buffer_m": 15,            # half a Landsat pixel
    "max_buffer_m": 1000,          # coordinates rounded to 0.1 degree (+-5 km) are capped here
                                   # and flagged with coarse_location = 1
    "buffer_m": 500,               # used only when a site has no buffer_m (old site tables)
    "tpi_outer_m": 3000,           # TPI = mean elevation in the buffer minus mean within 3 km
    "tri_radius_m": 1000,          # TRI = standard deviation of elevation within 1 km
    "fire_radius_m": 1000,         # a fire day = a FIRMS detection within 1 km of the site
    "fire_min_confidence": 30,     # drops the MODIS low-confidence class (0-29 %)

    # --- air temperature downscaling (elevation correction of ERA5-Land) ---------
    # T_site = T_cell + lapse_rate * (z_site - z_cell) / 1000, applied to every day before
    # TDD, FDD, heat-wave days... are computed (see temperature_from_daily). z_site is the
    # mean Copernicus GLO-30 elevation of the site buffer, z_cell the mean of the land part
    # of the site's ERA5-Land 0.1 degree cell. Summer: the standard -6.5 K/km. Oct-Apr: no
    # correction, because winter inversions make valley floors colder than the slopes above
    # them (e.g. the Fairbanks valley sites, 300 m below their cell mean), so a standard
    # lapse rate would warm them, the wrong way. Set a winter value to change that.
    "lapse_rate_k_per_km": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -6.5, 6: -6.5, 7: -6.5,
                            8: -6.5, 9: -6.5, 10: 0.0, 11: 0.0, 12: 0.0},
    "temperature_for_causal": "downscaled",  # TDD, FDD, MAAT... hold the "downscaled" or the
                                             # "grid" values (both are always written)
    "heat_wave_percentile": 90,    # heat_wave_days_p90: JJA days above the site's own 90th
    "heat_wave_ref_years": (1991, 2020),     # percentile of daily max temperature, 1991-2020

    # --- MODIS snow-off date (500 m, 2000 onward) ----------------------------------
    "modis_snow_months": (3, 8),   # Mar-Aug: wide enough for July melt in the high Arctic
    "modis_ndsi_snow": 40,         # snow = NDSI snow cover >= 40 (NDSI 0.4, the classic MODIS
                                   # snow threshold, about half the pixel snow covered)
    "modis_snow_min_clear": 10,    # at least 10 clear observations in the window
    "snow_off_for_causal": "era5land",  # what snow_off_doy holds: "era5land" (1990 onward,
                                        # 9 km), "modis" (2000 onward, 500 m) or "combined"
                                        # (MODIS, ERA5-Land shifted to MODIS before 2000)

    # --- choices ---------------------------------------------------------------
    "ndwi_for_causal": "gao_summer",   # what NDWI_annual_mean holds: "gao_summer" (manuscript
                                       # table) or "mcfeeters_summer"
    "ndvi_for_causal": "landsat",      # what NDVI_summer holds: "landsat" (30 m, one sensor
                                       # family 1985 onward) or "modis_avhrr" (250 m from 2000,
                                       # 5 km AVHRR adjusted to MODIS before)
    "landsat_min_obs": 2,          # Landsat summer means from fewer clear observations per
                                   # pixel are left empty
    "harmonize_landsat": True,     # TM/ETM+ reflectance to OLI before the indices (Roy et al. 2016)
    "min_overlap_years": 5,        # AVHRR-to-MODIS offset per site needs >= 5 shared years
    "persistent_fire_years": 5,    # fire detections in >= 5 different years = possible gas flare

    # --- outputs ---------------------------------------------------------------
    "export_folder": "ALDI_GEE_exports",   # Google Drive folder that Earth Engine writes to
}

K0 = 273.15  # 0 C in kelvin

# Roy et al. (2016, Remote Sens. Environ. 185, 57-70, doi:10.1016/j.rse.2015.12.024), Table 2:
# OLI = intercept + slope * ETM+, for green, NIR and SWIR1 surface reflectance (TM uses the same).
ROY2016 = {"red": (0.0061, 0.9047), "green": (0.0088, 0.8483), "nir": (0.0412, 0.8462),
           "swir1": (0.0254, 0.8937)}

# First and last ALT year each group can cover (from the Earth Engine catalog dates).
GROUP_YEARS = {
    "era5": (1951, 2100),          # ERA5-Land daily from 1950-01-02
    "modis": (2000, 2100),         # MOD13Q1 / MOD15A2H / MCD43A3 from Feb 2000, MCD64A1 from Nov 2000
    "avhrr": (1982, 2013),         # NOAA CDR AVHRR NDVI and LAI V5: 1981-06-24 to 2013-12-31
    "landsat": (1985, 2100),       # Landsat 5 from 1984
    "fire": (2001, 2100),          # FIRMS from 2000-11-01
    "t2m_daily": (1951, 2100),     # daily ERA5-Land air temperature, for the downscaling
    "snowmelt": (2000, 2100),      # MOD10A1 from 2000-02-24 (MYD10A1 from mid-2002)
}

ERA5_BANDS = [
    "MAAT", "T_JJA_mean", "T_summer_mean", "T_summer_max", "T_thaw_mean", "T_winter_mean",
    "T_annual_range", "TDD", "FDD", "heat_wave_days",
    "P_annual", "P_summer", "rain_summer", "snowfall_winter",
    "SWE_max", "snow_depth_max", "snow_depth_winter", "snow_off_doy", "snow_off_after_window",
    "snow_off_doy_last",
    "ROS_days", "soil_moisture", "soil_moisture_summer",
    "soil_T", "soil_T_winter_mean", "soil_T_winter_min",
    "albedo_annual", "SW_down_summer",
    "wind_speed_winter_mean", "wind_speed_winter_max", "wind_speed_winter_p90",
    "u10_winter", "v10_winter",
]
MODIS_BANDS = ["NDVI_summer_modis", "NDVI_max_modis", "LAI_summer_modis",
               "albedo_summer_modis", "burned_frac", "LST_JJA_mean", "LST_thaw_index"]
SNOWMELT_BANDS = ["snow_off_doy_modis", "snow_off_modis_censored", "snow_n_clear"]
AVHRR_BANDS = ["NDVI_summer_avhrr", "NDVI_max_avhrr", "LAI_summer_avhrr"]
LANDSAT_BANDS = ["NDWI_gao_summer", "NDWI_mcfeeters_summer", "NDWI_n_obs", "NDVI_landsat_summer"]
STATIC_BANDS = [
    "elevation_m", "slope_deg", "northness", "eastness", "elev_mean_outer", "tri",
    "landcover_esa_2020", "landcover_igbp", "landform_alos", "mtpi_alos",
    "soil_texture_0cm", "soil_texture_30cm", "soc_0", "soc_10", "soc_30",
    "bulk_density_0cm", "bulk_density_30cm", "clay_pct_0cm", "sand_pct_0cm",
    "z_cell_era5", "twi", "hand_m",
]
ALL_GROUPS = ("static", "era5", "t2m_daily", "modis", "snowmelt", "avhrr", "landsat", "fire")


# =============================================================================
# SITES
# =============================================================================
def load_sites(alt_csv, first_year=None, footprints_csv=None, cfg=CONFIG):
    """One row per site (site_id, lat, lon, method, buffer_m...) from alt_multisource_for_ALDI.csv.

    Only sites with at least one ALT year from first_year onward are kept. footprints_csv
    (optional, from build_site_footprints) gives the CALM grid sizes; without it the
    footprint comes from the measurement method alone (see site_buffers)."""
    first_year = cfg["first_year"] if first_year is None else first_year
    alt = pd.read_csv(alt_csv)
    alt = alt[alt["Year"] >= first_year]
    agg = dict(lat=("Lat", "first"), lon=("Long", "first"), Location=("Location", "first"))
    if "method" in alt.columns:
        agg["method"] = ("method", lambda m: m.mode().iat[0] if m.notna().any() else "")
    sites = alt.groupby("site_id", as_index=False).agg(**agg)
    fp = pd.read_csv(footprints_csv) if footprints_csv else None
    sites = site_buffers(sites, fp, cfg)
    print(f"{len(sites)} sites with ALT values from {first_year}; buffer radius median "
          f"{sites['buffer_m'].median():.0f} m, {int(sites['coarse_location'].sum())} sites "
          f"with coordinates rounded to more than {cfg['max_buffer_m']} m")
    return sites


# Steps a coordinate can be rounded to, coarse to fine (degrees): 0.1 degree, 1 arc-minute,
# 0.01 degree, 1 arc-second, then decimals.
_COORD_STEPS = [0.1, 1 / 60, 0.01, 1 / 3600, 1e-3, 1e-4, 1e-5]


def coord_step_deg(x):
    """Coarsest rounding step that x is a whole multiple of (e.g. 65.28333 -> 1/60, a
    position given in degrees and minutes). Very fine coordinates return 1e-6."""
    for step in _COORD_STEPS:
        q = x / step
        if abs(q - round(q)) < 1e-3:
            return step
    return 1e-6


def site_buffers(sites, footprints=None, cfg=CONFIG):
    """Adds the columns that set the area each site's values are averaged over.

    footprint_m: half the side of the probing grid (CALM grids from footprints, e.g. 500 m
    for a 1 km grid, 50 m for a 100 m grid), else CONFIG footprint_radius_m by method
    (15 m for a borehole or thaw tube, which are points).
    coord_uncert_m: half the rounding step of the coordinates (for example +-925 m north-south
    for a position given to the minute, +-5.6 km for one given to 0.1 degree).
    buffer_m: the larger of the two, within [min_buffer_m, max_buffer_m].
    coarse_location: 1 when the coordinate uncertainty is larger than max_buffer_m. Fine
    covariates (Landsat, terrain) of these sites may describe ground next to the site,
    not the site itself."""
    s = sites.copy()
    fp_by_method = cfg["footprint_radius_m"]
    s["footprint_m"] = s.get("method", pd.Series("", index=s.index)).map(fp_by_method)
    s["footprint_source"] = np.where(s["footprint_m"].notna(), "method", "default")
    if footprints is not None and len(footprints):
        known = footprints.set_index("site_id")["footprint_m"].dropna()
        hit = s["site_id"].isin(known.index)
        s.loc[hit, "footprint_m"] = s.loc[hit, "site_id"].map(known)
        s.loc[hit, "footprint_source"] = "grid_size"
    s["footprint_m"] = s["footprint_m"].fillna(cfg["default_footprint_m"]).astype(float)
    step = np.maximum(s["lat"].map(coord_step_deg), s["lon"].map(coord_step_deg))
    s["coord_step_deg"] = step
    s["coord_uncert_m"] = step / 2 * 111320.0          # north-south; east-west is smaller
    s["buffer_m"] = (np.maximum(s["footprint_m"], s["coord_uncert_m"])
                     .clip(cfg["min_buffer_m"], cfg["max_buffer_m"]).round(0))
    s["coarse_location"] = (s["coord_uncert_m"] > cfg["max_buffer_m"]).astype(int)
    return s


def build_site_footprints(sites_csv, pangaea_972777_tab, out_csv=None):
    """site_footprints.csv: probing grid size of the CALM sites, from the event list of the
    PANGAEA CALM collection (doi:10.1594/PANGAEA.972777, 'Method: Spatially-oriented
    mechanical probing at 100 m Grid or Transect'). footprint_m = half the grid side.
    Needs to be run once (the file is in the project's alt_multisource folder); load_sites
    reads it."""
    import re
    text = open(pangaea_972777_tab, encoding="utf-8").read()
    events = text.split("Event(s):", 1)[1].split("\nParameter(s):", 1)[0]
    rows = []
    for m in re.finditer(r"(CALM_[A-Za-z0-9_]+) \(.*?Method: ([^\n*]+)", events):
        code, method = m.group(1), m.group(2).strip()
        g = re.search(r"probing at (\d+(?:\.\d+)?) m", method)
        rows.append(dict(site_id=code, calm_method=method,
                         grid_side_m=float(g.group(1)) if g else np.nan))
    ev = pd.DataFrame(rows).drop_duplicates("site_id")
    sites = pd.read_csv(sites_csv)[["site_id"]]
    out = sites.merge(ev, on="site_id", how="left")
    out["footprint_m"] = out["grid_side_m"] / 2
    print(f"grid size found for {out['footprint_m'].notna().sum()} of {len(out)} sites")
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out


def make_sites_fc(sites):
    """Earth Engine points, one per site, with the site_id and buffer_m properties."""
    has_buf = "buffer_m" in sites.columns
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                        {"site_id": str(r.site_id),
                         "buffer_m": float(r.buffer_m) if has_buf else CONFIG["buffer_m"]})
             for r in sites.itertuples()]
    return ee.FeatureCollection(feats)


# =============================================================================
# DATES
# =============================================================================
def alt_year_months(cfg=CONFIG):
    m = cfg["year_start_month"]
    return (1, 12) if m == 1 else (m, m - 1)


def _window_ymd(year, months):
    """(start, end) as (y, m) tuples; end is exclusive (first day after the window)."""
    m0, m1 = months
    y0 = year - 1 if m0 > m1 else year
    y1, e1 = (year + 1, 1) if m1 == 12 else (year, m1 + 1)
    return (y0, m0), (y1, e1)


def _window(year, months):
    (y0, m0), (y1, m1) = _window_ymd(year, months)
    return ee.Date.fromYMD(y0, m0, 1), ee.Date.fromYMD(y1, m1, 1)


def _n_days(year, months):
    (y0, m0), (y1, m1) = _window_ymd(year, months)
    return (pd.Timestamp(y1, m1, 1) - pd.Timestamp(y0, m0, 1)).days


def _empty(name):
    """A fully masked one-band image, used when a collection has no images."""
    return ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename(name)


def _safe(col, reduce_fn, name):
    """reduce_fn(col) renamed, or a masked band when col is empty (avoids a band-less image)."""
    return ee.Image(ee.Algorithms.If(col.size().gt(0), reduce_fn(col).rename(name), _empty(name)))


# =============================================================================
# EARTH ENGINE IMAGES, ONE PER GROUP AND YEAR
# =============================================================================
def era5_image(year, cfg=CONFIG):
    """ERA5-Land daily aggregates (0.1 degree): climate, snow, soil, wind."""
    daily = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

    def sub(months):
        s, e = _window(year, months)
        return daily.filterDate(s, e)

    yr = sub(alt_year_months(cfg))
    thaw, jja = sub(cfg["thaw_months"]), sub(cfg["summer_months"])
    win, snow = sub(cfg["winter_months"]), sub(cfg["snow_season_months"])
    melt = sub(cfg["snow_off_months"])
    t = "temperature_2m"

    def doy(img):
        return ee.Image.constant(ee.Number(img.date().getRelative("day", "year")).add(1)).toFloat()

    # Land cells of ERA5-Land (sea cells are masked in every band). A land cell that is
    # masked in one band below has a reason (no snow-off in the window, ...) and must not
    # be filled from its neighbours like a sea cell (see _sample_grid).
    land = yr.select(t).mean().mask().gt(0)
    (y0, m0), (y1, m1) = _window_ymd(year, cfg["snow_off_months"])
    first_doy = (pd.Timestamp(y0, m0, 1) - pd.Timestamp(y0, 1, 1)).days + 1
    after_doy = (pd.Timestamp(y1, m1, 1) - pd.Timestamp(y0, 1, 1)).days + 1

    # SDO: first day of the window with snow depth below the threshold (manuscript
    # definition). Snow all through the window (late melt in the high Arctic): the day
    # after the window, flagged by snow_off_after_window = 1 (a lower bound, not a gap,
    # so these years are not lost).
    off = melt.map(lambda i: doy(i).updateMask(
        i.select("snow_depth").lt(cfg["snow_off_depth_m"]))).min()
    snow_off = off.unmask(after_doy).updateMask(land)
    snow_off_late = off.mask().eq(0).updateMask(land)
    # extra: the day after the last snow day of the window (ignores short melt spells);
    # no snow day in the window: its first day
    snow_last = melt.map(lambda i: doy(i).updateMask(
        i.select("snow_depth").gte(cfg["snow_off_depth_m"]))).max().add(1) \
        .unmask(first_doy).updateMask(land)

    def ros_day(i):
        return (i.select("snow_depth").gt(cfg["ros_snow_depth_m"])
                .And(i.select("total_precipitation_sum").gt(cfg["ros_precip_m"]))
                .And(i.select(t).gt(K0 + cfg["ros_tmean_c"])))

    # daily wind speed from the daily mean wind components
    ws = win.map(lambda i: i.select("u_component_of_wind_10m")
                 .hypot(i.select("v_component_of_wind_10m")).rename("ws"))

    ys, ye = _window(year, alt_year_months(cfg))
    monthly = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(ys, ye).select(t)

    bands = [
        yr.select(t).mean().subtract(K0).rename("MAAT"),
        jja.select(t).mean().subtract(K0).rename("T_JJA_mean"),
        jja.select(t).mean().subtract(K0).rename("T_summer_mean"),
        jja.select("temperature_2m_max").max().subtract(K0).rename("T_summer_max"),
        thaw.select(t).mean().subtract(K0).rename("T_thaw_mean"),
        win.select(t).mean().subtract(K0).rename("T_winter_mean"),
        monthly.max().subtract(monthly.min()).rename("T_annual_range"),
        thaw.select(t).map(lambda i: i.subtract(K0).max(0)).sum().rename("TDD"),
        yr.select(t).map(lambda i: i.subtract(K0).min(0).multiply(-1)).sum().rename("FDD"),
        jja.select("temperature_2m_max").map(
            lambda i: i.gt(K0 + cfg["heat_wave_tmax_c"])).sum().rename("heat_wave_days"),
        yr.select("total_precipitation_sum").sum().multiply(1000).rename("P_annual"),
        jja.select("total_precipitation_sum").sum().multiply(1000).rename("P_summer"),
        jja.map(lambda i: i.select("total_precipitation_sum").subtract(i.select("snowfall_sum"))
                .max(0)).sum().multiply(1000).rename("rain_summer"),
        win.select("snowfall_sum").sum().multiply(1000).rename("snowfall_winter"),
        snow.select("snow_depth_water_equivalent").max().rename("SWE_max"),
        snow.select("snow_depth").max().rename("snow_depth_max"),
        win.select("snow_depth").mean().rename("snow_depth_winter"),
        snow_off.rename("snow_off_doy"),
        snow_off_late.rename("snow_off_after_window"),
        snow_last.rename("snow_off_doy_last"),
        win.map(ros_day).sum().rename("ROS_days"),
        yr.select("volumetric_soil_water_layer_1").mean().rename("soil_moisture"),
        jja.select("volumetric_soil_water_layer_1").mean().rename("soil_moisture_summer"),
        yr.select("soil_temperature_level_1").mean().subtract(K0).rename("soil_T"),
        win.select("soil_temperature_level_1").mean().subtract(K0).rename("soil_T_winter_mean"),
        win.select("soil_temperature_level_1_min").min().subtract(K0).rename("soil_T_winter_min"),
        yr.select("forecast_albedo").mean().rename("albedo_annual"),
        jja.select("surface_solar_radiation_downwards_sum").mean().divide(1e6).rename("SW_down_summer"),
        ws.mean().rename("wind_speed_winter_mean"),
        ws.max().rename("wind_speed_winter_max"),
        ws.reduce(ee.Reducer.percentile([90])).rename("wind_speed_winter_p90"),
        win.select("u_component_of_wind_10m").mean().rename("u10_winter"),
        win.select("v_component_of_wind_10m").mean().rename("v10_winter"),
    ]
    return ee.Image.cat(bands)


def modis_image(year, cfg=CONFIG):
    """MODIS NDVI (250 m), LAI, summer albedo and burned area (500 m), land surface
    temperature (1 km)."""
    s, e = _window(year, cfg["summer_months"])
    ys, ye = _window(year, alt_year_months(cfg))

    vi = ee.ImageCollection("MODIS/061/MOD13Q1").merge(ee.ImageCollection("MODIS/061/MYD13Q1"))

    def vi_clean(i):  # SummaryQA 0 good, 1 marginal; 2 snow and 3 cloud are dropped
        return i.select("NDVI").updateMask(i.select("SummaryQA").lte(1)).multiply(0.0001)

    def lai_clean(i):  # main algorithm, no significant clouds, no fill values (> 100)
        qc = i.select("FparLai_QC")
        cloud = qc.rightShift(3).bitwiseAnd(3)
        ok = qc.bitwiseAnd(1).eq(0).And(cloud.eq(0).Or(cloud.eq(3)))
        lai = i.select("Lai_500m")
        return lai.updateMask(ok).updateMask(lai.lte(100)).multiply(0.1)

    def alb_clean(i):  # full BRDF inversions only
        return (i.select("Albedo_WSA_shortwave")
                .updateMask(i.select("BRDF_Albedo_Band_Mandatory_Quality_shortwave").eq(0))
                .multiply(0.001))

    burn = ee.ImageCollection("MODIS/061/MCD64A1").filterDate(ys, ye).map(
        lambda i: i.select("BurnDate").gt(0))

    # Land surface temperature, 1 km, 8-day composites of Terra and Aqua. The daily mean is
    # taken as the mean of the day and night values (about 10:30 and 22:30 for Terra, 13:30
    # and 01:30 for Aqua). Only clear-sky days enter a composite, so this is the temperature
    # of clear days; it describes the surface of the site's 1 km pixel, not the air.
    def lst_clean(i):
        def one(band, qc):
            q = i.select(qc)
            ok = q.bitwiseAnd(3).lte(1).And(q.rightShift(6).bitwiseAnd(3).lte(1))  # produced, error <= 2 K
            return i.select(band).updateMask(ok).multiply(0.02).subtract(K0)
        day, night = one("LST_Day_1km", "QC_Day"), one("LST_Night_1km", "QC_Night")
        return ee.Image(day.add(night).divide(2).rename("lst").copyProperties(i, ["system:time_start"]))

    lst = (ee.ImageCollection("MODIS/061/MOD11A2").merge(ee.ImageCollection("MODIS/061/MYD11A2"))
           .map(lst_clean))
    ts, te = _window(year, cfg["thaw_months"])
    n_thaw = _n_days(year, cfg["thaw_months"])
    return ee.Image.cat([
        _safe(lst.filterDate(s, e), lambda c: c.mean(), "LST_JJA_mean"),
        # surface thawing index: mean of the positive 8-day means times the days of the season
        _safe(lst.filterDate(ts, te).map(lambda i: i.max(0)),
              lambda c: c.mean().multiply(n_thaw), "LST_thaw_index"),
        _safe(vi.filterDate(s, e).map(vi_clean), lambda c: c.mean(), "NDVI_summer_modis"),
        _safe(vi.filterDate(ys, ye).map(vi_clean), lambda c: c.max(), "NDVI_max_modis"),
        _safe(ee.ImageCollection("MODIS/061/MOD15A2H").filterDate(s, e).map(lai_clean),
              lambda c: c.mean(), "LAI_summer_modis"),
        _safe(ee.ImageCollection("MODIS/061/MCD43A3").filterDate(s, e).map(alb_clean),
              lambda c: c.mean(), "albedo_summer_modis"),
        _safe(burn, lambda c: c.max(), "burned_frac"),
    ]).select(MODIS_BANDS)


def avhrr_image(year, cfg=CONFIG):
    """NOAA CDR AVHRR NDVI and LAI V5 (0.05 degree, daily), for the years before MODIS."""
    s, e = _window(year, cfg["summer_months"])
    ys, ye = _window(year, alt_year_months(cfg))
    nd = ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5")
    la = ee.ImageCollection("NOAA/CDR/AVHRR/LAI_FAPAR/V5")
    bad_bits = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 6)   # cloud, cloud shadow, water, night

    def nd_clean(i):
        v = i.select("NDVI")
        return (v.updateMask(i.select("QA").bitwiseAnd(bad_bits).eq(0))
                 .updateMask(v.gte(-1000)).multiply(0.0001))

    def la_clean(i):  # QA bits 0-1 = 0: quality OK
        v = i.select("LAI")
        return v.updateMask(i.select("QA").bitwiseAnd(3).eq(0)).updateMask(v.gte(0)).multiply(0.001)

    return ee.Image.cat([
        _safe(nd.filterDate(s, e).map(nd_clean), lambda c: c.mean(), "NDVI_summer_avhrr"),
        _safe(nd.filterDate(ys, ye).map(nd_clean), lambda c: c.max(), "NDVI_max_avhrr"),
        _safe(la.filterDate(s, e).map(la_clean), lambda c: c.mean(), "LAI_summer_avhrr"),
    ])


def t2m_daily_dates(year, cfg=CONFIG):
    """The days of ALT year `year` as YYYYMMDD strings (the band names of t2m_daily_image)."""
    (y0, m0), (y1, m1) = _window_ymd(year, alt_year_months(cfg))
    days = pd.date_range(pd.Timestamp(y0, m0, 1), pd.Timestamp(y1, m1, 1), inclusive="left")
    return [d.strftime("%Y%m%d") for d in days]


def t2m_daily_image(year, cfg=CONFIG):
    """Daily mean and maximum air temperature (C) of every day of the ALT year, one band per
    day ('20141001_tm', '20141001_tx', ...). The elevation correction and the temperature
    indices are computed from these in temperature_from_daily(), so the thresholds and lapse
    rates can be changed without running Earth Engine again."""
    s, e = _window(year, alt_year_months(cfg))
    daily = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(s, e)
             .map(lambda i: i.select(["temperature_2m", "temperature_2m_max"], ["tm", "tx"])
                  .subtract(K0)))
    return daily.toBands()     # band names: <system:index = YYYYMMDD>_tm, _tx


def snowmelt_image(year, cfg=CONFIG):
    """Snow-off date at 500 m from MODIS daily snow cover (MOD10A1 and MYD10A1, Collection 6.1).

    Every clear observation of the window (Mar-Aug) is snow (NDSI snow cover >= 40) or snow
    free; clouds, night and poor quality are left out. The snow-off date is the day that best
    splits the season into "snow before, no snow after": the day D with the fewest snow-free
    observations before D plus snow observations from D on. Short late snowfalls and missed
    clouds then move the date little, and a cloudy spell around melt is split in the middle
    (the midpoint of all equally good days is used).
    snow_off_modis_censored = 1 when the best day is the first or the day after the last day
    of the window (no snow at the start of March, or snow all through August)."""
    (y0, m0), (y1, m1) = _window_ymd(year, cfg["modis_snow_months"])
    start = pd.Timestamp(y0, m0, 1)
    n = (pd.Timestamp(y1, m1, 1) - start).days
    first_doy = start.dayofyear
    thr = cfg["modis_ndsi_snow"]
    col = (ee.ImageCollection("MODIS/061/MOD10A1").merge(ee.ImageCollection("MODIS/061/MYD10A1"))
           .filterDate(ee.Date(start.strftime("%Y-%m-%d")), ee.Date(start.strftime("%Y-%m-%d")).advance(n, "day")))

    def obs(i):
        v = i.select("NDSI_Snow_Cover")
        ok = v.lte(100).And(i.select("NDSI_Snow_Cover_Basic_QA").lte(2))   # 0 best, 1 good, 2 ok
        snow = v.gte(thr).updateMask(ok)
        return (snow.eq(0).rename("free").addBands(snow.rename("snow"))
                .unmask(0).copyProperties(i, ["system:time_start"]))

    obs_col = col.map(lambda i: ee.Image(obs(i)))
    d0 = ee.Date(start.strftime("%Y-%m-%d"))

    def per_day(k):   # Terra and Aqua of one day added: 0, 1 or 2 observations
        d = d0.advance(k, "day")
        day = obs_col.filterDate(d, d.advance(1, "day"))
        return ee.Image(ee.Algorithms.If(day.size().gt(0), day.sum(),
                                         ee.Image.constant([0, 0]).rename(["free", "snow"])))\
            .toFloat()

    days = ee.ImageCollection(ee.List.sequence(0, n - 1).map(per_day))
    arr = days.toArray()                                    # n x 2 per pixel
    free = arr.arraySlice(1, 0, 1).arrayProject([0])
    snow = arr.arraySlice(1, 1, 2).arrayProject([0])
    zero = ee.Image(ee.Array([0.0]))
    # cumulative counts before day k, for k = 0..n (n = snow until the end of the window)
    free_before = zero.arrayCat(free.arrayAccum(0, ee.Reducer.sum()), 0)
    snow_before = zero.arrayCat(snow.arrayAccum(0, ee.Reducer.sum()), 0)
    snow_total = snow.arrayReduce(ee.Reducer.sum(), [0]).arrayRepeat(0, n + 1)
    err = free_before.add(snow_total).subtract(snow_before)
    err_min = err.arrayReduce(ee.Reducer.min(), [0]).arrayRepeat(0, n + 1)
    k = ee.Image(ee.Array(ee.List.sequence(0, n)))
    best = k.arrayMask(err.lte(err_min))
    k_first = best.arrayReduce(ee.Reducer.min(), [0]).arrayGet([0])
    k_last = best.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0])
    n_clear = free.add(snow).arrayReduce(ee.Reducer.sum(), [0]).arrayGet([0])
    enough = n_clear.gte(cfg["modis_snow_min_clear"])
    off = k_first.add(k_last).divide(2).add(first_doy)
    censored = k_first.eq(0).Or(k_last.eq(n))
    return ee.Image.cat([
        off.updateMask(enough).rename("snow_off_doy_modis"),
        censored.updateMask(enough).rename("snow_off_modis_censored"),
        n_clear.rename("snow_n_clear"),
    ])


def _landsat_prep(img, sensor, cfg=CONFIG):
    """Clear-sky, snow-free, water-free Gao and McFeeters NDWI and NDVI for one Landsat scene."""
    src = (["SR_B2", "SR_B3", "SR_B4", "SR_B5"] if sensor == "tm"
           else ["SR_B3", "SR_B4", "SR_B5", "SR_B6"])
    names = ["green", "red", "nir", "swir1"]
    refl = img.select(src, names).multiply(2.75e-05).add(-0.2)
    if sensor == "tm" and cfg["harmonize_landsat"]:
        refl = (refl.multiply(ee.Image.constant([ROY2016[n][1] for n in names]))
                    .add(ee.Image.constant([ROY2016[n][0] for n in names])))
    # QA_PIXEL bits 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow, 5 snow, 7 water
    clear = img.select("QA_PIXEL").bitwiseAnd(0b10111111).eq(0)
    valid = refl.reduce(ee.Reducer.min()).gt(0).And(refl.reduce(ee.Reducer.max()).lte(1))
    refl = refl.updateMask(clear).updateMask(valid)
    return (refl.normalizedDifference(["nir", "swir1"]).rename("NDWI_gao")
                .addBands(refl.normalizedDifference(["green", "nir"]).rename("NDWI_mcfeeters"))
                .addBands(refl.normalizedDifference(["nir", "red"]).rename("NDVI")))


def landsat_image(year, sites_fc, cfg=CONFIG):
    """Summer mean NDWI and NDVI from Landsat 5/7/8/9 Collection 2 surface reflectance (30 m).

    TM and ETM+ are converted to OLI reflectance first (Roy et al. 2016), so the series is
    one sensor family from 1985 to now at the scale of the site, with no AVHRR-to-MODIS
    change in 2000. Few clear summer scenes exist at some sites in the 1990s (NDWI_n_obs)."""
    s, e = _window(year, cfg["summer_months"])
    parts = []
    if year <= 2011:
        parts.append(("LANDSAT/LT05/C02/T1_L2", "tm"))
    if 1999 <= year <= 2016:   # Landsat 7 after 2016 drifted in orbit (as in the EE composites)
        parts.append(("LANDSAT/LE07/C02/T1_L2", "tm"))
    if year >= 2013:
        parts.append(("LANDSAT/LC08/C02/T1_L2", "oli"))
    if year >= 2022:
        parts.append(("LANDSAT/LC09/C02/T1_L2", "oli"))
    col = None
    for cid, sensor in parts:
        c = (ee.ImageCollection(cid).filterBounds(sites_fc).filterDate(s, e)
             .map(lambda i, sensor=sensor: _landsat_prep(i, sensor, cfg)))
        col = c if col is None else col.merge(c)
    return ee.Image.cat([
        _safe(col.select("NDWI_gao"), lambda c: c.mean(), "NDWI_gao_summer"),
        _safe(col.select("NDWI_mcfeeters"), lambda c: c.mean(), "NDWI_mcfeeters_summer"),
        _safe(col.select("NDWI_gao"), lambda c: c.count(), "NDWI_n_obs"),
        _safe(col.select("NDVI"), lambda c: c.mean(), "NDVI_landsat_summer"),
    ])


def static_image_set():
    """Terrain, land cover and soil images (no time dimension)."""
    dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM")

    def tile_terrain(img):
        # slope and aspect per DEM tile, in the tile's own grid (the grid spacing changes
        # with latitude in GLO-30); northness and eastness are weighted by sin(slope) so that
        # flat ground has no aspect
        prod = ee.Terrain.products(img)
        slope = prod.select("slope")
        asp = prod.select("aspect").multiply(math.pi / 180)
        w = slope.multiply(math.pi / 180).sin()
        return (prod.select("DEM").rename("elevation_m")
                .addBands(slope.rename("slope_deg"))
                .addBands(asp.cos().multiply(w).rename("northness"))
                .addBands(asp.sin().multiply(w).rename("eastness")))

    # MERIT Hydro (90 m, to 90 N): upstream area and height above the nearest drainage
    merit = ee.Image("MERIT/Hydro/v1_0_1")
    mslope = ee.Terrain.slope(merit.select("elv")).multiply(math.pi / 180).tan().max(0.001)
    # specific catchment area: upstream area (km2) per metre of contour (about one 90 m pixel)
    sca = merit.select("upa").multiply(1e6).divide(ee.Image.pixelArea().sqrt())
    hydro = (sca.divide(mslope).log().rename("twi")
             .addBands(merit.select(["hnd"], ["hand_m"])))

    olm = "OpenLandMap/SOL/"
    return {
        "hydro": hydro,
        "terrain": dem_col.map(tile_terrain).mosaic(),
        "dem": dem_col.mosaic(),
        "worldcover": ee.ImageCollection("ESA/WorldCover/v100").first().select(["Map"], ["landcover_esa_2020"]),
        "igbp": ee.ImageCollection("MODIS/061/MCD12Q1").select("LC_Type1").mode().rename("landcover_igbp"),
        "alos_lf": ee.Image("CSP/ERGo/1_0/Global/ALOS_landforms").select(["constant"], ["landform_alos"]),
        "alos_mtpi": ee.Image("CSP/ERGo/1_0/Global/ALOS_mTPI").select(["AVE"], ["mtpi_alos"]),
        "texture": ee.Image(olm + "SOL_TEXTURE-CLASS_USDA-TT_M/v02").select(
            ["b0", "b30"], ["soil_texture_0cm", "soil_texture_30cm"]),
        "soil": (ee.Image(olm + "SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select(
                     ["b0", "b10", "b30"], ["soc_0", "soc_10", "soc_30"]).multiply(5)        # g/kg
                 .addBands(ee.Image(olm + "SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select(
                     ["b0", "b30"], ["bulk_density_0cm", "bulk_density_30cm"]).multiply(10))  # kg/m3
                 .addBands(ee.Image(olm + "SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select(["b0"], ["clay_pct_0cm"]))
                 .addBands(ee.Image(olm + "SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select(["b0"], ["sand_pct_0cm"]))),
    }


# =============================================================================
# SAMPLING: images -> one table row per site
# =============================================================================
def _sample_grid(img, sites_fc, proj, flag_name):
    """Value of the grid cell holding each site. A cell outside the land mask (sea or a
    large lake: every band masked) takes the mean of the land cells around it: 3 x 3
    cells, else 5 x 5, else 11 x 11 (about +-0.5 degree, the earlier 50 km fill).
    flag_name = 0 for the site's own cell, 1, 2 or 3 for the fill used, so the filled
    sites can be left out in a sensitivity run. A land cell masked in one band only
    (e.g. no snow-off date) is not filled."""
    img = img.setDefaultProjection(proj)
    names = img.bandNames()
    sea = img.select(0).mask().eq(0)

    def near(r):
        return img.reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(r),
                                      skipMasked=False).rename(names)

    n1, n2, n5 = near(1), near(2), near(5)
    fill = n1.unmask(n2, False).unmask(n5, False).updateMask(sea)
    filled = img.unmask(fill, False)
    flag = (ee.Image.constant(3).where(n2.select(0).mask().gt(0), 2)
            .where(n1.select(0).mask().gt(0), 1).multiply(sea).rename(flag_name))
    return filled.addBands(flag).reduceRegions(collection=sites_fc, reducer=ee.Reducer.first(), crs=proj)


def _buffers(sites_fc, radius_m):
    """A fixed radius around every site (fire search radius)."""
    return sites_fc.map(lambda f: f.buffer(radius_m))


def _site_buffers(sites_fc, scale):
    """Each site's own buffer (property buffer_m), but at least 0.75 pixel of the product,
    so that at least one pixel centre always falls inside (otherwise Earth Engine returns
    no value for a 15 m buffer on a 250 m grid)."""
    rmin = 0.75 * scale
    return sites_fc.map(lambda f: f.buffer(ee.Number(f.get("buffer_m")).max(rmin)))


def group_collection(group, year, sites_fc, cfg=CONFIG):
    """(FeatureCollection, column list) for one group and year."""
    if group == "era5":
        proj = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").first().projection()
        fc = _sample_grid(era5_image(year, cfg), sites_fc, proj, "era5_filled")
        cols = ERA5_BANDS + ["era5_filled"]
    elif group == "avhrr":
        proj = ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5").first().projection()
        fc = _sample_grid(avhrr_image(year, cfg), sites_fc, proj, "avhrr_filled")
        cols = AVHRR_BANDS + ["avhrr_filled"]
    elif group == "t2m_daily":
        proj = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").first().projection()
        fc = _sample_grid(t2m_daily_image(year, cfg), sites_fc, proj, "era5_filled")
        dates = t2m_daily_dates(year, cfg)
        cols = [f"{d}_{v}" for v in ("tm", "tx") for d in dates] + ["era5_filled"]
    elif group == "modis":
        # 250 m sampling: the 500 m and 1 km products are read through their 250 m
        # sub-pixels, so a small buffer still returns the pixel it falls in
        fc = modis_image(year, cfg).reduceRegions(
            collection=_site_buffers(sites_fc, 250), reducer=ee.Reducer.mean(), scale=250)
        cols = MODIS_BANDS
    elif group == "snowmelt":
        fc = snowmelt_image(year, cfg).reduceRegions(
            collection=_site_buffers(sites_fc, 500), reducer=ee.Reducer.mean(), scale=500,
            tileScale=4)
        cols = SNOWMELT_BANDS
    elif group == "landsat":
        fc = landsat_image(year, sites_fc, cfg).reduceRegions(
            collection=_site_buffers(sites_fc, 30), reducer=ee.Reducer.mean(), scale=30,
            tileScale=4)
        cols = LANDSAT_BANDS
    elif group == "fire":
        fc, cols = fire_collection(year, sites_fc, cfg)
    else:
        raise ValueError(group)
    return fc, ["site_id"] + cols


def fire_collection(year, sites_fc, cfg=CONFIG):
    """One row per site and day with a FIRMS detection within fire_radius_m of the site.
    The days are counted per site and year in assemble_features()."""
    s, e = _window(year, alt_year_months(cfg))
    buffers = _buffers(sites_fc, cfg["fire_radius_m"])

    def per_day(img):
        fire = img.select("confidence").gte(cfg["fire_min_confidence"]).unmask(0).rename("fire")
        hits = fire.reduceRegions(collection=buffers, reducer=ee.Reducer.max(), scale=1000)
        date = img.date().format("YYYY-MM-dd")
        return hits.filter(ee.Filter.gt("max", 0)).map(lambda f: f.set("date", date))

    fc = ee.FeatureCollection(ee.ImageCollection("FIRMS").filterDate(s, e).map(per_day)).flatten()
    return fc, ["date"]


def static_collection(sites_fc, cfg=CONFIG):
    imgs = static_image_set()

    def per_site(f):
        g = f.geometry()
        r = ee.Number(f.get("buffer_m"))

        def red(img, reducer, geom, scale):
            return img.reduceRegion(reducer=reducer, geometry=geom, scale=scale, maxPixels=1e9)

        def b(scale):   # the site buffer, at least 0.75 pixel (see _site_buffers)
            return g.buffer(r.max(0.75 * scale))

        # the land part of the site's ERA5-Land cell (0.1 degree, centred on multiples of
        # 0.1 degree), for the elevation correction of air temperature
        xy = g.coordinates()
        cx = ee.Number(xy.get(0)).multiply(10).round().divide(10)
        cy = ee.Number(xy.get(1)).multiply(10).round().divide(10)
        cell = ee.Geometry.Rectangle([cx.subtract(0.05), cy.subtract(0.05),
                                      cx.add(0.05), cy.add(0.05)], None, False)
        dem = imgs["dem"]
        z_cell = red(dem.updateMask(dem.gt(0)), ee.Reducer.mean(), cell, 250).get("DEM")

        mean, mode = ee.Reducer.mean(), ee.Reducer.mode()
        return (f.set(red(imgs["terrain"], mean, b(30), 30))
                 .set("elev_mean_outer", red(dem, mean, g.buffer(cfg["tpi_outer_m"]), 90).get("DEM"))
                 .set("tri", red(dem, ee.Reducer.stdDev(), g.buffer(cfg["tri_radius_m"]), 30).get("DEM"))
                 .set("z_cell_era5", z_cell)
                 .set(red(imgs["hydro"], mean, b(90), 90))
                 .set(red(imgs["worldcover"], mode, b(10), 10))
                 .set(red(imgs["igbp"], mode, b(500), 500))
                 .set(red(imgs["alos_lf"], mode, b(90), 90))
                 .set(red(imgs["alos_mtpi"], mean, b(270), 270))
                 .set(red(imgs["texture"], mode, b(250), 250))
                 .set(red(imgs["soil"], mean, b(250), 250)))

    return sites_fc.map(per_site), ["site_id"] + STATIC_BANDS


# =============================================================================
# RUNNING: interactive test, or export tasks to Google Drive
# =============================================================================
def _years_for(group, years, cfg=CONFIG):
    lo, hi = GROUP_YEARS[group]
    return [y for y in years if lo <= y <= hi]


def _check_era5_complete(year, cfg=CONFIG):
    s, e = _window(year, alt_year_months(cfg))
    n = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterDate(s, e).size().getInfo()
    need = _n_days(year, alt_year_months(cfg))
    if n < need - 1:
        print(f"  ERA5-Land has {n} of {need} days for ALT year {year}; year skipped")
        return False
    return True


def run_interactive(sites, years, groups=ALL_GROUPS,
                    chunk=20, cfg=CONFIG):
    """Compute in this session (no export). For tests or a few sites; large runs should use
    export_all(). Returns {group: DataFrame}."""
    out = {}
    for group in groups:
        rows = []
        for i in range(0, len(sites), chunk):
            fc_sites = make_sites_fc(sites.iloc[i:i + chunk])
            if group == "static":
                fc, cols = static_collection(fc_sites, cfg)
                rows += [dict(f["properties"]) for f in fc.getInfo()["features"]]
                continue
            for y in _years_for(group, years, cfg):
                fc, cols = group_collection(group, y, fc_sites, cfg)
                rows += [dict(f["properties"], year=y) for f in fc.getInfo()["features"]]
        out[group] = pd.DataFrame(rows)
        print(f"{group}: {len(rows)} rows")
    return out


def quick_test(sites, year=2015, n_sites=3, cfg=CONFIG):
    """A few sites and one year, every group, to check that everything runs."""
    res = run_interactive(sites.head(n_sites), [year], cfg=cfg)
    for g, df in res.items():
        print(f"\n--- {g} ---")
        if g == "t2m_daily" and len(df):   # 730 daily columns: print a summary
            tm = df.filter(regex=r"_tm$")
            print(pd.DataFrame({"site_id": df["site_id"], "days": tm.notna().sum(axis=1),
                                "mean_C": tm.mean(axis=1).round(2),
                                "era5_filled": df.get("era5_filled")}).to_string(index=False))
            continue
        print(df.T.to_string() if len(df) else "(no rows)")
    return res


def export_all(sites, years=None, groups=ALL_GROUPS,
               drive_dir=None, cfg=CONFIG):
    """Start one Earth Engine export task per group and year (CSV to Google Drive, folder
    CONFIG['export_folder']). Groups and years already exported (file in drive_dir) or still
    running are skipped, so the function can be run again to fill gaps."""
    years = list(range(cfg["first_year"], cfg["last_year"] + 1)) if years is None else list(years)
    sites_fc = make_sites_fc(sites)
    active = {t.config.get("description") for t in ee.batch.Task.list()
              if t.state in ("READY", "RUNNING")}
    started, skipped = 0, 0
    for group in groups:
        todo = [None] if group == "static" else _years_for(group, years, cfg)
        for y in todo:
            name = group if y is None else f"{group}_{y}"
            desc = f"aldi_{name}"
            if desc in active or (drive_dir and glob.glob(os.path.join(drive_dir, name + "*.csv"))):
                skipped += 1
                continue
            if group in ("era5", "t2m_daily") and not _check_era5_complete(y, cfg):
                continue
            if group == "static":
                fc, cols = static_collection(sites_fc, cfg)
            else:
                fc, cols = group_collection(group, y, sites_fc, cfg)
            task = ee.batch.Export.table.toDrive(
                collection=fc, description=desc, folder=cfg["export_folder"],
                fileNamePrefix=name, fileFormat="CSV", selectors=cols)
            task.start()
            started += 1
    print(f"Started {started} export tasks; skipped {skipped} (already in Drive or running).")
    print("Follow them with task_status() or at https://code.earthengine.google.com/tasks")


def task_status(prefix="aldi_"):
    """Counts of this script's Earth Engine tasks by state, and the failed ones."""
    tasks = [t for t in ee.batch.Task.list() if str(t.config.get("description", "")).startswith(prefix)]
    states = pd.Series([t.state for t in tasks]).value_counts()
    print(states.to_string() if len(states) else "no tasks found")
    for t in tasks:
        if t.state == "FAILED":
            print("FAILED:", t.config.get("description"), t.status().get("error_message", ""))
    return states


# =============================================================================
# OFFLINE STEPS (pandas): read the exports, derived features, augmented files
# =============================================================================
def _read_group(drive_dir, group):
    """All exported CSVs of one group as one table with a year column (newest copy wins
    when Drive holds duplicates such as 'era5_1990 (1).csv')."""
    import re
    files = {}
    pattern = re.compile(rf"^{re.escape(group)}_(\d{{4}})(?: \(\d+\))?\.csv$")
    for path in sorted(glob.glob(os.path.join(drive_dir, f"{group}_*.csv")), key=os.path.getmtime):
        m = pattern.match(os.path.basename(path))   # era5_1990.csv, not era5_daily_1990.csv
        if m:
            files[int(m.group(1))] = path
    parts = []
    for year, path in sorted(files.items()):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=["site_id"])
        df["year"] = year
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), sorted(files)


def harmonize_to_modis(df, modis_col, avhrr_col, out_col, cfg=CONFIG):
    """MODIS where it exists; before that AVHRR shifted by the site's median MODIS-AVHRR
    difference over the years both exist (2000-2013), or by the median over all sites when
    a site has fewer than min_overlap_years shared years. Without this the sensor change in
    2000 is a step in every site's series, which site fixed effects cannot remove."""
    both = df[[modis_col, avhrr_col]].notna().all(axis=1)
    diff = (df[modis_col] - df[avhrr_col]).where(both)
    per_site = diff.groupby(df["site_id"]).agg(["median", "count"])
    pooled = diff.median()
    off = df["site_id"].map(per_site["median"].where(per_site["count"] >= cfg["min_overlap_years"]))
    off = off.fillna(pooled)
    df[out_col] = df[modis_col].where(df[modis_col].notna(), df[avhrr_col] + off)
    df[out_col + "_source"] = np.where(df[modis_col].notna(), "MODIS",
                                       np.where(df[avhrr_col].notna(), "AVHRR_adjusted", ""))
    print(f"{out_col}: {avhrr_col} shifted by the pooled median {pooled:+.3f}; "
          f"{(per_site['count'] >= cfg['min_overlap_years']).sum()} sites with their own offset")
    return df


TEMP_VARS = ["MAAT", "T_JJA_mean", "T_summer_mean", "T_summer_max", "T_thaw_mean",
             "T_winter_mean", "TDD", "FDD", "heat_wave_days"]


def _in_window(dates, year, months):
    (y0, m0), (y1, m1) = _window_ymd(year, months)
    return (dates >= pd.Timestamp(y0, m0, 1)) & (dates < pd.Timestamp(y1, m1, 1))


def _temperature_indices(tm, tx, dates, year, cfg=CONFIG):
    """Temperature indices of one ALT year; tm, tx = sites x days arrays (C). Same
    definitions as era5_image()."""
    def w(months):
        return _in_window(dates, year, months)
    thaw, jja, win = w(cfg["thaw_months"]), w(cfg["summer_months"]), w(cfg["winter_months"])
    with np.errstate(all="ignore"):
        out = {
            "MAAT": np.nanmean(tm, axis=1),
            "T_JJA_mean": np.nanmean(tm[:, jja], axis=1),
            "T_summer_max": np.nanmax(tx[:, jja], axis=1),
            "T_thaw_mean": np.nanmean(tm[:, thaw], axis=1),
            "T_winter_mean": np.nanmean(tm[:, win], axis=1),
            "TDD": np.nansum(np.clip(tm[:, thaw], 0, None), axis=1),
            "FDD": np.nansum(np.clip(-tm, 0, None), axis=1),
            "heat_wave_days": np.sum(tx[:, jja] > cfg["heat_wave_tmax_c"], axis=1).astype(float),
        }
    out["T_summer_mean"] = out["T_JJA_mean"]
    bad = np.isnan(tm).all(axis=1)       # site without values (no export row)
    for v in out.values():
        v[bad] = np.nan
    return out


def temperature_from_daily(drive_dir, dz_m, cfg=CONFIG):
    """Grid and elevation-corrected ("downscaled") temperature indices per site and year,
    from the daily exports (group t2m_daily).

    dz_m: Series site_id -> z_site - z_cell (m); NaN = no correction (sea cell filled from
    its neighbours, or no DEM). The correction of each day is lapse_rate(month) * dz / 1000,
    added before the thresholds, so TDD and heat-wave days change as they would at the
    site's elevation, not by a constant. heat_wave_days_p90 counts the JJA days above the
    site's own percentile of daily maximum temperature (heat_wave_ref_years); a relative
    threshold does not depend on the elevation offset, and in the high Arctic the fixed
    20 C threshold is reached on almost no days."""
    import re
    raw, got = _read_group(drive_dir, "t2m_daily")
    if not len(raw):
        return pd.DataFrame()
    lapse = cfg["lapse_rate_k_per_km"]
    rows, jja_tx = [], {}
    for year in got:
        g = raw[raw["year"] == year].drop_duplicates("site_id")
        cols = [c for c in g.columns if re.fullmatch(r"\d{8}_tm", str(c))]
        days = sorted(c[:8] for c in cols)
        dates = pd.to_datetime(days, format="%Y%m%d")
        tm = g[[d + "_tm" for d in days]].to_numpy(float)
        tx = g[[d + "_tx" for d in days]].to_numpy(float)
        sid = g["site_id"].astype(str).to_numpy()
        dz = pd.Series(sid).map(dz_m).to_numpy(float)
        rate = np.array([lapse[m] for m in dates.month], float)          # K/km per day
        corr = np.nan_to_num(dz)[:, None] * rate[None, :] / 1000.0
        grid = _temperature_indices(tm, tx, dates, year, cfg)
        ds = _temperature_indices(tm + corr, tx + corr, dates, year, cfg)
        part = pd.DataFrame({"site_id": sid, "year": year})
        for v in TEMP_VARS:
            part[v + "_grid"] = grid[v]
            part[v + "_ds"] = ds[v]
        part["temperature_corrected"] = (~np.isnan(dz)).astype(int)
        rows.append(part)
        jja = _in_window(dates, year, cfg["summer_months"])
        for i, s_ in enumerate(sid):
            jja_tx[(s_, year)] = tx[i, jja] + corr[i, jja]
    out = pd.concat(rows, ignore_index=True)

    # site thresholds for heat_wave_days_p90 from the reference years (all years when the
    # site has fewer than 10 reference years)
    y0, y1 = cfg["heat_wave_ref_years"]
    thr = {}
    for s_ in out["site_id"].unique():
        ref = [v for (k, y), v in jja_tx.items() if k == s_ and y0 <= y <= y1]
        if len(ref) < 10:
            ref = [v for (k, y), v in jja_tx.items() if k == s_]
        vals = np.concatenate(ref) if ref else np.array([np.nan])
        thr[s_] = np.nanpercentile(vals, cfg["heat_wave_percentile"]) if np.isfinite(vals).any() else np.nan
    out["heat_wave_tmax_p90_C"] = out["site_id"].map(thr)
    out["heat_wave_days_p90"] = [np.sum(jja_tx[(s_, y)] > thr[s_]) if np.isfinite(thr[s_]) else np.nan
                                 for s_, y in zip(out["site_id"], out["year"])]
    return out


def potential_insolation_ratio(lat, slope_deg, aspect_deg, months=(5, 9), step_min=10):
    """Clear-sky direct sunlight on the sloping site over the thaw season, relative to flat
    ground at the same latitude (1 = like flat ground, below 1 = less sun, e.g. a north
    slope). Sun geometry only: no atmosphere and no shading by surrounding hills. aspect in
    degrees clockwise from north."""
    lat = np.radians(np.atleast_1d(np.asarray(lat, float)))[:, None, None]
    s = np.radians(np.atleast_1d(np.asarray(slope_deg, float)))[:, None, None]
    a = np.radians(np.atleast_1d(np.asarray(aspect_deg, float)))[:, None, None]
    doy = np.arange(pd.Timestamp(2001, months[0], 1).dayofyear,
                    pd.Timestamp(2001, months[1], 1).dayofyear
                    + pd.Timestamp(2001, months[1], 1).days_in_month)
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365.0)[None, :, None]
    h = np.radians(np.arange(-180, 180, step_min / 4.0))[None, None, :]   # hour angle
    sin_el = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(h)
    cos_el = np.sqrt(np.clip(1 - sin_el ** 2, 0, 1))
    # sun azimuth clockwise from north
    cos_az = (np.sin(decl) - sin_el * np.sin(lat)) / np.maximum(cos_el * np.cos(lat), 1e-9)
    az = np.arccos(np.clip(cos_az, -1, 1))
    az = np.where(h > 0, 2 * np.pi - az, az)
    cos_inc = sin_el * np.cos(s) + cos_el * np.sin(s) * np.cos(az - a)
    up = sin_el > 0
    on_slope = np.where(up, np.clip(cos_inc, 0, None), 0).sum(axis=(1, 2))
    flat = np.where(up, sin_el, 0).sum(axis=(1, 2))
    return on_slope / flat


def distance_to_coast_km(lat, lon, coastline=None):
    """Great-circle distance (km) to the nearest vertex of the Natural Earth 1:10m coastline
    (vertex spacing is about 1-5 km, which bounds the error)."""
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString
    from sklearn.neighbors import BallTree
    coastline = coastline or "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
    coast = gpd.read_file(coastline).to_crs("EPSG:4326")
    pts = []
    for geom in coast.geometry:
        lines = geom.geoms if isinstance(geom, MultiLineString) else [geom]
        for line in lines:
            if isinstance(line, LineString):
                pts.extend(line.coords)
    pts = np.array(pts)[:, :2]
    pts = pts[pts[:, 1] > 30]                      # sites are north of 50 N
    tree = BallTree(np.radians(pts[:, ::-1]), metric="haversine")
    d, _ = tree.query(np.radians(np.c_[lat, lon]), k=1)
    return d[:, 0] * 6371.0088


def _decode_combo(combo):
    """IPA map COMBO code -> (permafrostType, groundIceType, landformType), as the MATLAB
    assignEnvironmentalCategories/decodeCombo functions."""
    code = str(combo).lower() if isinstance(combo, str) else ""
    if len(code) < 3:
        return "Unknown", "Unknown", "Unknown"
    pf = {"c": "Continuous", "d": "Discontinuous", "s": "Sporadic", "i": "Isolated"}.get(code[0], "Unknown")
    gi = {"h": "High", "m": "Medium", "l": "Low"}.get(code[1], "Unknown")
    lf = {"f": "Lowlands/thickOverburdenCover", "r": "Mountains/thinOverburdenCover"}.get(code[2], "Unknown")
    return pf, gi, lf


def permafrost_classes(sites, shapefile, max_snap_km=20.0):
    """Permafrost extent, ground ice and landform from the IPA Circum-Arctic Map
    (Brown et al. 2002, NSIDC GGD318, permaice.shp).

    A site inside a permafrost polygon takes its COMBO code. A site outside (in a lake,
    on the coast, or on a polygon edge) takes the nearest permafrost polygon within
    max_snap_km; otherwise "Unknown". (The MATLAB version copied labels from the nearest
    labelled site, but its index pointed into the wrong array.)"""
    import geopandas as gpd
    shp = gpd.read_file(shapefile)
    if shp.crs is None:
        raise ValueError("permaice.shp has no projection file (.prj); keep the .prj next to it")
    shp = shp.to_crs("EPSG:6931")                 # EASE-Grid 2.0 North, metres
    codes = shp["COMBO"].astype(str).str.lower()
    pf_poly = shp[(codes.str.len() >= 3) & codes.str[0].isin(list("cdsi"))][["COMBO", "geometry"]]
    pts = gpd.GeoDataFrame(sites[["site_id"]].copy(),
                           geometry=gpd.points_from_xy(sites["lon"], sites["lat"]),
                           crs="EPSG:4326").to_crs("EPSG:6931")
    inside = gpd.sjoin(pts, pf_poly, how="left", predicate="within").drop_duplicates("site_id")
    inside = inside.set_index("site_id")["COMBO"]
    out = pd.DataFrame({"site_id": sites["site_id"].values})
    out["ipa_combo"] = out["site_id"].map(inside)
    out["ipa_match"] = np.where(out["ipa_combo"].notna(), "inside", "none")
    miss = out["ipa_combo"].isna().values
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], pf_poly, how="left", max_distance=max_snap_km * 1000,
                                 distance_col="snap_m").drop_duplicates("site_id").set_index("site_id")
        for i in np.where(miss)[0]:
            sid = out.at[i, "site_id"]
            if sid in near.index and pd.notna(near.at[sid, "COMBO"]):
                out.at[i, "ipa_combo"] = near.at[sid, "COMBO"]
                out.at[i, "ipa_match"] = f"nearest_{near.at[sid, 'snap_m'] / 1000:.1f}km"
    dec = out["ipa_combo"].map(_decode_combo)
    out["permafrostType"] = [d[0] for d in dec]
    out["groundIceType"] = [d[1] for d in dec]
    out["landformType"] = [d[2] for d in dec]
    print("permafrostType:", out["permafrostType"].value_counts().to_dict())
    return out


def region_clusters(sites, km=50.0):
    """Region id per site: sites closer than km to another site of the region (single
    linkage). Sites of one region share weather and often one ERA5-Land cell, so their
    years are not independent; use region_id instead of site_id to cluster standard
    errors in a robustness run."""
    lat, lon = np.radians(sites["lat"].to_numpy(float)), np.radians(sites["lon"].to_numpy(float))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    near = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) <= km
    label = -np.ones(len(sites), int)
    for i in range(len(sites)):
        if label[i] >= 0:
            continue
        stack, label[i] = [i], i
        while stack:
            j = stack.pop()
            for k in np.where(near[j] & (label < 0))[0]:
                label[k] = i
                stack.append(k)
    first = {lab: n for n, lab in enumerate(dict.fromkeys(label))}
    return pd.Series([f"R{first[l]:03d}" for l in label], index=sites["site_id"].to_numpy())


def assemble_features(sites, drive_dir, coastline=None, ipa_shapefile=None, cfg=CONFIG,
                      out_csv=None):
    """Read all exports and return one row per site and year with every feature.
    out_csv (optional) also writes it, e.g. ALDI_causal_features_complete.csv."""
    years = list(range(cfg["first_year"], cfg["last_year"] + 1))
    grid = pd.MultiIndex.from_product([sites["site_id"], years], names=["site_id", "year"]).to_frame(index=False)
    site_cols = [c for c in ("site_id", "lat", "lon", "buffer_m", "footprint_m", "coord_uncert_m",
                             "coarse_location") if c in sites.columns]
    df = grid.merge(sites[site_cols], on="site_id")
    for group in ("era5", "modis", "avhrr", "landsat", "snowmelt"):
        g, got = _read_group(drive_dir, group)
        want = _years_for(group, years, cfg)
        missing = sorted(set(want) - set(got))
        if missing:
            print(f"WARNING {group}: no export for years {missing}")
        if len(g):
            df = df.merge(g.drop_duplicates(["site_id", "year"]), on=["site_id", "year"], how="left")

    # fire days: distinct detection days per site and ALT year; 0 when the year was exported
    fires, got = _read_group(drive_dir, "fire")
    counts = (fires.dropna(subset=["date"]).groupby(["site_id", "year"])["date"].nunique()
              if len(fires) and "date" in fires else pd.Series(dtype=float))
    df["FIRMS_fire_days"] = [counts.get((s, y), 0) if y in got else np.nan
                             for s, y in zip(df["site_id"], df["year"])]
    df = df.sort_values(["site_id", "year"]).reset_index(drop=True)
    roll = df.groupby("site_id")["FIRMS_fire_days"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=5).sum())
    df["FIRMS_fire_days_prev5"] = roll
    fire_years = df.groupby("site_id")["FIRMS_fire_days"].transform(lambda x: (x > 0).sum())
    df["fire_persistent_source"] = (fire_years >= cfg["persistent_fire_years"]).astype(int)

    # vegetation: one series per variable across the AVHRR-MODIS change
    for var in ("NDVI_summer", "NDVI_max", "LAI_summer"):
        m, a = f"{var}_modis", f"{var}_avhrr"
        for c in (m, a):
            if c not in df:
                df[c] = np.nan
        df = harmonize_to_modis(df, m, a, var, cfg)

    # Landsat: summer means from too few clear observations are left empty
    if "NDWI_n_obs" in df:
        few = df["NDWI_n_obs"] < cfg["landsat_min_obs"]
        for c in ("NDWI_gao_summer", "NDWI_mcfeeters_summer", "NDVI_landsat_summer"):
            if c in df:
                df.loc[few, c] = np.nan

    # NDVI used as the causal treatment: Landsat (30 m, 1985 onward, one sensor family) or
    # MODIS with adjusted AVHRR before 2000; both are kept
    df["NDVI_summer_modis_avhrr"] = df["NDVI_summer"]
    if cfg["ndvi_for_causal"] == "landsat" and "NDVI_landsat_summer" in df:
        df["NDVI_summer"] = df["NDVI_landsat_summer"]
        df["NDVI_summer_source"] = np.where(df["NDVI_summer"].notna(), "Landsat", "")

    # snow-off date used as the causal treatment (ERA5-Land, MODIS, or MODIS with ERA5-Land
    # shifted to MODIS before 2000); all three are kept
    if "snow_off_doy" in df:
        df["snow_off_doy_era5land"] = df["snow_off_doy"]
    if {"snow_off_doy_modis", "snow_off_doy_era5land"} <= set(df.columns):
        df = harmonize_to_modis(df, "snow_off_doy_modis", "snow_off_doy_era5land",
                                "snow_off_doy_combined", cfg)
        choice = cfg["snow_off_for_causal"]
        if choice in ("modis", "combined"):
            df["snow_off_doy"] = df[f"snow_off_doy_{choice}"]

    # NDWI used as the causal treatment
    df["NDWI_annual_mean"] = df.get("NDWI_" + cfg["ndwi_for_causal"], np.nan)

    # winter wind direction (degrees, direction the wind comes from) from the mean vector
    if {"u10_winter", "v10_winter"} <= set(df.columns):
        df["wind_direction_winter"] = (np.degrees(np.arctan2(-df["u10_winter"], -df["v10_winter"])) + 360) % 360

    # static features
    st, _ = _read_group_static(drive_dir)
    if len(st):
        st["aspect_deg"] = (np.degrees(np.arctan2(st["eastness"], st["northness"])) + 360) % 360
        st["tpi"] = st["elevation_m"] - st["elev_mean_outer"]
        # depth-weighted 0-30 cm mean of the 0, 10 and 30 cm predictions (trapezoids)
        st["soc_gkg_0cm"], st["soc_gkg_30cm"] = st["soc_0"], st["soc_30"]
        st["soc_gkg_avg_0_30cm"] = (10 * (st["soc_0"] + st["soc_10"]) / 2
                                    + 20 * (st["soc_10"] + st["soc_30"]) / 2) / 30
        st["silt_pct_0cm"] = 100 - st["sand_pct_0cm"] - st["clay_pct_0cm"]
        lat = st["site_id"].map(sites.set_index("site_id")["lat"])
        st["insolation_ratio_thaw"] = potential_insolation_ratio(
            lat, st["slope_deg"].fillna(0), st["aspect_deg"].fillna(0), cfg["thaw_months"])
        if "z_cell_era5" in st:
            st["dz_era5_m"] = st["elevation_m"] - st["z_cell_era5"]
        st = st.drop(columns=["soc_0", "soc_10", "soc_30", "elev_mean_outer", "year"], errors="ignore")
        df = df.merge(st, on="site_id", how="left")
    else:
        print("WARNING: no static export found")

    # elevation correction of air temperature, from the daily exports
    df = _add_downscaled_temperature(df, drive_dir, cfg)

    # groupings for clustered standard errors (robustness runs): the ERA5-Land cell
    # (0.1 degree; sites of one cell get identical climate values) and 50 km regions
    df["era5_cell"] = (np.round(df["lat"] * 10).astype(int).astype(str) + "_"
                       + np.round(df["lon"] * 10).astype(int).astype(str))
    df["region_id"] = df["site_id"].map(region_clusters(sites))

    # distance to coast and continentality
    s = sites.copy()
    try:
        s["dist_to_coast_km"] = distance_to_coast_km(s["lat"].values, s["lon"].values, coastline)
        s["continentality_dist"] = np.log1p(s["dist_to_coast_km"])    # as in the earlier script
    except Exception as exc:  # no internet or no geopandas
        print("WARNING: distance to coast not computed:", exc)
        s["dist_to_coast_km"] = np.nan
        s["continentality_dist"] = np.nan
    df = df.merge(s[["site_id", "dist_to_coast_km", "continentality_dist"]], on="site_id", how="left")
    if "T_annual_range" in df:
        a = df.groupby("site_id")["T_annual_range"].transform("mean")
        df["conrad_continentality"] = 1.7 * a / np.sin(np.radians(df["lat"] + 10)) - 14

    # permafrost zone, ground ice and landform (IPA map)
    if ipa_shapefile:
        df = df.merge(permafrost_classes(sites, ipa_shapefile), on="site_id", how="left")
    else:
        print("WARNING: no IPA shapefile given; permafrostType, groundIceType and landformType = Unknown")
        for c in ("permafrostType", "groundIceType", "landformType"):
            df[c] = "Unknown"

    coverage = df.drop(columns=["site_id", "year", "lat", "lon"]).notna().mean().round(3)
    print("\nShare of site-years with a value:\n" + coverage.to_string())
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv}: {len(df)} rows, {df.shape[1]} columns")
    return df


def _add_downscaled_temperature(df, drive_dir, cfg=CONFIG):
    """Adds <var>_grid (ERA5-Land cell, from the era5 exports), <var>_ds (corrected to the
    site's elevation) and heat_wave_days_p90, and fills <var> with the version chosen in
    CONFIG temperature_for_causal (grid values where no corrected value exists)."""
    if "dz_era5_m" not in df:
        print("WARNING: no dz_era5_m (static export without z_cell_era5); temperature not downscaled")
        return df
    dz = df.groupby("site_id")["dz_era5_m"].first()
    if "era5_filled" in df:     # a sea cell filled from its neighbours: no correction
        filled = df.groupby("site_id")["era5_filled"].max()
        dz[filled.reindex(dz.index).fillna(0) > 0] = np.nan
    df["dz_era5_m"] = df["site_id"].map(dz)
    temp = temperature_from_daily(drive_dir, dz, cfg)
    if not len(temp):
        print("WARNING: no t2m_daily exports; temperature not downscaled")
        return df
    temp = temp.rename(columns={v + "_grid": v + "_daily_grid" for v in TEMP_VARS})
    df = df.merge(temp, on=["site_id", "year"], how="left")
    # check: the indices computed here from daily values must match those computed in
    # Earth Engine for the same cells
    if "TDD" in df:
        own = df["era5_filled"].fillna(0).eq(0) if "era5_filled" in df else True
        d = (df["TDD"] - df["TDD_daily_grid"]).abs().where(own)   # filled sea cells differ a little
        print(f"Check: TDD from daily exports vs Earth Engine, largest difference {d.max():.2f} "
              f"degree-days (should be about 0)")
    for v in TEMP_VARS:
        if v in df:
            df[v + "_grid"] = df[v]
            if cfg["temperature_for_causal"] == "downscaled":
                df[v] = df[v + "_ds"].where(df[v + "_ds"].notna(), df[v])
        elif v + "_daily_grid" in df:
            df[v + "_grid"] = df[v + "_daily_grid"]
            df[v] = df[v + "_ds"] if cfg["temperature_for_causal"] == "downscaled" else df[v + "_grid"]
    df = df.drop(columns=[v + "_daily_grid" for v in TEMP_VARS], errors="ignore")
    n_big = (dz.abs() > 100).sum()
    print(f"Temperature: {cfg['temperature_for_causal']} values used; {n_big} sites are more than "
          f"100 m above or below their ERA5-Land cell (about 0.65 C in summer)")
    return df


def _read_group_static(drive_dir):
    files = sorted(glob.glob(os.path.join(drive_dir, "static*.csv")), key=os.path.getmtime)
    if not files:
        return pd.DataFrame(), None
    return pd.read_csv(files[-1]), files[-1]


def build_augmented(features, alt_csv, aldi_results_csv, out_dir, cfg=CONFIG):
    """Join ALT, features and ALDI classes into the files ALDI_causality.ipynb reads.

    The ALT column is named 'Max' so the notebook runs unchanged (outcome="Max"); here it
    holds ALT_cm, the site-mean end-of-season thaw depth of the multi-source dataset (the
    ALDI notebook does the same internally).

    Use a new out_dir (for example .../UND/Index/multisource/) so the v4 files with the
    same names stay as they are; ALDI_causality.ipynb reads from that folder."""
    alt = pd.read_csv(alt_csv).rename(columns={"Year": "year"})
    alt = alt[alt["year"] >= cfg["first_year"]]
    # obs_doy (survey day, probing sources) and flag (information flags such as
    # burned_site or unusual_year_*) come with newer builds of the ALT file
    extra = [c for c in ("obs_doy", "flag") if c in alt.columns]
    alt = alt[["site_id", "Location", "year", "ALT_cm", "source", "method"] + extra].rename(
        columns={"ALT_cm": "Max", "flag": "alt_flag"})
    res = pd.read_csv(aldi_results_csv)[["site_id", "class_classification", "class_confidence"]]
    data = alt.merge(features, on=["site_id", "year"], how="inner")
    n0 = data["site_id"].nunique()
    data = data.merge(res, on="site_id", how="inner")
    print(f"{len(data)} site-years at {data['site_id'].nunique()} sites "
          f"({n0 - data['site_id'].nunique()} sites without an ALDI class left out)")
    front = ["site_id", "Location", "lat", "lon", "year", "Max", "source", "method",
             "class_classification", "class_confidence", "permafrostType", "groundIceType", "landformType"]
    data = data[front + [c for c in data.columns if c not in front]]

    os.makedirs(out_dir, exist_ok=True)

    def write(df, suffix=""):
        path = os.path.join(out_dir, f"ALDI_withYearlyData_augmented{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"  {os.path.basename(path)}: {len(df)} rows, {df['site_id'].nunique()} sites")

    # Every subset the notebook reads is written, empty (header only) when no site falls in
    # it, so an older file with the same name is never picked up by mistake.
    write(data)
    classes = ["rapid_thickening", "gradual_thickening", "no_trend", "gradual_thinning",
               "rapid_thinning", "transitional"]
    for cls in classes + sorted(set(data["class_classification"]) - set(classes)):
        write(data[data["class_classification"] == cls], f"_{cls}")
    zones = ["Continuous", "Discontinuous", "Sporadic", "Isolated", "Unknown"]
    for pf in zones:
        write(data[data["permafrostType"] == pf], f"_{pf}Permafrost")
    write(data[data["permafrostType"].isin(["Discontinuous", "Sporadic", "Isolated"])],
          "_nonContinuousPermafrost")
    return data


# =============================================================================
# WHAT EACH COLUMN IS (also written to feature_dictionary.csv by write_dictionary)
# =============================================================================
FEATURES = [
    # column, definition, source, units
    ("TDD", "Sum of daily mean air temperature above 0 C, May-Sep; corrected to the site elevation (CONFIG temperature_for_causal; TDD_grid = ERA5-Land cell, TDD_ds = corrected)", "ERA5-Land daily + GLO-30", "degree-days"),
    ("FDD", "Sum of |daily mean air temperature| below 0 C over the ALT year (same choice as TDD; no winter correction by default)", "ERA5-Land daily + GLO-30", "degree-days"),
    ("heat_wave_days", "JJA days with daily maximum air temperature > 20 C (same choice as TDD)", "ERA5-Land daily + GLO-30", "days"),
    ("heat_wave_days_p90", "JJA days with daily maximum air temperature above the site's own 90th percentile of 1991-2020 (extra; works in the high Arctic, where 20 C is rare)", "ERA5-Land daily", "days"),
    ("heat_wave_tmax_p90_C", "The site threshold used by heat_wave_days_p90", "ERA5-Land daily", "C"),
    ("dz_era5_m", "Site elevation (buffer mean) minus mean land elevation of its ERA5-Land cell; empty where the cell is sea", "Copernicus GLO-30", "m"),
    ("z_cell_era5", "Mean land elevation of the site's ERA5-Land 0.1 degree cell (stands in for the model orography)", "Copernicus GLO-30", "m"),
    ("temperature_corrected", "1 = the temperature columns of this row are corrected to the site elevation", "derived", "0/1"),
    ("SWE_max", "Maximum daily snow water equivalent, Oct-Jun", "ERA5-Land daily", "m w.e."),
    ("snow_off_doy", "First day of Mar-Jun with snow depth < 2 mm", "ERA5-Land daily", "day of year"),
    ("snow_off_doy_era5land", "snow_off_doy from ERA5-Land (always kept)", "ERA5-Land daily", "day of year"),
    ("snow_off_doy_modis", "Snow-off date at 500 m: the day that best splits Mar-Aug clear observations into snow before and no snow after (mean of the buffer pixels); from 2000", "MOD10A1 + MYD10A1 C6.1", "day of year"),
    ("snow_off_modis_censored", "1 = no snow at the start of March or snow all through August (date is a bound)", "MOD10A1 + MYD10A1", "0/1"),
    ("snow_n_clear", "Clear MODIS observations in Mar-Aug", "MOD10A1 + MYD10A1", "count"),
    ("snow_off_doy_combined", "MODIS snow-off from 2000; before, ERA5-Land shifted by the site's median MODIS minus ERA5-Land difference", "MODIS / ERA5-Land", "day of year"),
    ("snow_off_after_window", "1 = snow depth stayed >= 2 mm all of Mar-Jun; snow_off_doy is then the day after the window (a lower bound)", "ERA5-Land daily", "0/1"),
    ("snow_off_doy_last", "Day after the last day of Mar-Jun with snow depth >= 2 mm (extra); first day of the window when there was no snow", "ERA5-Land daily", "day of year"),
    ("P_summer", "Total JJA precipitation", "ERA5-Land daily", "mm"),
    ("P_annual", "Total precipitation over the ALT year", "ERA5-Land daily", "mm"),
    ("rain_summer", "JJA precipitation minus snowfall (extra)", "ERA5-Land daily", "mm"),
    ("snowfall_winter", "Nov-Apr snowfall (extra)", "ERA5-Land daily", "mm w.e."),
    ("soil_moisture", "Mean volumetric soil water 0-7 cm over the ALT year", "ERA5-Land daily", "m3/m3"),
    ("soil_moisture_summer", "JJA mean volumetric soil water 0-7 cm (extra)", "ERA5-Land daily", "m3/m3"),
    ("NDWI_annual_mean", "Copy of NDWI_gao_summer (CONFIG ndwi_for_causal); name kept for the notebook", "Landsat C2 SR", "-"),
    ("NDWI_gao_summer", "JJA mean Gao NDWI (NIR-SWIR1)/(NIR+SWIR1), clear snow-free land pixels in the site buffer", "Landsat 5/7/8/9 C2 SR", "-"),
    ("NDWI_mcfeeters_summer", "JJA mean McFeeters NDWI (green-NIR)/(green+NIR), same pixels", "Landsat 5/7/8/9 C2 SR", "-"),
    ("NDWI_n_obs", "Mean number of clear JJA Landsat observations per pixel", "Landsat C2", "count"),
    ("NDVI_summer", "JJA mean NDVI; Landsat 30 m by default (CONFIG ndvi_for_causal), else NDVI_summer_modis_avhrr", "Landsat C2 SR / MODIS / AVHRR", "-"),
    ("NDVI_landsat_summer", "JJA mean NDVI from clear snow-free land Landsat pixels in the site buffer, TM/ETM+ converted to OLI", "Landsat 5/7/8/9 C2 SR", "-"),
    ("NDVI_summer_modis_avhrr", "JJA mean NDVI; MODIS 250 m from 2000, AVHRR 5 km adjusted to MODIS before", "MOD13Q1+MYD13Q1 / AVHRR CDR V5", "-"),
    ("NDVI_max", "Maximum NDVI over the ALT year; MODIS from 2000, adjusted AVHRR before", "MOD13Q1+MYD13Q1 / AVHRR CDR V5", "-"),
    ("LAI_summer", "JJA mean LAI; MODIS from 2000, adjusted AVHRR before", "MOD15A2H / AVHRR CDR LAI V5", "m2/m2"),
    ("FIRMS_fire_days", "Days with a FIRMS detection (confidence >= 30) within 1 km, ALT year; from 2001", "FIRMS (MODIS)", "days"),
    ("FIRMS_fire_days_prev5", "Fire days in the 5 previous ALT years (extra)", "FIRMS", "days"),
    ("fire_persistent_source", "1 if detections in >= 5 different years (possible gas flare; check)", "FIRMS", "0/1"),
    ("burned_frac", "Share of the site buffer mapped as burned in the ALT year (extra)", "MCD64A1", "0-1"),
    ("ROS_days", "Nov-Apr days with snow depth > 5 mm, precipitation > 1 mm and daily mean T > 0 C", "ERA5-Land daily", "days"),
    ("MAAT", "Mean air temperature over the ALT year", "ERA5-Land daily", "C"),
    ("T_JJA_mean", "JJA mean air temperature (T_summer_mean is the same)", "ERA5-Land daily", "C"),
    ("T_summer_max", "Highest daily maximum air temperature of JJA", "ERA5-Land daily", "C"),
    ("T_thaw_mean", "May-Sep mean air temperature", "ERA5-Land daily", "C"),
    ("T_winter_mean", "Nov-Apr mean air temperature", "ERA5-Land daily", "C"),
    ("T_annual_range", "Warmest minus coldest monthly mean air temperature of the ALT year", "ERA5-Land monthly", "C"),
    ("snow_depth_winter", "Nov-Apr mean snow depth", "ERA5-Land daily", "m"),
    ("snow_depth_max", "Maximum daily snow depth, Oct-Jun", "ERA5-Land daily", "m"),
    ("soil_T", "Mean soil temperature 0-7 cm over the ALT year", "ERA5-Land daily", "C"),
    ("soil_T_winter_mean", "Nov-Apr mean soil temperature 0-7 cm", "ERA5-Land daily", "C"),
    ("soil_T_winter_min", "Lowest daily minimum soil temperature 0-7 cm, Nov-Apr", "ERA5-Land daily", "C"),
    ("albedo_annual", "Mean forecast albedo over the ALT year", "ERA5-Land daily", "0-1"),
    ("LST_JJA_mean", "JJA mean land surface temperature, mean of day and night, clear-sky 8-day composites, 1 km; from 2000 (extra)", "MOD11A2 + MYD11A2 C6.1", "C"),
    ("LST_thaw_index", "May-Sep mean of positive 8-day LST times the days of the season (surface thawing index, extra)", "MOD11A2 + MYD11A2 C6.1", "degree-days"),
    ("albedo_summer_modis", "JJA mean white-sky shortwave albedo, full inversions, from 2000 (extra)", "MCD43A3", "0-1"),
    ("SW_down_summer", "JJA mean daily downward shortwave radiation (extra)", "ERA5-Land daily", "MJ/m2/day"),
    ("wind_speed_winter_mean", "Nov-Apr mean of daily wind speed (from daily mean u, v)", "ERA5-Land daily", "m/s"),
    ("wind_speed_winter_max", "Nov-Apr maximum daily wind speed", "ERA5-Land daily", "m/s"),
    ("wind_speed_winter_p90", "Nov-Apr 90th percentile of daily wind speed", "ERA5-Land daily", "m/s"),
    ("wind_direction_winter", "Direction the Nov-Apr mean wind vector comes from (0 = north)", "ERA5-Land daily", "degrees"),
    ("u10_winter", "Nov-Apr mean eastward wind (direction as a component; use in regressions)", "ERA5-Land daily", "m/s"),
    ("v10_winter", "Nov-Apr mean northward wind", "ERA5-Land daily", "m/s"),
    ("era5_cell", "ERA5-Land 0.1 degree cell of the site (sites of one cell get the same climate values)", "derived", "id"),
    ("region_id", "Group of sites within 50 km of each other (single linkage), for clustering standard errors", "derived", "id"),
    ("era5_filled", "0 = the site's own ERA5-Land cell; 1, 2, 3 = the cell is sea and the mean of the land cells in the 3 x 3, 5 x 5 or 11 x 11 cells around it is used", "ERA5-Land", "0-3"),
    ("lat", "Site latitude", "site table", "degrees"),
    ("lon", "Site longitude", "site table", "degrees"),
    ("elevation_m", "Mean elevation in the site buffer (buffer_m)", "Copernicus GLO-30 (2024_1)", "m"),
    ("slope_deg", "Mean slope in the site buffer", "Copernicus GLO-30", "degrees"),
    ("aspect_deg", "Dominant aspect in the site buffer (mean slope-weighted direction)", "Copernicus GLO-30", "degrees"),
    ("northness", "Mean cos(aspect) x sin(slope) in the site buffer (regression-friendly aspect)", "Copernicus GLO-30", "-1 to 1"),
    ("eastness", "Mean sin(aspect) x sin(slope) in the site buffer", "Copernicus GLO-30", "-1 to 1"),
    ("tpi", "Mean elevation in the site buffer minus mean elevation within 3 km", "Copernicus GLO-30", "m"),
    ("tri", "Standard deviation of elevation within 1 km", "Copernicus GLO-30", "m"),
    ("twi", "Topographic wetness index ln(a / tan slope), mean in the buffer (high = water collects)", "MERIT Hydro v1.0.1 (90 m)", "-"),
    ("hand_m", "Height above the nearest drainage, mean in the buffer (low = wet, poorly drained)", "MERIT Hydro v1.0.1 (90 m)", "m"),
    ("insolation_ratio_thaw", "Clear-sky direct sun on the site's slope over May-Sep relative to flat ground (below 1 = less sun, e.g. north slopes)", "derived from slope, aspect, latitude", "-"),
    ("buffer_m", "Radius over which the 10 m-1 km values of the site are averaged: max(footprint_m, coord_uncert_m) within 15-1000 m", "derived", "m"),
    ("footprint_m", "Half the side of the probing grid (CALM grid size), else 15 m for a borehole or tube and 50 m for other probing", "PANGAEA 972777 / method", "m"),
    ("coord_uncert_m", "Half the rounding step of the site coordinates, north-south (e.g. 925 m for degrees and minutes, 5.6 km for 0.1 degree)", "derived", "m"),
    ("coarse_location", "1 = coordinates rounded to more than 1 km; fine covariates may describe ground next to the site", "derived", "0/1"),
    ("mtpi_alos", "Multi-scale topographic position index (extra)", "CSP ERGo ALOS mTPI", "m"),
    ("landform_alos", "Most common ALOS landform class in the site buffer (extra)", "CSP ERGo ALOS landforms", "class"),
    ("dist_to_coast_km", "Distance to the nearest sea coast", "Natural Earth 1:10m coastline", "km"),
    ("continentality_dist", "log(1 + dist_to_coast_km), as in the earlier script", "derived", "-"),
    ("conrad_continentality", "Conrad index 1.7 A / sin(lat + 10) - 14, A = site-mean T_annual_range (extra)", "derived", "-"),
    ("landcover_esa_2020", "Most common class in the site buffer", "ESA WorldCover v100 (2020)", "class"),
    ("landcover_igbp", "Most common IGBP class in the site buffer (per-pixel mode of 2001-2024)", "MCD12Q1", "class"),
    ("soil_texture_0cm", "Most common USDA texture class in the site buffer, 0 cm (1 clay ... 12 sand)", "OpenLandMap", "class"),
    ("soil_texture_30cm", "Same at 30 cm", "OpenLandMap", "class"),
    ("soc_gkg_0cm", "Soil organic carbon at 0 cm, mean in the site buffer", "OpenLandMap", "g/kg"),
    ("soc_gkg_30cm", "Soil organic carbon at 30 cm", "OpenLandMap", "g/kg"),
    ("soc_gkg_avg_0_30cm", "Depth-weighted mean SOC 0-30 cm", "OpenLandMap", "g/kg"),
    ("bulk_density_0cm", "Bulk density at 0 cm", "OpenLandMap", "kg/m3"),
    ("bulk_density_30cm", "Bulk density at 30 cm", "OpenLandMap", "kg/m3"),
    ("clay_pct_0cm", "Clay at 0 cm", "OpenLandMap", "%"),
    ("sand_pct_0cm", "Sand at 0 cm", "OpenLandMap", "%"),
    ("silt_pct_0cm", "100 - sand - clay at 0 cm", "OpenLandMap", "%"),
    ("Max", "ALT: site-mean end-of-season thaw depth (ALT_cm of the multi-source file; named Max for the notebook)", "multi-source ALT dataset", "cm"),
    ("obs_doy", "Day of year of the probing survey, where the source gives a date (later surveys read deeper; usable as a control)", "multi-source ALT dataset", "day of year"),
    ("alt_flag", "Information flags of the ALT value (burned_site, unusual_year_regional, unusual_year_isolated)", "multi-source ALT dataset", "text"),
    ("permafrostType", "Permafrost extent class of the site's polygon", "IPA map (Brown et al. 2002)", "class"),
    ("groundIceType", "Ground ice content class", "IPA map", "class"),
    ("landformType", "Lowland (thick overburden) or mountain (thin overburden)", "IPA map", "class"),
]


def write_dictionary(path):
    pd.DataFrame(FEATURES, columns=["column", "definition", "source", "units"]).to_csv(path, index=False)
    print(f"Wrote {path}")
