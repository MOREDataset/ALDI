"""
STEP 1: the circumarctic active layer thickness (ALT) target: download, read, build.

    py -3.12 1_alt_target.py              rebuild from the raw files already on this PC
    py -3.12 1_alt_target.py --download   first download new or missing source files

Reads   data/raw/alt_sources/   (downloads land here too)
Writes  data/1_alt_target/      alt_multisource_annual.csv is the input of step 3,
                                alt_multisource_for_ALDI.csv gives the site list of step 2
A new build goes to data/1_alt_target_building and replaces data/1_alt_target only when
it finished without error; the build before is kept in data/previous/1_alt_target.
NEON needs your own token for the session (PowerShell: $env:NEON_TOKEN="..."); without it
NEON is skipped. MET Norway's Frost API likewise needs your free client ID
($env:FROST_CLIENT_ID="..."; the client secret is not needed). Neither is ever written to a
file.

This one script makes the ALT target used by ALDI and the causal analysis, from every
source and every measuring method (probing grids, thaw tubes, ground-temperature
profiles). It has three stages, run in this order:

  1. DOWNLOAD  Every source file is fetched into the raw folder. Files whose link is known
               are listed in SOURCE_FILES (end of this script, with checksums); the
               sources served through an API (GTN-P data platform, Arctic Data Center,
               NEON, Zenodo, NSIDC, MET Norway THREDDS and Frost, CALM website) are queried here. Files
               already in the raw folder are not downloaded again. Sources that need a
               personal login or a data request are listed at the end of the run, with
               where to get them; put those files in the raw folder by hand.
  2. READ      One reader per source turns its files into one value per site and thaw
               season (cm), with the same rules for every source: the end-of-season
               survey for probing, and the 0 C crossing of the annual maximum profile for
               ground temperatures (see CONFIG for every rule and its reason).
  3. BUILD     Keeps circumarctic sites, joins the same physical site found in several
               sources, applies the quality flags, keeps sites with at least
               CONFIG["min_years"] valid years, and writes the outputs.

Python 3.12 with numpy pandas requests xlrd openpyxl pdfplumber truststore, and for
NEON and MET Norway also neonutilities xarray netCDF4.

Options:
    --download             download new or missing files first (default: use the
                           files already in data/raw/alt_sources)
    --download-only        stop after stage 1
    --only <readers>       run only some readers, e.g. --only gtnp_platform neon
    --min-years 8          threshold of valid annual values per site

Outputs (in data/1_alt_target):
    alt_multisource_all_records.csv  every annual record from every source, before merging
    alt_multisource_annual.csv       one row per site and year (kept sites), all flags, 'use'
    alt_multisource_sites.csv        one row per kept site: position, sources, method, years
    alt_multisource_for_ALDI.csv     site_id, Location, Lat, Long, Year, ALT_cm, source, method,
                                     obs_doy, flag: valid years only (the ALDI input)
    check_source_agreement.csv       for sites read from two sources, the yearly differences
                                     (e.g. GTN-P platform grids against PANGAEA 972777)
    check_pre2000_g02189.csv         NSIDC G02189 (pre-2000 Russian soil temperatures) against
                                     the Russian ALT series of the same stations; never used
                                     as data
    source_inventory.csv             every source: what was read and used, or why not
    download_report.csv              every file: downloaded, already there, failed, or by hand
    data_credits.md                  dataset citations and the data authors to credit
    build_log.txt                    everything the run printed

Units: cm. A "year" is the thaw season (calendar year of the end-of-season value).
The raw files are not in the repository: they belong to their data owners.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import quote, urljoin

import warnings

import numpy as np
import pandas as pd

# the readers parse many date layouts; pandas warns each time it falls back to dateutil
warnings.filterwarnings("ignore", message="Could not infer format")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "alt_sources"
OUT = DATA / "1_alt_target"


# =============================================================================
# SETTINGS (all decisions in one place; the reason is next to each value)
# =============================================================================
CONFIG = {
    # A site needs this many valid annual values (uncensored, not flagged) to be kept.
    "min_years": 8,
    # Circumarctic: north of 50 N. Mountain permafrost of central Asia and the Alps is
    # outside the scope even if north of 50 N (Mongolia, Altai, Kazakhstan, China).
    "min_lat": 50.0,
    # Downloads of new boreholes (GTN-P platform) are limited to datasets spanning at
    # least min_years seasons from this year (the ALDI analysis starts in 1990). Older
    # data already read from other sources are kept as before.
    "first_year": 1990,
    # A temperature profile can give ALT only with at least two sensors in the top
    # max_shallow_cm (deeper boreholes are not read).
    "max_shallow_cm": 150.0,
    # Syrdakh (Yakutia): the files and the Zenodo record give no coordinates, so every
    # Syrdakh series gets this approximate centre of the study area (lat, lon), as given
    # in the data paper. Logger positions may be in the record's .gpkg layers (not read).
    "syrdakh_position": (62.56, 130.9),
    "exclude_countries": ["Mongolia", "Kazakhstan", "China", "Switzerland", "Italy",
                          "France", "Austria", "Spain", "Kyrgyzstan", "Argentina",
                          "Antarctica", "Germany", "Japan"],
    # Plausible ALT range (cm). Values outside are dropped as entry errors. Borehole
    # (temperature) sites may thaw deeper than a probe can reach (e.g. Snoheim, Norway,
    # 5-7 m), so they get a wider range.
    "valid_alt_cm": (2.0, 500.0),
    "valid_alt_cm_borehole": (2.0, 1500.0),
    # End-of-season window for dated probing surveys (day of year). A season's value is
    # the deepest survey mean inside the window; seasons with surveys only before the
    # window start are not end-of-season and are dropped.
    "eos_doy_window": (213, 290),        # 1 Aug - 17 Oct
    # Probing grids: a survey counts only if it covers at least this fraction of the
    # season's fullest survey (partial surveys bias the mean).
    "min_node_fraction": 0.5,
    # A grid probed over several days counts as one survey: dates of one site no more
    # than this many days apart are pooled (dated by the last day).
    "survey_merge_days": 3,
    # Temperature profiles: in a given year a sensor is used only when it has data on at
    # least this fraction of the days from 1 August to 30 September (the end of the thaw
    # season, when the thaw is deepest; same window start as for probing surveys).
    "min_summer_cover": 0.8,
    # Temperature profiles: the two sensors around the 0 C crossing must be close enough
    # for a linear interpolation to mean something: spacing <= max(50 cm, 50 % of ALT),
    # and never more than 100 cm. Wider brackets are kept but flagged
    # 'wide_sensor_spacing' and not used.
    "max_bracket_cm": 50.0,
    "max_bracket_frac": 0.5,
    "max_bracket_abs_cm": 100.0,
    # Temperature profiles: the deepest thawed sensor must refreeze clearly in the
    # following winter (1 Oct - 30 Jun): its minimum must be below -refreeze_margin_c
    # (0.1 C, the usual accuracy of ground-temperature sensors). If not, the thaw did not
    # end at the base of a seasonally frozen layer (unfrozen ground, or sensors hovering
    # at 0 C), and the year is flagged 'no_refreeze' and not used. Checked only when that
    # sensor has data on at least min_winter_cover of the days of January-March.
    "refreeze_margin_c": 0.1,
    "min_winter_cover": 0.5,
    # Temperature profiles: the annual maximum temperature falls with depth (the seasonal
    # wave is damped as it goes down). When the last thawed sensor, the one the 0 C
    # crossing is interpolated from, reads more than inversion_tol_c warmer than the
    # sensor above it, its depth label or the sensor itself is wrong (GSC KP182M 2012:
    # 13.8 C at 125 cm under 0.2 C at 112 cm; Clearwater Lake LECCT02 2001-2004: 2.9 C at
    # 100 cm under 0.2 C at 80 cm). Such years are flagged 'profile_inversion' and not
    # used. Other sensors are not checked: a cool moss-layer sensor near the surface
    # (Parks Canada 2.5 cm) or a faulty deep one does not enter the interpolation.
    "inversion_tol_c": 1.0,
    # Two records are the same physical site when they share a site key (CALM code or
    # station id), or lie within this distance and use the same method family.
    "dup_distance_km": 1.0,
    # Two sources agree on a shared site when the median absolute difference over the
    # overlapping years is within this many cm (or this fraction of the mean ALT).
    "dup_agree_cm": 3.0,
    "dup_agree_frac": 0.05,
    # Temperature series at the same position (within 100 m) are the same borehole
    # processed by two groups (e.g. CALM and USGS for the NPR-A stations); their values
    # differ by a few cm, so the tolerance is wider: 5 cm or 10 %, on >= 5 shared years.
    "same_borehole_km": 0.1,
    "same_borehole_agree_cm": 5.0,
    "same_borehole_agree_frac": 0.10,
    # Copies of one series under two names (e.g. CALM U17 Wickersham and BNZ LTER 2130):
    # within 2 km, >= 5 shared years, median |difference| <= 1 cm and correlation >= 0.98.
    # Joined whatever method each source names (the same tube can be called a probe).
    "copy_km": 2.0,
    "copy_agree_cm": 1.0,
    "copy_min_r": 0.98,
    # Series of one source are never joined (the source lists them as different plots),
    # except in these collections of many files, where one station can be published
    # twice (a station file and a network table; a daily workbook and its NSIDC copy).
    # There, only rules (c) and (d) of assign_physical_sites join them: same position
    # and close agreement on >= 5 shared years.
    "join_within_source": ("tsp_alaska", "uaf_sites"),
    # Frost stations that are a CALM borehole read by MET Norway: the CALM code joins them
    # (rule a). Juvvasshoe SN15270 lies about 10 m from CALM N2 but reads 25-30 cm deeper
    # every year (a different way of placing the 0 C depth), so the agreement rules kept
    # the same borehole apart as two sites; CALM's values stay first (priority).
    "frost_calm_code": {"SN15270": "N2"},
    # Series read for comparison only (flag check_only, never used and never passing
    # flags to other sources). Frost SN99875 is the 102 m Janssonhaugen borehole under a
    # concrete lid: its upper metres read a damped annual cycle and ALT 17 cm (median)
    # shallower than the 15 m borehole P11 next to it, which CALM N1 reports (within 0-4 cm
    # in 1999-2024) and which metno_janssonhaugen reads.
    "check_only_series": {"metno_frost": ["METNO_SN99875"]},
    # The UiO file of the same Juvvasshoe PACE borehole (juvP, 10 m from SN15270) joins
    # CALM N2 the same way.
    "uio_calm_code": {"juvP": "N2"},
    # Roshydromet soil-temperature stations that are a station of the 1930-1990 Russian ALT
    # product (ru_hms_alt): WMO index -> its site key, so the two are one site (rule a).
    # Matched by name and position (within 3 km).
    "rusoil_ruhms": {24959: "RUHMS_003", 24266: "RUHMS_011", 25206: "RUHMS_012",
                     24688: "RUHMS_013", 24790: "RUHMS_133", 25428: "RUHMS_165",
                     24143: "RUHMS_237", 24951: "RUHMS_239", 24136: "RUHMS_246",
                     24382: "RUHMS_249"},
    # Same borehole served by two products (rule e of assign_physical_sites): the series on
    # the left joins the site of the series on the right, which ranks first. The UiO file of
    # Juvflye BH1 matches Frost SN15265 within 0-3 cm in 2009-2012 but reads 40-100 cm
    # deeper from 2014, so the agreement rules made it a second site of one borehole. The
    # other two pairs join by agreement anyway; listing them keeps that from depending on
    # the tolerance.
    "force_join": {"uio_boreholes:UIO_juv1": "metno_frost:METNO_SN15265",
                   "uio_boreholes:UIO_juv3": "metno_frost:METNO_SN15290",
                   "uio_boreholes:UIO_isk2": "metno_frost:METNO_SN97710"},
    # NEON re-surveys its sensors and writes a new position row each time, mostly with
    # the same depth. A sensor counts as moved (frost heave) when its depth changes by
    # more than this within the plot-year.
    "neon_move_m": 0.02,
    # A UAF station file whose header position is this far from the station lists has a
    # typo in its header; the station lists are used (see _uaf_station_file).
    "station_typo_km": 5.0,
    # Within-site screen on log(ALT): robust z beyond this is flagged (not removed).
    "outlier_robust_z": 4.0,
    # Source priority when the same site-year is in several sources (lower = preferred).
    # Network-curated tables first, then original data files, then compilations.
    "priority": {"calm_summary": 1, "pangaea_calm": 2, "gtnp": 3,
                 "gsc_thaw_tube": 3, "parks_canada": 3, "ru_hms_alt": 3,
                 "calm_site_files": 3, "kwbs": 3, "allena": 3, "kytalyk": 3,
                 "samoylov_calm": 3, "arc_lter": 3, "bnz_lter": 3, "itex": 3,
                 "illisarvik": 3, "stordalen": 3, "gem": 3,
                 "usgs_npra": 4, "gsc_cable": 4, "awi_soil_temp": 4, "ngee_barrow": 4,
                 "uaf_stations": 4, "umiujaq_cable": 4,
                 "canada_gt": 5, "above_field": 5, "cusp": 6,
                 # sources added in November 2026: their copies of known sites rank after
                 # the original data files; GTN-P grids rank as the old GTN-P exports
                 # (UAF site files and NEON rank after the sources they repeat, so site
                 # ids stay as they were)
                 "gtnp_platform_alt": 3, "gtnp_platform": 4, "tsp_alaska": 4, "uaf_sites": 5,
                 "neon": 5, "syrdakh": 4, "metno": 4, "metno_frost": 4, "g02189": 8,
                 # UiO Norway/Iceland boreholes rank after Frost, which serves the same
                 # loggers of four of them up to 2025
                 "tarfala_pace": 4, "uio_boreholes": 5,
                 # Roshydromet daily soil temperatures rank before the monthly-based
                 # 1930-1990 ALT product of the same stations (it reads shallow)
                 "roshydromet_soil": 2,
                 # Janssonhaugen P11 is the borehole behind CALM N1, so CALM stays first
                 "metno_janssonhaugen": 4, "other": 7},
    # Known site problems, from earlier ALDI v5 QC: CALM code -> first affected year.
    # Pearl Creek (U19) and Bonanza Creek (U18) thaw into a talik: thaw depth stops
    # being ALT. Values from that year are flagged "talik".
    "talik_from": {"U18": 2011, "U19": 2015},
    # CALM sites whose borehole value sits at the deepest sensor most years (lower bound).
    "sensor_cap_cm": {"U30": 120.0},
    # Interior Alaska probing (Bonanza Creek LTER): the probe, with extensions, is pushed
    # down to the first hard frozen ground. Once the winter frost no longer reaches the
    # permafrost (a talik), that is the permafrost table or leftover seasonal frost, not
    # ALT, and the series jump between ~70 and ~300 cm from one year to the next. CALM
    # flags its two Fairbanks-area sites (U18, U19) as talik from a thaw depth of about
    # 1 m. Here a talik is assumed from the first year a series of these sources is
    # deeper than this (cm); that year and all later ones are flagged 'possible_talik'.
    "talik_onset_cm": {"bnz_lter": 150.0},
    # New boreholes: at the GSC Mackenzie cables the first year of a record reads about
    # 6 % shallower than the second (median of 44 cables; the second and third years
    # differ by 1.5 %), a sign that the ground around the new hole had not settled
    # thermally. The first year of each series of these sources is flagged
    # 'borehole_equilibration'.
    "equilibration_years": {"gsc_cable": 1},
    # The Canada ground temperature collection has no installation dates, but a record
    # that starts in May-August had its cable put in that summer, so its first late-summer
    # maximum is read within weeks of drilling. For those series the first year is flagged
    # 'borehole_equilibration' too (36 such series with three good first years: the first
    # year is shallower than the second in 58 %, the third in 28 %; e.g. UAF Mould Bay,
    # drilled July 2004: 27 cm, then 40-69 cm).
    "install_months": (5, 8),
    # Canada collection series with years whose sensors are not comparable with the rest
    # of the record (series id -> (first, last) years); flagged 'sensor_problem', not used.
    #   LECCT02 and LECTOUR (Clearwater Lake, Nunavik, same coordinates): both cables were
    #   re-instrumented in 2005 (new depth sets). Before, LECCT02's profile is not ordered
    #   by depth (60 cm warmer than 50 cm, 100 cm than 80 cm) and the two series swap
    #   values at the change (LECCT02 116 -> 45 cm, LECTOUR 62 -> 121 cm): the old records
    #   mix the two cables. Years to 2005 not used.
    #   LECTOUR from 2017: the whole profile is colder than in any year since 2006 (25 cm
    #   maximum 11 C in 2017 and 6 C in 2018, 12-18 C before), in a normal summer at the
    #   other Nunavik sites (2017).
    #   BYLOTPD (Bylot Island, low-centred polygon): the summer maximum at 10 cm is
    #   3.0-4.5 C in 1999-2012, then 2.2 C (2013) and 0.4-1.4 C in 2014-2022, while the
    #   20 cm sensor of BYLOTPB 50 m away stays at 9-13 C; ALT falls from 50-66 to 30-39 cm.
    #   The last two look like a change of the sensors' depth below the surface (burial,
    #   ponding) or of the sensors, not of the ground.
    # uio_boreholes: from August 2013 the UiO file of Juvflye BH1 (juv1.dat) carries the
    #   depth labels one sensor too deep (two from May 2018): each column then equals the
    #   Frost SN15265 sensor one label higher (0.00 C difference), and its 0.2 m column
    #   holds a channel that matches no sensor. Its ALT reads 30-100 cm too deep, so
    #   2013 onward is not used (the site keeps the Frost values).
    "sensor_problem_years": {"uio_boreholes": {"UIO_juv1": [(2013, 2100)]},
                             "canada_gt": {"LECCT02": [(1900, 2005)],
                                           "LECTOUR": [(1900, 2005), (2017, 2100)],
                                           "BYLOTPD": [(2013, 2100)]}},
}

# Flags that make a value unusable for trend analysis (it stays in the files).
# Matched as whole flags, so an informational flag such as 'deep_thaw_check_talik'
# does not count as 'talik'.
INVALID_FLAG_NAMES = ("calm_footnote", "talik", "possible_talik", "sensor_cap",
                      "wide_sensor_spacing", "lower_bound", "treatment", "early_measurement",
                      "no_refreeze", "borehole_equilibration", "disturbed_site",
                      "profile_inversion", "sensor_problem", "sensor_moved", "check_only")
INVALID_FLAGS = r"(?:^|;)(?:" + "|".join(INVALID_FLAG_NAMES) + r")(?:;|$)"

# Flags that belong to the physical site, not to one source's copy of a value (applied
# across sources after the series of one site are joined, in site_state_flags):
#   STATE_FLAGS: the ground is in a talik from that year on -> every later year of the site;
#   YEAR_STATE_FLAGS: the ground did not refreeze that winter -> every record of that year;
#   COPY_FLAGS: something about that survey or borehole (a new borehole's first year, a
#   disturbed site) -> records of that year with the same value (the same data passed on
#   through another source, e.g. a GSC cable also in the Canada collection).
STATE_FLAGS = ("talik", "possible_talik")
YEAR_STATE_FLAGS = ("no_refreeze",)
COPY_FLAGS = ("calm_footnote", "sensor_cap", "lower_bound", "treatment", "early_measurement",
              "borehole_equilibration", "disturbed_site")

COUNTRY_ALIASES = {"United States": "USA", "United States of America": "USA", "US": "USA",
                   "United States (Alaska)": "USA", "Alaska": "USA",
                   "Russian Federation": "Russia", "Svalbard": "Norway (Svalbard)",
                   "Svalbard and Jan Mayen": "Norway (Svalbard)"}

RECORD_COLUMNS = [
    "source", "source_site_id", "site_name", "calm_code", "site_key", "lat", "lon",
    "country", "method", "year", "alt_cm", "alt_sd_cm", "n_points", "obs_date",
    "censored", "bracket_cm", "flag", "site_note", "source_file",
]

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg)
    LOG_LINES.append(msg)


# =============================================================================
# small helpers
# =============================================================================
def as_float(s):
    try:
        return float(str(s).replace(",", ".").strip())
    except (TypeError, ValueError):
        return np.nan


def parse_censored(v):
    """'>120' or '120+' -> (120.0, True); '<56' -> (56.0, True); '85' -> (85.0, False)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, False
    if isinstance(v, (int, float, np.number)):
        return float(v), False
    s = str(v).strip()
    cens = s.startswith((">", "<")) or s.endswith("+")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return (float(m.group()) if m else np.nan), cens


def clean_cell(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (int, float, np.number)):
        return repr(float(v))
    return str(v).replace("\xa0", " ").replace("�", "").strip()


def norm_calm_code(code: str) -> str:
    """'U12 A' -> 'U12A'; '' stays ''."""
    return re.sub(r"\s+", "", str(code or "")).upper()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def method_family(m: str) -> str:
    """probe / tube / temperature: only series of the same family can be merged."""
    m = str(m)
    if m.startswith(("borehole", "soil_temp")):
        return "temperature"
    if m in ("thaw_tube", "frost_tube"):
        return "tube"
    return "probe"


def is_temperature_method(methods: pd.Series) -> np.ndarray:
    return methods.astype(str).str.startswith(("borehole", "soil_temp")).to_numpy()


def empty_records() -> pd.DataFrame:
    return pd.DataFrame(columns=RECORD_COLUMNS)


def add_flag(flags: pd.Series, mask, text: str) -> pd.Series:
    """Append 'text' to the ';'-separated flags where mask is True."""
    flags = flags.fillna("").astype(str).copy()
    mask = np.asarray(mask, bool)
    flags[mask] = (flags[mask] + ";" + text).str.strip(";")
    return flags


def finish(df: pd.DataFrame, source: str, source_file: str = "") -> pd.DataFrame:
    """Fill missing record columns, set types, and drop implausible values."""
    df = df.copy()
    for c in RECORD_COLUMNS:
        if c not in df:
            df[c] = "" if c in ("flag", "calm_code", "country", "site_name", "site_key",
                                "site_note", "source_file") else np.nan
    df["source"] = source
    if source_file:
        df["source_file"] = df["source_file"].fillna("").replace("", source_file)
    for c in ("flag", "site_name", "site_key", "site_note", "source_file", "country"):
        df[c] = df[c].fillna("").astype(str)
    df["calm_code"] = df["calm_code"].fillna("").map(norm_calm_code)
    df["site_key"] = np.where((df.site_key == "") & (df.calm_code != ""),
                              "CALM_" + df.calm_code, df.site_key)
    df["country"] = df["country"].replace(COUNTRY_ALIASES)
    # Compiled tables sometimes give two countries ('Sweden/Svalbard', 'Poland/Svalbard'):
    # decide by position. Svalbard is the only land at 74-81.5 N, 0-40 E.
    lat_ = pd.to_numeric(df.lat, errors="coerce")
    lon_ = pd.to_numeric(df.lon, errors="coerce")
    svalbard = lat_.between(74, 81.5) & lon_.between(0, 40)
    two = df.country.str.contains("/")
    df.loc[svalbard, "country"] = "Norway (Svalbard)"
    df.loc[two & ~svalbard, "country"] = df.loc[two & ~svalbard, "country"].str.split("/").str[0]
    df["censored"] = df["censored"].fillna(False).astype(bool)
    for c in ("year", "alt_cm", "alt_sd_cm", "n_points", "lat", "lon", "bracket_cm"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    lo, hi = CONFIG["valid_alt_cm"]
    hi_b = CONFIG["valid_alt_cm_borehole"][1]
    top = np.where(is_temperature_method(df.method), hi_b, hi)
    bad = (df.alt_cm.isna() | (df.alt_cm < lo) | (df.alt_cm > top) | df.year.isna()
           | df.lat.isna() | df.lon.isna())
    if bad.any():
        log(f"  {source}: {int(bad.sum())} records dropped (missing value or position, or "
            f"outside {lo:g}-{hi:g} cm for probes/tubes, {lo:g}-{hi_b:g} cm for temperature)")
    df = df[~bad].copy()
    df["year"] = df["year"].astype(int)
    out = df[RECORD_COLUMNS].reset_index(drop=True)
    if len(out):
        log(f"  {source}: {out.source_site_id.nunique()} series, {len(out)} site-years "
            f"({int(out.year.min())}-{int(out.year.max())})")
    return out


# =============================================================================
# end of season (probing surveys)
# =============================================================================
def survey_means(points: pd.DataFrame, key: str = "source_site_id") -> pd.DataFrame:
    """
    points: one row per probe reading (key, obs_date, alt_cm, censored optional, other
    site columns). Returns one row per site and survey date: mean, SD, number of points,
    and the share of censored readings (probe hit its length, rock, ...).
    Surveys with fewer than min_node_fraction of the season's fullest survey are dropped.
    """
    p = points.copy()
    p["obs_date"] = pd.to_datetime(p["obs_date"], errors="coerce")
    p = p[p.obs_date.notna() & p.alt_cm.notna()]
    if "censored" not in p:
        p["censored"] = False
    # pool dates of one survey (grid probed over two or three days)
    p = p.sort_values([key, "obs_date"])
    gap = p.groupby(key).obs_date.diff().dt.days
    survey_no = (gap.isna() | (gap > CONFIG["survey_merge_days"])).groupby(p[key]).cumsum()
    p["obs_date"] = p.groupby([p[key], survey_no]).obs_date.transform("max")
    site_cols = [c for c in ("site_name", "calm_code", "site_key", "lat", "lon", "country",
                             "method", "site_note", "source_file") if c in p]
    agg = {"alt_cm": ("alt_cm", "mean"), "alt_sd_cm": ("alt_cm", "std"),
           "n_points": ("alt_cm", "size"), "cens_share": ("censored", "mean")}
    for c in site_cols:
        agg[c] = (c, "median") if c in ("lat", "lon") else (c, "first")
    s = p.groupby([key, "obs_date"]).agg(**agg).reset_index()
    nmax = s.groupby([s[key], s.obs_date.dt.year]).n_points.transform("max")
    partial = s.n_points < CONFIG["min_node_fraction"] * nmax
    if partial.any():
        log(f"    {int(partial.sum())} partial surveys dropped (< "
            f"{CONFIG['min_node_fraction']:.0%} of the season's fullest survey)")
    s = s[~partial].copy()
    # a survey mean is a lower bound when many of its points were capped
    s["censored"] = s.cens_share >= 0.2
    return s.drop(columns="cens_share")


def end_of_season(surveys: pd.DataFrame, key: str = "source_site_id",
                  min_last_doy: int | None = None) -> pd.DataFrame:
    """
    surveys: one row per site and survey date (alt_cm = survey mean, obs_date, n_points).
    Returns one row per site and year: the deepest survey inside the end-of-season window.
    Seasons whose surveys are all before the window are dropped (logged).
    min_last_doy: for thaw-depth monitoring series (repeated readings through the
    summer), also drop seasons whose last reading is before this day of year, because
    the thaw had not reached its maximum yet.
    """
    s = surveys.copy()
    s["obs_date"] = pd.to_datetime(s["obs_date"], errors="coerce")
    s = s[s.obs_date.notna() & s.alt_cm.notna()]
    s["year"] = s.obs_date.dt.year
    doy = s.obs_date.dt.dayofyear
    d0, d1 = CONFIG["eos_doy_window"]
    inwin = doy.between(d0, d1)
    early_only = ~inwin.groupby([s[key], s.year]).transform("max")
    n_early = s[early_only].groupby([key, "year"]).ngroups
    if n_early:
        log(f"    {n_early} site-seasons with surveys only before day {d0} dropped "
            f"(not end of season)")
    if min_last_doy is not None:
        last = doy.groupby([s[key], s.year]).transform("max")
        short = (last < min_last_doy) & ~early_only
        n_short = s[short].groupby([key, "year"]).ngroups
        if n_short:
            log(f"    {n_short} site-seasons whose last reading is before day "
                f"{min_last_doy} dropped (thaw not finished)")
        inwin &= ~short
    s = s[inwin]
    s = s.sort_values("alt_cm").groupby([key, "year"], as_index=False).tail(1)
    return s


# =============================================================================
# ALT from ground temperatures
# =============================================================================
def alt_from_temperature_profile(temps: pd.DataFrame, min_season_cover: float | None = None,
                                 zero_c: float = 0.0, monthly: bool = False) -> pd.DataFrame:
    """
    Annual ALT from ground temperatures at several depths (boreholes, soil stations).

    temps: columns 'time' (datetime), 'depth_cm' (positive down), 'temp_c'.
    For each calendar year: take each sensor's maximum temperature of the year; ALT is the
    depth where that maximum-temperature profile first crosses 0 C going down, by linear
    interpolation between the last thawed sensor and the next sensor below it.

    In a given year a sensor is used only when it has data on >= min_season_cover of the
    days from 1 August to 30 September (the end of the thaw season, when the thaw is
    deepest); a year needs at least two such sensors. With monthly=True (monthly mean
    temperatures) a sensor needs August and September values instead. Monthly means are smoother than daily values, so ALT
    from monthly data is biased shallow; those records carry method 'soil_temp_monthly'.
    Censored (ALT deeper than the deepest sensor): all sensors above 0 C -> value is the
    deepest sensor depth, censored=True. No sensor above 0 C -> no value (thaw shallower
    than the shallowest sensor; not resolvable).
    Refreeze check: the deepest thawed sensor must go below -refreeze_margin_c (0.1 C) in
    the following winter (1 Oct - 30 Jun). If it does not, the year gets
    no_refreeze=True. When that sensor has too little January-March data the check is
    skipped (no_refreeze=False).
    Profile check: inversion=True when the last thawed sensor reads more than
    inversion_tol_c warmer than the sensor above it (mislabelled depth or faulty sensor).
    Returns one row per year: year, alt_cm, censored, n_sensors, deepest_cm, bracket_cm,
    nonmonotonic (a deeper sensor is above 0 C again below the crossing), no_refreeze,
    inversion.
    """
    if min_season_cover is None:
        min_season_cover = CONFIG["min_summer_cover"]
    t = temps.dropna(subset=["temp_c", "depth_cm"]).copy()
    t = t[t.depth_cm > 0]
    if t.empty:
        return pd.DataFrame(columns=["year", "alt_cm", "censored", "n_sensors", "deepest_cm",
                                     "bracket_cm", "nonmonotonic", "no_refreeze", "inversion"])
    t["time"] = pd.to_datetime(t["time"])
    t["year"] = t.time.dt.year
    late = t[t.time.dt.month.between(8, 9)]                 # 1 Aug - 30 Sep, 61 days
    if monthly:
        cover = late.assign(m=late.time.dt.month).groupby(["year", "depth_cm"]).m.nunique() / 2.0
        need = 1.0
    else:
        cover = late.assign(d=late.time.dt.normalize()).groupby(["year", "depth_cm"]).d.nunique() / 61.0
        need = min_season_cover
    tmax = t.groupby(["year", "depth_cm"]).temp_c.max()
    # following winter: Oct-Dec of the year and Jan-Jun of the next year
    m = t.time.dt.month
    w = t[(m >= 10) | (m <= 6)].copy()
    w["wyear"] = np.where(w.time.dt.month >= 10, w.year, w.year - 1)
    wmin = w.groupby(["wyear", "depth_cm"]).temp_c.min()
    jfm = w[w.time.dt.month <= 3]
    if monthly:
        wcover = jfm.assign(k=jfm.time.dt.month).groupby(["wyear", "depth_cm"]).k.nunique() / 3.0
    else:
        wcover = jfm.assign(k=jfm.time.dt.normalize()).groupby(["wyear", "depth_cm"]).k.nunique() / 90.0
    wmin = wmin[wcover.reindex(wmin.index).fillna(0) >= CONFIG["min_winter_cover"]]

    def no_refreeze(yr, depth):
        v = wmin.get((yr, depth), np.nan)
        return bool(np.isfinite(v) and v >= zero_c - CONFIG["refreeze_margin_c"])

    out = []
    for yr, prof in tmax.groupby(level=0):
        prof = prof.droplevel(0).sort_index()
        ok = cover.reindex(pd.MultiIndex.from_product([[yr], prof.index])).droplevel(0)
        prof = prof[ok.fillna(0).to_numpy() >= need - 1e-9]
        if len(prof) < 2:
            continue
        d, v = prof.index.to_numpy(float), prof.to_numpy(float)
        above = v > zero_c
        if not above.any():
            continue
        # Go down from the shallowest thawed sensor to the first sensor at or below 0 C.
        # Deeper sensors slightly above 0 C after that (common in permafrost close to
        # 0 C, from sensor offsets of a few tenths of a degree) do not move the value;
        # the year is flagged 'nonmonotonic_profile' instead.
        i = int(np.argmax(above))
        while i + 1 < len(v) and above[i + 1]:
            i += 1
        if i == len(v) - 1:
            out.append(dict(year=yr, alt_cm=d[i], censored=True, n_sensors=len(v),
                            deepest_cm=d[-1], bracket_cm=np.nan, nonmonotonic=False,
                            no_refreeze=no_refreeze(yr, d[i]), inversion=False))
            continue
        # last thawed sensor warmer than the thawed sensor above it
        inversion = bool(i > int(np.argmax(above)) and v[i] > v[i - 1] + CONFIG["inversion_tol_c"])
        # the crossing lies between sensor i (above 0) and i+1 (at or below 0)
        z = d[i] + (d[i + 1] - d[i]) * (v[i] - zero_c) / (v[i] - v[i + 1])
        # bracket_cm: spacing of the two sensors around the crossing = interpolation
        # uncertainty (the true profile is not linear between sensors)
        out.append(dict(year=yr, alt_cm=z, censored=False, n_sensors=len(v),
                        deepest_cm=d[-1], bracket_cm=d[i + 1] - d[i],
                        nonmonotonic=bool(above[i + 1:].any()), no_refreeze=no_refreeze(yr, d[i]),
                        inversion=inversion))
    return pd.DataFrame(out)


def records_from_profiles(temps: pd.DataFrame, sites: pd.DataFrame, method: str,
                          monthly: bool = False, key: str = "source_site_id") -> pd.DataFrame:
    """
    temps: long table key, time, depth_cm, temp_c (one or many sites).
    sites: one row per key with site_name, lat, lon, country (and optional site_key,
    calm_code, site_note, source_file).
    Runs alt_from_temperature_profile per site, flags wide sensor spacing, lower bounds,
    non-monotonic profiles and missing winter refreeze, and returns
    records (not yet finished).
    """
    rows = []
    for k, g in temps.groupby(key):
        a = alt_from_temperature_profile(g, monthly=monthly)
        if a.empty:
            continue
        a[key] = k
        rows.append(a)
    if not rows:
        return pd.DataFrame(columns=[key])
    a = pd.concat(rows, ignore_index=True)
    a = a.merge(sites, on=key, how="left")
    a["method"] = method
    a["n_points"] = a.pop("n_sensors")
    allowed = np.minimum(np.maximum(CONFIG["max_bracket_cm"], CONFIG["max_bracket_frac"] * a.alt_cm),
                         CONFIG["max_bracket_abs_cm"])
    wide = a.bracket_cm > allowed
    a["flag"] = add_flag(a.get("flag", pd.Series("", index=a.index)), wide, "wide_sensor_spacing")
    a["flag"] = add_flag(a["flag"], a.censored, "lower_bound")
    a["flag"] = add_flag(a["flag"], a.pop("nonmonotonic").fillna(False).astype(bool),
                         "nonmonotonic_profile")
    a["flag"] = add_flag(a["flag"], a.pop("no_refreeze").fillna(False).astype(bool),
                         "no_refreeze")
    a["flag"] = add_flag(a["flag"], a.pop("inversion").fillna(False).astype(bool),
                         "profile_inversion")
    return a.drop(columns=["deepest_cm"])


# =============================================================================
# PANGAEA tab files
# =============================================================================
def read_pangaea_tab(path: Path) -> tuple[str, pd.DataFrame]:
    """Returns (header text, data table) of a PANGAEA .tab file."""
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    head, body = "", txt
    if txt.lstrip().startswith("/*"):
        i = txt.find("*/")
        head, body = txt[:i], txt[i + 2:]
    df = pd.read_csv(pd.io.common.StringIO(body.lstrip("\r\n")), sep="\t", low_memory=False)
    return head, df


def pangaea_events(head: str) -> dict[str, dict]:
    """Event label -> {lat, lon, elev, location, comment, name} from a PANGAEA header."""
    m = re.search(r"Event\(s\):\t(.*?)(?:\n[A-Z][A-Za-z ()/]*:\t|\Z)", head, re.S)
    if not m:
        m = re.search(r"Event:\t(.*?)(?:\n[A-Z][A-Za-z ()/]*:\t|\Z)", head, re.S)
    out = {}
    if not m:
        return out
    for ev in re.split(r"\n\t", m.group(1)):
        ev = ev.strip()
        if not ev:
            continue
        label = ev.split(" * ")[0].strip()
        name = ""
        mm = re.match(r"(\S+)\s*\((.*?)\)", label)
        if mm:
            label, name = mm.group(1), mm.group(2)
        get = lambda k: (re.search(rf"\b{k}: ([^*]+)", ev) or [None, ""])[1].strip()
        out[label] = dict(lat=as_float(get("LATITUDE")), lon=as_float(get("LONGITUDE")),
                          elev=as_float(get("ELEVATION").replace(" m", "")),
                          location=get("LOCATION"), comment=get("COMMENT"), name=name)
    return out


def utm_to_latlon(easting, northing, zone: int, north: bool = True):
    """UTM (WGS84/NAD83) to latitude/longitude in degrees (standard series formulas)."""
    e = np.asarray(easting, float) - 500000.0
    n = np.asarray(northing, float) - (0.0 if north else 10000000.0)
    a, f, k0 = 6378137.0, 1 / 298.257223563, 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    m = n / k0
    mu = m / (a * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    p = (mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * np.sin(2 * mu)
         + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * np.sin(4 * mu)
         + (151 * e1 ** 3 / 96) * np.sin(6 * mu) + (1097 * e1 ** 4 / 512) * np.sin(8 * mu))
    c1, t1 = ep2 * np.cos(p) ** 2, np.tan(p) ** 2
    n1 = a / np.sqrt(1 - e2 * np.sin(p) ** 2)
    r1 = a * (1 - e2) / (1 - e2 * np.sin(p) ** 2) ** 1.5
    d = e / (n1 * k0)
    lat = p - (n1 * np.tan(p) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
                                        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720)
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2) * d ** 5 / 120) / np.cos(p)
    return np.degrees(lat), (zone - 1) * 6 - 180 + 3 + np.degrees(lon)


def eml_sites(xml_text: str) -> dict[str, tuple[float, float, str]]:
    """
    Site code -> (lat, lon, description) from the geographic coverages of an EML
    (LTER / EDI metadata) file, where each description starts with '<code> -- ...'.
    """
    out = {}
    for m in re.finditer(r"<geographicDescription>(.*?)</geographicDescription>.*?"
                         r"<westBoundingCoordinate>(.*?)</west.*?<northBoundingCoordinate>(.*?)</north",
                         xml_text, re.S):
        desc = re.sub(r"\s+", " ", m.group(1)).strip()
        if " -- " not in desc:
            continue
        code, text = desc.split(" -- ", 1)
        out[code.strip()] = (as_float(m.group(3)), as_float(m.group(2)), text)
    return out


def files(raw, pattern: str) -> list[Path]:
    """
    All files under the raw folder(s) matching pattern, sorted. 'raw' is one folder or a
    list of folders. Known duplicate download folders are skipped.
    """
    skip = ("ParksCanada_ALT",)
    roots = raw if isinstance(raw, (list, tuple)) else [raw]
    return sorted(p for r in roots for p in Path(r).rglob(pattern)
                  if not any(s in p.parts for s in skip))


# =============================================================================
# STAGE 1: DOWNLOAD
# =============================================================================
# Where the sources served through an API live, and how they are screened before
# download (the scope rules themselves are in CONFIG).
API = {
    "gtnp": "https://data.gtn-p.org/api",                 # GTN-P data platform (AWI), CC BY
    "dataone": "https://arcticdata.io/metacat/d1/mn/v2",  # Arctic Data Center
    "tsp_alaska_doi": "doi:10.18739/A2J34P",              # Network of Permafrost Observatories: Alaska
    "uaf_site": "https://permafrost.gi.alaska.edu/site/{code}",
    "uaf_sites": {"gl1": "Galbraith Lake", "wd1": "West Dock", "hv1": "Happy Valley",
                  "cs1": "Chandalar Shelf"},
    "neon_product": "DP1.00041.001",                      # NEON soil temperature
    "neon_sites": ["BARR", "TOOL", "HEAL"],
    "syrdakh_zenodo": "14619854",
    "g02189": "https://noaadata.apps.nsidc.org/NOAA/G02189/",
    "metno_thredds": "https://thredds.met.no/thredds",
    "metno_catalog": "arcticdata/frost2netcdf-permafrost",
    "frost": "https://frost.met.no",                      # MET Norway Frost API (client ID)
    "calm_page": "https://www2.gwu.edu/~calm/data/north.htm",
    # A download never exceeds this size (MB); larger files are listed, not fetched.
    "max_file_mb": 300,
}
# Folder of each API source inside the raw folder (the readers look there).
API_FOLDERS = {"gtnp": "GTNP_platform", "tsp_alaska": "ADC_TSP_Alaska", "uaf_sites": "UAF_sites",
               "neon": "NEON_soil_temperature", "syrdakh": "Zenodo_Syrdakh",
               "g02189": "NSIDC_G02189", "metno": "METNO_permafrost", "metno_frost": "METNO_frost",
               "calm_table": "CALM_GWU", "tarfala_pace": "Tarfala_PACE",
               "uio_boreholes": "UiO_Norway_Iceland",
               "roshydromet_soil": "Roshydromet_soil",
               "metno_janssonhaugen": "METNO_Janssonhaugen_shallow"}
TABLE_EXT = (".csv", ".txt", ".dat", ".xls", ".xlsx", ".tab")

DOWNLOAD_REPORT: list[dict] = []
_SESSION = None


def _http():
    global _SESSION
    if _SESSION is None:
        try:
            # check TLS certificates against the computer's own certificate store (e.g. the
            # Windows store), which knows the intermediate certificates that some servers
            # do not send (permafrost.gi.alaska.edu failed on Windows without this).
            # Verification stays on. pip install truststore
            import truststore
            truststore.inject_into_ssl()
        except ImportError:
            pass
        import requests                                   # only needed for stage 1
        _SESSION = requests.Session()
        _SESSION.headers["User-Agent"] = ("ALDI-alt-target/1.0 (research; "
                                          "https://github.com/MOREDataset/ALDI)")
    return _SESSION


def http_get(url, params=None, stream=False, tries=4, headers=None):
    """GET with retries on rate limits and server errors (2, 4, 8 s). TLS certificates are
    always verified."""
    for k in range(tries):
        r = _http().get(url, params=params, timeout=300, stream=stream, headers=headers)
        if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(2 ** (k + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(url)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, source: str, params=None, sha256: str = "") -> Path | None:
    """
    Download url to dest unless dest is already there. Every call is logged in
    download_report.csv. When a checksum is known and the server now serves a different
    file, the new file is kept and the change is reported (the data owner updated it).
    """
    row = dict(source=source, file=str(dest), url=url, status="", note="")
    if dest.exists() and dest.stat().st_size > 0:
        row["status"] = "already there"
        DOWNLOAD_REPORT.append(row)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = http_get(url, params, stream=True)
        size = int(r.headers.get("Content-Length", 0) or 0)
        if size > API["max_file_mb"] * 1e6:
            row.update(status="too large", note=f"{size / 1e6:.0f} MB")
            DOWNLOAD_REPORT.append(row)
            return None
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
        row["status"] = "downloaded"
        if sha256 and sha256_of(dest) != sha256:
            row["note"] = "changed since the 2026-10-24 download (checksum differs)"
    except Exception as e:                       # one file failing must not stop the others
        row.update(status="failed", note=f"{type(e).__name__}: {e}"[:300])
        dest = None
    DOWNLOAD_REPORT.append(row)
    return dest


def download_listed(raw: Path, only: list[str] | None = None) -> None:
    """Every file in SOURCE_FILES with a direct link ('open'); the others are reported."""
    lst = pd.read_csv(io.StringIO(SOURCE_FILES), dtype=str).fillna("")
    for r in lst.itertuples():
        readers = r.reader.split(";")
        if only and not set(readers) & set(only):
            continue
        dest = raw / Path(r.local_path)
        if r.access != "open":
            DOWNLOAD_REPORT.append(dict(source=r.reader, file=str(dest), url=r.url,
                                        status="already there" if dest.exists() else "get by hand",
                                        note=r.note))
            continue
        fetch(r.url, dest, r.reader, sha256=r.sha256)


def download_calm_table(raw: Path) -> None:
    """CALM summary table: the newest CALM_Summary_table*.xls linked from the CALM page."""
    folder = raw / API_FOLDERS["calm_table"]
    try:
        html = http_get(API["calm_page"]).text
    except Exception as e:
        DOWNLOAD_REPORT.append(dict(source="calm_summary", file=str(folder), url=API["calm_page"],
                                    status="failed", note=str(e)[:300]))
        return
    links = sorted({urljoin(API["calm_page"], h) for h in re.findall(r'href="([^"]+)"', html, re.I)
                    if re.search(r"summary.*\.xlsx?$", h, re.I)})
    if not links:
        DOWNLOAD_REPORT.append(dict(source="calm_summary", file=str(folder), url=API["calm_page"],
                                    status="failed",
                                    note="the CALM page no longer links the summary table: put your "
                                         "saved CALM_Summary_table*.xls in the raw folder"))
    for u in links:
        name = u.split("/")[-1]
        if not name.startswith("CALM_Summary_table"):
            name = "CALM_Summary_table_" + name
        fetch(u, folder / name, "calm_summary")


def _json_cached(url, cache: Path, params=None):
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(http_get(url, params).json()))
    return json.loads(cache.read_text())


def download_gtnp_platform(raw: Path) -> None:
    """
    GTN-P data platform (data.gtn-p.org, open API, CC BY; replaces gtnpdatabase.org).
      Boreholes: sites north of CONFIG min_lat, outside the excluded countries, with
        has_data and a ground-temperature dataset that spans at least min_years seasons
        from CONFIG first_year. The depth check (at least two sensors in the top
        max_shallow_cm) is done by the reader, which sees the sensor depths.
      Active-layer datasets (grids and transects, mostly CALM): all of them; the build
        joins them to the CALM tables by CALM code and checks the agreement.
    The site metadata (JSON) is kept next to the data: the reader takes positions and the
    data authors from it.
    """
    base, folder = API["gtnp"], raw / API_FOLDERS["gtnp"]
    for kind in ("borehole", "activelayer"):
        lst = _json_cached(f"{base}/list-sites", folder / f"list_sites_{kind}.json",
                           {"lat_min": CONFIG["min_lat"], "metadata": "true", f"{kind}_data": "true"})
        for site in lst:
            if (site.get("country_name") or "") in CONFIG["exclude_countries"]:
                continue
            items = site.get("boreholes" if kind == "borehole" else "activelayers", [])
            for it in items:
                iid = it["id"]
                meta_url = f"{base}/pt/{iid}" if kind == "borehole" else f"{base}/alt/{iid}"
                sub = "pt" if kind == "borehole" else "alt"
                try:
                    meta = _json_cached(meta_url, folder / f"{sub}_meta" / f"{iid}.json")
                except Exception as e:
                    DOWNLOAD_REPORT.append(dict(source="gtnp_platform", file=f"{sub} {iid}",
                                                url=meta_url, status="failed", note=str(e)[:300]))
                    continue
                if kind == "borehole" and not meta.get("has_data"):
                    continue
                for d in meta.get("datasets", []):
                    var = json.dumps(d.get("variable")).lower()
                    if d.get("is_retired"):
                        continue
                    if kind == "borehole":
                        if "ground" not in var or _gtnp_dataset_years(d, meta) < CONFIG["min_years"]:
                            continue
                        params = {"pt_data": d["id"], "variable": "gt", "combined": "false"}
                    else:
                        if "active layer" not in var:
                            continue
                        params = {"alt_data": d["id"], "combined": "false",
                                  "variable": "alt-gt" if "grid" in var else "alt-btt"}
                    fetch(f"{base}/data", folder / f"{sub}_data" / f"{sub}_{d['id']}.zip",
                          "gtnp_platform", params=params)


def _gtnp_dataset_years(d: dict, meta: dict) -> int:
    """Seasons a GTN-P dataset spans from CONFIG first_year (from its statistics, else the
    borehole's dates)."""
    st = d.get("statistics") or {}
    d0 = pd.to_datetime(st.get("date_start") or st.get("date_min") or meta.get("date_start"),
                        errors="coerce")
    d1 = pd.to_datetime(st.get("date_end") or st.get("date_max") or meta.get("date_end"),
                        errors="coerce")
    if pd.isna(d0) or pd.isna(d1):
        return 0
    return d1.year - max(d0.year, CONFIG["first_year"]) + 1


def _solr(q, fl, rows=1000):
    return http_get(f"{API['dataone']}/query/solr/",
                    {"q": q, "fl": fl, "rows": rows, "wt": "json"}).json()["response"]["docs"]


def download_tsp_alaska(raw: Path) -> None:
    """
    Arctic Data Center: the newest version of 'Network of Permafrost Observatories in North
    America and Russia: Alaska' (doi:10.18739/A2J34P, followed through obsoletedBy) and the
    other current packages of V. Romanovsky with permafrost temperatures (the yearly
    thermal state of permafrost releases). Data tables and each package's EML metadata
    (the reader takes the station positions from it).
    """
    folder = raw / API_FOLDERS["tsp_alaska"]
    pid = API["tsp_alaska_doi"]
    while True:
        d = _solr(f'id:"{pid}"', "id,obsoletedBy")
        if not d or not d[0].get("obsoletedBy"):
            break
        pid = d[0]["obsoletedBy"]
    pkgs = {pid}
    pkgs |= {d["id"] for d in _solr("origin:*Romanovsky* AND title:*ermafrost* AND -obsoletedBy:* "
                                    "AND formatType:METADATA", "id")}
    for p in sorted(pkgs):
        safe = re.sub(r"[^\w.-]", "_", p)
        fetch(f"{API['dataone']}/object/{quote(p, safe='')}", folder / safe / "metadata_eml.xml",
              "tsp_alaska")
        for f in _solr(f'resourceMap:"resource_map_{p}" AND formatType:DATA', "id,fileName,size"):
            name = f.get("fileName") or re.sub(r"\W", "_", f["id"])
            if not name.lower().endswith(TABLE_EXT + (".zip",)):
                continue
            if (f.get("size") or 0) > API["max_file_mb"] * 1e6:
                DOWNLOAD_REPORT.append(dict(source="tsp_alaska", file=name, url=f["id"],
                                            status="too large", note=f"{f['size'] / 1e6:.0f} MB"))
                continue
            fetch(f"{API['dataone']}/object/{quote(f['id'], safe='')}", folder / safe / name,
                  "tsp_alaska")


def download_uaf_sites(raw: Path) -> None:
    """UAF Permafrost Lab site pages (Galbraith Lake, West Dock, Happy Valley, Chandalar
    Shelf): every data file linked from the page."""
    folder = raw / API_FOLDERS["uaf_sites"]
    for code, name in API["uaf_sites"].items():
        url = API["uaf_site"].format(code=code)
        try:
            html = http_get(url).text
        except Exception as e:
            DOWNLOAD_REPORT.append(dict(source="uaf_sites", file=code, url=url, status="failed",
                                        note=str(e)[:300]))
            continue
        (folder / code).mkdir(parents=True, exist_ok=True)
        (folder / code / "site_page.html").write_text(html, encoding="utf-8")
        links = {urljoin(url, h) for h in re.findall(r'href=["\']([^"\']+)["\']', html)
                 if h.lower().split("?")[0].endswith(TABLE_EXT + (".zip",))}
        if not links:
            DOWNLOAD_REPORT.append(dict(source="uaf_sites", file=code, url=url, status="failed",
                                        note="no data links in the page"))
        for u in sorted(links):
            fn = re.sub(r"[^\w.-]", "_", u.split("/")[-1].split("?")[0])
            fetch(u, folder / code / fn, "uaf_sites")


def download_neon(raw: Path) -> None:
    """
    NEON soil temperature (DP1.00041.001) at BARR, TOOL and HEAL with neonutilities:
    30-minute means with finalQF = 0, reduced to the daily mean per plot and depth (as the
    other temperature sources, which are daily means),
    plus the sensor positions (depths and position changes) and the product's change log
    (frost-heave entries). The reader applies the heave rules.
    """
    import os
    folder = raw / API_FOLDERS["neon"]
    folder.mkdir(parents=True, exist_ok=True)
    # NEON asks for a personal API token (free, from your neonscience.org account page).
    # The script never stores it: set it for the session, e.g. in Windows
    #   set NEON_TOKEN=<your token>     and on Linux/macOS   export NEON_TOKEN=<your token>
    token = os.environ.get("NEON_TOKEN", "")
    if not token:
        DOWNLOAD_REPORT.append(dict(source="neon", file=str(folder), url="neonutilities",
                                    status="get by hand",
                                    note="NEON needs your personal API token: create one in your "
                                         "neonscience.org account and set NEON_TOKEN before the run"))
        return
    import neonutilities as nu                     # pip install neonutilities
    prod = http_get(f"https://data.neonscience.org/api/v0/products/{API['neon_product']}",
                    headers={"X-API-Token": token}).json()["data"]
    (folder / "product.json").write_text(json.dumps(prod))
    for site in API["neon_sites"]:
        cache = folder / f"{site}_daily_mean.csv"
        if cache.exists():
            DOWNLOAD_REPORT.append(dict(source="neon", file=str(cache), url="neonutilities",
                                        status="already there", note=""))
            continue
        try:
            d = nu.load_by_product(dpid=API["neon_product"], site=site, package="basic",
                                   timeindex="30", check_size=False, include_provisional=True,
                                   progress=False, token=token)
        except Exception as e:
            DOWNLOAD_REPORT.append(dict(source="neon", file=str(cache), url="neonutilities",
                                        status="failed", note=str(e)[:300]))
            continue
        st = d["ST_30_minute"]
        st = st[st["finalQF"] == 0].copy()
        st["date"] = pd.to_datetime(st["startDateTime"]).dt.tz_localize(None).dt.normalize()
        (st.groupby(["horizontalPosition", "verticalPosition", "date"])["soilTempMean"].mean()
         .reset_index().to_csv(cache, index=False))
        d["sensor_positions_00041"].to_csv(folder / f"{site}_sensor_positions.csv", index=False)
        DOWNLOAD_REPORT.append(dict(source="neon", file=str(cache), url="neonutilities",
                                    status="downloaded", note=""))


def download_syrdakh(raw: Path) -> None:
    """Syrdakh observatory, Yakutia (Zenodo record): every file, zips unpacked."""
    folder = raw / API_FOLDERS["syrdakh"]
    rec = _json_cached(f"https://zenodo.org/api/records/{API['syrdakh_zenodo']}", folder / "record.json")
    for f in rec.get("files", []):
        p = fetch(f["links"].get("self") or f["links"].get("content"), folder / f["key"], "syrdakh")
        if p is not None and p.suffix.lower() == ".zip" and not (folder / p.stem).exists():
            with zipfile.ZipFile(p) as zf:
                zf.extractall(folder / p.stem)


def download_g02189(raw: Path) -> None:
    """NSIDC G02189 (Russian soil temperatures to 2000): every table in the HTTPS listing."""
    folder, root = raw / API_FOLDERS["g02189"], API["g02189"]

    def walk(url, depth=0):
        html = http_get(url).text
        for h in re.findall(r'href="([^"?#]+)"', html):
            if h.startswith(("/", "..", "http")) and not h.startswith(url):
                continue
            u = urljoin(url, h)
            if h.endswith("/") and depth < 2:
                walk(u, depth + 1)
            elif u.lower().endswith(TABLE_EXT + (".pdf", ".zip")):
                fetch(u, folder / u[len(root):], "g02189")
    walk(root)


def download_metno(raw: Path) -> None:
    """MET Norway permafrost stations (THREDDS, OPeNDAP; pip install xarray netCDF4):
    daily mean per depth (as the other temperature sources), with the station position
    and name."""
    import xarray as xr
    folder = raw / API_FOLDERS["metno"]
    folder.mkdir(parents=True, exist_ok=True)
    cat = http_get(f"{API['metno_thredds']}/catalog/{API['metno_catalog']}/catalog.xml").text
    (folder / "catalog.xml").write_text(cat)
    for st in sorted(set(re.findall(r'xlink:href="([^"]+)/catalog\.xml"', cat))):
        name = st.strip("/").split("/")[-1]
        cache = folder / f"{name}_daily_mean.csv"
        url = f"{API['metno_thredds']}/dodsC/{API['metno_catalog']}/{name}/{name}-aggregated.ncml"
        if cache.exists():
            DOWNLOAD_REPORT.append(dict(source="metno", file=str(cache), url=url,
                                        status="already there", note=""))
            continue
        try:
            ds = xr.open_dataset(url)
            var = next(v for v in ds.data_vars
                       if "soil_temperature" in str(ds[v].attrs.get("standard_name", "")))
            df = ds[var].to_dataframe().reset_index()
            dcol = next(c for c in df.columns if "depth" in c.lower())
            df = df.rename(columns={var: "temp_c", dcol: "depth_m"})
            df["date"] = pd.to_datetime(df["time"]).dt.normalize()
            (df.groupby(["depth_m", "date"]).temp_c.mean().reset_index()
             .assign(lat=float(ds.attrs.get("geospatial_lat_min", np.nan)),
                     lon=float(ds.attrs.get("geospatial_lon_min", np.nan)),
                     title=ds.attrs.get("title", name))
             .to_csv(cache, index=False))
            DOWNLOAD_REPORT.append(dict(source="metno", file=str(cache), url=url,
                                        status="downloaded", note=""))
        except Exception as e:
            DOWNLOAD_REPORT.append(dict(source="metno", file=str(cache), url=url, status="failed",
                                        note=f"{type(e).__name__}: {e}"[:300]))


def download_frost(raw: Path) -> None:
    """
    MET Norway Frost API (frost.met.no): every soil and ground temperature series of the
    MET stations on permafrost, as a daily mean per depth (as the other temperature
    sources), with the station position and name. "On permafrost": in Svalbard (north of
    74 N) or with a sensor deeper than 2 m (the permafrost boreholes; weather stations
    measure soil temperature to 1-2 m and on the mainland stand on seasonal frost only).
    The MET permafrost stations on THREDDS (download_metno) come from the same database;
    the reader keeps one copy of a station.
    Needs a free Frost client ID (frost.met.no/auth/requestCredentials.html) for the
    session, never stored:  PowerShell  $env:FROST_CLIENT_ID="<client ID>"
    """
    import base64
    import os
    folder = raw / API_FOLDERS["metno_frost"]
    folder.mkdir(parents=True, exist_ok=True)
    cid = os.environ.get("FROST_CLIENT_ID", "")
    if not cid:
        DOWNLOAD_REPORT.append(dict(source="metno_frost", file=str(folder), url=API["frost"],
                                    status="get by hand",
                                    note="Frost needs your client ID (free, frost.met.no/auth/"
                                         "requestCredentials.html): set FROST_CLIENT_ID before the run"))
        return
    auth = {"Authorization": "Basic " + base64.b64encode(f"{cid}:".encode()).decode()}

    def frost(path, **params):
        """One Frost request; None when Frost has no data for it (404 / 412)."""
        r = _http().get(f"{API['frost']}/{path}", params=params, headers=auth, timeout=300)
        for k in range(4):
            if r.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(2 ** (k + 1))
            r = _http().get(f"{API['frost']}/{path}", params=params, headers=auth, timeout=300)
        if r.status_code in (404, 412):
            return None
        r.raise_for_status()
        return r.json().get("data", [])

    # 1. every soil / ground temperature series Frost holds, and where its station is
    elements = frost("elements/v0.jsonld", fields="id,name,unit", lang="en-US") or []
    wanted = sorted(e["id"] for e in elements
                    if re.search(r"(soil|ground)_temperature", e["id"]) and "P1M" not in e["id"]
                    and not re.match(r"(min|max)\(", e["id"]))
    series = []
    for el in wanted:
        for s in frost("observations/availableTimeSeries/v0.jsonld", elements=el) or []:
            lev = s.get("level") or {}
            series.append(dict(source_id=s["sourceId"].split(":")[0], element=s["elementId"],
                               resolution=s.get("timeResolution", ""), valid_from=s.get("validFrom", ""),
                               valid_to=s.get("validTo", ""), level_type=lev.get("levelType", ""),
                               level=lev.get("value", np.nan), level_unit=lev.get("unit", "")))
    series = pd.DataFrame(series)
    if series.empty:
        DOWNLOAD_REPORT.append(dict(source="metno_frost", file=str(folder), url=API["frost"],
                                    status="failed", note=f"no soil temperature series (elements {wanted})"))
        return
    to_m = {"m": 1.0, "cm": 0.01, "mm": 0.001}
    series["depth_m"] = series.level.astype(float).abs() * series.level_unit.map(to_m)
    ids = sorted(series.source_id.unique())
    st = []
    for i in range(0, len(ids), 50):
        for s in frost("sources/v0.jsonld", ids=",".join(ids[i:i + 50]), fields="id,name,geometry,masl") or []:
            lon, lat = (s.get("geometry") or {}).get("coordinates", [np.nan, np.nan])[:2]
            st.append(dict(source_id=s["id"], name=s.get("name", ""), lat=lat, lon=lon, masl=s.get("masl", np.nan)))
    st = pd.DataFrame(st)
    series = series.merge(st, on="source_id", how="left")
    series.to_csv(folder / "available_series.csv", index=False)       # what Frost holds (all stations)
    deep = series.groupby("source_id").depth_m.max()
    keep = st[(st.lat >= 74) | st.source_id.map(deep).gt(2.0)].source_id
    log(f"  Frost: {series.source_id.nunique()} stations with soil temperature, {len(keep)} on permafrost")

    # 2. per station: daily mean per depth, one request per series and year (Frost serves
    #    at most 100,000 values per request; a year of 10-minute values is 52,560)
    for sid in keep:
        cache = folder / f"{sid}_daily_mean.csv"
        if cache.exists():
            DOWNLOAD_REPORT.append(dict(source="metno_frost", file=str(cache), url=API["frost"],
                                        status="already there", note=""))
            continue
        ss = series[(series.source_id == sid) & series.depth_m.notna()]
        # per depth: a daily-mean element where Frost has one, else the sub-daily values
        ss = ss.assign(daily=ss.element.str.contains("P1D"))
        ss = ss[ss.daily == ss.groupby("depth_m").daily.transform("max")]
        rows = []
        try:
            for s in ss.drop_duplicates(["element", "level", "level_unit"]).itertuples():
                y0 = pd.Timestamp(s.valid_from).year if s.valid_from else 1950
                y1 = pd.Timestamp(s.valid_to).year if s.valid_to else pd.Timestamp.now().year
                lev = f"{float(s.level):g}"                  # 10.0 -> "10"
                for y in range(y0, y1 + 1):
                    obs = frost("observations/v0.jsonld", sources=sid, elements=s.element,
                                levels=lev, referencetime=f"{y}-01-01/{y + 1}-01-01",
                                fields="referenceTime,value,level,qualityCode") or []
                    for o in obs:
                        for v in o.get("observations", []):
                            rows.append((o["referenceTime"][:10], s.depth_m, v.get("value"),
                                         v.get("qualityCode", np.nan)))
                    time.sleep(0.2)                      # be kind to the server
        except Exception as e:
            DOWNLOAD_REPORT.append(dict(source="metno_frost", file=str(cache), url=API["frost"],
                                        status="failed", note=f"{type(e).__name__}: {e}"[:300]))
            continue
        if not rows:
            DOWNLOAD_REPORT.append(dict(source="metno_frost", file=str(cache), url=API["frost"],
                                        status="failed", note="no observations"))
            continue
        d = pd.DataFrame(rows, columns=["date", "depth_m", "temp_c", "quality"])
        d["temp_c"] = pd.to_numeric(d.temp_c, errors="coerce")
        info = st.set_index("source_id").loc[sid]
        (d.groupby(["depth_m", "date"]).agg(temp_c=("temp_c", "mean"), n=("temp_c", "size"),
                                           quality_max=("quality", "max")).reset_index()
         .assign(lat=info.lat, lon=info.lon, title=f"{info['name']} ({sid})", station_id=sid,
                 element=",".join(sorted(ss.element.unique())))
         .to_csv(cache, index=False))
        DOWNLOAD_REPORT.append(dict(source="metno_frost", file=str(cache), url=API["frost"],
                                    status="downloaded", note=f"{d.depth_m.nunique()} depths"))


def download_uio_boreholes(raw: Path) -> None:
    """Unpack 'Data Storage Paper.7z' (fetched from SOURCE_FILES) with py7zr, or with the
    tar.exe of Windows 10/11 (bsdtar reads 7z), when it is not unpacked yet."""
    base = raw / API_FOLDERS["uio_boreholes"]
    arc = base / "Data Storage Paper.7z"
    dest = base / "unpacked"
    if not arc.exists() or any(dest.rglob("*.dat")):
        return
    dest.mkdir(parents=True, exist_ok=True)
    try:
        import py7zr
        with py7zr.SevenZipFile(arc) as z:
            z.extractall(dest)
    except ImportError:
        tar = shutil.which("tar")
        if not tar:
            DOWNLOAD_REPORT.append(dict(source="uio_boreholes", file=str(arc), url="", status="get by hand",
                                        note=f"unpack {arc.name} into {dest} (7-Zip, or pip install py7zr)"))
            return
        subprocess.run([tar, "-xf", str(arc), "-C", str(dest)], check=True)
    log(f"  unpacked {arc.name}: {len(list(dest.rglob('*.dat')))} .dat files")


DOWNLOADERS = {
    "calm_summary": download_calm_table,
    "gtnp_platform": download_gtnp_platform,
    "tsp_alaska": download_tsp_alaska,
    "uaf_sites": download_uaf_sites,
    "neon": download_neon,
    "syrdakh": download_syrdakh,
    "g02189": download_g02189,
    "metno": download_metno,
    "metno_frost": download_frost,
    "uio_boreholes": download_uio_boreholes,
}


def download_all(raw: Path, out: Path, only: list[str] | None = None) -> None:
    log("== STAGE 1: download")
    download_listed(raw, only)
    for name, fn in DOWNLOADERS.items():
        if only and name not in only:
            continue
        log(f"[{name}]")
        try:
            fn(raw)
        except Exception as e:                      # one source failing must not stop the others
            DOWNLOAD_REPORT.append(dict(source=name, file="", url="", status="failed",
                                        note=f"{type(e).__name__}: {e}"[:300]))
    rep = pd.DataFrame(DOWNLOAD_REPORT, columns=["source", "file", "url", "status", "note"])
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out / "download_report.csv", index=False)
    log("  " + ", ".join(f"{k} {v}" for k, v in rep.status.value_counts().items()))
    for r in rep[rep.status == "failed"].itertuples():
        log(f"  FAILED {r.source}: {r.file} ({r.note})")
    hand = rep[rep.status == "get by hand"]
    if len(hand):
        log("  Files to put in the raw folder by hand (login or data request):")
        for r in hand.drop_duplicates("note").itertuples():
            log(f"    {r.source}: {r.note}")


# =============================================================================
# STAGE 2: READ (one reader per source)
# =============================================================================
# =============================================================================
# CALM network tables
# =============================================================================
def read_calm_summary(raw: Path) -> pd.DataFrame:
    """
    CALM 'north' summary table (CALM_Summary_table.xls from www2.gwu.edu/~calm).
    Blocks of sites per region; each block has a header row with LAT, LONG and one
    column per year. Values are site averages of end-of-season thaw depth (cm);
    '-' missing, '>x' censored (thaw deeper than the probe/sensor), 'x*' footnoted
    (CALM: not a regular end-of-season value). Method column: number of points / P,
    TT (thaw tube), T/B (temperature in borehole).
    """
    found = files(raw, "CALM_Summary_table*.xls*")
    if not found:
        return empty_records()
    path = found[-1]
    sheet = pd.read_excel(path, header=None, sheet_name=0)
    rows = [[clean_cell(v) for v in r] for r in sheet.itertuples(index=False)]
    recs, year_of_col, country = [], None, ""
    country_words = {"Alaska": "USA", "Canada": "Canada", "Greenland": "Greenland",
                     "Svalbard": "Norway (Svalbard)", "Norway": "Norway", "Sweden": "Sweden",
                     "Russia": "Russia", "Mongolia": "Mongolia", "Kazakhstan": "Kazakhstan",
                     "China": "China", "Switzerland": "Switzerland", "Finland": "Finland",
                     "Iceland": "Iceland", "Denmark": "Greenland"}
    for row in rows:
        first = row[0] if row else ""
        nonempty = [c for c in row if c]
        if len(nonempty) == 1 and first and not re.match(r"^[A-Z]{1,2}\d", first):
            for w, cty in country_words.items():          # region heading rows
                if w.lower() in first.lower():
                    country = cty
            continue
        if any(c.upper() == "LAT" for c in row):
            yrs = [(j, int(as_float(c))) for j, c in enumerate(row)
                   if 1900 <= as_float(c) <= 2100]
            year_of_col = dict(yrs) if yrs else None
            continue
        if year_of_col is None or len(row) < 5:
            continue
        code, name, lat, lon = row[0], row[1], as_float(row[2]), as_float(row[3])
        if not code or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue
        m = row[4].upper()
        if "TT" in m:
            method = "thaw_tube"
        elif re.match(r"^\d", m) or "P" in m:
            method = "probe"
        elif "B" in m or m.startswith("T"):
            method = "borehole_temp"
        else:
            method = "probe"
        cc = norm_calm_code(code)
        cap = CONFIG["sensor_cap_cm"].get(cc)
        for j, yr in year_of_col.items():
            if j >= len(row):
                continue
            s = row[j]
            if s in ("", "-", "*") or "inactive" in s.lower():
                continue
            v, cens = parse_censored(s.replace("*", ""))
            if np.isnan(v):
                continue
            flags = []
            if "*" in s:
                flags.append("calm_footnote")
            if cap is not None and v >= cap - 0.5:
                cens = True
                flags.append("sensor_cap")
            if yr >= CONFIG["talik_from"].get(cc, 9999):
                flags.append("talik")
            recs.append(dict(source_site_id="CALM_" + cc, site_name=name, calm_code=cc,
                             lat=lat, lon=lon, country=country, method=method, year=yr,
                             alt_cm=v, censored=cens, flag=";".join(flags),
                             source_file=path.name))
    out = pd.DataFrame(recs)
    return finish(out, "calm_summary") if len(out) else empty_records()


def read_pangaea_calm(raw: Path) -> pd.DataFrame:
    """
    GTN-P CALM compilation on PANGAEA (Streletskiy et al. 2025, doi:10.1594/PANGAEA.972777):
    263 CALM site series, 1990-2024, one value per site and year (ALD, cm). Same sites as
    the CALM summary table (deduplicated by CALM code), but more recent years for some.
    Method per site is in the event comment ('Method: ...').
    """
    found = files(raw, "PANGAEA_972777*.tab")
    if not found:
        return empty_records()
    head, d = read_pangaea_tab(found[-1])
    ev = pangaea_events(head)
    alt_col = next(c for c in d.columns if c.startswith("ALD"))
    d = d.rename(columns={alt_col: "ald", "Date/Time": "year"})

    def method_of(event):
        c = ev.get(event, {}).get("comment", "").lower()
        if "thaw tube" in c or "frost tube" in c:
            return "thaw_tube"
        if "temperature" in c or "borehole" in c or "thermistor" in c:
            return "borehole_temp"
        return "probe"

    vals = d.ald.map(parse_censored)
    out = pd.DataFrame(dict(
        source_site_id="CALM_" + d.Site.map(norm_calm_code), calm_code=d.Site,
        site_name=d.Name, lat=d.Latitude, lon=d.Longitude, country=d.Country,
        method=d.Event.map(method_of), year=pd.to_numeric(d.year.astype(str).str[:4], errors="coerce"),
        alt_cm=[v for v, _ in vals], censored=[c for _, c in vals],
        source_file=found[-1].name))
    cc = out.calm_code.map(norm_calm_code)
    talik = out.year >= cc.map(CONFIG["talik_from"]).fillna(9999)
    out["flag"] = add_flag(pd.Series("", index=out.index), talik, "talik")
    cap = cc.map(CONFIG["sensor_cap_cm"])
    capped = cap.notna() & (out.alt_cm >= cap - 0.5)
    out.loc[capped, "censored"] = True
    out["flag"] = add_flag(out.flag, capped, "sensor_cap")
    # the per-value 'Sample comment' marks surveys that are not end-of-season values
    note = d["Sample comment"].fillna("").str.lower() if "Sample comment" in d else pd.Series("", index=d.index)
    early = note.str.contains(r"early|end of july|mid-july|august ?1\b")
    out["flag"] = add_flag(out.flag, early, "early_measurement")
    not_reached = note.str.contains("not reached")
    out.loc[not_reached, "censored"] = True
    out["flag"] = add_flag(out.flag, note.str.contains("burn"), "burned")
    out["flag"] = add_flag(out.flag, note.str.contains("small|limited sample"), "small_sample")
    return finish(out, "pangaea_calm")


def read_gtnp_grids(raw: Path) -> pd.DataFrame:
    """
    GTN-P active-layer exports (gtnpdatabase.org): one CSV per grid/transect with one row
    per node (offset_x, offset_y) and one column per probing date; -999/-888 = missing;
    a matching '<stem>.metadata.txt' holds coordinates and the CALM 'Ancillary Code'.
    Survey mean over nodes, then the end-of-season value per year (end_of_season()).
    Exports of the old gtnpdatabase.org (kept by the corridor project); the GTN-P data
    platform serves the same grids now (read_gtnp_platform_alt).
    """
    recs = []
    for path in files(raw, "Activelayer_*timeserie*.csv"):
        stem = path.name.split("-Dataset_")[0]
        meta = next(path.parent.glob(f"{stem}.metadata.txt"), None)
        if meta is None:
            log(f"    gtnp: SKIP {path.name} (no metadata file with coordinates)")
            continue
        txt = meta.read_text(encoding="utf-8", errors="replace")
        get = lambda k: (re.search(rf"^{k}\s*:\s*(.+)$", txt, re.M) or [None, ""])[1].strip()
        lat, lon = as_float(get("Latitude")), as_float(get("Longitude"))
        code, name, cty = norm_calm_code(get("Ancillary Code")), get("Name"), get("Country")
        if "Ground_Subsidence" in path.name:
            continue                                   # surface height, not thaw depth
        method = "thaw_tube" if ("Thaw_Tube" in path.name or "Frost_Tube" in path.name) else "probe"
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            rows = list(csv.reader(fh))
        head = rows[0]
        dcols = [j for j, c in enumerate(head) if re.match(r"\d{4}-\d{2}-\d{2}", c)]
        if not dcols:
            continue
        K, j0 = len(dcols), dcols[0]
        vals = []
        for r in rows[1:]:
            v = np.array([as_float(x) for x in r[j0:]], float)
            if len(v) == K:
                vals.append(v)
            elif len(v) and len(v) % K == 0:
                # several nodes with the same offsets on one line, interleaved per date
                # (value j belongs to date j // m), as in some GTN-P exports
                vals.extend(v.reshape(K, -1).T)
        if not vals:
            log(f"    gtnp: SKIP {path.name} (rows do not match the date header)")
            continue
        vals = np.array(vals)
        vals[(vals < 0) | (vals > 900)] = np.nan       # -999/-888 missing codes
        dates = pd.to_datetime([head[j][:10] for j in dcols])
        n = np.isfinite(vals).sum(0)
        n_max = pd.Series(n).groupby(dates.year.values).transform("max").to_numpy()
        keep = (n > 0) & (n >= CONFIG["min_node_fraction"] * n_max)
        sid = "GTNP_" + re.sub(r"^Activelayer_", "", stem)
        for k, d in enumerate(dates):
            if keep[k]:
                recs.append(dict(source_site_id=sid, site_name=name, calm_code=code, lat=lat,
                                 lon=lon, country=cty, method=method, obs_date=d,
                                 alt_cm=np.nanmean(vals[:, k]),
                                 alt_sd_cm=np.nanstd(vals[:, k], ddof=1) if n[k] > 1 else np.nan,
                                 n_points=int(n[k]), censored=False, flag="",
                                 source_file=path.name))
    if not recs:
        return empty_records()
    out = end_of_season(pd.DataFrame(recs), "source_site_id")
    return finish(out, "gtnp")


# =============================================================================
# Canada
# =============================================================================
def _gsc_id(s: str) -> str:
    """'90-TT-01' and '90TT01' -> '90TT01'."""
    return re.sub(r"[\s\-]", "", str(s)).upper()


def _of9340_text(raw: Path):
    """Site table (Table 1) and Appendix B rows of GSC Open File 9340, or None."""
    z = files(raw, "of_9340e.zip")
    if not z:
        return None
    try:
        import pdfplumber
    except ImportError as e:
        # without it the GSC thaw tubes 2006-2022 and all GSC cables are lost, so the
        # reader fails (and the run ends with an error) instead of skipping quietly
        raise ImportError("pdfplumber is needed to read GSC Open File 9340 "
                          "(pip install pdfplumber)") from e
    with zipfile.ZipFile(z[-1]) as zf:
        pdf_name = next(n for n in zf.namelist() if n.lower().endswith(".pdf"))
        pdf = pdfplumber.open(io.BytesIO(zf.read(pdf_name)))
    sites, alt_rows = {}, []
    site_re = re.compile(r"(\S+)\s+(?:<?\d+(?:\.\d+)?\s+)?(\d{2}\.\d{3,})\s+(\d{2,3}\.\d{3,})\s+"
                         r"(Continuous|Discontinuous|Extensive|Sporadic|Isolated)")
    for page in pdf.pages:
        txt = page.extract_text() or ""
        if "Table 1" in txt[:400]:
            for line in txt.split("\n"):
                m = site_re.search(line)
                if m:
                    sites[_gsc_id(m.group(1))] = (float(m.group(2)), -float(m.group(3)))
        if "APPENDIX B" in txt[:300]:
            words = page.extract_words()
            years = [(w["x1"], int(w["text"])) for w in words if re.fullmatch(r"(19|20)\d\d", w["text"])]
            hdr_top = min((w["top"] for w in words if re.fullmatch(r"(19|20)\d\d", w["text"])), default=0)
            lines = {}
            for w in words:
                if w["top"] > hdr_top + 5:
                    lines.setdefault(round(w["top"]), []).append(w)
            for top, ws in sorted(lines.items()):
                ws = sorted(ws, key=lambda w: w["x0"])
                sid = ws[0]["text"]
                if not re.fullmatch(r"\d\dTT\w+|\d\dTC\w+", sid):
                    continue
                name = " ".join(w["text"] for w in ws[1:] if not re.fullmatch(r"[<>]?\d+(\.\d+)?", w["text"]))
                for w in ws[1:]:
                    if re.fullmatch(r"[<>]?\d+(\.\d+)?", w["text"]) and w["x0"] > 190:
                        x1, yr = min(years, key=lambda y: abs(y[0] - w["x1"]))
                        if abs(x1 - w["x1"]) < 12:
                            alt_rows.append((_gsc_id(sid), name, yr, w["text"]))
    return sites, alt_rows, z[-1].name


def read_gsc_thaw_tubes(raw: Path) -> pd.DataFrame:
    """
    Geological Survey of Canada thaw-tube network, Mackenzie Valley and Delta (Nixon,
    Smith et al.). Thaw tubes record the maximum thaw of the season (glass marker trapped
    at freeze-back); the table value is ALT (cm) for the thaw season.
      * GSC Open File 6287 (Smith et al. 2009): 'Active Layer.txt' 1991-2007, site
        coordinates in 'Metadata.txt'.
      * GSC Open File 9340 (2006-2022): Appendix B of the report (PDF table), site
        coordinates in Table 1.
    '<x' and '>x' (thaw below the tube, or tube damaged) are kept as censored.
    The years 2006-2007 are in both files; OF 9340 is the later, reviewed value.
    """
    recs = []
    z = files(raw, "of_6287.zip")
    meta_coords = {}
    if z:
        with zipfile.ZipFile(z[-1]) as zf:
            names = zf.namelist()
            alt_txt = zf.read(next(n for n in names if n.endswith("Active Layer.txt"))).decode("latin-1")
            meta_txt = zf.read(next(n for n in names if n.endswith("Metadata.txt"))).decode("latin-1")
        mrows = [r.split("\t") for r in re.split(r"[\r\n]+", meta_txt) if r.strip()]
        mh = mrows[0]
        for r in mrows[1:]:
            r = dict(zip(mh, r))
            meta_coords[_gsc_id(r["Site"])] = (as_float(r.get("Lat")), as_float(r.get("Long")),
                                               r.get("Site Name", ""), r.get("Comments", ""))
        arows = [r.split("\t") for r in re.split(r"[\r\n]+", alt_txt) if r.strip()]
        ah = arows[0]
        for r in arows[1:]:
            sid = _gsc_id(r[0])
            for j, c in enumerate(ah):
                if re.fullmatch(r"(19|20)\d\d", c.strip()) and j < len(r) and r[j].strip():
                    v, cens = parse_censored(r[j])
                    recs.append(dict(sid=sid, name=r[1], year=int(c), alt_cm=v, censored=cens,
                                     source_file=f"{z[-1].name}/Active Layer.txt", order=1))
    of = _of9340_text(raw)
    coords9340 = {}
    if of:
        coords9340, alt_rows, fname = of
        for sid, name, yr, s in alt_rows:
            v, cens = parse_censored(s)
            recs.append(dict(sid=sid, name=name, year=yr, alt_cm=v, censored=cens,
                             source_file=f"{fname}/Appendix B", order=2))
    if not recs:
        return empty_records()
    d = pd.DataFrame(recs)
    # one value per site and year: the later report wins where both have it
    d = d.sort_values("order").groupby(["sid", "year"], as_index=False).tail(1)

    def pos(sid):
        if sid in coords9340:
            return coords9340[sid]
        m = meta_coords.get(sid)
        return (m[0], m[1]) if m else (np.nan, np.nan)

    p = d.sid.map(pos)
    d["lat"], d["lon"] = [a for a, _ in p], [b for _, b in p]
    names = d.groupby("sid").name.transform(lambda x: max(x, key=len))
    calm = d.sid.map(lambda s: re.search(r"CALM Site (C-?\d+\w?)", meta_coords.get(s, ("", "", "", ""))[3]))
    out = pd.DataFrame(dict(source_site_id="GSC_" + d.sid, site_name=names, lat=d.lat,
                            lon=d.lon, country="Canada", method="thaw_tube", year=d.year,
                            alt_cm=d.alt_cm, censored=d.censored, source_file=d.source_file,
                            site_note=calm.map(lambda m: f"thaw tube on CALM site {m.group(1)}" if m else "")))
    out["flag"] = add_flag(pd.Series("", index=out.index), out.censored, "lower_bound")
    nopos = out.lat.isna()
    if nopos.any():
        log(f"    gsc: {out[nopos].source_site_id.nunique()} tubes without coordinates skipped: "
            f"{sorted(out[nopos].source_site_id.unique())[:10]}")
    return finish(out, "gsc_thaw_tube")


def read_gsc_cables(raw: Path) -> pd.DataFrame:
    """
    GSC Open File 9340, temperature cables (folder CAB): daily ground temperatures at
    several depths ('0.53 m', ...) for ~90 boreholes, 2006-2022. ALT = 0 C crossing of the
    annual maximum-temperature profile (alt_from_temperature_profile).
    Coordinates from Table 1 of the report.
    """
    z = files(raw, "of_9340e.zip")
    if not z:
        return empty_records()
    of = _of9340_text(raw)
    coords = of[0] if of else {}
    parts = []
    with zipfile.ZipFile(z[-1]) as zf:
        for n in zf.namelist():
            if "/CAB/" not in n or not n.endswith(".csv"):
                continue
            d = pd.read_csv(io.BytesIO(zf.read(n)))
            dcols = [c for c in d.columns if re.fullmatch(r"-?\d+(\.\d+)?\s*m", c.strip())]
            if len(dcols) < 2:
                continue
            long = d.melt(id_vars=["SITE", "DATE"], value_vars=dcols, var_name="depth", value_name="temp_c")
            long["depth_cm"] = long.depth.str.replace("m", "").astype(float) * 100
            long["source_site_id"] = "GSCcable_" + _gsc_id(Path(n).stem)
            long["time"] = pd.to_datetime(long.DATE, errors="coerce")
            long["sid"] = _gsc_id(Path(n).stem)
            long["source_file"] = f"{z[-1].name}/{n}"
            parts.append(long[["source_site_id", "sid", "time", "depth_cm", "temp_c", "source_file"]])
    if not parts:
        return empty_records()
    t = pd.concat(parts, ignore_index=True)
    t["temp_c"] = pd.to_numeric(t.temp_c, errors="coerce")
    sites = t.groupby("source_site_id").agg(sid=("sid", "first"), source_file=("source_file", "first")).reset_index()
    sites["lat"] = sites.sid.map(lambda s: coords.get(s, (np.nan, np.nan))[0])
    sites["lon"] = sites.sid.map(lambda s: coords.get(s, (np.nan, np.nan))[1])
    sites["site_name"], sites["country"] = sites.sid, "Canada"
    out = records_from_profiles(t, sites.drop(columns="sid"), "borehole_temp")
    return finish(out, "gsc_cable")


# =============================================================================
# Alaska
# =============================================================================
def _read_dbf(data: bytes) -> pd.DataFrame:
    """Minimal dBase III reader (the attribute table of a shapefile)."""
    n, hl, rl = struct.unpack("<IHH", data[4:12])
    fields, o = [], 32
    while data[o] != 0x0D:
        fields.append((data[o:o + 11].split(b"\0")[0].decode(), data[o + 16]))
        o += 32
    rows = []
    for i in range(n):
        r, p, row = data[hl + i * rl: hl + (i + 1) * rl], 1, {}
        for name, ln in fields:
            row[name] = r[p:p + ln].decode("latin-1").strip()
            p += ln
        rows.append(row)
    return pd.DataFrame(rows)


def read_usgs_npra(raw: Path) -> pd.DataFrame:
    """
    USGS DOI/GTN-P climate and active-layer stations, National Petroleum Reserve-Alaska
    and Arctic National Wildlife Refuge (Urban 2021, doi:10.5066/F7VX0FGB), 1998-2019.
    One text file per station; ground temperatures Tg5cm ... Tg120cm (hourly or 2-hourly).
    ALT = 0 C crossing of the annual maximum-temperature profile. Coordinates from the
    station shapefile (AlaskaPoints_shapefile.zip).
    """
    shp = files(raw, "AlaskaPoints_shapefile.zip")
    if not shp:
        return empty_records()
    with zipfile.ZipFile(shp[-1]) as zf:
        st = _read_dbf(zf.read(next(n for n in zf.namelist()
                                    if n.endswith(".dbf") and "__MACOSX" not in n)))
    st["lat"], st["lon"] = st.Lat_dd.astype(float), st.Lon_dd.astype(float)
    parts, sites = [], []
    for r in st.itertuples():
        f = next((p for p in files(raw, f"{r.Station}.txt")), None)
        if f is None:
            continue
        d = pd.read_csv(f, sep=r"\s+", na_values=["NaN"], low_memory=False)
        tg = [c for c in d.columns if re.fullmatch(r"Tg\d+cm", c)]
        d["time"] = pd.to_datetime(dict(year=d.Year, month=d.Mon, day=d.Day), errors="coerce")
        long = d.melt(id_vars=["time"], value_vars=tg, var_name="c", value_name="temp_c").dropna()
        long["depth_cm"] = long.c.str.extract(r"(\d+)").astype(float)
        sid = "USGS_" + r.Station.replace(" ", "_")
        long["source_site_id"] = sid
        # daily means first: an hourly maximum is noisier than the daily envelope
        long = long.groupby(["source_site_id", "time", "depth_cm"], as_index=False).temp_c.mean()
        parts.append(long)
        sites.append(dict(source_site_id=sid, site_name=r.Station, lat=r.lat, lon=r.lon,
                          country="USA", source_file=f.name))
    if not parts:
        return empty_records()
    out = records_from_profiles(pd.concat(parts), pd.DataFrame(sites), "soil_temp")
    return finish(out, "usgs_npra")


# =============================================================================
# Russia
# =============================================================================
def read_ru_hms_alt(raw: Path) -> pd.DataFrame:
    """
    Time series of ALT in the Russian Arctic, 1930-1990 (Frauenfeld, Zhang et al.;
    NSIDC/ADC arcss160): annual maximum thaw depth at 31 hydrometeorological stations,
    from monthly soil temperatures (linear interpolation of the 0 C isotherm). Values in
    metres in files TMD2.<station id>.max; station list in the xls.
    """
    z = files(raw, "arcss160.zip")
    if not z:
        return empty_records()
    recs = []
    with zipfile.ZipFile(z[-1]) as zf:
        lst = pd.read_excel(io.BytesIO(zf.read(next(n for n in zf.namelist() if n.endswith(".xls")))))
        lst.columns = [str(c).strip() for c in lst.columns]
        info = {int(r.Station_ID): r for r in lst.itertuples()}
        for n in zf.namelist():
            m = re.match(r".*TMD2\.(\d+)\.max$", n)
            if not m:
                continue
            sid = int(m.group(1))
            r = info.get(sid)
            if r is None:
                continue
            for line in zf.read(n).decode("latin-1").split("\n"):
                p = line.split()
                if len(p) == 2 and as_float(p[1]) > 0:
                    recs.append(dict(source_site_id=f"RUHMS_{sid:03d}", site_key=f"RUHMS_{sid:03d}",
                                     site_name=str(r.Station_Name).strip(), lat=r.Latitude,
                                     lon=r.Longitude, country="Russia", method="soil_temp_monthly",
                                     year=int(as_float(p[0])), alt_cm=as_float(p[1]) * 100,
                                     censored=False, source_file=f"{z[-1].name}/{Path(n).name}"))
    return finish(pd.DataFrame(recs), "ru_hms_alt") if recs else empty_records()


# =============================================================================
# compilations (lowest priority; only sources not read from their originals)
# =============================================================================
CUSP_SKIP = {
    "CALM": "same as the CALM tables", "Streletskiy_2026": "CALM copy (and Alps/Mongolia)",
    "CALM_South": "Southern Hemisphere", "PERMOS_2024": "Alps", "Zhao_2021": "Tibetan Plateau",
    "Barrow_CALM_U1": "single nodes of CALM grid U1 (the grid mean is in CALM)",
    "Wang_2018": "ALT given only as 25/50/75 cm classes in CUSP; derived from the "
                 "original station temperatures instead",
    "Ruess_2025": "read from the original Bonanza Creek LTER file",
    "Chapin_2025": "read from the original Bonanza Creek LTER files",
    "Kling_2016": "read from the original Arctic LTER file",
    "Kling_2025": "read from the original Arctic LTER file",
    "Veremeeva_etal_2025": "read from the original ALLena file (PANGAEA 974408)",
}


def read_cusp(raw: Path) -> pd.DataFrame:
    """
    CUSP v1.2 (Zenodo/GitHub, 2026), an observation-level compilation of thaw-depth
    measurements from 60 sources. Used only for sources not read from their originals
    (CUSP_SKIP lists the others and why). GPR values are not used (different method).
    'LB' (lower bound) and pf_observed = 0 (no frozen ground reached) are censored.
    Observations are averaged per site and date, then the end-of-season rule applies.
    """
    z = files(raw, "CUSP_v*.zip")
    if not z:
        return empty_records()
    with zipfile.ZipFile(z[-1]) as zf:
        n = next(n for n in zf.namelist() if re.search(r"cusp_v[\d.]+\.csv$", n))
        d = pd.read_csv(io.BytesIO(zf.read(n)), low_memory=False)
    d = d[~d.source.isin(CUSP_SKIP) & (d.method != "gp")].copy()
    d["censored"] = d.quality_flags.fillna("").str.contains(r"\bLB\b") | (d.pf_observed == 0)
    d["alt_cm"] = d.thaw_depth.fillna(d.obs_limit)
    d = d[d.alt_cm.notna()]
    d["source_site_id"] = "CUSP_" + d.source + "_" + d.site_id.astype(str).str.replace(r"\s+", "_", regex=True)
    d["obs_date"] = pd.to_datetime(d.date, errors="coerce")
    d["method"] = d.method.map({"tp": "probe", "tt": "thaw_tube", "temp": "soil_temp"}).fillna("probe")
    d["site_name"], d["source_file"] = d.site_id.astype(str), Path(n).name
    d["site_note"] = "via CUSP compilation, original source " + d.source
    # country from position (CUSP has none): Alaska west of 141 W, Canada between 141 W
    # and 73 W (and Labrador/Nunavik south of 60.5 N), Greenland east of 60 W; the rest
    # (northern Baffin Bay, Eurasia) stays blank
    lon, lat = d.lon, d.lat
    d["country"] = np.select([lon < -141, lon < -73, (lon < -55) & (lat < 60.5),
                              (lon >= -60) & (lon < -10) & (lat > 59.5)],
                             ["USA", "Canada", "Canada", "Greenland"], "")
    s = survey_means(d[["source_site_id", "obs_date", "alt_cm", "censored", "site_name", "lat",
                        "lon", "country", "method", "site_note", "source_file"]])
    out = end_of_season(s)
    out["flag"] = add_flag(pd.Series("", index=out.index), out.censored, "lower_bound")
    return finish(out, "cusp")


# =============================================================================
# CALM sites with their own files (node-level grids, longer records)
# =============================================================================
def _calm_positions(raw) -> dict[str, tuple[float, float, str, str]]:
    """CALM code -> (lat, lon, name, country) from the PANGAEA CALM compilation."""
    found = files(raw, "PANGAEA_972777*.tab")
    if not found:
        return {}
    head, d = read_pangaea_tab(found[-1])
    d = d.drop_duplicates("Site")
    return {norm_calm_code(r.Site): (r.Latitude, r.Longitude, r.Name, r.Country)
            for r in d.itertuples()}


def _calm_record(code, pos, **kw):
    lat, lon, name, cty = pos.get(code, (np.nan, np.nan, "", ""))
    return dict(source_site_id="CALM_" + code, calm_code=code, site_name=name, lat=lat,
                lon=lon, country=cty, **kw)


def read_calm_site_files(raw) -> pd.DataFrame:
    """
    Site workbooks published on the CALM website for single CALM sites, which go back
    before 1990 (the CALM summary table and the PANGAEA compilation start in 1990):
      S1 Kapp Linne, Svalbard 1972-2011 ('summary' sheet, m); S2 Abisko 1978-2016 ('summary',
      m); P1 Calypsostranda, Svalbard 1986-2009 (point table, cm); G1 and G2 Zackenberg
      ZEROCALM 1996-2017 ('summary' / 'EoS ALT Statistic', cm); U2 Barrow 1962-2024
      (CRREL summary csv, cm).
    Same sites as the CALM tables (same CALM code), so they only add missing years.
    """
    pos = _calm_positions(raw)
    recs = []

    def from_summary(pattern, code, sheet, col_hint, scale, method="probe"):
        for f in files(raw, pattern):
            x = pd.read_excel(f, sheet_name=sheet, header=None)
            hdr = next(i for i in range(len(x)) if str(x.iat[i, 0]).strip().lower() == "year")
            cols = [str(c).lower() for c in x.iloc[hdr]]
            j = next(k for k, c in enumerate(cols) if col_hint in c)
            for r in x.iloc[hdr + 1:].itertuples(index=False):
                yr, v = as_float(r[0]), as_float(r[j])
                if 1900 < yr < 2100 and v == v:
                    recs.append(_calm_record(code, pos, method=method, year=int(yr),
                                             alt_cm=v * scale, censored=False, source_file=f.name))
            return

    from_summary("CALM_S1_KappLinne_alt_*.xls", "S1", "summary", "average", 100)
    from_summary("CALM_S2_Abisko_alt_*.xls", "S2", "summary", "mean", 100)
    from_summary("CALM_G1_ZEROCALM1_*.xls*", "G1", "summary", "mean", 1)
    from_summary("CALM_G2_ZEROCALM2_*.xls*", "G2", "EoS ALT Statistic", "mean", 1)
    for f in files(raw, "CALM_P1_Calypsostranda_alt_*.xls"):
        x = pd.read_excel(f, sheet_name=0, header=None)
        years = {j: int(as_float(v)) for j, v in enumerate(x.iloc[0]) if 1900 < as_float(v) < 2100}
        pts = x.iloc[1:]
        nmax = max(pd.to_numeric(pts[j], errors="coerce").notna().sum() for j in years)
        for j, yr in years.items():
            v = pd.to_numeric(pts[j], errors="coerce").dropna()
            if len(v) >= CONFIG["min_node_fraction"] * nmax:
                recs.append(_calm_record("P1", pos, method="probe", year=yr, alt_cm=v.mean(),
                                         alt_sd_cm=v.std(), n_points=len(v), censored=False,
                                         source_file=f.name))
    for f in files(raw, "U2_alt_1962_2024summary.csv"):
        d = pd.read_csv(f)
        for r in d.itertuples():
            recs.append(_calm_record("U2", pos, method="probe", year=int(r.year), alt_cm=r.Mean,
                                     alt_sd_cm=r._4, n_points=r._2, censored=False,
                                     source_file=f.name))
        break
    return finish(pd.DataFrame(recs), "calm_site_files") if recs else empty_records()


def read_samoylov_calm(raw) -> pd.DataFrame:
    """
    Samoylov Island CALM grid (CALM R51, Lena Delta; Boike et al., AWI): one file per year
    with one column per grid node ('Dal_rr_cc') and one row per probing date, 2002-2024
    (calm_lv1_V201908.zip for 2002-2018, PANGAEA files for later years). Survey mean
    over nodes, then the end-of-season rule.
    """
    pos = _calm_positions(raw)
    tables = []
    for z in files(raw, "calm_lv1_V*.zip") + files(raw, "PANGAEA_947069_Calm_lv1_*.zip"):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if re.search(r"Calm_lv1_\d{4}\.dat$", n):
                    tables.append((f"{z.name}/{Path(n).name}", pd.read_csv(io.BytesIO(zf.read(n)))))
    for f in files(raw, "PANGAEA_*_Calm_lv1_*.dat"):
        tables.append((f.name, pd.read_csv(f)))
    pts = []
    for name, d in tables:
        tcol = next(c for c in d.columns if c.upper().startswith("UTC") or "date" in c.lower())
        nodes = [c for c in d.columns if c.startswith("Dal_")]
        long = d.melt(id_vars=[tcol], value_vars=nodes, value_name="alt_cm").rename(columns={tcol: "obs_date"})
        long["alt_cm"] = pd.to_numeric(long.alt_cm, errors="coerce")
        long["source_file"] = name
        pts.append(long.dropna(subset=["alt_cm"]))
    if not pts:
        return empty_records()
    p = pd.concat(pts)
    p = p[p.alt_cm >= 0]
    lat, lon, nm, cty = pos.get("R51", (72.369775, 126.48063, "Samoylov", "Russia"))
    p = p.assign(source_site_id="CALM_R51", calm_code="R51", site_name=nm, lat=lat, lon=lon,
                 country="Russia", method="probe")
    s = survey_means(p.drop_duplicates(["obs_date", "variable"]))
    return finish(end_of_season(s), "samoylov_calm")


def read_parks_canada(raw) -> pd.DataFrame:
    """
    Parks Canada tundra active-layer monitoring (open.canada.ca): CALM-style 100 m grids in
    Auyuittuq (CALM C27 Owl River), Quttinirpaaq (C18), Sirmilik (C26 Borden Peninsula,
    C28 Bylot Island) and Ukkusiksalik (C25) national parks. Each grid cell has one or two
    readings; their mean is the cell value, then survey mean and end-of-season rule.
    Coordinates are those of the CALM sites. Ground-temperature stations at the same
    grids (2.5 cm to 1.65 m) give a temperature-derived ALT ('parks_canada' sensors).
    Thaw tubes are read but give at most 13 years.
    """
    pos = _calm_positions(raw)
    park_code = {"auyuittuq": "C27", "quttinirpaaq": "C18", "ukkusiksalik": "C25"}
    site_code = {"borden": "C26", "bylot": "C28"}
    pts, temps, sites = [], [], []

    def code_of(fname, site):
        park = fname.split("_")[0]
        if park == "sirmilik":
            return site_code.get(str(site).split()[0].lower())
        return park_code.get(park)

    for f in sorted(set(files(raw, "*calm-grid*.csv"))):
        d = pd.read_csv(f, skiprows=[1], encoding="latin-1")
        d.columns = [c.strip().lstrip("﻿").lstrip("ï»¿") for c in d.columns]
        v1 = next(c for c in d.columns if c.lower().startswith("first"))
        v2 = next(c for c in d.columns if c.lower().startswith("second"))
        d["alt_cm"] = d[[v1, v2]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        d["obs_date"] = pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day), errors="coerce")
        site = d["Site"] if "Site" in d else ""
        d["calm"] = [code_of(f.name, s) for s in (site if isinstance(site, pd.Series) else [""] * len(d))]
        for code, g in d.dropna(subset=["calm"]).groupby("calm"):
            g = g.drop_duplicates(["obs_date", "Grid Cell Number"])
            pts.append(pd.DataFrame(dict(**_calm_record(code, pos, method="probe", source_file=f.name),
                                         obs_date=g.obs_date.values, alt_cm=g.alt_cm.values)))
    out = []
    if pts:
        p = pd.concat(pts, ignore_index=True)
        out.append(end_of_season(survey_means(p)))
    for f in files(raw, "*temperature*.csv"):
        if not f.name.split("_")[0] in ("auyuittuq", "sirmilik", "quttinirpaaq"):
            continue
        d = pd.read_csv(f, skiprows=[1], encoding="latin-1", low_memory=False)
        d.columns = [c.strip().lstrip("﻿").lstrip("ï»¿") for c in d.columns]
        tcols = {c: as_float(re.search(r"-(\d+(?:\.\d+)?)\s*m below", c).group(1)) * 100
                 for c in d.columns if re.search(r"-\d+(?:\.\d+)?\s*m below", c)}
        if "Site" not in d:
            d["Site"] = ""
        d["time"] = pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day), errors="coerce")
        for site, g in d.groupby("Site"):
            code = code_of(f.name, site)
            if code is None:
                continue
            sid = f"PARKS_T_{code}"
            long = g.melt(id_vars=["time"], value_vars=list(tcols), var_name="c", value_name="temp_c")
            long["depth_cm"] = long.c.map(tcols)
            long["temp_c"] = pd.to_numeric(long.temp_c, errors="coerce")
            long["source_site_id"] = sid
            temps.append(long.groupby(["source_site_id", "time", "depth_cm"], as_index=False).temp_c.mean())
            lat, lon, nm, cty = pos.get(code, (np.nan, np.nan, "", ""))
            sites.append(dict(source_site_id=sid, site_name=f"{nm} (temperature station)",
                              lat=lat, lon=lon, country="Canada", source_file=f.name,
                              site_note=f"ground temperature station at CALM grid {code}"))
    if temps:
        t = pd.concat(temps)
        out.append(records_from_profiles(t, pd.DataFrame(sites).drop_duplicates("source_site_id"),
                                         "soil_temp"))
    if not out:
        return empty_records()
    return finish(pd.concat(out, ignore_index=True), "parks_canada")


def read_illisarvik(raw) -> pd.DataFrame:
    """
    Illisarvik drained-lake experiment, Richards Island, NWT (Mackay; NSIDC GGD642):
    thaw-depth courses (tundra control, 400 m and 600 m lines), 15 points per course,
    probed in August 1983-1999+. Survey mean per date, then the end-of-season rule.
    The drained-lake courses record permafrost aggradation after drainage (1978), so
    they carry a site note.
    """
    recs = []
    for f in files(raw, "ggd642_thawdepth_*.txt"):
        course = re.search(r"thawdepth_(\w+)\.txt", f.name).group(1)
        for line in f.read_text(encoding="latin-1").split("\n"):
            p = line.rstrip("\r").split("\t")
            m = re.match(r"(\d{1,2})/(\d{1,2})/(\d{2,4})$", p[0].strip())
            if not m:
                continue
            yr = int(m.group(3)) + (1900 if len(m.group(3)) == 2 else 0)
            date = pd.Timestamp(yr, int(m.group(1)), int(m.group(2)))
            for v in p[1:16]:
                x = as_float(v)
                if x == x and 2 <= x <= 300:                 # '53153' style typos dropped
                    recs.append(dict(obs_date=date, alt_cm=x, course=course))
    if not recs:
        return empty_records()
    p = pd.DataFrame(recs)
    p["source_site_id"] = "ILLISARVIK_" + p.course
    p["site_name"] = "Illisarvik " + p.course
    p["lat"], p["lon"], p["country"], p["method"] = 69.4825, -134.5662, "Canada", "probe"
    p["site_note"] = np.where(p.course == "tundra", "undisturbed tundra course",
                              "drained lake bottom (drained 1978), permafrost aggrading")
    p["source_file"] = "ggd642_thawdepth_" + p.course + ".txt"
    return finish(end_of_season(survey_means(p)), "illisarvik")


# =============================================================================
# Alaska: LTER and plot studies
# =============================================================================
def read_arc_lter(raw) -> pd.DataFrame:
    """
    Arctic LTER, Toolik Lake (EDI knb-lter-arc):
      1019 Toolik tussock watershed thaw 1990-present and 1626 Imnavait thaw grid
      2003-present (Kling): survey mean, SD and N per date.
      1610 experimental plots 1993-present: only the control plots (Treatment CT/Ct/CT2)
      are used; the mean of the plot means per site and date.
    End-of-season rule per site. Coordinates from the EML files.
    """
    out = []
    for f in files(raw, "knb-lter-arc.1019*TW_thaw*.csv") + files(raw, "knb-lter-arc.1626*Thaw*.csv"):
        eml = next(iter(files(raw, f.name.split("_")[0] + "_EML.xml")), None)
        txt = eml.read_text(encoding="utf-8", errors="replace") if eml else ""
        m = re.search(r"<westBoundingCoordinate>(.*?)</west.*?<northBoundingCoordinate>(.*?)</north", txt, re.S)
        lat, lon = (as_float(m.group(2)), as_float(m.group(1))) if m else (np.nan, np.nan)
        d = pd.read_csv(f, encoding="utf-8-sig")
        d = d[pd.to_datetime(d.Date, errors="coerce", format="mixed").notna()]
        name = "Toolik tussock watershed (Kling)" if "1019" in f.name else "Imnavait thaw grid (Kling)"
        s = pd.DataFrame(dict(source_site_id="ARCLTER_" + ("TW" if "1019" in f.name else "IMNAVAIT"),
                              site_name=name, lat=lat, lon=lon, country="USA", method="probe",
                              obs_date=pd.to_datetime(d.Date), alt_cm=pd.to_numeric(d.Mean_thaw_cm, errors="coerce"),
                              alt_sd_cm=pd.to_numeric(d.Standard_Devation_cm, errors="coerce"),
                              n_points=pd.to_numeric(d.N, errors="coerce"), censored=False,
                              source_file=f.name))
        nmax = s.n_points.groupby(s.obs_date.dt.year).transform("max")
        s = s[~(s.n_points < CONFIG["min_node_fraction"] * nmax)]
        out.append(end_of_season(s.dropna(subset=["alt_cm"])))
    for f in files(raw, "knb-lter-arc.1610*soilthaw*.csv"):
        eml = next(iter(files(raw, "knb-lter-arc.1610*_EML.xml")), None)
        txt = eml.read_text(encoding="utf-8", errors="replace") if eml else ""
        coords = {}
        for m in re.finditer(r"<geographicDescription>(.*?)</geographicDescription>.*?<westBoundingCoordinate>(.*?)</west.*?<northBoundingCoordinate>(.*?)</north", txt, re.S):
            desc = m.group(1)
            for code in re.findall(r"\b(MAT\d\d|MNT\d\d|NNT\d\d|MNN\d\d|WSG\d\d|Wet Sedge Tundra Block \d)", desc):
                coords[code] = (as_float(m.group(3)), as_float(m.group(2)))
        # sites without their own coverage get the position of the neighbouring block of
        # the same tundra type (MAT89 plots lie next to the MAT81 plots)
        alias = {"WSG89": "Wet Sedge Tundra Block 1", "MNN97": "NNT89", "MNT97": "MNT89", "MAT06": "MAT81",
                 "MAT89": "MAT81"}
        d = pd.read_csv(f)
        d = d[d.Treatment.astype(str).str.strip().str.upper().isin(["CT", "CT2"])]
        d["alt"] = pd.to_numeric(d["Average Thaw"], errors="coerce")
        s = d.groupby(["Site", "Date"]).agg(alt_cm=("alt", "mean"), alt_sd_cm=("alt", "std"),
                                            n_points=("alt", "size")).reset_index()
        pos = s.Site.map(lambda c: coords.get(c) or coords.get(alias.get(c, ""), (np.nan, np.nan)))
        s = s.assign(source_site_id="ARCLTER_CT_" + s.Site, site_name="Toolik LTER control plots " + s.Site,
                     lat=[p[0] for p in pos], lon=[p[1] for p in pos], country="USA", method="probe",
                     obs_date=pd.to_datetime(s.Date, errors="coerce"), censored=False, source_file=f.name,
                     site_note="control plots of the LTER experiment")
        out.append(end_of_season(s.dropna(subset=["alt_cm"])))
    if not out:
        return empty_records()
    return finish(pd.concat(out, ignore_index=True), "arc_lter")


def read_bnz_lter(raw) -> pd.DataFrame:
    """
    Bonanza Creek LTER (EDI knb-lter-bnz) thaw-depth series near Fairbanks and Eight Mile
    Lake: 11 (BCEF core sites, 1991-), 7 (Rosie Creek fire, 1984-), 8 (Wickersham fire,
    1972-), 239 (Boundary fire, 2004-), 240 (Survey Line fire, 2002-), 605 (Regional Site
    Network, 2001-), 519 (Eight Mile Lake gradient, 2004-), 480 (CiPEHR warming
    experiment, controls only). One reading per stake or probe point. Readings coded
    -9999 or flagged M (missing) are dropped; hit type Rock / None / Gravel or flag R / N
    (rock, no frozen ground within the probe length) are censored. The EML site
    description is kept as a note, and burned sites get the information flag
    'burned_site'. Coordinates per site from the EML files. Not used: 496 (DryPEHR
    drought experiment on the CiPEHR plots) and 518 (2004-2012 subset of 519).
    """
    pts = []
    for f in files(raw, "knb-lter-bnz.*.csv"):
        pkg = re.match(r"(knb-lter-bnz\.\d+)\.", f.name).group(1)
        eml = next(iter(files(raw, f"{pkg}.*_EML.xml")), None)
        sites = eml_sites(eml.read_text(encoding="utf-8", errors="replace")) if eml else {}
        sites = {k.upper(): v for k, v in sites.items()}
        d = pd.read_csv(f, low_memory=False, encoding="latin-1")
        d.columns = [c.strip() for c in d.columns]
        low = {c.lower(): c for c in d.columns}
        if "dry" in low:                                                        # 496 DryPEHR
            continue
        if "depth" in low and "site" in low:                                   # 7, 8, 11, 239, 240, 605
            val = pd.to_numeric(d[low["depth"]], errors="coerce")
            hit = d[low.get("hit type", low.get("hit_type", ""))] if ("hit type" in low or "hit_type" in low) \
                else pd.Series("", index=d.index)
            hit = hit.astype(str).str.lower().str.strip()
            flag = d[low["flag"]].astype(str).str.upper().str.strip() if "flag" in low else pd.Series("", index=d.index)
            val[(val < 0) | (flag == "M") | (hit == "missing")] = np.nan
            cens = hit.isin(["rock", "none", "gravel"]) | flag.isin(["R", "N"])
            site = d[low["site"]].astype(str).str.strip().str.upper()
            date = pd.to_datetime(d[low["date"]], errors="coerce")
        elif "td" in low and "plot" in low and "ww" in low:                     # 480 CiPEHR
            ctl = (d.ww.astype(str).str.lower() == "c") & (d.sw.astype(str).str.lower() == "c")
            d = d[ctl & (d.get("id", "plot") == "plot")]
            val, cens = pd.to_numeric(d.td, errors="coerce"), pd.Series(False, index=d.index)
            site, date = pd.Series("CIPEHR_CONTROL", index=d.index), pd.to_datetime(d.date, errors="coerce")
            sites.setdefault("CIPEHR_CONTROL", sites.get("EML", (63.8807, -149.2396, "")))
        elif "td" in low and "site" in low:                                     # 519 gradient
            val, cens = pd.to_numeric(d.td, errors="coerce"), pd.Series(False, index=d.index)
            site = "EML_" + d.site.astype(str).str.upper()
            date = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day), errors="coerce")
            for s in site.unique():
                sites.setdefault(s, sites.get("EML", (63.8807, -149.2396, "")))
        else:
            continue
        g = pd.DataFrame(dict(site=site, obs_date=date, alt_cm=val, censored=cens))
        g["source_file"], g["pkg"] = f.name, pkg
        g["lat"] = g.site.map(lambda s: sites.get(s, (np.nan,))[0])
        g["lon"] = g.site.map(lambda s: sites.get(s, (np.nan, np.nan))[1])
        g["site_note"] = g.site.map(lambda s: sites.get(s, ("", "", ""))[2][:160])
        pts.append(g)
    if not pts:
        return empty_records()
    p = pd.concat(pts, ignore_index=True).dropna(subset=["alt_cm"])
    p["source_site_id"] = "BNZ_" + p.site.str.replace(r"\s+", "_", regex=True)
    p["site_name"], p["country"], p["method"] = p.site, "USA", "probe"
    s = survey_means(p)
    out = end_of_season(s)
    out["flag"] = add_flag(pd.Series("", index=out.index), out.censored, "lower_bound")
    # information only: burned sites of the fire studies (controls and unburned lines not)
    note = out.site_note.str.lower()
    burned = note.str.contains(r"(?<!un)burn") & ~note.str.contains("control|unburned")
    out["flag"] = add_flag(out.flag, burned, "burned_site")
    return finish(out, "bnz_lter")


def read_itex(raw) -> pd.DataFrame:
    """
    ITEX Utqiagvik (Barrow) and Atqasuk thaw depths 1995-2024 (Hollister, Webber; Arctic
    Data Center): 24 control and 24 open-top-chamber plots at four sites (dry heath, wet
    meadow). Only control plots are used: mean of the plot readings per site and date,
    then the end-of-season rule.
    """
    fs = files(raw, "*_Barrow_Atqasuk_ITEX_Thaw_v*.csv")
    if not fs:
        return empty_records()
    d = pd.concat([pd.read_csv(f, dtype=str).assign(source_file=f.name) for f in fs])
    d = d[d.strTrea == "C"]
    d["alt_cm"] = pd.to_numeric(d.numThaw, errors="coerce")
    d["obs_date"] = pd.to_datetime(d.strUTC, format="%Y%m%d", errors="coerce")
    names = {"AD": "Atqasuk dry heath", "AW": "Atqasuk wet meadow",
             "BD": "Utqiagvik dry heath", "BW": "Utqiagvik wet meadow"}
    d["site_name"] = d.strSitCom.map(names)
    d["lat"] = np.where(d.strSitCom.str[0] == "B", 71.3167, 70.4833)
    d["lon"] = np.where(d.strSitCom.str[0] == "B", -156.6167, -157.4167)
    d = d.assign(source_site_id="ITEX_" + d.strSitCom, country="USA", method="probe",
                 site_note="ITEX control plots")
    return finish(end_of_season(survey_means(d)), "itex")


def read_ngee_barrow(raw) -> pd.DataFrame:
    """
    NGEE-Arctic Utqiagvik (Barrow) intensive areas A-D, thermal transects 2012-2023:
    five temperature profiles per area (polygon centre, margin, rim, trough, second rim),
    16 sensors each from 2 cm to 1.50 m. ALT per profile from the 0 C crossing, then the
    mean of the profiles of an area (needs >= 3 profiles that year).
    Depths: most files label the sensors with their installation depths (0.02-1.50 m).
    Some files (mostly 2015-2018) label them relative to a re-measured ground surface,
    shifted by a few cm. To keep one depth reference through the series, every profile
    is put back on the installation depths: shift = 1.50 m minus the deepest label.
    Sensors that then collide (top sensors only) are dropped for that file.
    Coordinates from the sensor location table (UTM zone 4N).
    """
    fs = files(raw, "NGEE_BRW_*_Thermal_Transect*.csv")
    loc = next(iter(files(raw, "NGEE-BRW-ABCD-sensor-locations_w-o_figs_*.csv")), None)
    if not fs or loc is None:
        return empty_records()
    L = pd.read_csv(loc)
    L = L[L.Code == "TP"]
    lat, lon = utm_to_latlon(L.Easting, L.Northing, 4)
    pos = pd.DataFrame(dict(area=L.AREA, lat=lat, lon=lon)).groupby("area").median()
    temps = []
    for f in fs:
        area = re.search(r"NGEE_BRW_([A-D])_", f.name).group(1)
        lines = f.read_text(encoding="latin-1").split("\n")
        h = next(i for i, l in enumerate(lines) if l.lstrip("\ufeffï»¿").startswith(("Timestamp,", "Time,")))
        d = pd.read_csv(io.StringIO("\n".join(lines[h:])), na_values=["NULL"], low_memory=False,
                        header=None, skiprows=1)
        names = lines[h].rstrip("\r").split(",")
        prof = {}
        for k, c in enumerate(names):
            m = re.search(r"VATP(\d)_(-?\d+\.\d+)m_", c)
            if m:
                prof.setdefault(m.group(1), []).append((k, float(m.group(2))))
        time = pd.to_datetime(d[0], errors="coerce").dt.normalize()
        for pr, cols in prof.items():
            shift = 1.50 - max(z for _, z in cols)
            depth = pd.Series({k: round((z + shift) * 100, 1) for k, z in cols})
            depth = depth[~depth.duplicated(keep=False)]
            for k, z in depth.items():
                temps.append(pd.DataFrame(dict(source_site_id=f"{area}_{pr}", time=time, depth_cm=z,
                                               temp_c=pd.to_numeric(d[k], errors="coerce"))))
    t = pd.concat(temps).dropna(subset=["time", "temp_c"])
    t = t.groupby(["source_site_id", "time", "depth_cm"], as_index=False).temp_c.mean()
    sites = pd.DataFrame(dict(source_site_id=t.source_site_id.unique()))
    a = records_from_profiles(t, sites, "soil_temp")
    if a.empty:
        return empty_records()
    a["area"] = a.source_site_id.str[0]
    ok = a[~a.censored & ~a.flag.str.contains("wide")]
    g = ok.groupby(["area", "year"]).agg(alt_cm=("alt_cm", "mean"), alt_sd_cm=("alt_cm", "std"),
                                         n_points=("alt_cm", "size"), bracket_cm=("bracket_cm", "max")).reset_index()
    g = g[g.n_points >= 3]
    g = g.assign(source_site_id="NGEE_BRW_" + g.area, site_name="NGEE Utqiagvik area " + g.area,
                 lat=g.area.map(pos.lat), lon=g.area.map(pos.lon), country="USA", method="soil_temp",
                 censored=False, source_file="NGEE_BRW_*_Thermal_Transect*.csv",
                 site_note="mean of the polygon-position temperature profiles of the area")
    return finish(g, "ngee_barrow")


def _location_from_rows(rows: list[str]):
    """(lat, lon) from the 'Location:' line of a UAF station file (the first rows, as
    text): 'N 70o09.677' W148o27.918'', '148.4653 W; 70.161283 N' or 'W 149.58; N 68.07'.
    Without a 'Location' line, row 3 is used. None when no position is found."""
    loc = next((r for r in rows if "location" in r.lower()), rows[2] if len(rows) > 2 else "")
    dm = re.findall(r"([NW])\s*(\d+)o(\d+(?:\.\d+)?)", loc)
    if dm:
        v = {h: int(d) + float(m) / 60 for h, d, m in dm}
        if "N" in v and "W" in v:
            return (v["N"], -v["W"])
        return None
    # decimal degrees: the letter is either before every number ('N 64.86781  W 147.78',
    # 'W 149.58; N 68.07', 'N 68.07  E -149.58') or after it ('149.50 W, 68.47 N'). The
    # side is taken from the first number, so a number is never paired with the next
    # number's letter ('N 64.87  W 147.78' must not give W 64.87).
    body = loc.split(":", 1)[-1].strip()
    if re.match(r"[NSEW]\s*-?\d", body):
        pairs = [(h, float(v)) for h, v in re.findall(r"([NSEW])\s*(-?\d+(?:\.\d+)?)", body)]
    else:
        pairs = [(h, float(v)) for v, h in re.findall(r"(-?\d+(?:\.\d+)?)\s*o?\s*([NSEW])\b", body)]
    lat = next((v if h == "N" else -abs(v) for h, v in pairs if h in "NS"), None)
    lon = next((v if h == "E" else -abs(v) for h, v in pairs if h in "EW"), None)
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return (lat, lon)


def uaf_station_sheet(x: pd.DataFrame):
    """
    One sheet of a UAF Permafrost Lab station file, read without a header: title lines,
    a position line in row 3 ('N 70o09.677' W148o27.918'' or '148.4653 W; 70.161283 N'),
    then a header row with the sensor depths in m ('0.025 m', 'Temp_MRC_0.125') and daily
    rows. Returns ((lat, lon) or None, long table time/depth_cm/temp_c or None).
    """
    # the position line: row 3 in the workbooks; the csv versions split it into cells
    pos = _location_from_rows([" ".join(str(c) for c in x.iloc[i] if pd.notna(c))
                               for i in range(min(8, len(x)))])
    # header row: the first row with at least three depth labels
    lab = lambda c: re.fullmatch(r"(?:Temp_MRC_)?(-?\d+\.\d+)\s*m?", str(c).strip())
    hr = next((i for i in range(min(30, len(x))) if sum(bool(lab(c)) for c in x.iloc[i]) >= 3), None)
    if hr is None:
        return pos, None
    body = x.iloc[hr + 1:]

    def as_dates(col):
        v = pd.to_numeric(col, errors="coerce")
        if v.notna().mean() > 0.5:                   # plain numbers (year, day, YYYYMMDD)
            return pd.Series(pd.NaT, index=col.index)
        d = pd.to_datetime(col.astype(str).str[:10], format="%Y-%m-%d", errors="coerce")
        if d.notna().mean() < 0.5:                   # csv versions: other date layouts
            d = pd.to_datetime(col, errors="coerce")
        return d
    tcol = max(range(min(5, x.shape[1])), key=lambda j: as_dates(body[j]).notna().mean())
    time = as_dates(body[tcol])
    parts = []
    for j, c in enumerate(x.iloc[hr]):
        m = lab(c)
        if not m or j == tcol or float(m.group(1)) <= 0:
            continue
        v = pd.to_numeric(body[j], errors="coerce")
        v[v.abs() >= 99] = np.nan
        parts.append(pd.DataFrame(dict(time=time, depth_cm=round(float(m.group(1)) * 100, 1), temp_c=v)))
    return pos, (pd.concat(parts) if parts else None)


def read_uaf_stations(raw) -> pd.DataFrame:
    """
    UAF Permafrost Laboratory (V. Romanovsky) soil-temperature stations on the North Slope
    along the Dalton Highway: Deadhorse (files DH 2001-2005 and DH1 2006-2014, same
    position, so one station), Deadhorse 2 (DH2, 2009-2018), Franklin Bluffs (FB1),
    Sagwon MAT and MNT, Imnavait (IM1). One file per year (June-May), daily means at
    10-14 depths to about 1 m (sensors 7-15 cm apart). Depth labels are given relative
    to the ground surface of each year (they change slowly as the surface subsides) and
    are used as given; sensors at or above the surface are not used. '999' and 'NULL'
    are missing. The deep borehole files (u23, u24, us23, us24, u60: one profile from
    4 m down) cannot give ALT and are not read.
    """
    station = {"DH": "DH", "DH1": "DH", "DH2": "DH2", "FB1": "FB1", "SAGMAT": "SAGMAT",
               "SAGMNT": "SAGMNT", "IM1": "IM1"}
    names = {"DH": "Deadhorse (UAF DH/DH1)", "DH2": "Deadhorse 2 (UAF DH2)", "FB1": "Franklin Bluffs (UAF FB1)",
             "SAGMAT": "Sagwon MAT (UAF)", "SAGMNT": "Sagwon MNT (UAF)", "IM1": "Imnavait (UAF IM1)"}
    temps, pos = [], {}
    for f in files(raw, "*.xls") + files(raw, "*.csv"):
        if "UAF_PermafrostLab" not in f.parts:
            continue
        code = station.get(re.match(r"([A-Za-z]+\d?)", f.name).group(1).upper())
        if code is None:
            continue
        x = (pd.read_excel(f, header=None, sheet_name=0) if f.suffix == ".xls" else
             pd.read_csv(f, header=None, encoding="latin-1", on_bad_lines="skip", engine="python"))
        p, long = uaf_station_sheet(x)
        if p is not None:
            pos.setdefault(code, p)
        if long is not None:
            temps.append(long.assign(source_site_id="UAF_" + code))
    if not temps:
        return empty_records()
    t = pd.concat(temps).dropna(subset=["time", "temp_c"])
    # a sensor's label can change between files; keep one value per sensor depth and day
    t = t.groupby(["source_site_id", "time", "depth_cm"], as_index=False).temp_c.mean()
    sites = pd.DataFrame([dict(source_site_id="UAF_" + k, site_name=names[k], lat=v[0], lon=v[1],
                               country="USA", source_file="UAF Permafrost Lab station files",
                               site_note="sensor depths relative to the ground surface of each year")
                          for k, v in pos.items()])
    return finish(records_from_profiles(t, sites, "soil_temp"), "uaf_stations")


# =============================================================================
# Russia: Kolyma, Indigirka, Lena Delta, Yakutia
# =============================================================================
def read_kwbs(raw) -> pd.DataFrame:
    """
    Kolyma Water-Balance Station, Magadan region (Makarieva et al. 2018; PANGAEA 881754):
    thaw depth measured repeatedly through the summer at ~37 plots, 1954-1997. The value
    is the lower boundary of the thawed layer. A season counts when the last reading is
    on or after 1 September (thaw complete); the season's deepest reading is ALT.
    The soil-temperature record of the Nizhnyaya plot (881792, 1974-1981) is used too.
    """
    out = []
    for f in files(raw, "PANGAEA_881754*.tab"):
        head, d = read_pangaea_tab(f)
        ev = pangaea_events(head)
        low = next(c for c in d.columns if "lower boundary" in c)
        s = pd.DataFrame(dict(source_site_id="KWBS_" + d.Event.str.replace("KWBS_cryo_", "", regex=False),
                              site_name="KWBS plot " + d.Event.str.replace("KWBS_cryo_", "", regex=False) + " "
                              + d.Event.map(lambda e: ev.get(e, {}).get("comment", "")),
                              lat=d.Latitude, lon=d.Longitude, country="Russia", method="probe",
                              obs_date=pd.to_datetime(d["Date/Time"], errors="coerce"),
                              alt_cm=pd.to_numeric(d[low], errors="coerce"), censored=False,
                              source_file=f.name))
        out.append(end_of_season(s.dropna(subset=["alt_cm"]), min_last_doy=244))
    for f in files(raw, "PANGAEA_881792*.tab"):
        head, d = read_pangaea_tab(f)
        cols = {c: as_float(re.search(r"at ([\d.]+) m", c).group(1)) * 100 for c in d.columns if "T soil" in c}
        d["time"] = pd.to_datetime(d["Date/Time"], errors="coerce")
        t = d.melt(id_vars=["time"], value_vars=list(cols), var_name="c", value_name="temp_c")
        t["depth_cm"], t["source_site_id"] = t.c.map(cols), "KWBS_T_Nizhnyaya"
        sites = pd.DataFrame([dict(source_site_id="KWBS_T_Nizhnyaya", site_name="KWBS Nizhnyaya soil temperature",
                                   lat=61.85, lon=147.67, country="Russia", source_file=f.name)])
        out.append(records_from_profiles(t, sites, "soil_temp"))
    if not out:
        return empty_records()
    return finish(pd.concat(out, ignore_index=True), "kwbs")


def read_kytalyk(raw) -> pd.DataFrame:
    """
    Allaikha (Kytalyk area), Indigirka lowland (Liang et al. 2023; PANGAEA 961879): a grid
    of ~120 points probed once a year (Aug-Sep) 2009-2018; one column per date.
    """
    out = []
    for f in files(raw, "PANGAEA_961879*.tab"):
        head, d = read_pangaea_tab(f)
        dcols = [c for c in d.columns if c.startswith("Thaw depth")]
        long = d.melt(id_vars=["Site", "Latitude", "Longitude"], value_vars=dcols, value_name="alt_cm")
        long["obs_date"] = pd.to_datetime(long.variable.str.extract(r"\((\d{4}-\d{2}-\d{2})\)")[0])
        long["alt_cm"] = pd.to_numeric(long.alt_cm, errors="coerce")
        long = long.assign(source_site_id="KYTALYK_" + long.Site, site_name=long.Site + " (Kytalyk)",
                           lat=long.Latitude, lon=long.Longitude, country="Russia", method="probe",
                           source_file=f.name)
        out.append(end_of_season(survey_means(long)))
    return finish(pd.concat(out), "kytalyk") if out else empty_records()


def read_allena(raw) -> pd.DataFrame:
    """
    ALLena, Lena Delta thaw-depth compilation 1998-2022 (Veremeeva et al. 2025; PANGAEA
    974408, main table). Mostly one-off campaign points. Points repeated at the same place
    are grouped by position rounded to 0.001 deg (about 100 m); per group: mean per date,
    then the end-of-season rule. The Samoylov CALM grid itself is read separately.
    """
    z = files(raw, "PANGAEA_974408*.zip")
    if not z:
        return empty_records()
    with zipfile.ZipFile(z[-1]) as zf:
        n = next(n for n in zf.namelist() if n.endswith("Veremeeva_main_table.tab"))
        tmp = zf.read(n).decode("utf-8", errors="replace")
    i = tmp.find("*/")
    d = pd.read_csv(io.StringIO(tmp[i + 2:].lstrip("\r\n")), sep="\t", low_memory=False)
    col = lambda p: next(c for c in d.columns if c.startswith(p))
    lat, lon = pd.to_numeric(d[col("Latitude")], errors="coerce"), pd.to_numeric(d[col("Longitude")], errors="coerce")
    p = pd.DataFrame(dict(obs_date=pd.to_datetime(d[col("Date/Time")], errors="coerce"),
                          alt_cm=pd.to_numeric(d[col("Thaw depth mean")], errors="coerce"),
                          lat=lat, lon=lon, site_name=d[col("Location")].astype(str)))
    p["source_site_id"] = "ALLENA_" + lat.round(3).astype(str) + "_" + lon.round(3).astype(str)
    p = p.assign(country="Russia", method="probe", source_file=f"{z[-1].name}/Veremeeva_main_table.tab")
    s = survey_means(p.dropna(subset=["alt_cm", "lat"]))
    out = end_of_season(s)
    return finish(out, "allena")


def read_awi_soil_temp(raw) -> pd.DataFrame:
    """
    AWI soil-temperature stations with ALT from the 0 C crossing:
      Bayelva, Svalbard (homogenised hourly product 1998-2024, PANGAEA 982254;
      profile 3-135 cm plus borehole to 9 m).
      Samoylov, Lena Delta (soil_lv2 2002-2018 and yearly lv1 files 2019-2021; the
      polygon-rim profile, 6-71 cm).
    """
    out = []
    for f in files(raw, "PANGAEA_982254*.tab"):
        d = pd.read_csv(f, comment="#", na_values=["NA"], low_memory=False)
        cols = {c: as_float(re.search(r"_(\d+)cm", c).group(1)) for c in d.columns if c.startswith("Tsoil_degC_")}
        d["time"] = pd.to_datetime(d.Datetime_UTC, errors="coerce").dt.normalize()
        t = d.melt(id_vars=["time"], value_vars=list(cols), var_name="c", value_name="temp_c")
        t["depth_cm"] = t.c.map(cols)
        t = t.groupby(["time", "depth_cm"], as_index=False).temp_c.mean().assign(source_site_id="AWI_BAYELVA")
        sites = pd.DataFrame([dict(source_site_id="AWI_BAYELVA", site_name="Bayelva soil station",
                                   lat=78.92094, lon=11.83333, country="Norway (Svalbard)", source_file=f.name)])
        out.append(records_from_profiles(t, sites, "soil_temp"))
        break
    # Samoylov: the polygon-rim profile (to 60-71 cm) is used; the centre profile ends at
    # 40 cm, above the thaw depth in most years. Readings with a quality flag are dropped.
    parts = []
    tables = []
    for z in files(raw, "soil_lv2_V*.zip") + files(raw, "PANGAEA_*_soil_lv1_*.zip"):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith(("soil_lv2.dat", ".dat")) and "soil_lv" in n:
                    tables.append(pd.read_csv(io.BytesIO(zf.read(n)), na_values=["NA", "NaN"], low_memory=False))
    for f in files(raw, "PANGAEA_*_soil_lv1_*.dat"):
        tables.append(pd.read_csv(f, na_values=["NA", "NaN"], low_memory=False))
    for d in tables:
        cols = {c: as_float(re.search(r"_(\d+)$", c).group(1)) for c in d.columns
                if re.match(r"Ts_rim_(lv2_)?\d+$", c)}
        long = d.melt(id_vars=["UTC"], value_vars=list(cols), var_name="c", value_name="temp_c")
        fl = [c + "_fl" for c in cols if c + "_fl" in d]
        if fl:
            flags = d.melt(id_vars=["UTC"], value_vars=fl, value_name="fl").fl.to_numpy()
            long.loc[pd.to_numeric(flags, errors="coerce") != 0, "temp_c"] = np.nan
        long["depth_cm"] = long.c.map(cols)
        long["time"] = pd.to_datetime(long.UTC, errors="coerce").dt.normalize()
        long["temp_c"] = pd.to_numeric(long.temp_c, errors="coerce")
        parts.append(long.groupby(["time", "depth_cm"], as_index=False).temp_c.mean())
    if parts:
        t = pd.concat(parts).groupby(["time", "depth_cm"], as_index=False).temp_c.mean()
        t["source_site_id"] = "AWI_SAMOYLOV"
        sites = pd.DataFrame([dict(source_site_id="AWI_SAMOYLOV", site_name="Samoylov soil station (polygon rim)",
                                   lat=72.3696, lon=126.4806, country="Russia",
                                   source_file="soil_lv2_V201908.zip + PANGAEA soil_lv1 2019-2021")])
        out.append(records_from_profiles(t, sites, "soil_temp"))
    if not out:
        return empty_records()
    return finish(pd.concat(out, ignore_index=True), "awi_soil_temp")


# =============================================================================
# Nordic, eastern Canada
# =============================================================================
def read_stordalen(raw) -> pd.DataFrame:
    """
    Stordalen mire, Abisko, Sweden (Lund University; 2003-2017): thaw depth ('A L', cm,
    negative down) every few days at plot groups (dry, mesic, wet, tussock, E, F).
    Season's deepest reading after 1 September, as for other repeated-reading series.
    """
    f = next(iter(files(raw, "Stordalen_Active_Layer_Water_Table_*.xlsx")), None)
    if f is None:
        return empty_records()
    x = pd.read_excel(f, sheet_name="DATA", header=None)
    groups = {j: str(x.iat[0, j]).strip() for j in range(x.shape[1]) if str(x.iat[2, j]).strip() == "A L"}
    recs = []
    for r in x.iloc[3:].itertuples(index=False):
        date = pd.to_datetime(r[0], errors="coerce")
        if pd.isna(date):
            continue
        for j, g in groups.items():
            v = as_float(r[j])
            if v == v:
                recs.append(dict(obs_date=date, alt_cm=abs(v), grp=g))
    if not recs:
        return empty_records()
    p = pd.DataFrame(recs)
    p["source_site_id"] = "STORDALEN_" + p.grp.str.split().str[0]
    p = p.assign(site_name="Stordalen mire " + p.grp, lat=68.3556, lon=19.0453, country="Sweden",
                 method="probe", censored=False, source_file=f.name)
    return finish(end_of_season(p, min_last_doy=244), "stordalen")


def read_umiujaq_cables(raw) -> pd.DataFrame:
    """
    Umiujaq, Tasiapik Valley permafrost mound (lithalsa), Nunavik (CEN; Borealis
    doi:10.5683/SP3/QSRW0I): daily temperatures in five boreholes (TA1-TA5), 2001-2024,
    post-cleaning files only. TA1's cable was replaced in 2014; both periods are the same
    borehole. ALT from the 0 C crossing.
    """
    fs = [f for f in files(raw, "data_ta*_post_cleaning.xlsx")]
    if not fs:
        return empty_records()
    temps = []
    for f in fs:
        ta = re.search(r"data_(ta\d)_", f.name).group(1).upper()
        d = pd.read_excel(f)
        cols = {c: as_float(str(c).replace("m", "")) * 100 for c in d.columns if re.fullmatch(r"\d+(\.\d+)?\s*m", str(c).strip())}
        t = d.melt(id_vars=["Date_Time"], value_vars=list(cols), var_name="c", value_name="temp_c")
        t["depth_cm"] = t.c.map(cols)
        t["time"] = pd.to_datetime(t.Date_Time, errors="coerce")
        t["temp_c"] = pd.to_numeric(t.temp_c, errors="coerce")
        t["source_site_id"] = "UMIUJAQ_" + ta
        temps.append(t)
    t = pd.concat(temps)
    sites = pd.DataFrame(dict(source_site_id=t.source_site_id.unique()))
    sites = sites.assign(site_name="Umiujaq Tasiapik mound " + sites.source_site_id.str[-3:], lat=56.5585, lon=-76.4800,
                         country="Canada", source_file="data_ta*_post_cleaning.xlsx",
                         site_note="permafrost mound (lithalsa), discontinuous permafrost")
    return finish(records_from_profiles(t, sites, "borehole_temp"), "umiujaq_cable")


def _merge_borehole_copies(rec: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """
    The collection sometimes holds one borehole twice: from two contributors (Beaver Creek
    'HPW_Shakwak_orig_natural' and 'BC-RES_YG4_natural_*' give the same ALT to 0.05 cm),
    Two series are copies when they lie within 300 m and their ALT agrees
    within copy_agree_cm (1 cm, median) on >= 3 shared years with an uncensored value
    (lower bounds are not compared: two nearby cables whose deepest sensor sits at the
    same depth would 'agree' every year without being copies). The members of a group
    become one series, named after the member with the most valid years; it keeps all its
    own years and takes from the other members only the years it does not have.
    Series that the metadata calls the same borehole but with new or other sensors
    (BYLOTPB_a, SALTSIL_T109) are not merged: a sensor change shifts the ALT (BYLOTPB
    reads 78-81 cm in 2007-2015, BYLOTPB_a 96-107 cm from 2016), so splicing them would
    make a step look like a trend. They stay separate series.
    """
    ids = sorted(rec.source_site_id.unique())
    parent = {i: i for i in ids}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    pos = sites.set_index("source_site_id").loc[ids, ["lat", "lon"]]
    ser = {k: g.set_index("year").alt_cm for k, g in rec[~rec.censored].groupby("source_site_id")}
    ser = {i: ser.get(i, pd.Series(dtype=float)) for i in ids}
    for i, a in enumerate(ids):
        d = haversine_km(pos.lat.iat[i], pos.lon.iat[i], pos.lat.to_numpy(), pos.lon.to_numpy())
        for j in np.where(d <= 0.3)[0]:
            b = ids[j]
            if j <= i or root(a) == root(b):
                continue
            common = ser[a].index.intersection(ser[b].index)
            if len(common) >= 3 and (ser[a][common] - ser[b][common]).abs().median() <= CONFIG["copy_agree_cm"]:
                parent[root(a)] = root(b)
    groups = pd.Series({i: root(i) for i in ids})
    rec = rec.copy()
    rec["grp"] = rec.source_site_id.map(groups)
    rec["valid"] = ~rec.censored & ~rec.flag.str.contains(INVALID_FLAGS)
    out = []
    for g, r in rec.groupby("grp"):
        members = r.source_site_id.unique()
        if len(members) == 1:
            out.append(r)
            continue
        size = r.groupby("source_site_id").agg(v=("valid", "sum"), n=("year", "size"))
        order = size.sort_values(["v", "n"], ascending=False).index.tolist()
        r = r.assign(o=r.source_site_id.map({m: k for k, m in enumerate(order)}))
        r = r.sort_values(["year", "o"]).drop_duplicates("year").drop(columns="o")
        main, others = order[0], order[1:]
        site = sites.set_index("source_site_id").loc[main]
        r["source_site_id"], r["site_name"], r["lat"], r["lon"] = main, site.site_name, site.lat, site.lon
        r["site_note"] = (f"{site.site_note}; same borehole as {', '.join(others)} (years missing "
                          f"in {main} taken from them)")
        log(f"    {main}: merged with {', '.join(others)} (same borehole)")
        out.append(r)
    return pd.concat(out, ignore_index=True).drop(columns=["grp", "valid"])


def read_canada_gt(raw) -> pd.DataFrame:
    """
    Canadian permafrost ground temperature collection 2025 (Meier-Legault et al., Zenodo
    doi:10.5281/zenodo.21042525): daily mean ground temperatures from ~430 boreholes and
    cables of many groups (GSC, CEN, Yukon and NWT geological surveys, ECCC climate
    stations, UAF, universities), cleaned and put in one format, 1960-2025. Only ground
    temperature sites (type 'GT', not surface loggers). ALT from the 0 C crossing.
      - The compilers mark each site as disturbed or not (highway embankments, mine
        sites, burnt or cleared ground). Disturbed sites (disturbance = 1) are flagged
        'disturbed_site' and not used: their ALT follows the disturbance, not the climate.
        Unknown (-1) is kept and noted in site_note.
      - Cables installed that summer (record starts in May-August): first year flagged
        'borehole_equilibration' (CONFIG install_months).
      - Years whose sensors are not comparable with the rest of the record (CONFIG
        sensor_problem_years): flagged 'sensor_problem'.
      - Copies of one borehole are merged (_merge_borehole_copies).
    The same boreholes can also be in sources read directly (GSC Open File 9340 cables,
    Umiujaq): the build joins them by position and agreement, and the direct source is
    preferred (CONFIG priority).
    """
    zs = files(raw, "all_observations.zip")
    metas = files(raw, "site_metadata.csv")
    metas = [m for m in metas if (m.parent / "all_observations.zip").exists()]
    if not zs or not metas:
        return empty_records()
    meta = pd.read_csv(metas[-1], encoding="utf-8-sig")
    meta = meta[(meta.type == "GT") & (meta.latitude >= CONFIG["min_lat"])]
    with zipfile.ZipFile(zs[-1]) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".csv"))
        d = pd.read_csv(zf.open(name), usecols=["timestamp", "depth", "temperature", "site_id"],
                        dtype={"depth": "float64", "temperature": "float64", "site_id": "str"})
    d = d[d.site_id.isin(meta.site_id)]
    t = pd.DataFrame(dict(source_site_id=d.site_id.to_numpy(),
                          time=pd.to_datetime(d.timestamp.str.slice(0, 10), errors="coerce"),
                          depth_cm=(d.depth * 100).round(1), temp_c=d.temperature))
    t = t[t.depth_cm <= CONFIG["valid_alt_cm_borehole"][1]]
    del d

    def note(r):
        parts = [str(r.institution)]
        if r.disturbance == 1:
            parts.append("disturbed site (collection metadata)")
        elif r.disturbance == -1:
            parts.append("disturbance unknown")
        if r.bedrock == 1:
            parts.append("borehole in bedrock")
        if "ECCC" in str(r.institution):
            parts.append("climate station soil thermometers read twice daily, 50 cm apart "
                         "near the thaw front: values often fall on a sensor depth")
        env = str(r.environment_description)
        if env and env != "nan":
            parts.append(re.sub(r"\s+", " ", env)[:150])
        return "; ".join(parts)

    sites = pd.DataFrame(dict(source_site_id=meta.site_id, site_name=meta.site_id,
                              lat=meta.latitude, lon=meta.longitude, country="Canada",
                              site_note=meta.apply(note, axis=1),
                              source_file=metas[-1].parent.name + "|" + meta.site_id))
    rec = records_from_profiles(t, sites, "borehole_temp")
    if rec.empty:
        return empty_records()
    disturbed = rec.source_site_id.isin(meta.site_id[meta.disturbance == 1])
    rec["flag"] = add_flag(rec.flag, disturbed, "disturbed_site")
    start = t.groupby("source_site_id").time.min()
    m0, m1 = CONFIG["install_months"]
    installed = start[start.dt.month.between(m0, m1)].dt.year
    first_year = rec.groupby("source_site_id").year.transform("min")
    new_hole = (rec.year == first_year) & (rec.source_site_id.map(installed) == rec.year)
    rec["flag"] = add_flag(rec.flag, new_hole, "borehole_equilibration")
    bad = pd.Series(False, index=rec.index)
    for sid, periods in CONFIG["sensor_problem_years"]["canada_gt"].items():
        for y0, y1 in periods:
            bad |= (rec.source_site_id == sid) & rec.year.between(y0, y1)
    rec["flag"] = add_flag(rec.flag, bad, "sensor_problem")
    log(f"  canada_gt: {int(disturbed.sum())} site-years at disturbed sites, {int(new_hole.sum())} "
        f"installation years and {int(bad.sum())} years with sensor problems flagged")
    rec = _merge_borehole_copies(rec, sites)
    rec["source_file"] = rec.source_file.str.replace("|", "/all_observations.csv: ", regex=False)
    return finish(rec, "canada_gt")


# =============================================================================
# Greenland: GEM (Greenland Ecosystem Monitoring) downloads
# =============================================================================
def _gem_table(zpath: Path) -> pd.DataFrame:
    """The '_data.txt' table of a GEM download zip (tab separated, -9999 = missing)."""
    with zipfile.ZipFile(zpath) as zf:
        name = next(n for n in zf.namelist() if n.endswith("_data.txt"))
        d = pd.read_csv(io.BytesIO(zf.read(name)), sep="\t", encoding="utf-8-sig",
                        na_values=[-9999, "-9999"], low_memory=False)
    d.columns = [str(c).strip() for c in d.columns]
    return d


def read_gem(raw) -> pd.DataFrame:
    """
    GEM downloads (data.g-e-m.dk, CC-BY-SA 4.0), one zip per dataset:
      ZEROCALM-1 and ZEROCALM-2 active layer (CALM G1 and G2, Zackenberg): probe readings
        at every grid node through the thaw season, 1996-2025. Survey mean and end-of-
        season rule, as for the other grids. Same CALM codes as the CALM tables, so they
        add the years those tables lack.
      Ground temperatures, 0 C crossing rule: GeoBasis Disko AWS1 (5, 60, 175 cm,
        1991-2018) and AWS2 profiles P1 (10-140 cm, 2012-2024) and P2 (5-350 cm,
        2013-2024); ZEROCALM-2 upper, middle and lower profiles (to 2-2.5 m, 2012-2017).
        Sensors at or above the surface ('0 cm', '+3cm') are not used.
    Not read: shallow soil sensors (AWS2/3/4 soil temperature to 40 cm, Nuuk SoilEmp to
    30 cm), Nuuk SoilVUE (2021-2024), and the meteorology, snow and moisture files.
    """
    zips = {z.name: z for z in files(raw, "*_10.17897_*.zip")}
    if not zips:
        return empty_records()
    pos = _calm_positions(raw)
    out = []

    # active layer grids
    pts = []
    for key, code in (("ZEROCALM-1_Active_layer", "G1"), ("ZEROCALM-2_Active_layer", "G2")):
        z = next((zips[n] for n in zips if n.startswith(key)), None)
        if z is None:
            continue
        d = _gem_table(z)
        d["obs_date"] = pd.to_datetime(d.Date, errors="coerce")
        d["alt_cm"] = pd.to_numeric(d.Active_layer_cm, errors="coerce")
        rec = _calm_record(code, pos, method="probe", source_file=z.name)
        if not np.isfinite(rec["lat"]):
            rec.update(lat=74.47, lon=-20.57, country="Greenland")
        pts.append(pd.DataFrame(dict(**rec, obs_date=d.obs_date.values, alt_cm=d.alt_cm.values)))
    if pts:
        out.append(end_of_season(survey_means(pd.concat(pts, ignore_index=True))))

    # ground-temperature profiles
    profiles = {  # zip name start: (site id, name, lat, lon, method)
        "AWS1-GroundTemperature": ("GEM_DISKO_AWS1", "Disko, Arctic Station AWS1", 69.25328, -53.52003,
                                   "borehole_temp"),
        "AWS2-GroundTemperatureP1": ("GEM_DISKO_AWS2_P1", "Disko, Arctic Station AWS2 profile P1",
                                     69.25349, -53.51413, "borehole_temp"),
        "AWS2-GroundTemperatureP2": ("GEM_DISKO_AWS2_P2", "Disko, Arctic Station AWS2 profile P2",
                                     69.25349, -53.51413, "borehole_temp"),
        "ZEROCALM-2 upper_Soil_temperature": ("GEM_ZC2_UPPER", "Zackenberg ZEROCALM-2 upper profile",
                                              74.46571, -20.56367, "soil_temp"),
        "ZEROCALM-2 middle_Soil_temperature": ("GEM_ZC2_MIDDLE", "Zackenberg ZEROCALM-2 middle profile",
                                               74.46526, -20.56447, "soil_temp"),
        "ZEROCALM-2_lower_Soil_temperature": ("GEM_ZC2_LOWER", "Zackenberg ZEROCALM-2 lower profile",
                                              74.46505, -20.56554, "soil_temp"),
    }
    for start, (sid, name, lat, lon, method) in profiles.items():
        z = next((zips[n] for n in zips if n.startswith(start)), None)
        if z is None:
            continue
        d = _gem_table(z)
        # depth from the column name: '..._60cm_..', '.. 100 cm_P2..', 'temperature_25_cm_..'
        depth = {}
        for c in d.columns:
            m = re.search(r"(?<![+\d])(\d+(?:\.\d+)?)\s*_?cm", c)
            if m and "+" not in c and c not in ("Date", "Time"):
                depth[c] = float(m.group(1))
        depth = {c: z_ for c, z_ in depth.items() if z_ > 0}
        if len(depth) < 2:
            continue
        d["time"] = pd.to_datetime(d.Date, errors="coerce").dt.normalize()
        t = d.melt(id_vars=["time"], value_vars=list(depth), var_name="c", value_name="temp_c")
        t["depth_cm"] = t.c.map(depth)
        t["temp_c"] = pd.to_numeric(t.temp_c, errors="coerce")
        t = t.groupby(["time", "depth_cm"], as_index=False).temp_c.mean()      # daily means
        t["source_site_id"] = sid
        site = pd.DataFrame([dict(source_site_id=sid, site_name=name, lat=lat, lon=lon,
                                  country="Greenland", source_file=z.name)])
        out.append(records_from_profiles(t, site, method))
    if not out:
        return empty_records()
    return finish(pd.concat(out, ignore_index=True), "gem")


# =============================================================================
# readers kept from the corridor project
# =============================================================================
def read_above_field(raw: Path) -> pd.DataFrame:
    """
    NASA ABoVE soil thaw depth validation table (ORNL DAAC, ABoVE_Soil_ThawDepth_
    Moisture_Validation_V2.csv). Point probes from many teams. Only repeat-visited plots
    can give a time series: points are grouped by team + site name, survey mean per date,
    then end-of-season value per year. Probe-length caps are flagged as censored.
    """
    found = files(raw, "ABoVE_Soil_ThawDepth_Moisture_Validation*.csv")
    if not found:
        return empty_records()
    f = [x for x in found if re.search(r"_V\d+\.csv$", x.name)] or found
    f = f[-1]
    d = pd.read_csv(f, usecols=["site_name", "plot", "point", "team_name", "latitude",
                                "longitude", "date", "ALT_instrument", "ALT"],
                    na_values=["NA", "-9999", "-999"], low_memory=False)
    d = d[d.ALT.notna()].drop_duplicates()
    d = d[d.ALT_instrument.astype(str).str.upper() != "GPR"]   # probes only
    d["team_name"] = d.team_name.astype(str).str.strip("'\" ")
    d["obs_date"] = pd.to_datetime(d.date, errors="coerce")
    d = d[d.obs_date.notna()]
    grp = d.team_name + "|" + d.site_name.astype(str)
    # probe-length cap: the season maximum repeated by >= 3 points
    mx = d.groupby([grp, d.obs_date.dt.year]).ALT.transform("max")
    at = d.ALT >= mx - 1e-6
    n_at = at.groupby([grp, d.obs_date.dt.year]).transform("sum")
    d["cap"] = at & (n_at >= 3) & (mx >= 80)
    s = d.groupby([grp.rename("gid"), "obs_date"]).agg(
        alt_cm=("ALT", "mean"), alt_sd_cm=("ALT", "std"), n_points=("ALT", "size"),
        cens=("cap", "mean"), lat=("latitude", "mean"), lon=("longitude", "mean"),
        site_name=("site_name", "first")).reset_index()
    years = s.groupby("gid").obs_date.apply(lambda x: x.dt.year.nunique())
    s = s[s.gid.isin(years[years >= 3].index)]        # one-off plots cannot form a series
    s["source_site_id"] = "ABOVE_" + s.gid.str.replace(r"[^A-Za-z0-9]+", "_", regex=True)
    s["censored"] = s.cens >= 0.2
    s["flag"] = np.where(s.alt_cm > 150, "deep_thaw_check_talik", "")
    s["method"], s["country"] = "probe", np.where(s.lon < -141, "USA", "Canada")
    out = end_of_season(s, "source_site_id")
    out["source_file"] = f.name
    return finish(out, "above_field")


# =============================================================================
# Sources added in November 2026 (fetched in stage 1 through their APIs)
# =============================================================================
# Their file layouts were not all seen before this reader was written. Tables a reader
# cannot read, or that have no station position, are listed in the build log and in
# unread_tables.csv, so a specific reader can be added; they never enter the data.
UNREAD_TABLES: list[dict] = []

_DEPTH_RE = re.compile(r"([-+]?\d+(?:[.,]\d+)?)\s*(cm|m)?\b", re.I)


def _depth_from_name(name: str):
    """
    Depth (cm, or a unitless number) from a column name such as '0.2', '-0.50 m',
    'T_20cm', '1.5m', '0.2_meters', 'Temp_TMC_+000.50m', 'TP1_150cm'. A number with a
    unit wins over one without (so 'TP1_2cm' is 2 cm, not 1); without a unit the last
    number is taken. None when the column is not a depth.
    """
    s = str(name).strip().replace(",", ".").replace("_", " ")
    s = re.sub(r"(?i)(centi)?met(er|re)s?\b", lambda m: "cm" if m.group(1) else "m", s)
    if re.search(r"(date|time|year|month|day|hour|flag|qf|qual|id|lat|lon|air|surf|snow)", s, re.I):
        return None
    found = list(_DEPTH_RE.finditer(s))
    if not found:
        return None
    m = next((f for f in found if f.group(2)), found[-1])
    v = abs(float(m.group(1)))
    unit = (m.group(2) or "").lower()
    return v if unit == "cm" else v * 100 if unit == "m" else v


def _depth_has_unit(name) -> bool:
    """True when a depth column name carries its unit ('0.2_meters', '20cm', '+000.50m')."""
    s = str(name).replace(",", ".").replace("_", " ")
    s = re.sub(r"(?i)(centi)?met(er|re)s?\b", lambda m: "cm" if m.group(1) else "m", s)
    return any(f.group(2) for f in _DEPTH_RE.finditer(s))


def temperature_long(df: pd.DataFrame, unit_hint: str | None = None) -> pd.DataFrame | None:
    """
    A temperature table, wide (a time column and one column per depth) or long (time,
    depth, temperature), as a long table time, depth_cm, temp_c. Unitless depths are
    metres when every depth is <= 30, else cm (unit_hint 'm' or 'cm' overrides). Values
    outside -60..40 C are no-data codes. None when the layout is not recognised.
    """
    if df.columns.duplicated().any():
        # a repeated column name (e.g. two sensors labelled with the same depth) would
        # make df[name] a table: number the repeats ('0.5', '0.5 #2'); same depth, and
        # the readers average the readings of one depth
        seen: dict = {}
        names = []
        for c in df.columns:
            seen[c] = seen.get(c, 0) + 1
            names.append(c if seen[c] == 1 else f"{c} #{seen[c]}")
        df = df.copy()
        df.columns = names
    cols = {str(c).lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in cols if c in ("time", "datetime", "date_time", "timestamp",
                                                 "date", "startdatetime")), None)
    tcol = tcol or next((cols[c] for c in cols if "time" in c or "date" in c), None)
    if tcol is None:
        return None
    t = pd.to_datetime(df[tcol], errors="coerce")
    if t.notna().mean() < 0.9:
        return None
    dcol = next((cols[c] for c in cols if c.startswith("depth")), None)
    vcol = next((cols[c] for c in cols if c.startswith(("temp", "value", "gt", "ground"))), None)
    if dcol is not None and vcol is not None:
        d = pd.to_numeric(df[dcol], errors="coerce").abs()
        unit = unit_hint or ("m" if d.max() <= 30 else "cm")
        long = pd.DataFrame({"time": t, "depth_cm": d * (100 if unit == "m" else 1),
                             "temp_c": pd.to_numeric(df[vcol], errors="coerce")})
    else:
        depth = {c: _depth_from_name(c) for c in df.columns if c != tcol}
        depth = {c: v for c, v in depth.items()
                 if v is not None and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.1}
        if len(depth) < 2:
            return None
        unitless = [c for c in depth if not _depth_has_unit(c)]
        if unitless:
            unit = unit_hint or ("m" if max(depth[c] for c in unitless) <= 30 else "cm")
            for c in unitless:
                depth[c] *= 100 if unit == "m" else 1
        long = df[[tcol] + list(depth)].melt(id_vars=tcol, var_name="col", value_name="temp_c")
        long["depth_cm"] = long["col"].map(depth)
        long["time"] = pd.to_datetime(long[tcol], errors="coerce")
        long["temp_c"] = pd.to_numeric(long["temp_c"], errors="coerce")
    long = long[["time", "depth_cm", "temp_c"]].dropna()
    return long[long.temp_c.between(-60, 40)]


def _promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """Spreadsheet read without a header: the header is the first row that is mostly text."""
    for i in range(min(40, len(df) - 1)):
        row = df.iloc[i]
        if row.notna().sum() >= 3 and row.astype(str).str.contains(r"[A-Za-z]").mean() > 0.5:
            body = df.iloc[i + 1:].copy()
            body.columns = [str(c) for c in row.fillna("").astype(str)]
            return body.reset_index(drop=True)
    return df


def read_any_table(path: Path) -> list[tuple[str, pd.DataFrame]]:
    """Every table of a file as (sheet or member name, table): spreadsheets give one per
    sheet, zips one per table inside, text files are read from the first row with at
    least three separators."""
    suf = path.suffix.lower()
    if suf in (".xls", ".xlsx"):
        return [(n, _promote_header(d))
                for n, d in pd.read_excel(path, sheet_name=None, header=None).items()]
    if suf == ".zip":
        out = []
        with zipfile.ZipFile(path) as zf:
            for n in zf.namelist():
                if n.lower().endswith((".csv", ".txt", ".dat", ".tab")):
                    out += [(f"{n}", d) for _, d in _text_tables(zf.read(n).decode("utf-8", "replace"))]
        return out
    return _text_tables(path.read_text(encoding="utf-8", errors="replace"))


def _text_tables(text: str) -> list[tuple[str, pd.DataFrame]]:
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    for sep in (",", "\t", ";", r"\s+"):
        hdr = next((i for i, ln in enumerate(lines[:200])
                    if len(re.split(sep, ln.strip())) >= 3 and not ln.startswith("/*")), None)
        if hdr is None:
            continue
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=sep, engine="python",
                             on_bad_lines="skip")
        except (pd.errors.ParserError, ValueError):
            continue
        if df.shape[1] >= 3:
            return [("", df)]
    return []


def _unread(source: str, file: str, why: str, columns="") -> None:
    UNREAD_TABLES.append(dict(source=source, file=file, reason=why, columns=str(columns)[:500]))


def _profiles_from_folder(raw, folder: str, source: str, position, method="borehole_temp",
                          unit_hint=None, special=None, series_id=None) -> pd.DataFrame:
    """
    Every table under raw/<folder> that is a temperature profile with at least two
    sensors in the top CONFIG max_shallow_cm becomes one series; ALT by the 0 C crossing
    rule (records_from_profiles). position(path, sheet) returns (name, lat, lon, country)
    or None when the position of that table is not known (the table is then not used).
    special(path) may read a file itself: it returns a list of (name, long table, position)
    (an empty list for files that are metadata, not data), or None to use the generic
    reading. Sub-daily values are reduced to daily means, as the other temperature
    sources are daily means. series_id(site_name) may give a stable id for a series (e.g.
    its station code); otherwise the series is named after its first file.
    """
    temps, sites = [], []

    def add(rel, long, pos):
        key = re.sub(r"\W+", "_", rel)
        temps.append(long.assign(source_site_id=key))
        sites.append(dict(source_site_id=key, site_name=pos[0], lat=pos[1], lon=pos[2],
                          country=pos[3], source_file=rel))

    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        base = Path(root) / folder
        for p in sorted(base.rglob("*")) if base.exists() else []:
            if not p.is_file() or p.suffix.lower() not in TABLE_EXT + (".zip",):
                continue
            if special is not None:
                try:
                    got = special(p)
                except Exception as e:
                    _unread(source, str(p.relative_to(base)), f"cannot read: {type(e).__name__}: {e}"[:200])
                    continue
                if isinstance(got, str):                # a file that cannot give ALT, and why
                    _unread(source, str(p.relative_to(base)), got)
                    continue
                if got is not None:
                    for name, long, pos in got:
                        rel = f"{p.relative_to(base)}{(':' + name) if name else ''}"
                        if pos is None:
                            _unread(source, rel, "no station position found")
                        elif (long.depth_cm.drop_duplicates() <= CONFIG["max_shallow_cm"]).sum() >= 2:
                            add(rel, long, pos)
                    continue
            try:
                tabs = read_any_table(p)
            except Exception as e:
                _unread(source, str(p.relative_to(base)), f"cannot open: {e}"[:200])
                continue
            for sheet, df in tabs:
                rel = f"{p.relative_to(base)}{(':' + sheet) if sheet else ''}"
                lc = [str(c).lower() for c in df.columns]
                if any(c.startswith("depth") for c in lc) and not any(
                        ("date" in c or "time" in c) and df[df.columns[i]].nunique() > 3
                        for i, c in enumerate(lc)):
                    # one logged profile (depth, temperature) of a single day: cannot give ALT
                    _unread(source, rel, "single-date profile (cannot give ALT)")
                    continue
                try:
                    long = temperature_long(df, unit_hint)
                except Exception as e:          # listed in unread_tables.csv, not lost silently
                    _unread(source, rel, f"cannot read: {type(e).__name__}: {e}"[:200],
                            "|".join(map(str, df.columns[:30])))
                    continue
                if long is None:
                    n = sum((_depth_from_name(c) or 0) > 0 for c in df.columns)
                    why = "fewer than two sensors (cannot give ALT)" if n == 1 else "layout not recognised"
                    _unread(source, rel, why, "|".join(map(str, df.columns[:30])))
                    continue
                if (long.depth_cm.drop_duplicates() <= CONFIG["max_shallow_cm"]).sum() < 2:
                    continue                            # too deep to resolve ALT: not an error
                pos = position(p, sheet)
                if pos is None:
                    _unread(source, rel, "no station position found")
                    continue
                add(rel, long, pos)
    if not temps:
        return empty_records()
    t = pd.concat(temps, ignore_index=True)
    t["time"] = pd.to_datetime(t.time).dt.normalize()
    # several tables of one station (one per year) are one series: same name and position
    s = pd.DataFrame(sites)
    s["name_key"] = s.site_name.map(_name_key)      # 'inter boil' = 'interboil'
    s["series"] = s.groupby(["name_key", "lat", "lon"]).source_site_id.transform("first")
    if series_id is not None:
        first = s.drop_duplicates("series")
        new, used = {}, set()
        for ser, name in zip(first.series, first.site_name):
            sid = series_id(name) or ser
            if sid in used:                 # one code at two positions: keep them apart
                sid = ser
            used.add(sid)
            new[ser] = sid
        s["series"] = s.series.map(new)
    t["source_site_id"] = t.source_site_id.map(s.set_index("source_site_id").series)
    s = (s.drop_duplicates("series").drop(columns=["source_site_id", "name_key"])
          .rename(columns={"series": "source_site_id"}))
    t = t.groupby(["source_site_id", "time", "depth_cm"], as_index=False).temp_c.mean()
    return finish(records_from_profiles(t, s, method, monthly=method.endswith("monthly")), source)


def _eml_positions(xml: str) -> list[tuple[str, float, float]]:
    """(description, lat, lon) of each geographic coverage of an EML file."""
    out = []
    for m in re.finditer(r"<geographicDescription>(.*?)</geographicDescription>.*?"
                         r"<westBoundingCoordinate>(.*?)</west.*?<northBoundingCoordinate>(.*?)</north",
                         xml, re.S):
        out.append((re.sub(r"\s+", " ", m.group(1)).strip(), as_float(m.group(3)), as_float(m.group(2))))
    return out


def _match_position(name: str, coverages) -> tuple[str, float, float] | None:
    """The coverage whose description names the station of a file: the longest station
    code or name word (>= 3 characters) shared by both."""
    toks = {t for t in re.split(r"[\W_]+", name.lower()) if len(t) >= 3 and not t.isdigit()}
    best = None
    for desc, lat, lon in coverages:
        words = set(re.split(r"[\W_]+", desc.lower()))
        hit = max((len(t) for t in toks if t in words), default=0)
        if hit and (best is None or hit > best[0]):
            best = (hit, desc, lat, lon)
    return best[1:] if best else None


def read_gtnp_platform(raw) -> pd.DataFrame:
    """
    GTN-P data platform boreholes (data.gtn-p.org; files from download_gtnp_platform):
    daily ground temperatures, ALT by the 0 C crossing rule. Position, name and country
    from the borehole's metadata; boreholes the platform marks as disturbed are flagged
    'disturbed_site'. The same boreholes can be in other sources (USGS, UAF, GSC, Canada
    collection): the build joins them by position and agreement.
    """
    temps, sites = [], []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        folder = Path(root) / API_FOLDERS["gtnp"]
        country = {}
        lst = folder / "list_sites_borehole.json"
        if lst.exists():
            for s in json.loads(lst.read_text()):
                for b in s.get("boreholes", []):
                    country[b["id"]] = s.get("country_name") or ""
        meta_of = {}
        for m in sorted((folder / "pt_meta").glob("*.json")) if folder.exists() else []:
            meta = json.loads(m.read_text())
            for d in meta.get("datasets", []):
                meta_of[str(d["id"])] = (int(m.stem), meta)
        for z in sorted((folder / "pt_data").glob("pt_*.zip")) if folder.exists() else []:
            dsid = z.stem.split("_", 1)[1]
            if dsid not in meta_of:
                _unread("gtnp_platform", z.name, "no metadata for this dataset")
                continue
            bid, meta = meta_of[dsid]
            parts = []
            for name, df in read_any_table(z):
                # GTN-P gives depths in metres; without the hint, a borehole deeper than
                # 30 m would have its unitless depths read as cm
                long = temperature_long(df, unit_hint="m")
                if long is None:
                    _unread("gtnp_platform", f"{z.name}:{name}", "layout not recognised",
                            "|".join(map(str, df.columns[:30])))
                else:
                    parts.append(long)
            if not parts:
                continue
            long = pd.concat(parts)
            if (long.depth_cm.drop_duplicates() <= CONFIG["max_shallow_cm"]).sum() < 2:
                continue
            loc = meta.get("location") or {}
            dist = json.dumps(meta.get("disturbance") or "").lower()
            key = f"GTNP_PT_{bid}_{dsid}"
            temps.append(long.assign(source_site_id=key))
            sites.append(dict(source_site_id=key, site_name=meta.get("name") or f"GTN-P borehole {bid}",
                              lat=loc.get("latitude"), lon=loc.get("longitude"),
                              country=country.get(bid, ""), source_file=z.name,
                              site_note=f"GTN-P borehole {bid}; disturbance: {dist[:80]}",
                              disturbed=any(w in dist for w in ("disturbed", "anthropogenic",
                                                                "fire", "burn", "construction"))))
    if not temps:
        return empty_records()
    s = pd.DataFrame(sites)
    a = records_from_profiles(pd.concat(temps, ignore_index=True), s.drop(columns="disturbed"),
                              "borehole_temp")
    bad = a.source_site_id.isin(s.source_site_id[s.disturbed])
    a["flag"] = add_flag(a.flag, bad, "disturbed_site")
    return finish(a, "gtnp_platform")


def read_gtnp_platform_alt(raw) -> pd.DataFrame:
    """
    GTN-P data platform active-layer datasets (grids and transects, mostly CALM): node
    readings -> survey means -> end-of-season value, as for every other grid. The CALM
    code from the metadata joins them to the CALM tables (PANGAEA 972777); their yearly
    differences are in check_source_agreement.csv. Values above 5 are taken as cm,
    tables whose largest value is 5 or less as metres.
    """
    pts = []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        folder = Path(root) / API_FOLDERS["gtnp"]
        meta_of = {}
        for m in sorted((folder / "alt_meta").glob("*.json")) if folder.exists() else []:
            meta = json.loads(m.read_text())
            for d in meta.get("datasets", []):
                meta_of[str(d["id"])] = (int(m.stem), meta)
        for z in sorted((folder / "alt_data").glob("alt_*.zip")) if folder.exists() else []:
            dsid = z.stem.split("_", 1)[1]
            if dsid not in meta_of:
                _unread("gtnp_platform_alt", z.name, "no metadata for this dataset")
                continue
            aid, meta = meta_of[dsid]
            loc = meta.get("location") or {}
            for name, df in read_any_table(z):
                cols = {str(c).lower(): c for c in df.columns}
                tcol = next((cols[c] for c in cols if "date" in c or "time" in c), None)
                vcol = next((cols[c] for c in cols
                             if c.startswith(("alt", "value", "thaw", "thickness", "depth"))), None)
                if tcol is None or vcol is None:
                    _unread("gtnp_platform_alt", f"{z.name}:{name}", "layout not recognised",
                            "|".join(map(str, df.columns[:30])))
                    continue
                v = pd.to_numeric(df[vcol], errors="coerce")
                v = v.where(v > -100)                                   # -999 / -888 codes
                if v.max() <= 5:
                    v = v * 100
                pts.append(pd.DataFrame(dict(
                    source_site_id=f"GTNP_ALT_{aid}_{dsid}", obs_date=df[tcol], alt_cm=v.abs(),
                    site_name=meta.get("name") or "", calm_code=meta.get("calm_code") or "",
                    lat=loc.get("latitude"), lon=loc.get("longitude"),
                    country=meta.get("country_name") or "", method="probe", source_file=z.name)))
    if not pts:
        return empty_records()
    return finish(end_of_season(survey_means(pd.concat(pts, ignore_index=True))), "gtnp_platform_alt")


def _code_tokens(s: str) -> tuple:
    """'US_A09_012' and 'US_A09_012_2016_06_01_daily.csv' -> ('us','a','09','012',...);
    'ru13_0001_12' -> ('ru','13','0001','12'); 'U22' -> ('u','22'). Letters and digits are
    split, so station codes match file names whatever their separators."""
    s = re.sub(r"\.(csv|txt|dat|xlsx?|tab)$", "", str(s).strip().lstrip("﻿").lower())
    s = re.sub(r"([a-z])(\d)", r"\1_\2", s)
    return tuple(t for t in re.split(r"[^a-z0-9]+", s) if t)


_COUNTRY_CODES = {"US": "USA", "USA": "USA", "CA": "Canada", "CANADA": "Canada", "RU": "Russia",
                  "RUSSIA": "Russia", "GL": "Greenland", "NO": "Norway", "SE": "Sweden",
                  "FI": "Finland", "MN": "Mongolia", "KZ": "Kazakhstan"}


def _tsp_site_tables(base: Path) -> pd.DataFrame:
    """
    Every station list in the Arctic Data Center packages: tables with a latitude and a
    longitude column and a station code, file name or station name column (the yearly
    SitesRoster files of the TSP packages, site_and_file_metadata.csv, Site_Information.csv,
    the Russian metadata.csv, cadis_list.xls, network_sampling_sites.csv). One row per
    listed station or file: code, file and name tokens, name, lat, lon, country.
    """
    rows = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".csv", ".xls", ".xlsx"):
            continue
        try:
            head = (pd.read_excel(p, header=None, nrows=3) if p.suffix.lower() != ".csv"
                    else pd.read_csv(p, header=None, nrows=3, encoding="utf-8-sig",
                                     on_bad_lines="skip", engine="python"))
        except Exception:
            continue
        text = " ".join(map(str, head.values.ravel())).lower()
        if "lat" not in text or "lon" not in text:
            continue
        for _, df in read_any_table(p):
            cols = {str(c).strip().lstrip("﻿").lower(): c for c in df.columns}
            lat = next((cols[c] for c in cols if c.startswith("lat")), None)
            lon = next((cols[c] for c in cols if c.startswith(("lon", "long"))), None)
            code = next((cols[c] for c in cols if c in ("sitecode", "sitecode_new", "site code",
                                                        "borehole id")), None)
            fcol = next((cols[c] for c in cols if c in ("filename", "file name", "filename_startdate")), None)
            ncol = next((cols[c] for c in cols if c in ("sitename", "site name", "borehole name")), None)
            ccol = next((cols[c] for c in cols if c in ("sitecountry", "country")), None)
            if lat is None or lon is None or (code is None and fcol is None and ncol is None):
                continue
            df = df.reset_index(drop=True)
            la = pd.to_numeric(df[lat], errors="coerce")
            lo = pd.to_numeric(df[lon], errors="coerce")
            ok = la.between(-90, 90) & lo.between(-180, 180)
            keys = [c for c in (code, fcol, ncol) if c is not None]
            d = df[ok].assign(lat_=la[ok], lon_=lo[ok]).drop_duplicates(keys + ["lat_", "lon_"])
            for i in d.index:
                rows.append(dict(
                    code=str(d.at[i, code]).strip() if code is not None else "",
                    file=_code_tokens(d.at[i, fcol]) if fcol is not None else (),
                    name=str(d.at[i, ncol]).strip().rstrip(" *") if ncol is not None else "",
                    lat=float(d.at[i, "lat_"]), lon=float(d.at[i, "lon_"]),
                    country=_COUNTRY_CODES.get(str(d.at[i, ccol]).strip().upper(), "")
                    if ccol is not None else "", table=p.name))
    t = pd.DataFrame(rows, columns=["code", "file", "name", "lat", "lon", "country", "table"])
    t.loc[t["code"].str.upper().isin(["NAN", "N/A", "NONE"]), "code"] = ""
    t.loc[t["name"].str.lower().isin(["nan", "n/a"]), "name"] = ""
    t["code_tokens"] = t["code"].map(_code_tokens)
    t["name_tokens"] = t["name"].map(_code_tokens)
    return t


def _tsp_locator(base: Path):
    """
    position(path) for a TSP data file: by its file name in a station list, else by the
    longest listed station code that its name starts with (US_A09_012_2016_..._daily.csv
    -> US_A09_012; ru_03_0008_16.csv -> RU_03_0008; u22_05.csv -> U22), else by a
    station name equal to its name (Kugurak_Cabin.csv -> 'Kugurak Cabin'), else by the one
    station whose name words contain the file's name words or are contained in them
    (Selawik_Thaw_Slump.csv -> 'Selawik ...'; only when all such stations are within 1 km).
    A station listed in several years gets its median position, so its files form one
    series.
    """
    t = _tsp_site_tables(base)
    by_code = {}
    for k, g in t[t["code_tokens"].map(len) > 0].groupby("code_tokens"):
        names = [n for n in g["name"] if n and n.lower() != "nan"]
        label = f"{names[0]} ({g['code'].iloc[0].upper()})" if names else g["code"].iloc[0].upper()
        by_code[k] = (label, g["lat"].median(), g["lon"].median(),
                      next((c for c in g["country"] if c), ""))
    by_file = {r.file: r for r in t[t["file"].map(len) > 0].itertuples()}
    by_name = {}
    for k, g in t[t["name_tokens"].map(len) > 0].groupby("name_tokens"):
        by_name[k] = (g["name"].iloc[0], g["lat"].median(), g["lon"].median(),
                      next((c for c in g["country"] if c), ""))

    def country_of(cty, lon):
        if cty:
            return cty
        return "Russia" if (lon > 0 or lon < -168.5) else "USA" if lon < -141 else ""

    def position(p: Path, sheet: str = ""):
        tok = _code_tokens(p.name)
        r = by_file.get(tok)
        if r is not None:
            if r.code_tokens in by_code:
                return by_code[r.code_tokens]
            return (r.name or p.stem, r.lat, r.lon, country_of(r.country, r.lon))
        best = max((k for k in by_code if tok[:len(k)] == k), key=len, default=None)
        if best is not None:
            n, la, lo, c = by_code[best]
            return (n, la, lo, country_of(c, lo))
        if tok in by_name:
            n, la, lo, c = by_name[tok]
            return (n, la, lo, country_of(c, lo))
        words = _name_words(tok)
        if words:
            hits = [v for k, v in by_name.items()
                    if (kw := _name_words(k)) and (kw <= words or words <= kw)]
            if hits and max(haversine_km(hits[0][1], hits[0][2], h[1], h[2]) for h in hits) <= 1.0:
                n, la, lo, c = hits[0]
                return (n, la, lo, country_of(c, lo))
        return None

    return position, t


# words that say nothing about which station a file belongs to
_GENERIC_WORDS = {"daily", "hourly", "data", "temp", "temps", "temperature", "temperatures", "soil",
                  "ground", "borehole", "boreholes", "site", "sites", "station", "csv", "xls", "xlsx",
                  "the", "and", "all", "final", "raw", "qc", "level", "nsidc", "cadis", "arcdc"}


def _name_words(tokens) -> set:
    """The distinctive words of a name or file name (tokens from _code_tokens): letters
    only, at least 3 characters, not generic."""
    return {t for t in tokens if t.isalpha() and len(t) >= 3 and t not in _GENERIC_WORDS}


def read_tsp_alaska(raw) -> pd.DataFrame:
    """
    Arctic Data Center permafrost-observatory packages (V. Romanovsky and the TSP network;
    files from download_tsp_alaska): daily ground temperatures of the continuous TSP
    stations (US_*, CA_* daily files), the Russian stations (ru_*), the UAF borehole and
    station files (u##, SL#, ...) and the network_sampling_sites table. ALT by the 0 C
    crossing rule, only where at least two sensors are in the top CONFIG max_shallow_cm.
      Positions: the packages' EML gives only a region, so stations are placed from the
      station lists in the packages (_tsp_locator). A file that cannot be placed is listed
      in unread_tables.csv and not used.
      Not data: station lists, single-day logged profiles (u##_##_NSIDC, the 'occasional'
      TSP files: one day, often from 6 m down) and monthly compilations without dates.
      UAF station files (title lines, a 'Location:' line, then daily rows; e.g.
      SL1_07.xls) are read as in read_uaf_stations (_uaf_station_file).
      A station placed by its code keeps that code as its id (e.g. US_A09_012), so the
      id stays the same when new years are published.
    """
    out = []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        base = Path(root) / API_FOLDERS["tsp_alaska"]
        if not base.exists():
            continue
        position, lists = _tsp_locator(base)
        list_files = set(lists.table)
        log(f"    tsp_alaska: {len(lists)} station-list rows from {len(list_files)} tables")

        def special(p: Path):
            low = p.name.lower()
            if p.name in list_files and "network_sampling_sites" not in low:
                return []                                   # a station list, not data
            got = _uaf_station_file(p, position)
            if got is not None:
                return got
            if p.suffix.lower() == ".csv":
                with open(p, encoding="utf-8-sig", errors="replace") as f:
                    head = re.sub(r"[^a-z,]", "", f.readline().lower()).split(",")
                if "sitecode" in head and "depth" in head and ("temp" in head or "temperature" in head):
                    return _long_station_table(p, position)
            if "network_sampling_sites" in low:
                # one long daily table: siteCode, siteName, latitude, longitude, temp3cm ...
                d = pd.read_csv(p, encoding="utf-8-sig", na_values=["N/A", "NA"])
                got = []
                for code, g in d.groupby("siteCode"):
                    keep = ["date"] + [c for c in g.columns if re.fullmatch(r"temp\d+cm", c)]
                    long = temperature_long(g[keep].reset_index(drop=True), "cm")
                    if long is not None:
                        got.append((str(code), long, (f"{g.siteName.iloc[0]} ({code})",
                                                     g.latitude.median(), g.longitude.median(), "USA")))
                return got
            return None

        out.append(_profiles_from_folder(base.parent, base.name, "tsp_alaska", position,
                                         special=special, series_id=_station_code))
    out = [o for o in out if len(o)]
    return pd.concat(out, ignore_index=True) if out else empty_records()


def _long_station_table(p: Path, position):
    """
    A long table of several stations (e.g. pf_borehole_daily_temps_nirpo_2021_2024.csv:
    Date, Site.code, Depth, Temp): one series per station code, placed by its code in the
    station lists. Depths have no unit: metres when the deepest is <= 30, else cm (as in
    temperature_long); the sign is ignored (negative = below the surface).
    """
    d = pd.read_csv(p, encoding="utf-8-sig")
    c = {re.sub(r"[^a-z]", "", str(k).lower()): k for k in d.columns}
    code_c, dep_c = c["sitecode"], c["depth"]
    tmp_c = c.get("temp") or c.get("temperature")
    date_c = next((c[k] for k in ("date", "timestamp", "datetime", "time") if k in c), None)
    if date_c is None:
        return "layout not recognised"
    got = []
    for code, g in d.groupby(code_c):
        dep = pd.to_numeric(g[dep_c], errors="coerce").abs()
        long = pd.DataFrame(dict(time=pd.to_datetime(g[date_c], errors="coerce"),
                                 depth_cm=dep * 100 if dep.max() <= 30 else dep,
                                 temp_c=pd.to_numeric(g[tmp_c], errors="coerce"))).dropna()
        got.append((str(code), long, position(Path(f"{code}.csv"))))
    return got


def _station_code(site_name: str):
    """A stable series id from a station's name:
      'APP 2009-012 (US_A09_012)' -> 'US_A09_012' (the code a station list gave, see
      _tsp_locator); 'Chandalar Shelf (UAF CS1)' -> 'UAF_CS1' (a UAF site page);
      'UAF West Dock' -> 'UAF_WEST_DOCK' (the name line of a UAF station file).
    None without a code or name."""
    s = str(site_name)
    if s.startswith("UAF "):
        return re.sub(r"\W+", "_", s).strip("_").upper()
    m = re.search(r"\((UAF )?([A-Za-z0-9_]+)\)\s*$", s)
    if not m:
        return None
    return ("UAF_" if m.group(1) else "") + m.group(2).upper()


def _grid(p: Path) -> pd.DataFrame:
    """A table read as it is, without a header (title lines kept): the first sheet of a
    workbook, or a csv split at its commas."""
    if p.suffix.lower() in (".xls", ".xlsx"):
        return pd.read_excel(p, header=None, sheet_name=0)
    rows = list(csv.reader(io.StringIO(p.read_text(encoding="utf-8-sig", errors="replace"))))
    width = max((len(r) for r in rows), default=0)
    return pd.DataFrame([r + [None] * (width - len(r)) for r in rows])


def _uaf_station_file(p: Path, position):
    """
    A UAF Permafrost Lab station file (workbook, or the same layout as csv): title lines,
    a 'Location:' line, then daily rows (uaf_station_sheet). Returns [(name, long table,
    position)]; a reason (str) for a single-date log ('Data date:' and a 'Depth, m'
    column: one borehole log, which cannot give ALT); or None when the file has another
    layout. The position is the file's own 'Location:' line, else position(p) (the
    station lists). A header more than CONFIG station_typo_km from the median header of
    the same station's other files in the folder (or, without 3 such files, from the
    station lists) has a typo, and that reference position is used: WD1_06.xls gives
    N 70.166, the latitude of Deadhorse, for West Dock at 70.38 N.
    """
    if p.suffix.lower() not in (".xls", ".xlsx", ".csv"):
        return None
    if p.suffix.lower() == ".csv":
        with open(p, encoding="utf-8-sig", errors="replace") as f:
            head = " ".join(f.readline() for _ in range(8)).lower()
        if "location" not in head:
            return None                               # an ordinary table: generic reader
    x = _grid(p)
    top = " ".join(str(v) for v in x.iloc[:20, :2].values.ravel() if pd.notna(v)).lower()
    if "data date" in top and re.search(r"depth,?\s*m", top):
        return "single-date profile (cannot give ALT)"
    pos, long = uaf_station_sheet(x)
    if long is None:
        return None
    listed = position(p)
    if pos is None:
        return [("", long, listed)]
    name = _uaf_station_name(x, p)
    cty = "USA" if pos[1] < -141 else ""
    # reference: the median header of this station's files in the folder (3 or more),
    # else the station lists
    same = _station_headers(p.parent).get(_name_key(name))
    ref = (f"its {same[2]} files in {p.parent.name}", same[0], same[1]) if same else (
        (f"the station lists ({listed[0]})", listed[1], listed[2]) if listed else None)
    if ref is not None and haversine_km(pos[0], pos[1], ref[1], ref[2]) > CONFIG["station_typo_km"]:
        key = (round(pos[0], 4), round(pos[1], 4), ref[0])
        if key not in _TYPO_LOGGED:                   # once per header position and station
            _TYPO_LOGGED.add(key)
            log(f"    {p.name}: header position {pos[0]:.4f}, {pos[1]:.4f} is "
                f"{haversine_km(pos[0], pos[1], ref[1], ref[2]):.0f} km from {ref[0]}; "
                f"the file is placed at {ref[1]:.4f}, {ref[2]:.4f} instead of its header")
        # the file keeps its own station name, so it joins the station's other files
        return [("", long, (f"UAF {name}", round(ref[1], 4), round(ref[2], 4), cty))]
    return [("", long, (f"UAF {name}", round(pos[0], 4), round(pos[1], 4), cty))]


def _name_key(name: str) -> str:
    """A station name for matching: letters and digits only, lower case ('New inter boil'
    and 'New interboil' are the same station)."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_HEADERS = {}


def _station_headers(folder: Path) -> dict:
    """{name key: (median lat, median lon, number of files)} of the UAF station files in a
    folder that give a 'Location:' line, for stations with at least 3 such files. Only
    the first rows of each file are read."""
    if folder in _HEADERS:
        return _HEADERS[folder]
    pts = {}
    for q in sorted(folder.iterdir()):
        try:
            if q.suffix.lower() in (".xls", ".xlsx"):
                g = pd.read_excel(q, header=None, nrows=8, sheet_name=0)
            elif q.suffix.lower() == ".csv":
                with open(q, encoding="utf-8-sig", errors="replace") as f:
                    lines = [f.readline() for _ in range(8)]
                rows = list(csv.reader(lines))
                width = max((len(r) for r in rows), default=0)
                g = pd.DataFrame([r + [None] * (width - len(r)) for r in rows])
            else:
                continue
        except Exception:
            continue
        rows = [" ".join(str(c) for c in r if pd.notna(c)) for r in g.itertuples(index=False)]
        if not any("location" in r.lower() for r in rows):
            continue
        pos = _location_from_rows(rows)
        if pos is not None:
            pts.setdefault(_name_key(_uaf_station_name(g, q)), []).append(pos)
    _HEADERS[folder] = {k: (float(np.median([a for a, _ in v])), float(np.median([b for _, b in v])), len(v))
                        for k, v in pts.items() if len(v) >= 3}
    return _HEADERS[folder]


def _uaf_station_name(x: pd.DataFrame, p: Path) -> str:
    """The station name of a UAF station file: the first cell of the first rows that is
    not a title, position or site-description line (usually row 2: 'West Dock'), else
    the station code that starts the file name (SL4_07.xls -> SL4)."""
    for i in range(min(4, len(x))):
        v = x.iat[i, 0]
        t = "" if pd.isna(v) else str(v).strip()
        if t and t.lower() != "nan" and not re.search(
                r"location|temperature|elevation|slope|aspect|date|depth", t, re.I):
            return t
    m = re.match(r"[A-Za-z]+\d*", p.stem)
    return m.group(0).upper() if m else p.stem


_TSP_LOCATORS = {}
_TYPO_LOGGED = set()


def read_uaf_sites(raw) -> pd.DataFrame:
    """
    UAF Permafrost Lab site files (Galbraith Lake, West Dock, Happy Valley, Chandalar
    Shelf; files from download_uaf_sites). ALT as for read_tsp_alaska. Position, in order:
      - the file's own 'Location:' line (station files, workbook or csv);
      - the station code of the file name in the Arctic Data Center station lists
        (US_CHS_102_..._daily.csv; _tsp_locator), which name the borehole;
      - the site page ('Latitude' and 'Longitude' text), which gives one position for
        the site.
    Most of these stations are also in the Arctic Data Center packages (tsp_alaska); the
    build joins the copies.
    """
    roots = raw if isinstance(raw, (list, tuple)) else [raw]
    locators = []
    for r in roots:
        b = Path(r) / API_FOLDERS["tsp_alaska"]
        if b.exists():
            if b not in _TSP_LOCATORS:
                _TSP_LOCATORS[b] = _tsp_locator(b)[0]
            locators.append(_TSP_LOCATORS[b])

    def page_position(p: Path):
        page = next((q / "site_page.html" for q in p.parents if (q / "site_page.html").exists()), None)
        if page is None:
            return None
        html = re.sub(r"<[^>]+>", " ", page.read_text(encoding="utf-8", errors="replace"))
        la = re.search(r"Lat(?:itude)?\.?\s*[:=]?\s*(-?\d{2}\.\d+)", html, re.I)
        lo = re.search(r"Lon(?:gitude)?\.?\s*[:=]?\s*(-?\d{2,3}\.\d+)", html, re.I)
        if not (la and lo):
            return None
        code = page.parent.name
        return (f"{API['uaf_sites'].get(code, code)} (UAF {code.upper()})", float(la.group(1)),
                -abs(float(lo.group(1))), "USA")                # all four sites are in Alaska (W)

    def position(p: Path, sheet: str = ""):
        for loc in locators:
            q = loc(p)
            if q is not None:
                return q
        return page_position(p)

    return _profiles_from_folder(raw, API_FOLDERS["uaf_sites"], "uaf_sites", position,
                                 special=lambda p: _uaf_station_file(p, position),
                                 series_id=_station_code)


def read_neon(raw) -> pd.DataFrame:
    """
    NEON soil temperature (DP1.00041.001) at Utqiagvik (BARR), Toolik (TOOL) and Healy
    (HEAL): daily mean per plot and depth (files from download_neon). Depth from the
    sensor positions (|zOffset|). Frost heave moves NEON sensors, so a plot-year (1
    January to 1 July of the next year, i.e. the thaw season and the refreeze winter
    the 0 C rule checks) is not used when
      - any sensor of that plot changed depth by more than CONFIG neon_move_m in that
        period (sensor_positions; a new position row with the same depth is a
        re-survey, not a move), or
      - a change-log entry of the product about heave names the site and plot and
        overlaps the period.
    Rejected plot-years are flagged 'sensor_moved' and not used.
    """
    temps, sites, moved = [], [], []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        folder = Path(root) / API_FOLDERS["neon"]
        if not (folder / "product.json").exists():
            continue
        prod = json.loads((folder / "product.json").read_text())
        heave = [c for c in prod.get("changeLogs") or [] if "heav" in (c.get("issue") or "").lower()]
        for site in API["neon_sites"]:
            f = folder / f"{site}_daily_mean.csv"
            if not f.exists():
                continue
            daily = pd.read_csv(f, parse_dates=["date"])
            pos = pd.read_csv(folder / f"{site}_sensor_positions.csv", dtype={"HOR.VER": str})
            pos = pos[pos["siteID"] == site] if "siteID" in pos else pos
            # neonutilities names the plot position locationReferenceLatitude/-Longitude
            # (older versions: referenceLatitude/-Longitude)
            la_col = next((c for c in ("locationReferenceLatitude", "referenceLatitude") if c in pos), None)
            lo_col = next((c for c in ("locationReferenceLongitude", "referenceLongitude") if c in pos), None)
            hv = pos["HOR.VER"].astype(str).str.split(".", expand=True)
            pos["hor"], pos["ver"] = hv[0].astype(int), hv[1].astype(int)
            pos["start"] = pd.to_datetime(pos["positionStartDateTime"], utc=True).dt.tz_localize(None)
            pos["end"] = pd.to_datetime(pos["positionEndDateTime"], errors="coerce", utc=True).dt.tz_localize(None)
            daily["hor"], daily["ver"] = daily.horizontalPosition.astype(int), daily.verticalPosition.astype(int)
            for hor, g in daily.groupby("hor"):
                p = pos[pos.hor == hor]
                key = f"NEON_{site}_{hor:03d}"
                # depth of each record from the position valid on its date; zOffset is
                # negative below the reference surface, and a sensor heaved above it
                # (zOffset >= 0) is not measuring soil, so its values are left out
                below = p[pd.to_numeric(p.zOffset, errors="coerce") < 0]
                g = g.merge(below[["ver", "start", "end", "zOffset"]], on="ver")
                g = g[(g.date >= g.start) & (g.end.isna() | (g.date < g.end))]
                temps.append(pd.DataFrame(dict(source_site_id=key, time=g.date,
                                               depth_cm=-g.zOffset * 100, temp_c=g.soilTempMean)))
                sites.append(dict(source_site_id=key, site_name=f"NEON {site} soil plot {hor}",
                                  lat=pd.to_numeric(p[la_col], errors="coerce").median() if la_col else np.nan,
                                  lon=pd.to_numeric(p[lo_col], errors="coerce").median() if lo_col else np.nan,
                                  country="USA", source_file=f.name))
                for year in sorted(g.date.dt.year.unique()):
                    y0, y1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 7, 1)
                    pp = p[(p.start < y1) & (p.end.isna() | (p.end > y0))]
                    z = pd.to_numeric(pp.zOffset, errors="coerce")
                    if (z.groupby(pp.ver).agg(lambda v: v.max() - v.min()) > CONFIG["neon_move_m"]).any() or any(
                            _neon_heave_hit(c, site, hor, y0, y1) for c in heave):
                        moved.append((key, year))
    if not temps:
        return empty_records()
    a = records_from_profiles(pd.concat(temps, ignore_index=True), pd.DataFrame(sites), "soil_temp")
    bad = pd.Series(list(zip(a.source_site_id, a.year))).isin(set(moved)).to_numpy()
    a["flag"] = add_flag(a.flag, bad, "sensor_moved")
    return finish(a, "neon")


def _neon_heave_hit(c, site, hor, y0, y1) -> bool:
    """True when a heave change-log entry names this site and plot and overlaps [y0, y1)."""
    loc = c.get("locationAffected") or ""
    if site not in loc:
        return False
    plots = set()
    for a, b in re.findall(r"plots?\s*(\d)(?:\s*-\s*(\d))?", loc, re.I):
        plots |= set(range(int(a), int(b or a) + 1))
    plots |= {int(p) for p in re.findall(r"HOR[^0-9]*0*(\d)", loc)}
    if plots and hor not in plots:
        return False
    a = pd.to_datetime(c.get("dateRangeStart"), errors="coerce", utc=True)
    b = pd.to_datetime(c.get("dateRangeEnd"), errors="coerce", utc=True)
    a = a.tz_localize(None) if pd.notna(a) else a
    b = b.tz_localize(None) if pd.notna(b) else b
    return (pd.isna(a) or a < y1) and (pd.isna(b) or b > y0)


def _syrdakh_file(p: Path):
    """A Syrdakh database csv: (header dict, data table). The header is 'key,value' lines
    up to '# last row of header/meta_data'; the data follow with their own header."""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    end = next((i for i, ln in enumerate(lines) if ln.startswith("# last row of header")), None)
    if end is None:
        return None, None
    head = {}
    for ln in lines[:end]:
        if not ln.startswith("#") and "," in ln:
            k, v = ln.split(",", 1)
            head[k.strip()] = v.strip().strip(",").strip()
    data = pd.read_csv(io.StringIO("\n".join(lines[end + 1:])))
    return head, data


def read_syrdakh(raw) -> pd.DataFrame:
    """
    Syrdakh observatory, central Yakutia (Zenodo 14619854; files from download_syrdakh).
    Every csv has a 'key,value' header (variable_id, unit, source_id, time_start, ...)
    and then its data.
      Ground temperature: one file per sensor ('tsl0100' = 100 cm; time, value, QF; QF 0
        = good). The sensors of one logger (source_id, with the deployment years taken
        out of the name) form one profile; daily means; ALT by the 0 C crossing rule.
      Thaw depth ('td' files, one per survey, late September): one value per point
        (hobo-F1, drill-2012-J1-AL1, ...) with its method (rod, drill, pit, tube);
        max_exceed = 1 means deeper than the tool (censored).
    The files and the Zenodo record give no coordinates, so every series gets the centre
    of the study area (CONFIG syrdakh_position) and a note; the build then keeps
    different loggers and points apart (one source), and colocated_with lists them.
    """
    lat, lon = CONFIG["syrdakh_position"]
    note = "position: centre of the Syrdakh study area (the files give none)"
    temps, sites, td = [], {}, []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        base = Path(root) / API_FOLDERS["syrdakh"]
        for p in sorted(base.rglob("*.csv")) if base.exists() else []:
            try:
                head, d = _syrdakh_file(p)
            except Exception as e:
                _unread("syrdakh", p.name, f"cannot read: {type(e).__name__}: {e}"[:200])
                continue
            if head is None:
                continue
            var = head.get("variable_id", "")
            m = re.fullmatch(r"tsl(\d{4})", var)
            if m and int(m.group(1)) > 0 and {"time", "value"} <= set(d.columns):
                d = d[d.get("QF", 0) == 0]
                logger = re.sub(r"[-_]?(19|20)\d{2}(?=[-_]|$)", "", head.get("source_id", p.stem)).strip("-_")
                key = f"SYRDAKH_{logger}"
                temps.append(pd.DataFrame(dict(source_site_id=key, depth_cm=float(m.group(1)),
                                               time=pd.to_datetime(d.time, errors="coerce"),
                                               temp_c=pd.to_numeric(d.value, errors="coerce"))))
                sites[key] = dict(source_site_id=key, site_name=f"Syrdakh logger {logger}", lat=lat,
                                  lon=lon, country="Russia", source_file="ground/temp", site_note=note)
            elif var == "td" and {"point", "value"} <= set(d.columns):
                date = pd.to_datetime(head.get("time_start"), errors="coerce")
                meth = d.get("method", pd.Series("", index=d.index)).astype(str).str.upper()
                td.append(pd.DataFrame(dict(
                    source_site_id="SYRDAKH_TD_" + d.point.astype(str), site_name="Syrdakh " + d.point.astype(str),
                    lat=lat, lon=lon, country="Russia", year=date.year if pd.notna(date) else np.nan,
                    obs_date=date, alt_cm=pd.to_numeric(d.value, errors="coerce"),
                    censored=pd.to_numeric(d.get("max_exceed", 0), errors="coerce").fillna(0).astype(bool),
                    method=np.where(meth.str.contains("TUBE"), "thaw_tube", "probe"),
                    n_points=1, site_note=note + "; method " + meth.str.lower(), source_file=p.name)))
    out = []
    if temps:
        t = pd.concat(temps, ignore_index=True).dropna()
        t["time"] = t.time.dt.normalize()
        t = t.groupby(["source_site_id", "time", "depth_cm"], as_index=False).temp_c.mean()
        out.append(records_from_profiles(t, pd.DataFrame(sites.values()), "soil_temp"))
    if td:
        d = pd.concat(td, ignore_index=True)
        d["flag"] = add_flag(pd.Series("", index=d.index), d.censored, "lower_bound")
        out.append(d)
    out = [o for o in out if len(o)]
    return finish(pd.concat(out, ignore_index=True), "syrdakh") if out else empty_records()


def _frost_stations(raw) -> set[str]:
    """Station numbers (SNxxxxx) downloaded from Frost (download_frost)."""
    return {f.stem.replace("_daily_mean", "") for root in (raw if isinstance(raw, (list, tuple)) else [raw])
            for f in (Path(root) / API_FOLDERS["metno_frost"]).glob("*_daily_mean.csv")}


def read_metno(raw) -> pd.DataFrame:
    """MET Norway permafrost stations (files from download_metno): daily mean per
    depth; ALT by the 0 C crossing rule. Most stations start in 2019 (Janssonhaugen Vest
    SN99874), so they reach the year threshold only from about 2027. A station also
    downloaded from Frost is read there (read_metno_frost), not here."""
    temps, sites = [], []
    frost = _frost_stations(raw)
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        for f in sorted((Path(root) / API_FOLDERS["metno"]).glob("*_daily_mean.csv")):
            d = pd.read_csv(f, parse_dates=["date"])
            sn = re.search(r"SN\d+", f"{f.stem} {d.title.iloc[0] if len(d) else ''}")
            if sn and sn.group(0) in frost:
                continue
            key = f"METNO_{f.stem.replace('_daily_mean', '')}"
            temps.append(pd.DataFrame(dict(source_site_id=key, time=d.date,
                                           depth_cm=d.depth_m.abs() * 100, temp_c=d.temp_c)))
            sites.append(dict(source_site_id=key, site_name=str(d.title.iloc[0]), lat=d.lat.iloc[0],
                              lon=d.lon.iloc[0], country="Norway", source_file=f.name))
    if not temps:
        return empty_records()
    return finish(records_from_profiles(pd.concat(temps, ignore_index=True), pd.DataFrame(sites),
                                        "soil_temp"), "metno")


def read_metno_frost(raw) -> pd.DataFrame:
    """MET Norway stations on permafrost from the Frost API (files from download_frost):
    daily mean per depth; ALT by the 0 C crossing rule. Method 'borehole_temp' when the
    station has a sensor deeper than 2 m (a permafrost borehole), else 'soil_temp'."""
    temps, sites = [], []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        for f in sorted((Path(root) / API_FOLDERS["metno_frost"]).glob("*_daily_mean.csv")):
            d = pd.read_csv(f, parse_dates=["date"])
            if d.empty:
                continue
            key = f"METNO_{f.stem.replace('_daily_mean', '')}"
            temps.append(pd.DataFrame(dict(source_site_id=key, time=d.date,
                                           depth_cm=d.depth_m.abs() * 100, temp_c=d.temp_c)))
            country = "Norway (Svalbard)" if 74 <= d.lat.iloc[0] <= 81.5 else "Norway"
            sites.append(dict(source_site_id=key, site_name=str(d.title.iloc[0]), lat=d.lat.iloc[0],
                              lon=d.lon.iloc[0], country=country, source_file=f.name,
                              calm_code=CONFIG["frost_calm_code"].get(f.stem.replace("_daily_mean", ""), ""),
                              deep=d.depth_m.abs().max() > 2.0))
    if not temps:
        return empty_records()
    sites = pd.DataFrame(sites)
    out = []
    for deep, method in [(True, "borehole_temp"), (False, "soil_temp")]:
        s = sites[sites.deep == deep].drop(columns="deep")
        t = pd.concat(temps, ignore_index=True)
        t = t[t.source_site_id.isin(s.source_site_id)]
        if len(t):
            out.append(records_from_profiles(t, s, method))
    out = [o for o in out if len(o)]
    if not out:
        return empty_records()
    rec = finish(pd.concat(out, ignore_index=True), "metno_frost")
    chk = rec.source_site_id.isin(CONFIG["check_only_series"].get("metno_frost", []))
    rec["flag"] = add_flag(rec.flag, chk, "check_only")
    return rec


def read_metno_janssonhaugen(raw) -> pd.DataFrame:
    """
    Janssonhaugen, Adventdalen, Svalbard: the shallow (15 m) borehole P11 next to the 102 m
    PACE borehole, sensors at 20-500 cm, 6-hourly, April 1999 to 18 August 2025 (logger and
    thermistor string upgraded in August 2025). Sent to the user by K. Isaksen (MET Norway)
    by email on 10 November 2026; cite Isaksen et al. 2000, 2001 and 2022. He notes that
    this borehole is not affected by the concrete lid on top of the deep borehole (Frost
    SN99875, kept as check_only), so it suits near-surface temperatures better.
    File P11_Janssonhaugen-*.txt: ';'-separated, header YEAR;MONTH;DAY;TIME;X1TJ<depth cm>,
    TIME = hour (0, 6, 12, 18). Values outside -60..40 C are removed, readings averaged per
    day (at least 2 of 4), then the 0 C rule gives ALT. Joins CALM N1 by its CALM code;
    CALM N1 is this borehole's ALT, and CALM's values stay first. A later file (data after
    the upgrade) can be added to the same folder: a day found in two files counts once,
    the newer file's value kept.
    """
    parts = []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        base = Path(root) / API_FOLDERS["metno_janssonhaugen"]
        for f in sorted(base.rglob("P11_Janssonhaugen*.txt")) if base.exists() else []:
            d = pd.read_csv(f, sep=";")
            d.columns = [str(c).strip().upper() for c in d.columns]
            tcols = [c for c in d.columns if re.fullmatch(r"X1TJ\d+", c)]
            if not {"YEAR", "MONTH", "DAY", "TIME"} <= set(d.columns) or not tcols:
                _unread("metno_janssonhaugen", f.name, "expected YEAR;MONTH;DAY;TIME;X1TJ<cm> columns")
                continue
            d["time"] = (pd.to_datetime(dict(year=d.YEAR, month=d.MONTH, day=d.DAY), errors="coerce")
                         + pd.to_timedelta(pd.to_numeric(d.TIME, errors="coerce"), unit="h"))
            long = d.melt(id_vars="time", value_vars=tcols, var_name="col", value_name="temp_c")
            long["temp_c"] = pd.to_numeric(long.temp_c, errors="coerce")
            long.loc[~long.temp_c.between(-60, 40), "temp_c"] = np.nan
            long["depth_cm"] = long.col.str[4:].astype(float)
            long["file"] = f.name
            parts.append(long.dropna(subset=["time", "temp_c"]))
    if not parts:
        return empty_records()
    t = (pd.concat(parts, ignore_index=True).sort_values("file")
           .drop_duplicates(["time", "depth_cm"], keep="last"))
    t["day"] = t.time.dt.normalize()
    daily = t.groupby(["day", "depth_cm"]).temp_c.agg(["mean", "count"]).reset_index()
    daily = daily[daily["count"] >= 2]
    key = "METNO_JANSSONHAUGEN_P11"
    temps = pd.DataFrame(dict(source_site_id=key, time=daily.day, depth_cm=daily.depth_cm,
                              temp_c=daily["mean"]))
    site = pd.DataFrame([dict(source_site_id=key, site_name="Janssonhaugen shallow borehole P11 (15 m), "
                              "Adventdalen, Svalbard", lat=78.1794, lon=16.4669, country="Norway (Svalbard)",
                              calm_code="N1", site_note="15 m borehole next to the 102 m PACE borehole, "
                              "270 m a.s.l. (Isaksen et al. 2001); the borehole behind CALM N1",
                              source_file=f"{API_FOLDERS['metno_janssonhaugen']}/"
                                          + ", ".join(sorted(t.file.unique())))])
    rec = records_from_profiles(temps, site, "borehole_temp")
    return finish(rec, "metno_janssonhaugen") if len(rec) else empty_records()


# Tarfala PACE: position of the 100 m borehole Ta-1 and the 15 m borehole Ta-2 (GTN-P
# database; the data files give no coordinates). Ridge at about 1550 m a.s.l.
TARFALA_PACE = {
    "100m": dict(file="PACE_100_m_thermistor_string.csv", lat=67.916791, lon=18.63382,
                 depths_m=[0.2, 0.4, 0.8, 1.2, 1.6, 2, 2.5, 3, 3.5, 4, 5],
                 name="Tarfala PACE borehole Ta-1 (100 m, Tarfalaryggen)"),
    "15m": dict(file="PACE_15_m_thermistor_string.csv", lat=67.916678, lon=18.63352,
                depths_m=[0.2, 0.4, 0.8, 1.2, 1.6, 2, 2.5, 3, 3.5, 4, 5, 7, 9, 10, 11, 13, 15],
                name="Tarfala PACE borehole Ta-2 (15 m, Tarfalaryggen)"),
}


def read_tarfala_pace(raw) -> pd.DataFrame:
    """
    Tarfala PACE boreholes, Tarfalaryggen, northern Sweden (Stockholm University, figshare
    doi:10.17045/sthlmuni.3171646, 2000-2014). Two boreholes about 18 m apart (100 m and
    15 m deep), each with its own logger and thermistor string; 6-hourly readings, no
    header: time, battery, logger temperature, two control channels, then one column per
    thermistor (depths in TARFALA_PACE, from the data description). Logger error codes
    (-273, -231, -113, -106 C) are removed (anything outside -60..40 C). Readings are
    averaged per day (at least 2 of the 4 readings), then the 0 C rule gives ALT.
    The two boreholes are one site, and two series of one source are never joined, so only
    the borehole with more usable years is kept (the other is reported in the log).
    """
    out = []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        base = Path(root) / API_FOLDERS["tarfala_pace"]
        for hole, info in TARFALA_PACE.items():
            for f in sorted(base.rglob(info["file"])) if base.exists() else []:
                d = pd.read_csv(f, header=None, na_values=["NaN", "nan", ""])
                n = len(info["depths_m"])
                if d.shape[1] != 5 + n:
                    _unread("tarfala_pace", f.name, f"expected {5 + n} columns, found {d.shape[1]}")
                    continue
                d.columns = ["time", "batt", "tlogger", "c1", "c2"] + [f"t{i}" for i in range(n)]
                d["time"] = pd.to_datetime(d.time, errors="coerce")
                long = d.melt(id_vars="time", value_vars=[f"t{i}" for i in range(n)],
                              var_name="col", value_name="temp_c")
                long["temp_c"] = pd.to_numeric(long.temp_c, errors="coerce")
                long.loc[~long.temp_c.between(-60, 40), "temp_c"] = np.nan
                long["depth_cm"] = long.col.map({f"t{i}": round(z * 100, 1)
                                                 for i, z in enumerate(info["depths_m"])})
                long["day"] = long.time.dt.normalize()
                daily = (long.dropna(subset=["temp_c", "day"])
                             .groupby(["day", "depth_cm"]).temp_c.agg(["mean", "count"]).reset_index())
                daily = daily[daily["count"] >= 2]
                key = f"TARFALA_PACE_{hole}"
                t = pd.DataFrame(dict(source_site_id=key, time=daily.day, depth_cm=daily.depth_cm,
                                      temp_c=daily["mean"]))
                site = pd.DataFrame([dict(source_site_id=key, site_name=info["name"], lat=info["lat"],
                                          lon=info["lon"], country="Sweden",
                                          site_note="mountain permafrost, about 1550 m a.s.l.; PACE borehole",
                                          source_file=f"{f.parent.name}/{f.name}")])
                r = records_from_profiles(t, site, "borehole_temp")
                if len(r):
                    out.append(r)
    if not out:
        return empty_records()
    rec = finish(pd.concat(out, ignore_index=True), "tarfala_pace")
    if rec.empty:
        return rec
    ok = ~rec.flag.str.contains(INVALID_FLAGS) & ~rec.censored     # as counted in the build
    n_ok = ok.groupby(rec.source_site_id).sum().reindex(rec.source_site_id.unique(), fill_value=0)
    keep = n_ok.idxmax()
    log("  tarfala_pace: usable years per borehole " + ", ".join(f"{k} {int(v)}" for k, v in n_ok.items())
        + f"; kept {keep}")
    return rec[rec.source_site_id == keep].reset_index(drop=True)


# University of Oslo mountain permafrost boreholes in Norway and Iceland (Etzelmüller et
# al. 2023, NIRD doi:10.11582/2023.00128). The files give no coordinates: positions from
# the GTN-P database (NORPERM for juvP and tr1), elevation from Table 1 of the paper.
UIO_BOREHOLES = {
    "isk1": ("Iskoras BH1", "Norway", 69.3023, 25.3364, 585),
    "isk2": ("Iskoras BH2", "Norway", 69.3004, 25.3460, 591),
    "gu1": ("Guolasjavri", "Norway", 69.3537, 21.2106, 780),
    "jet1": ("Jetta", "Norway", 61.9008, 9.2847, 1560),
    "juv1": ("Juvflye BH1", "Norway", 61.6761, 8.3650, 1851),
    "juv3": ("Juvflye BH3 (Dugurdskampen)", "Norway", 61.6969, 8.3858, 1561),
    "juv4": ("Juvflye BH4", "Norway", 61.6998, 8.3851, 1547),
    "juv5": ("Juvflye BH5", "Norway", 61.7006, 8.3921, 1468),
    "juvp": ("Juvvasshoe PACE borehole", "Norway", 61.6776, 8.3691, 1894),
    "tr1": ("Tronfjell", "Norway", 62.1740, 10.7027, 1640),
    "hag": ("Hagongur", "Iceland", 64.5563, -18.2416, 899),
    "sau": ("Saudafell", "Iceland", 64.8836, -15.5022, 906),
    "vop": ("Vopnafjordur", "Iceland", 65.6775, -14.5161, 892),
    "gag": ("Gagnheidi", "Iceland", 65.2243, -14.2596, 931),
}


def _read_uio_dat(path: Path) -> pd.DataFrame:
    """One UiO borehole file: a header line ('Date' or 'depth', then the sensor depths in m,
    negative down), a line of '=', then one line per day. The date is always the first 11
    characters (dd-Mon-yyyy); values follow in 14-character right-aligned columns and can
    touch the date, so the rest of the line is split on blanks. Lines with the wrong
    number of values are skipped."""
    lines = path.read_text(encoding="latin-1").splitlines()
    depths = [abs(float(x)) for x in lines[0].split()[1:]]
    rows, bad = [], 0
    for ln in lines[2:]:
        if len(ln) < 12:
            continue
        vals = ln[11:].split()
        if len(vals) != len(depths):
            bad += 1
            continue
        rows.append([ln[:11]] + vals)
    if bad:
        log(f"    {path.name}: {bad} lines with a wrong number of values skipped")
    d = pd.DataFrame(rows, columns=["date"] + [f"{z:g}" for z in depths])
    d["date"] = pd.to_datetime(d.date, format="%d-%b-%Y", errors="coerce")
    long = d.melt(id_vars="date", var_name="depth_m", value_name="temp_c")
    long["temp_c"] = pd.to_numeric(long.temp_c, errors="coerce")
    long["depth_cm"] = (long.depth_m.astype(float) * 100).round(1)
    long.loc[~long.temp_c.between(-60, 40), "temp_c"] = np.nan
    return long.dropna(subset=["date", "temp_c"]).rename(columns={"date": "time"})[["time", "depth_cm", "temp_c"]]


def read_uio_boreholes(raw) -> pd.DataFrame:
    """
    UiO mountain permafrost boreholes in Norway and Iceland (Etzelmüller et al. 2023,
    NIRD doi:10.11582/2023.00128; daily ground temperatures, 1999-2023). ALT by the 0 C
    rule. Four of them are also served by Frost (isk2 = SN97710, juv1 = SN15265, juv3 =
    SN15290, juvP = SN15270): the build joins them by position and agreement (rule c), and
    juvP joins CALM N2 by its CALM code like SN15270 (CONFIG uio_calm_code). Ground and
    air surface files (*_gst, *_sat) are not read.
    """
    temps, sites = [], []
    for f in files(raw, "*.dat"):
        if API_FOLDERS["uio_boreholes"] not in f.parts or re.search(r"_(gst|sat)$", f.stem, re.I):
            continue
        code = f.stem.lower()
        if code not in UIO_BOREHOLES:
            _unread("uio_boreholes", f.name, "not in UIO_BOREHOLES")
            continue
        name, country, lat, lon, elev = UIO_BOREHOLES[code]
        key = f"UIO_{f.stem}"
        t = _read_uio_dat(f)
        if t.empty:
            continue
        temps.append(t.assign(source_site_id=key))
        sites.append(dict(source_site_id=key, site_name=name, lat=lat, lon=lon, country=country,
                          site_note=f"mountain permafrost borehole, {elev} m a.s.l. (University of Oslo)",
                          calm_code=CONFIG["uio_calm_code"].get(f.stem, ""),
                          source_file=f"{f.parent.name}/{f.name}"))
    if not temps:
        return empty_records()
    rec = records_from_profiles(pd.concat(temps, ignore_index=True), pd.DataFrame(sites), "borehole_temp")
    if rec.empty:
        return empty_records()
    rec = finish(rec, "uio_boreholes")
    bad = pd.Series(False, index=rec.index)
    for sid, periods in CONFIG["sensor_problem_years"]["uio_boreholes"].items():
        for y0, y1 in periods:
            bad |= (rec.source_site_id == sid) & rec.year.between(y0, y1)
    rec["flag"] = add_flag(rec.flag, bad, "sensor_problem")
    return rec


# Roshydromet daily soil temperature to 320 cm: station catalogue of the dataset
# (Каталог_Температура_почвы_до_320м.pdf, public on meteo.ru). The data export gives only
# the WMO index, so positions come from here.
RUSOIL_STATIONS = """\
wmo,name_ru,name,lat,lon,elev_m,first_year,last_year
20069,О.Визе,O.Vize,79.49,76.98,10,1989,2019
20087,О.Голомянный,O.Golomyannyy,79.55,90.61,8,1989,2021
20292,Им.Е.К.Федорова,Im.E.K.Fedorova,77.71,104.29,15,1977,2021
20674,О.Диксон,O.Dikson,73.51,80.41,46,1977,2022
21921,Кюсюр,Kyusyur,70.68,127.40,33,1977,2000
22217,Кандалакша,Kandalaksha,67.17,32.35,25,1963,2022
22235,Краснощелье,Krasnoshchele,67.35,37.05,157,1965,2022
22249,Каневка,Kanevka,67.13,39.66,151,1977,2022
22292,Индига,Indiga,67.68,48.68,4,1976,2022
22408,Калевала,Kalevala,65.21,31.15,120,1976,2022
22438,"Жижгин,остров",Zhizhgin ostrov,65.20,36.81,17,1965,2022
22471,Мезень,Mezen,65.86,44.21,19,1964,2022
22550,Архангельск,Arkhangelsk,64.50,40.73,8,1965,2022
22583,Койнас,Koynas,64.75,47.65,64,1963,2022
22602,Реболы,Reboly,63.83,30.81,182,1963,2022
22641,Онега,Onega,63.90,38.11,13,1963,2022
22676,Сура,Sura,63.58,45.63,62,1976,2022
22768,Шенкурск,Shenkursk,62.10,42.90,45,1976,2022
22802,Сортавала,Sortavala,61.71,30.71,19,1963,2022
22820,Петрозаводск,Petrozavodsk,61.82,34.27,111,1963,2022
22837,Вытегра,Vytegra,61.01,36.45,56,1963,2022
22845,Каргополь,Kargopol,61.51,38.93,126,1976,2022
22981,Великий Устюг,Velikiy Ustyug,60.76,46.30,99,1976,2004
23205,Нарьян-Мар,Naryan-Mar,67.63,53.02,12,1986,2022
23219,Хоседа-Хард,Khoseda-Khard,67.08,59.38,84,1963,1997
23274,Игарка,Igarka,67.46,86.56,31,1977,2022
23324,Петрунь,Petrun,66.43,60.76,62,1976,2022
23383,Агата,Agata,66.88,93.46,278,1977,1993
23405,Усть-Цильма,Ust-Tsilma,65.43,52.26,68,1963,2022
23412,Усть-Уса,Ust-Usa,65.96,56.91,78,1976,2022
23418,Печора,Pechora,65.13,57.14,59,1963,2022
23463,Янов Стан,Yanov Stan,65.98,84.26,40,1977,1993
23472,"Туруханск,ГМО",Turukhansk GMO,65.79,87.99,38,1963,2022
23527,Саранпауль,Saranpaul,64.28,60.88,29,1978,2022
23552,Тарко-Сале,Tarko-Sale,64.91,77.81,27,1978,2022
23631,Березово,Berezovo,63.93,65.05,32,1963,2022
23656,Халесовая,Khalesovaya,63.38,78.32,67,1978,1993
23662,Толька,Tolka,63.98,82.08,33,1978,2022
23678,Верхнеимбатск,Verkhneimbatsk,63.15,87.95,46,1977,1997
23711,Троицко-Печорское,Troitsko-Pechorskoe,62.70,56.20,139,1963,2022
23724,Няксимволь,Nyaksimvol,62.43,60.86,51,1963,2022
23734,Октябрьское,Oktyabrskoe,62.45,66.05,72,1978,2022
23804,Сыктывкар,Syktyvkar,61.68,50.78,119,1967,2022
23849,"Сургут,АМСГ",Surgut AMSG,61.25,73.50,56,1963,1978
23867,Ларьяк,Laryak,61.10,80.25,57,1978,2022
23884,Бор,Bor,61.58,90.03,58,1977,2022
23891,Байкит,Baykit,61.66,96.36,262,1963,1994
23921,Ивдель,Ivdel,60.68,60.45,94,1963,2022
23933,Ханты-Мансийск,Khanty-Mansiysk,61.03,69.12,44,1963,2022
23955,Александровское,Aleksandrovskoe,60.43,77.87,49,1963,2022
23986,Северо-Енисейский,Severo-Eniseyskiy,60.38,93.33,519,1977,2019
24125,Оленек,Olenek,68.50,112.43,220,1966,2017
24136,Сухана,Sukhana,68.61,118.33,77,1977,2002
24143,Джарджан,Dzhardzhan,68.73,124.00,39,1977,2022
24266,Верхоянск,Verkhoyansk,67.57,133.40,136,1964,2022
24343,Жиганск,Zhigansk,66.77,123.40,92,1964,2017
24382,Усть-Мома,Ust-Moma,66.45,143.23,196,1977,2022
24507,Тура,Tura,64.27,100.22,191,1963,1997
24606,Кислокан,Kislokan,63.58,103.96,211,1977,1999
24641,Вилюйск,Vilyuysk,63.78,121.62,111,1964,2022
24688,Оймякон,Oymyakon,63.25,143.15,740,1963,2022
24738,Сунтар,Suntar,62.15,117.65,133,1964,2022
24790,Сусуман,Susuman,62.78,148.16,655,1977,2022
24802,Стрелка Чуня,Strelka Chunya,61.75,102.80,358,1977,1999
24817,Ербогачен,Erbogachen,61.26,108.01,291,1963,2022
24908,Ванавара,Vanavara,60.34,102.29,260,1963,2022
24923,Ленск,Lensk,60.71,114.88,234,1985,2022
24944,Олекминск,Olekminsk,60.40,120.42,230,1964,2022
24951,Исить,Isit,60.81,125.31,118,1964,2022
24959,Якутск,Yakutsk,62.02,129.72,101,1964,2022
24966,Усть-Мая,Ust-Maya,60.38,134.45,170,1964,2022
25138,Островное,Ostrovnoe,68.11,164.16,94,1977,2022
25206,Среднеколымск,Srednekolymsk,67.45,153.71,21,1977,2009
25400,Зырянка,Zyryanka,65.73,150.90,43,1977,2022
25428,Омолон,Omolon,65.23,160.53,260,1988,2022
25551,Марково,Markovo,64.68,170.41,26,1963,2022
25705,Среднекан,Srednekan,62.45,152.31,260,1977,2005
25913,"Магадан,Нагаева бухта",Magadan Nagaeva bukhta,59.55,150.78,116,1963,2022
26063,Санкт-Петербург,Sankt-Peterburg,59.97,30.30,6,1963,2022
26069,Белогорка,Belogorka,59.35,30.13,90,1976,2022
26094,Тихвин,Tikhvin,59.65,33.55,61,1976,2022
26258,Псков,Pskov,57.81,28.41,45,1993,2022
26298,Бологое,Bologoe,57.88,34.03,187,1976,2022
26359,Пушкинские Горы,Pushkinskie Gory,57.01,28.90,108,1976,2022
26477,Великие Луки,Velikie Luki,56.35,30.62,103,1963,2022
26479,Торопец,Toropets,56.48,31.63,186,1976,2022
26781,Смоленск,Smolensk,54.75,32.07,238,1963,2022
26898,"Брянск,АМСГ",Bryansk AMSG,53.25,34.31,216,1963,2022
27037,"Вологда,Прилуки",Vologda Priluki,59.32,39.93,130,1971,2022
27051,Тотьма,Totma,59.88,42.75,134,1976,2022
27066,Никольск,Nikolsk,59.53,45.46,143,1976,2022
27164,Кологрив,Kologriv,58.82,44.32,151,1963,2022
27296,Кумены,Kumeny,58.10,49.92,177,1984,2022
27333,Кострома,Kostroma,57.73,40.78,126,1963,2022
27485,Йошкар-Ола,Yoshkar-Ola,56.70,47.91,106,1963,2022
27509,Можайск,Mozhaysk,55.51,36.00,185,1976,2022
27625,Коломна,Kolomna,55.13,38.73,114,1976,2022
27648,Елатьма,Elatma,54.95,41.76,136,1963,2022
27675,Порецкое,Poretskoe,55.18,46.33,136,1967,2022
27707,Сухиничи,Sukhinichi,54.10,35.35,238,1964,2022
27823,Павелец,Pavelets,53.78,39.25,209,1963,2022
27857,Земетчино,Zemetchino,53.48,42.63,131,1977,2022
27947,Тамбов,Tambov,52.80,41.33,128,1963,2022
27962,"Пенза,ГМО",Penza GMO,53.12,45.02,172,1977,2022
27995,Безенчук,Bezenchuk,52.98,49.43,45,1977,2022
28064,Леуши,Leushi,59.61,65.71,72,1963,2022
28138,Бисер,Biser,58.51,58.85,463,1963,2022
28224,Пермь,Perm,58.01,56.30,171,1963,2022
28255,Туринск,Turinsk,58.05,63.68,103,1978,2013
28275,Тобольск,Tobolsk,58.15,68.25,47,1963,2022
28367,Тюмень,Tyumen,57.11,65.43,102,1978,2022
28411,Ижевск,Izhevsk,56.83,53.45,159,1963,1989
28418,Сарапул,Sarapul,56.46,53.73,133,1963,2022
28434,Красноуфимск,Krasnoufimsk,56.65,57.78,206,1963,2022
28445,Верхнее дуброво,Verkhnee dubrovo,56.73,61.07,288,1965,2022
28493,Тара,Tara,56.90,74.38,73,1963,2022
28537,Дуван,Duvan,55.70,57.90,338,1967,2022
28552,Шадринск,Shadrinsk,56.06,63.65,89,1966,2022
28573,Ишим,Ishim,56.10,69.43,82,1978,2022
28630,Златоуст,Zlatoust,55.20,59.66,538,1963,2022
28661,"Курган,Вороновка",Kurgan Voronovka,55.47,65.42,74,1963,2022
28666,Макушино,Makushino,55.25,67.30,140,1965,2022
28698,"Омск,степная",Omsk stepnaya,55.01,73.38,122,1963,2022
28704,Чулпаново,Chulpanovo,54.50,50.41,78,1977,2022
28825,Стерлитамак,Sterlitamak,53.58,56.00,137,1963,2022
28833,Верхнеуральск,Verkhneuralsk,53.88,59.20,403,1963,2013
29023,Напас,Napas,59.85,81.95,76,1963,2022
29111,Средний Васюган,Sredniy Vasyugan,59.21,78.23,69,1963,2022
29231,Колпашево,Kolpashevo,58.32,82.95,75,1963,2022
29263,Енисейск,Eniseysk,58.45,92.20,79,1963,2022
29282,Богучаны,Boguchany,58.38,97.45,134,1963,2022
29313,Пудино,Pudino,57.53,79.36,97,1963,2022
29328,Бакчар,Bakchar,57.08,81.91,110,1963,2022
29348,Первомайское,Pervomayskoe,57.06,86.21,114,1977,2022
29393,Червянка,Chervyanka,57.65,99.53,220,1963,2022
29418,Северное,Severnoe,56.35,78.35,126,1963,2022
29430,Томск,Tomsk,56.50,84.91,139,1963,2022
29467,"Ачинск,ж.д.ст.",Achinsk zh.d.st.,56.28,90.52,266,1977,1992
29541,Тайга,Tayga,56.06,85.61,251,1982,2022
29570,"Красноярск,оп.п.",Krasnoyarsk op.p.,56.03,92.75,276,1963,2022
29580,Солянка,Solyanka,56.17,95.27,343,1977,2022
29594,Тайшет,Tayshet,55.95,98.00,307,1963,2022
29605,Татарск,Tatarsk,55.20,75.96,112,1977,2022
29612,Барабинск,Barabinsk,55.35,78.31,120,1963,2022
29638,Огурцово,Ogurtsovo,54.90,82.95,133,1977,2022
29645,"Кемерово,АГРО",Kemerovo AGRO,55.25,86.21,148,1966,2022
29664,Светлолобово,Svetlolobovo,55.10,90.83,325,1977,2022
29698,Нижнеудинск,Nizhneudinsk,54.88,99.03,411,1963,2022
29822,Камень-на-Оби,Kamen-na-Obi,53.81,81.26,129,1977,2022
29838,Барнаул,Barnaul,53.43,83.51,184,1963,2022
29849,Кузедеево,Kuzedeevo,53.33,87.16,293,1977,2022
29862,Абакан (Хак. ЦГМС),Abakan (Khak. TsGMS),53.77,91.32,256,1977,2022
29866,Минусинск,Minusinsk,53.70,91.70,254,1963,2022
29915,Славгород,Slavgorod,52.96,78.65,126,1963,2022
29923,Ребриха,Rebrikha,53.06,82.30,219,1984,2013
29939,Бийск-Зональная,Biysk-Zonalnaya,52.68,84.95,228,1977,2022
30054,Витим,Vitim,59.45,112.58,190,1963,2022
30230,Киренск,Kirensk,57.77,108.07,259,1963,2022
30252,Мамакан,Mamakan,57.81,114.16,249,1967,2022
30309,"Братск,обс.",Bratsk obs.,56.28,101.75,416,1963,2022
30328,Орлинга,Orlinga,56.05,105.83,338,1964,2022
30337,"Казачинское,АМСГ",Kazachinskoe AMSG,56.31,107.61,362,1970,2008
30372,Чара,Chara,56.90,118.27,711,1977,2017
30385,Усть-Нюкжа,Ust-Nyukzha,56.58,121.48,426,1977,2022
30433,Нижнеангарск,Nizhneangarsk,55.78,109.55,487,1977,2022
30469,Калакан,Kalakan,55.11,116.76,613,1977,2013
30493,Нагорный,Nagornyy,55.96,124.88,861,1977,2002
30504,Тулун,Tulun,54.60,100.63,522,1964,2022
30521,Жигалово,Zhigalovo,54.80,105.21,426,1963,2013
30555,Троицкий прииск,Troitskiy priisk,54.61,113.13,1314,1963,2022
30612,"Балаганск,раб.пос.",Balagansk rab.pos.,54.00,103.06,427,1967,2022
30627,Баяндай,Bayanday,53.10,105.53,762,1965,2022
30636,Баргузин,Barguzin,53.61,109.63,489,1963,2022
30673,Могоча,Mogocha,53.75,119.73,625,1963,2022
30692,Сковородино,Skovorodino,54.00,123.96,398,1963,2022
30710,"Иркутск,обс.",Irkutsk obs.,52.26,104.31,469,1963,2022
30745,Сосново-Озерское,Sosnovo-Ozerskoe,52.53,111.55,952,1977,2022
30758,Чита,Chita,52.08,113.48,671,1963,2022
30777,Сретенск,Sretensk,52.23,117.70,525,1963,2022
30811,Тунка,Tunka,51.73,102.53,721,1963,2022
30822,Бабушкин,Babushkin,51.71,105.85,467,1963,2022
30823,"Улан-Удэ,город",Ulan-Ude gorod,51.83,107.60,515,1963,2022
30859,Агинское,Aginskoe,51.10,114.51,681,1977,2022
30879,Нерчинский з-д,Nerchinskiy z-d,51.31,119.61,619,1977,2022
30925,Кяхта,Kyakhta,50.36,106.45,797,1963,2022
30935,Красный Чикой,Krasnyy Chikoy,50.37,108.75,771,1977,2022
30949,Кыра,Kyra,49.56,111.96,908,1963,2022
30965,Борзя,Borzya,50.40,116.52,676,1963,2022
31088,Охотск,Okhotsk,59.37,143.20,6,1963,1989
31168,Аян,Ayan,56.45,138.15,8,1963,2022
31174,Большой Шантар,Bolshoy Shantar,54.83,137.53,21,1977,2022
31253,Бомнак,Bomnak,54.71,128.93,357,1963,2022
31371,Черняево,Chernyaevo,52.78,126.00,210,1977,2022
31388,Норск,Norsk,52.35,129.91,208,1963,2019
31416,Им.П.Осипенко,Im.P.Osipenko,52.41,136.50,73,1963,2022
31439,Богородское,Bogorodskoe,52.38,140.46,35,1977,2022
31443,Мазаново,Mazanovo,51.63,128.81,161,1990,2022
31510,Благовещенск,Blagoveshchensk,50.28,127.48,169,1963,2014
31532,Чекунда,Chekunda,50.86,132.25,272,1963,2003
31594,Архара,Arkhara,49.41,130.08,135,1963,2022
31707,Екатерино-Никольское,Ekaterino-Nikolskoe,47.73,130.96,73,1963,2022
31725,Смидович,Smidovich,48.61,133.83,52,1977,2022
31733,Елабуга,Elabuga,48.81,135.88,58,1989,2022
31770,Советская Гавань,Sovetskaya Gavan,49.00,140.30,24,1977,2022
31788,Лермонтовка,Lermontovka,47.15,134.33,76,1977,2022
31829,Золотой,Zolotoy,47.31,138.98,26,1963,2010
31845,Красный Яр,Krasnyy Yar,46.53,135.31,130,1977,1998
31873,Дальнереченск,Dalnerechensk,45.87,133.73,105,1963,2022
31915,Пограничный,Pogranichnyy,44.40,131.38,211,1963,1996
31931,Свиягино,Sviyagino,44.80,133.08,99,1977,1997
31960,"Владивосток,порт",Vladivostok port,43.11,131.93,183,1963,2022
31961,Тимирязевский,Timiryazevskiy,43.88,131.96,37,1977,2022
31969,Посьет,Poset,42.65,130.80,35,1977,1998
32061,Александровск-Сахал.,Aleksandrovsk-Sakhal.,50.90,142.17,31,1963,2022
32071,Тымовское,Tymovskoe,50.73,142.71,95,1977,2022
32088,Углегорск,Uglegorsk,49.06,142.03,43,1977,2022
32099,Мыс Терпения,Mys Terpeniya,48.65,144.73,6,1977,2018
32150,Южно-Сахалинск,Yuzhno-Sakhalinsk,46.95,142.72,24,1977,2022
32165,Южно-Курильск,Yuzhno-Kurilsk,44.01,145.86,49,1963,2003
32213,Мыс Лопатка,Mys Lopatka,50.86,156.68,47,1988,2006
32287,Усть-Хайрюзово,Ust-Khayryuzovo,57.08,156.70,8,1987,2022
32389,Ключи,Klyuchi,56.32,160.83,29,1963,2022
32618,Никольское(О.Беринга),Nikolskoe(O.Beringa),55.20,165.98,18,1988,2016
34003,Поныри,Ponyri,52.31,36.30,251,1972,2022
34009,"Курск,АЭ",Kursk AE,51.77,36.17,247,1963,2022
34110,Богородицкое-Фенино,Bogoroditskoe-Fenino,51.16,37.35,226,1965,2022
34123,"Воронеж,АГРО",Voronezh AGRO,51.70,39.21,149,1963,2022
34139,Каменная степь,Kamennaya step,51.05,40.70,194,1963,2022
34163,Октябрьский Городок,Oktyabrskiy Gorodok,51.63,45.45,201,1963,2022
34172,"Саратов,цгмс",Saratov tsgms,51.56,46.03,156,1981,2022
34186,Ершов,Ershov,51.36,48.30,111,1977,2022
34240,Урюпинск,Uryupinsk,50.80,42.00,110,1977,2022
34321,Валуйки,Valuyki,50.21,38.10,112,1965,2022
34356,Фролово,Frolovo,49.80,43.66,119,1986,2022
34391,Александров Гай,Aleksandrov Gay,50.15,48.55,25,1964,2022
34432,Чертково,Chertkovo,49.38,40.16,145,1963,2022
34476,Эльтон,Elton,49.13,46.85,6,1977,2022
34579,Верхний Баскунчак,Verkhniy Baskunchak,48.21,46.73,34,1989,2022
34646,Цимлянск,Tsimlyansk,47.63,42.11,65,1966,2022
34720,Таганрог,Taganrog,47.20,38.90,32,1977,2022
34740,Гигант,Gigant,46.51,41.35,79,1977,2022
34824,Приморско-Ахтарск,Primorsko-Akhtarsk,46.03,38.15,5,1966,2022
34861,"Элиста,АМСГ",Elista AMSG,46.36,44.33,155,1963,2022
34866,Яшкуль,Yashkul,46.18,45.35,-7,1965,1988
34880,"Астрахань,ГМО",Astrakhan GMO,46.28,48.05,-23,1963,2022
34949,"Ставрополь,АМСГ",Stavropol AMSG,45.11,42.08,452,1977,2022
35026,Зилаир,Zilair,52.21,57.40,521,1963,2022
35041,Бреды,Bredy,52.43,60.35,310,1966,2022
35121,"Оренбург,ЗГМО",Orenburg ZGMO,51.68,55.10,118,1963,2022
36034,Рубцовск,Rubtsovsk,51.50,81.21,219,1963,2022
36038,Змеиногорск,Zmeinogorsk,51.15,82.20,355,1963,2022
36064,Яйлю,Yaylyu,51.76,87.60,480,1963,2022
36096,Кызыл (Тув. ЦГМС),Kyzyl (Tuv. TsGMS),51.71,94.49,632,1977,1995
36103,Тоора-Хем,Toora-Khem,52.46,96.10,920,1977,1992
36229,Усть-Кокса,Ust-Koksa,50.28,85.61,978,1980,2022
36259,Кош-Агач,Kosh-Agach,50.01,88.68,1759,1963,2022
36307,Эрзин,Erzin,50.26,95.11,1101,1977,1992
37061,Буденновск,Budennovsk,44.78,44.13,136,1977,2022
37099,"Сочи,ГМО",Sochi GMO,43.58,39.76,34,1963,2022
37126,Шаджатмаз,Shadzhatmaz,43.73,42.66,2056,1977,2022
37235,"Грозный,АМСГ",Groznyy AMSG,43.35,45.68,163,1963,1994
37470,Дербент,Derbent,42.06,48.30,-18,1988,2022
37472,Махачкала,Makhachkala,43.01,47.48,9,1963,2022
37663,Ахты,Akhty,41.46,47.75,1015,1977,2022
"""
RUSOIL_DEPTHS_CM = (2, 5, 10, 15, 20, 40, 60, 80, 120, 160, 240, 320)


def read_roshydromet_soil(raw) -> pd.DataFrame:
    """
    Roshydromet daily soil temperature at depths to 320 cm, version 3 (RIHMI-WDC,
    Sherstiukov; http://meteo.ru/data/soil-temperature/), 1963-2024. Served only after a
    personal AISORI-M login, so the user saves the export ZIPs (wr*.txt data, statlist*.txt,
    fld*.txt) unchanged in Roshydromet_soil/; every ZIP there is read, and a station-day
    found in two ZIPs counts once.
    wr*.txt: ';'-separated, no header: WMO index, year, month, day, then value and quality
    flag at 2, 5, 10, 15, 20, 40, 60, 80, 120, 160, 240 and 320 cm. Missing = 999.9 (flag
    9); every month has 31 rows and the impossible dates are empty.
      - Only quality flag 0 (reliable) is used. The RIHMI-WDC description defines 1-4 and
        8 as erroneous, 5-7 as doubtful (5 = test not possible) and 9 as missing.
      - 2-15 cm are not used: bare-soil thermometers read in the warm season only. 20-320
        cm are the exhaust thermometers under the natural surface, read once a day.
      - Position from RUSOIL_STATIONS; stations south of CONFIG min_lat are not read.
      - The stations of the 1930-1990 Russian ALT product (ru_hms_alt) carry its site key
        (CONFIG rusoil_ruhms), so the two are one site; the daily values rank first (monthly
        means smooth the late-summer peak, so ALT from them reads shallow).
    ALT by the 0 C rule (method soil_temp). Most station-years are lower bounds: the
    cleared, snow-covered station plot thaws below the deepest sensor in use, or there
    is no permafrost (first build: 157 valid of 5,264 site-years). Stations that read
    only 80 and 160 cm (Verkhoyansk) get 'wide_sensor_spacing'.
    """
    cat = pd.read_csv(io.StringIO(RUSOIL_STATIONS), dtype={"wmo": int}).set_index("wmo")
    use = [d for d in RUSOIL_DEPTHS_CM if d >= 20]
    names = ["wmo", "year", "month", "day"] + [f"{k}{d}" for d in RUSOIL_DEPTHS_CM for k in ("t", "q")]
    cols = ["wmo", "year", "month", "day"] + [f"{k}{d}" for d in use for k in ("t", "q")]
    parts, seen = [], set()
    for z in files(raw, "*.zip"):
        if API_FOLDERS["roshydromet_soil"] not in z.parts:
            continue
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if not re.match(r"wr.*\.txt$", Path(n).name, re.I):
                    continue
                d = pd.read_csv(zf.open(n), sep=";", header=None, names=names, usecols=cols,
                                skipinitialspace=True, dtype="float64")
                d = d[d.wmo.isin(cat.index[cat.lat >= CONFIG["min_lat"]])]
                d["time"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day), errors="coerce")
                d = d.dropna(subset=["time"])
                for dep in use:
                    ok = (d[f"q{dep}"] == 0) & (d[f"t{dep}"] < 900)
                    parts.append(pd.DataFrame(dict(wmo=d.wmo[ok].astype(int).to_numpy(), time=d.time[ok].to_numpy(),
                                                   depth_cm=float(dep), temp_c=d[f"t{dep}"][ok].to_numpy())))
                seen.add(f"{z.name}/{n}")
    if not parts:
        return empty_records()
    t = pd.concat(parts, ignore_index=True).drop_duplicates(["wmo", "time", "depth_cm"])
    del parts
    log(f"  roshydromet_soil: {t.wmo.nunique()} stations at or north of {CONFIG['min_lat']:g} N from "
        f"{', '.join(sorted(seen))}")
    out = []
    for wmo, g in t.groupby("wmo"):
        r = cat.loc[wmo]
        key = f"RUSOIL_{wmo}"
        site = pd.DataFrame([dict(source_site_id=key, site_name=f"{r['name']} ({r.name_ru}), WMO {wmo}",
                                  lat=r.lat, lon=r.lon, country="Russia",
                                  site_key=CONFIG["rusoil_ruhms"].get(int(wmo), ""),
                                  site_note=f"Roshydromet weather station, exhaust soil thermometers 20-320 cm, "
                                            f"{r.elev_m:g} m a.s.l.",
                                  source_file=f"{API_FOLDERS['roshydromet_soil']}/wr*.txt: {wmo}")])
        rec = records_from_profiles(g.drop(columns="wmo").assign(source_site_id=key), site, "soil_temp")
        if len(rec):
            out.append(rec)
    return finish(pd.concat(out, ignore_index=True), "roshydromet_soil") if out else empty_records()


def read_g02189(raw) -> pd.DataFrame:
    """
    NSIDC G02189 ('How permanent is permafrost'): soil temperatures at five Yakutian
    stations (Yakutsk, Verkhoyansk, Pokrovsk, Isit', Churapcha), one workbook per station
    with only January and July monthly means at 0.2-3.2 m (files from download_g02189;
    the 'subset' copies are not read). With no August or September the 0 C rule cannot
    give ALT, so each year's value is the July thaw depth (0 C crossing of the July mean
    profile): a lower limit of that year's ALT. Flagged 'check_only' and
    'july_thaw_depth', never used as data. check_pre2000_g02189.csv compares it with the
    Russian ALT series (ru_hms_alt) of the same station: ALT should be at least the July
    thaw depth.
    """
    recs = []
    for root in (raw if isinstance(raw, (list, tuple)) else [raw]):
        base = Path(root) / API_FOLDERS["g02189"]
        for p in sorted(base.rglob("*.xls*")) if base.exists() else []:
            if "subset" in p.name.lower():
                continue
            try:
                book = pd.read_excel(p, sheet_name=None, header=None)
            except Exception as e:
                _unread("g02189", p.name, f"cannot open: {e}"[:200])
                continue
            info = next((v for k, v in book.items() if "info" in k.lower()), None)
            jul = next((v for k, v in book.items() if k.lower().startswith("jul")), None)
            if info is None or jul is None:
                _unread("g02189", p.name, "no Info or July sheet")
                continue
            stations = {}
            for r in info.itertuples(index=False):
                if pd.notna(pd.to_numeric(r[0], errors="coerce")):
                    stations[str(int(float(r[0])))] = (str(r[1]), as_float(r[2]), as_float(r[3]))
            txt = " ".join(map(str, jul.iloc[:15, 0]))
            wmo = (re.search(r"WMO\s*=\s*(\d+)", txt) or [None, ""])[1]
            if wmo not in stations:
                _unread("g02189", p.name, "station not in the Info sheet")
                continue
            hr = next(i for i in range(len(jul)) if str(jul.iat[i, 0]).strip().upper() == "YEAR")
            depth = {j: _depth_from_name(c) for j, c in enumerate(jul.iloc[hr]) if j > 0}
            depth = {j: v for j, v in depth.items() if v is not None}       # '0.20 m' -> 20 cm
            name, la, lo = stations[wmo]
            for r in jul.iloc[hr + 1:].itertuples(index=False):
                yr = pd.to_numeric(r[0], errors="coerce")
                if pd.isna(yr):
                    continue
                prof = sorted((d, as_float(r[j])) for j, d in depth.items())
                prof = [(d, v) for d, v in prof if pd.notna(v) and v > -900]
                if len(prof) < 2 or prof[0][1] <= 0:
                    continue
                z, cens = np.nan, False
                for (d0, v0), (d1, v1) in zip(prof, prof[1:]):
                    if v0 > 0 >= v1:
                        z = d0 + (d1 - d0) * v0 / (v0 - v1)
                        break
                else:
                    z, cens = prof[-1][0], True             # thawed below the deepest sensor
                recs.append(dict(source_site_id=f"G02189_{wmo}", site_name=f"{name} (WMO {wmo}, G02189)",
                                 lat=la, lon=lo, country="Russia", method="soil_temp_monthly",
                                 year=int(yr), alt_cm=z, censored=cens, n_points=len(prof),
                                 flag="check_only;july_thaw_depth" + (";lower_bound" if cens else ""),
                                 source_file=p.name))
    return finish(pd.DataFrame(recs), "g02189") if recs else empty_records()


ADAPTERS = {
    "calm_summary": read_calm_summary,
    "pangaea_calm": read_pangaea_calm,
    "gtnp": read_gtnp_grids,
    "gsc_thaw_tube": read_gsc_thaw_tubes,
    "gsc_cable": read_gsc_cables,
    "usgs_npra": read_usgs_npra,
    "ru_hms_alt": read_ru_hms_alt,
    "calm_site_files": read_calm_site_files,
    "samoylov_calm": read_samoylov_calm,
    "parks_canada": read_parks_canada,
    "illisarvik": read_illisarvik,
    "arc_lter": read_arc_lter,
    "bnz_lter": read_bnz_lter,
    "itex": read_itex,
    "ngee_barrow": read_ngee_barrow,
    "uaf_stations": read_uaf_stations,
    "kwbs": read_kwbs,
    "kytalyk": read_kytalyk,
    "allena": read_allena,
    "awi_soil_temp": read_awi_soil_temp,
    "stordalen": read_stordalen,
    "umiujaq_cable": read_umiujaq_cables,
    "canada_gt": read_canada_gt,
    "gem": read_gem,
    "above_field": read_above_field,
    "cusp": read_cusp,
    # added November 2026 (stage 1 downloads them through their APIs)
    "gtnp_platform": read_gtnp_platform,
    "gtnp_platform_alt": read_gtnp_platform_alt,
    "tsp_alaska": read_tsp_alaska,
    "uaf_sites": read_uaf_sites,
    "neon": read_neon,
    "syrdakh": read_syrdakh,
    "metno": read_metno,
    "metno_frost": read_metno_frost,
    "g02189": read_g02189,
    "tarfala_pace": read_tarfala_pace,
    "uio_boreholes": read_uio_boreholes,
    "roshydromet_soil": read_roshydromet_soil,
    "metno_janssonhaugen": read_metno_janssonhaugen,
}


# =============================================================================
# STAGE 3: BUILD
# =============================================================================
# what each source is (for the inventory and the credits)
# =============================================================================
SOURCES = {
    "calm_summary": ("CALM site summary table, Circumpolar Active Layer Monitoring (GWU)",
                     "https://www2.gwu.edu/~calm/data/north.htm", "CALM data, free use with acknowledgement"),
    "pangaea_calm": ("GTN-P CALM: 35 years of Active Layer Thickness (ALT) across latitudinal and "
                     "elevational gradients in the Northern Hemisphere",
                     "https://doi.org/10.1594/PANGAEA.972777", "CC-BY 4.0"),
    "gtnp": ("GTN-P database active-layer grids and transects (exports from the Corridor-Scale "
             "Damage Estimation project)", "http://gtnpdatabase.org",
             "GTN-P data policy: inform and credit the data authors (listed in data_credits.md)"),
    "gsc_thaw_tube": ("Geological Survey of Canada, Mackenzie Valley active layer monitoring: Open File "
                      "6287 and Open File 9340", "https://doi.org/10.4095/248197 ; https://doi.org/10.4095/pxpyc4s8tf",
                      "Open Government Licence - Canada"),
    "gsc_cable": ("Geological Survey of Canada, Mackenzie corridor ground temperature cables (Open File 9340)",
                  "https://doi.org/10.4095/pxpyc4s8tf", "Open Government Licence - Canada"),
    "usgs_npra": ("USGS: DOI/GTN-P Climate and Active-Layer Data, National Petroleum Reserve and Arctic "
                  "National Wildlife Refuge, Alaska", "https://doi.org/10.5066/F7VX0FGB", "Public domain (USGS)"),
    "ru_hms_alt": ("Time Series of Active Layer Thickness in the Russian Arctic, 1930-1990 (NCAR/ARCSS)",
                   "https://doi.org/10.5065/D64747ZJ", "Arctic Data Center (cite dataset)"),
    "calm_site_files": ("CALM site workbooks S1 Kapp Linne, S2 Abisko, P1 Calypsostranda, G1/G2 Zackenberg "
                        "ZEROCALM, and U2 Barrow CRREL summary",
                        "https://www2.gwu.edu/~calm/data/north.htm ; https://doi.org/10.18739/A20G3H10S",
                        "CALM data / CC-BY 4.0 (Arctic Data Center)"),
    "samoylov_calm": ("Calm data at station Samoylov, 2002-2018 (v201908) and 2019-2024 (Boike et al., AWI)",
                      "https://doi.org/10.1594/PANGAEA.905229 ; 947069 ; 949381 ; 961867 ; 961875 ; 973715",
                      "CC-BY 4.0"),
    "parks_canada": ("Parks Canada active layer monitoring: Auyuittuq, Quttinirpaaq, Sirmilik, Ukkusiksalik",
                     "https://open.canada.ca (Active Layer - [park name], Parks Canada)", "Open Government Licence - Canada"),
    "illisarvik": ("Annual Thaw Depths in Illisarvik, Northwest Territories (NSIDC GGD642)",
                   "https://nsidc.org/data/ggd642", "NSIDC (cite dataset)"),
    "arc_lter": ("Arctic LTER, Toolik: tussock watershed (1019), Imnavait (1626) thaw surveys and "
                 "experimental-plot thaw depths (1610, control plots)",
                 "https://portal.edirepository.org (knb-lter-arc.1019, .1626, .1610)", "CC-BY 4.0 (EDI/LTER)"),
    "bnz_lter": ("Bonanza Creek LTER active layer depths: 7, 8, 11, 239, 240, 480 (controls), 519, 605",
                 "https://portal.edirepository.org (knb-lter-bnz)", "CC-BY 4.0 (EDI/LTER)"),
    "itex": ("Thaw depth in the ITEX plots at Utqiagvik and Atqasuk, Alaska (control plots)",
             "https://arcticdata.io/catalog/view/urn:uuid:7e55b70a-5f3f-4a77-af37-d7b48e7478ec",
             "CC-BY 4.0 (Arctic Data Center)"),
    "ngee_barrow": ("NGEE-Arctic: Subsurface temperature ... Areas A-D, Utqiagvik, 2012-2023",
                    "https://doi.org/10.5440/1126515", "CC-BY 4.0"),
    "uaf_stations": ("UAF Permafrost Laboratory soil temperature stations (Deadhorse, Franklin Bluffs, "
                     "Sagwon, Imnavait; V. Romanovsky)", "https://permafrost.gi.alaska.edu",
                     "UAF Permafrost Lab public data (cite Romanovsky et al.)"),
    "kwbs": ("Thaw depth and snow height time series, Kolyma Water-Balance Station 1954-1997; soil "
             "temperature at Nizhnyaya", "https://doi.org/10.1594/PANGAEA.881754 ; https://doi.org/10.1594/PANGAEA.881792",
             "CC-BY 4.0"),
    "kytalyk": ("Thaw depth of the landscape in northeastern Siberia (S5), Allaikha/Kytalyk",
                "https://doi.org/10.1594/PANGAEA.961879", "CC-BY 4.0"),
    "allena": ("ALLena: thaw depth compilation, Lena River Delta 1998-2022 (Veremeeva et al.)",
               "https://doi.org/10.1594/PANGAEA.974408", "CC-BY 4.0"),
    "awi_soil_temp": ("AWI soil stations: Bayelva homogenised product 1998-2024; Samoylov soil data "
                      "2002-2018 (v201908) and 2019-2021",
                      "https://doi.org/10.1594/PANGAEA.982254 ; https://doi.org/10.1594/PANGAEA.905234 ; 947034 ; 949198 ; 974420",
                      "CC-BY 4.0"),
    "stordalen": ("Manual active layer and water table depth, Stordalen Mire autochamber sites 2003-2017",
                  "https://doi.org/10.5281/zenodo.10420396", "CC-BY 4.0"),
    "umiujaq_cable": ("Subsurface ground temperature, instrumented permafrost mound, Tasiapik Valley, "
                      "Umiujaq (Nordicana D)", "https://doi.org/10.5683/SP3/QSRW0I", "CC-BY 4.0"),
    "canada_gt": ("A standardized permafrost ground temperature collection for Canada 2025, v0.9.3 "
                  "(Meier-Legault, Brown, Gruber et al., ESSD 2026): ground temperature boreholes (type GT)",
                  "https://doi.org/10.5281/zenodo.21042525 (concept doi:10.5281/zenodo.18022924)",
                  "see the Zenodo record; cite the collection and each site's own reference"),
    "gem": ("Greenland Ecosystem Monitoring (GEM): ZEROCALM-1/2 active layer (GeoBasis Zackenberg); ground "
            "temperatures at GeoBasis Disko AWS1, AWS2 P1/P2 and ZEROCALM-2 upper/middle/lower",
            "https://doi.org/10.17897/2Q7S-2876 ; 10.17897/4EE8-WE92 ; 10.17897/B2K2-0Q82 ; 10.17897/3T7W-7638 ; "
            "10.17897/8HPH-H716 ; 10.17897/KCP6-QG90 ; 10.17897/D2NV-GN54 ; 10.17897/FX4T-2H83",
            "CC-BY-SA 4.0 (GEM terms of use: cite each dataset DOI)"),
    "above_field": ("ABoVE: Soil thaw depth and moisture validation data (ORNL DAAC), repeat-visited plots",
                    "https://daac.ornl.gov (ABoVE_Soil_ThawDepth_Moisture_Validation)", "ORNL DAAC (cite dataset)"),
    "cusp": ("CUSP - CommUnity near-Surface Permafrost dataset v1.2 (only its sources not read elsewhere)",
             "https://doi.org/10.5281/zenodo.23171354", "CC-BY 4.0"),
    "gtnp_platform": ("GTN-P data platform (AWI): borehole ground temperatures, boreholes with at least two "
                      "sensors in the top 1.5 m", "https://data.gtn-p.org",
                      "CC BY 4.0; GTN-P data policy: credit the data authors (listed in data_credits.md)"),
    "gtnp_platform_alt": ("GTN-P data platform (AWI): active-layer grids and transects (mostly CALM)",
                          "https://data.gtn-p.org",
                          "CC BY 4.0; GTN-P data policy: credit the data authors (listed in data_credits.md)"),
    "tsp_alaska": ("Network of Permafrost Observatories in North America and Russia: Alaska, and the other "
                   "current permafrost-temperature packages of V. Romanovsky (Arctic Data Center)",
                   "https://doi.org/10.18739/A2J34P (newest version)", "Arctic Data Center (cite each package)"),
    "uaf_sites": ("UAF Permafrost Laboratory site data: Galbraith Lake, West Dock, Happy Valley, Chandalar Shelf",
                  "https://permafrost.gi.alaska.edu", "UAF Permafrost Lab public data (cite Romanovsky et al.)"),
    "neon": ("NEON soil temperature DP1.00041.001 at BARR, TOOL and HEAL (plot-years without sensor moves "
             "or frost-heave entries)", "https://data.neonscience.org/data-products/DP1.00041.001", "CC0 (NEON)"),
    "syrdakh": ("Syrdakh observatory, central Yakutia (Zenodo 14619854)", "https://doi.org/10.5281/zenodo.14619854",
                "see the Zenodo record"),
    "metno": ("MET Norway permafrost stations (Svalbard and mainland Norway), THREDDS frost2netcdf-permafrost",
              "https://thredds.met.no/thredds/catalog/arcticdata/frost2netcdf-permafrost/catalog.html",
              "CC BY 4.0 (MET Norway)"),
    "metno_frost": ("MET Norway Frost API: soil and ground temperatures of the MET stations in Svalbard and "
                    "the permafrost boreholes (sensors deeper than 2 m)", "https://frost.met.no",
                    "CC BY 4.0 (MET Norway; 'Data from The Norwegian Meteorological Institute')"),
    "metno_janssonhaugen": ("Janssonhaugen, Svalbard: shallow borehole P11, ground temperature 0.2-5 m, "
                            "1999-2025 (MET Norway, K. Isaksen; sent by email 10 November 2026; "
                            "Isaksen et al. 2000, 2001, 2022)", "https://cryo.met.no/en/permafrost/janssonhaugen",
                            "shared by the data owner for this analysis; cite Isaksen et al. 2000, 2001, 2022"),
    "g02189": ("NSIDC G02189 Russian station soil temperatures (check before 2000 only, never used as data)",
               "https://nsidc.org/data/g02189", "NSIDC (cite dataset)"),
    "tarfala_pace": ("PACE-1-2 borehole temperature data 2000-2014, Tarfalaryggen, Sweden (Stockholm University, "
                     "Tarfala Research Station)", "https://doi.org/10.17045/sthlmuni.3171646.v1", "CC BY 4.0"),
    "uio_boreholes": ("Ground temperatures of the University of Oslo mountain permafrost boreholes in Norway and "
                      "Iceland (Etzelmüller et al. 2023, The Cryosphere: 'Rapid warming and degradation of "
                      "mountain permafrost in Norway and Iceland')", "https://doi.org/10.11582/2023.00128",
                      "CC BY 4.0"),
    "roshydromet_soil": ("Roshydromet daily soil temperature at depths to 320 cm, version 3, 1963-2024 "
                         "(RIHMI-WDC, Sherstiukov 2012; downloaded by the user through AISORI-M)",
                         "http://meteo.ru/data/soil-temperature/",
                         "RIHMI-WDC terms for registered users; cite the dataset"),
}

# Reviewed but not used, with the reason (kept here so the inventory is complete).
# Why a source that was read gives no eligible site (shown in source_inventory.csv)
READ_NOT_USED = {
    "umiujaq_cable": "most years flagged: the deepest thawed sensor stays within 0.1 C of 0 C all "
                     "winter (no_refreeze), or the sensors around the 0 C crossing are 1-2 m apart",
}

NOT_USED = [
    ("InSAR ALT products (ReSALT, AirMOSS, ABoVE remote sensing)", "ORNL DAAC 1266, 2004, 1760",
     "no site with 8 or more annual values; model-assisted retrievals, not measurements"),
    ("ESA CCI Permafrost ALT and other gridded ALT products", "Zenodo 10070610, 21667583, 19148960",
     "model output, not measurements"),
    ("GTN-P database (old site)", "http://gtnpdatabase.org", "replaced by the GTN-P data platform "
     "(data.gtn-p.org, read as gtnp_platform and gtnp_platform_alt); the grids exported from the old "
     "site for the corridor project are still read when present (gtnp)"),
    ("GEM Greenland: shallow soil temperature (GeoBasis Disko AWS2/AWS3/AWS4, Nuuk SoilEmp, SoilEmpSa, "
     "SoilVUE Heath and Fen), meteorology, snow and soil moisture", "https://data.g-e-m.dk",
     "sensors only to 20-50 cm, or 2021-2024 only (SoilVUE Fen to 100 cm); not ALT"),
    ("Yukon Permafrost Database (direct download)", "https://data.geology.gov.yk.ca/permafrost",
     "only site locations could be downloaded; its temperature series come through the Canada "
     "ground temperature collection (canada_gt)"),
    ("Nordicana D (direct download)", "https://nordicana.cen.ulaval.ca", "server refused downloads (403); "
     "Umiujaq came from Borealis, and the CEN Nunavik and Bylot Island cables (D-series, e.g. "
     "doi:10.5885/45291SL-34F28A9491014AFD) come through the Canada ground temperature collection"),
    ("Roshydromet daily soil temperature to 320 cm, version 3, 1963-2024 (Sherstiukov, RIHMI-WDC)",
     "http://meteo.ru/data/soil-temperature/", "data only through a personal login (aisori-m.meteo.ru); "
     "not downloaded. Station list of candidates: roshydromet_stations_to_request.csv"),
    ("Pan-Arctic terrestrial in situ weather data set 1990-2023 (Rasmussen et al.)",
     "https://zenodo.org/records/15388336", "soil temperature at 22 places north of 50 N, most only "
     "to 10-50 cm; one (Marre-Sale, CALM R3, sensors to 200 cm) gives 8 thaw years before checks, and "
     "that site is in the dataset from CALM probing. Its ground data come from CALM, GEM, NGEE, FMI "
     "and CEN stations, which are read here from their originals or are too shallow"),
    ("Scotty Creek, NWT", "https://doi.org/10.5683/SP2/BTRLHO ; https://doi.org/10.5683/SP/OQDRJG",
     "thaw and talik studies shorter than 8 years; the long monitoring record is restricted"),
    ("Trail Valley Creek soil station (AWI)", "PANGAEA 962869-962875, 973138, 983873",
     "sensors only to 20 cm, above the thaw depth"),
    ("NWT Permafrost Database and GSC OF 8656 (Inuvik-Tuktoyaktuk Highway)", "https://doi.org/10.4095/321870",
     "6 years or fewer per site"),
    ("NPS Arctic and Central Alaska networks (active layer grids, soil temperature)", "https://irma.nps.gov",
     "6 years or fewer per grid; soil temperature at 2-3 depths only"),
    ("Toolik LTER main met station soil temperature (NSIDC GGD906)", "https://nsidc.org/data/ggd906",
     "three sensors (20, 50, 100 cm) recorded in whole degrees; thaw below the deepest sensor most years"),
    ("Sisimiut and Kangerlussuaq boreholes, Greenland (NSIDC GGD23, GGD632)", "https://nsidc.org/data/ggd23",
     "Sisimiut: ground within a few tenths of 0 C from 3 to 9 m (0 C crossing not defined); "
     "Kangerlussuaq: 7 years"),
    ("East Siberian ground temperatures 1882-1994 (NCAR ARCSS)", "https://arcticdata.io",
     "same stations as the 1930-1990 Russian ALT product; monthly means at 0.2-3.2 m (80-160 cm gaps)"),
    ("Vaigat borehole, Greenland", "https://doi.org/10.5281/zenodo.15478324", "2019-2022 only"),
    ("European mountain permafrost boreholes (Noetzli et al. 2024)", "https://doi.org/10.5281/zenodo.13628540",
     "temperatures from 4.5 m down only"),
    ("Schefferville boreholes (NSIDC GGD605), Salluit (GGD418), Svalbard GGD249/GGD239, Vaisjeaggi palsa "
     "(GGD622), Tanana transects (GGD623), Alaska GGD624", "https://nsidc.org/data",
     "first sensor at 2.9 m (Schefferville), or fewer than 8 years"),
    ("CALM grid node files on the Arctic Data Center (Alaska U-sites, R23, R24, R35, R51)", "https://arcticdata.io",
     "same sites and years as the CALM tables (checked: no additional years)"),
    ("CUSP sub-sources", "https://doi.org/10.5281/zenodo.23171354",
     "Wang_2018 given only as 25/50/75 cm classes; CALM copies; Alps, Tibet, Southern Hemisphere; "
     "sources read from their originals"),
]


# =============================================================================
# harmonise, deduplicate, QC, select
# =============================================================================
def in_scope(df: pd.DataFrame) -> pd.Series:
    ok = df.lat >= CONFIG["min_lat"]
    ok &= ~df.country.isin(CONFIG["exclude_countries"])
    return ok


def drop_short_series(rec: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """
    Series with fewer than min_n years and no site key cannot reach the year threshold
    unless they join another series; they are left out of the duplicate search (it is
    slow for the many one-off campaign points). They stay in all_records.
    """
    n = rec.groupby(["source", "source_site_id"]).year.transform("nunique")
    keep = (n >= min_n) | (rec.site_key != "")
    log(f"  {int((~keep).sum())} site-years in series shorter than {min_n} years set aside")
    return rec[keep].copy()


def assign_physical_sites(rec: pd.DataFrame) -> pd.DataFrame:
    """
    Give every source series a physical site id ('site_id'). Series are the same site if
      (a) they carry the same site key (CALM code, or a station id shared across
          products), or
      (b) they are within dup_distance_km, use the same method family (probe, tube,
          temperature), and agree on overlapping years (median |difference| <=
          dup_agree_cm or dup_agree_frac of the mean ALT), or
      (c) both are temperature series at the same position (same_borehole_km) and agree
          within same_borehole_agree_cm / _frac on >= 5 shared years, or
      (d) one is a copy of the other (within copy_km, >= 5 shared years, median
          |difference| <= copy_agree_cm and correlation >= copy_min_r), whatever the
          method names.
      (e) CONFIG force_join names it as the same borehole as a series of a preferred
          source (two products of one logger that disagree, so rules b-d keep them apart).
    When rules b-d pass for several sites, the series joins the one that agrees best
    (smallest median difference), then the nearest.
    Two different site keys are never joined, and neither are two series of the same
    source (the source itself lists them as different plots, e.g. the Eight Mile Lake
    thaw-gradient sites or the Kolyma water-balance plots, even when they share a
    coordinate). The exception is the collections in CONFIG join_within_source, where
    only rules (c) and (d) apply.
    Series that are that close but disagree, or do not overlap, stay separate sites (the
    sites table lists them in 'colocated_with'). The most preferred source (CONFIG
    priority), then the longest series, comes first and names the site.
    """
    ser = (rec.groupby(["source", "source_site_id"])
              .agg(lat=("lat", "median"), lon=("lon", "median"), site_key=("site_key", "first"),
                   method=("method", "first"), n=("year", "nunique"))
              .reset_index())
    ser["prio"] = ser.source.map(CONFIG["priority"]).fillna(CONFIG["priority"]["other"])
    ser = ser.sort_values(["prio", "n"], ascending=[True, False]).reset_index(drop=True)
    ser["fam"] = ser.method.map(method_family)
    by_key = {k: g.groupby("year").alt_cm.mean() for k, g in rec.groupby(["source", "source_site_id"])}
    reps = []                     # one dict per physical site
    site_of, n_close = {}, 0
    for r in ser.itertuples():
        key = (r.source, r.source_site_id)
        match = None
        if r.site_key:
            match = next((s for s in reps if s["site_key"] == r.site_key), None)
        forced = CONFIG["force_join"].get(f"{r.source}:{r.source_site_id}")
        if match is None and forced:                                         # rule (e)
            sid = site_of.get(tuple(forced.split(":", 1)))
            match = next((s for s in reps if s["site_id"] == sid), None)
            if match is not None:
                log(f"    {key[0]}:{key[1]} joined to {sid} (CONFIG force_join)")
        if match is None and reps:
            lat = np.array([s["lat"] for s in reps])
            lon = np.array([s["lon"] for s in reps])
            dist = haversine_km(r.lat, r.lon, lat, lon)
            near = np.where(dist <= max(CONFIG["dup_distance_km"], CONFIG["copy_km"]))[0]
            best = None                       # (median difference, distance, i, shared years)
            for i in near[np.argsort(dist[near])]:
                s = reps[i]
                if r.site_key and s["site_key"]:
                    continue
                same_src = r.source in s["sources"]
                if same_src and r.source not in CONFIG["join_within_source"]:
                    continue
                a, b = by_key[key], by_key[s["key"]]
                common = a.index.intersection(b.index)
                same_fam = s["fam"] == r.fam
                ok = False
                if len(common) >= 2:
                    diff = (a[common] - b[common]).abs().median()
                    tol = max(CONFIG["dup_agree_cm"], CONFIG["dup_agree_frac"] * b.mean())
                    ok = ((same_fam and dist[i] <= CONFIG["dup_distance_km"] and diff <= tol
                           and not same_src)                                 # rule (b)
                          or (same_fam and r.fam == "temperature" and len(common) >= 5
                              and dist[i] <= CONFIG["same_borehole_km"]
                              and diff <= max(CONFIG["same_borehole_agree_cm"],
                                              CONFIG["same_borehole_agree_frac"] * b.mean()))  # (c)
                          or (len(common) >= 5 and diff <= CONFIG["copy_agree_cm"]
                              and a[common].corr(b[common]) >= CONFIG["copy_min_r"]))      # (d)
                    if ok and (best is None or (diff, dist[i]) < best[:2]):
                        best = (diff, dist[i], i, len(common))
                if not ok and same_fam and dist[i] <= CONFIG["dup_distance_km"]:
                    n_close += 1
            # Of all candidates that pass, the one that agrees best (then the nearest): several
            # series can sit at one coordinate (the KP313 cables), or a copy can lie beyond a
            # nearer plot that also passes (Kytalyk Allaiha = CALM R31, not R31A).
            if best is not None:
                diff, d_km, i, n_common = best
                match = reps[i]
                log(f"    {key[0]}:{key[1]} joined to {match['site_id']} ({d_km:.2f} km, "
                    f"{n_common} shared years, median difference {diff:.1f} cm)")
        if match is None:
            sid = r.site_key or f"{r.source}:{r.source_site_id}"
            if any(s["site_id"] == sid for s in reps):
                sid = f"{sid}:{r.source}"
            reps.append(dict(site_id=sid, lat=r.lat, lon=r.lon, fam=r.fam, site_key=r.site_key, key=key,
                             sources={r.source}))
            site_of[key] = sid
        else:
            site_of[key] = match["site_id"]
            match["sources"].add(r.source)
    rec = rec.copy()
    rec["site_id"] = [site_of[k] for k in zip(rec.source, rec.source_site_id)]
    log(f"  {ser.shape[0]} source series -> {len(reps)} physical sites; {n_close} pairs of "
        f"same-method series within {CONFIG['dup_distance_km']:g} km kept apart (values differ "
        f"or years do not overlap)")
    return rec


def _has_flag(flags: pd.Series, name: str) -> pd.Series:
    return flags.fillna("").str.contains(r"(?:^|;)" + name + r"(?:;|$)")


def site_state_flags(rec: pd.DataFrame) -> pd.DataFrame:
    """
    Flags that belong to the physical site rather than to one source. Without this step a
    year that one source flags (for example CALM's talik years at Bonanza Creek U18) comes
    back through another source's unflagged copy of the same value (BNZ LTER, GTN-P).
      1. Talik onset (CONFIG talik_onset_cm): in the listed sources, a probe or tube series
         is flagged 'possible_talik' from the first year it is deeper than the threshold.
      2. Borehole equilibration (CONFIG equilibration_years): the first years of each
         series of the listed sources.
      3. STATE_FLAGS ('talik', 'possible_talik'): every record of the site from the first
         flagged year on.
      4. YEAR_STATE_FLAGS ('no_refreeze'): every record of the site in that year.
      5. COPY_FLAGS (CALM footnote, sensor cap, early survey, treatment, lower bound) and
         censoring: records of the same site and year whose value is a copy of the flagged
         one (within dup_agree_cm or dup_agree_frac); a lower bound also covers smaller
         values. A different, deeper measurement in another source stays usable.
    Steps 3-5 never start from a 'check_only' record (G02189 July thaw depths, Frost
    SN99875 under the concrete lid): a series kept only for comparison passes nothing on.
    """
    rec = rec.copy()
    rec["flag"] = rec.flag.fillna("")
    chk = _has_flag(rec.flag, "check_only")
    before = (~rec.censored & ~rec.flag.str.contains(INVALID_FLAGS)).sum()

    # 1. talik onset in the listed probing sources
    for src, thr in CONFIG["talik_onset_cm"].items():
        m = (rec.source == src) & rec.method.isin(["probe", "thaw_tube"])
        deep = rec[m & (rec.alt_cm > thr)]
        onset = deep.groupby("source_site_id").year.min()
        hit = m & (rec.year >= rec.source_site_id.map(onset).fillna(9999))
        rec["flag"] = add_flag(rec.flag, hit & ~_has_flag(rec.flag, "possible_talik"), "possible_talik")
        if len(onset):
            log(f"  {src}: {len(onset)} series deeper than {thr:g} cm in some year; "
                f"{int(hit.sum())} site-years from that year on flagged possible_talik")

    # 2. first years of new boreholes
    for src, n in CONFIG["equilibration_years"].items():
        m = rec.source == src
        first = rec[m].groupby("source_site_id").year.transform("min")
        hit = m & (rec.year < first.reindex(rec.index).fillna(9999) + n)
        rec["flag"] = add_flag(rec.flag, hit, "borehole_equilibration")
        log(f"  {src}: first {n} year(s) of {rec[m].source_site_id.nunique()} series flagged "
            f"borehole_equilibration ({int(hit.sum())} site-years)")

    # 3. talik state from the first flagged year on
    for f in STATE_FLAGS:
        has = _has_flag(rec.flag, f)
        onset = rec[has & ~chk].groupby("site_id").year.min()
        hit = (rec.year >= rec.site_id.map(onset).fillna(9999)) & ~has
        rec["flag"] = add_flag(rec.flag, hit, f)
        if hit.any():
            log(f"  {f}: {int(hit.sum())} records of other sources or later years flagged "
                f"({rec[hit].site_id.nunique()} sites)")

    # 4. no refreeze: the whole site-year
    for f in YEAR_STATE_FLAGS:
        has = _has_flag(rec.flag, f)
        sy = set(zip(rec.site_id[has & ~chk], rec.year[has & ~chk]))
        hit = pd.Series([k in sy for k in zip(rec.site_id, rec.year)], index=rec.index) & ~has
        rec["flag"] = add_flag(rec.flag, hit, f)
        if hit.any():
            log(f"  {f}: {int(hit.sum())} records of other sources in the same site-year flagged")

    # 5. copies of a flagged or censored value
    src_cols = ["site_id", "year", "alt_cm", "censored", "flag"]
    flagged = rec[~chk & (rec.censored | rec.flag.str.contains(
        r"(?:^|;)(?:" + "|".join(COPY_FLAGS) + r")(?:;|$)"))][src_cols]
    if len(flagged):
        pair = rec.reset_index().merge(flagged, on=["site_id", "year"], suffixes=("", "_f"))
        tol = np.maximum(CONFIG["dup_agree_cm"], CONFIG["dup_agree_frac"] * pair.alt_cm_f)
        same = (pair.alt_cm - pair.alt_cm_f).abs() <= tol
        below_bound = pair.censored_f & (pair.alt_cm <= pair.alt_cm_f + tol)
        n_flag = 0
        for f in COPY_FLAGS:
            hit = same & _has_flag(pair.flag_f, f) & ~_has_flag(pair.flag, f)
            idx = pair.loc[hit, "index"].unique()
            rec.loc[idx, "flag"] = add_flag(rec.loc[idx, "flag"], np.ones(len(idx), bool), f)
            n_flag += len(idx)
        idx = pair.loc[below_bound & ~pair.censored, "index"].unique()
        rec.loc[idx, "censored"] = True
        rec.loc[idx, "flag"] = add_flag(rec.loc[idx, "flag"], np.ones(len(idx), bool), "lower_bound")
        log(f"  {n_flag} copies of flagged values and {len(idx)} copies of lower bounds in "
            f"other sources flagged")
    after = (~rec.censored & ~rec.flag.str.contains(INVALID_FLAGS)).sum()
    log(f"  site-level flags: {before - after} more records not usable")
    return rec


def merge_site_years(rec: pd.DataFrame) -> pd.DataFrame:
    """
    One value per physical site and year: the valid record from the preferred source
    (a censored or flagged record is used only when no valid one exists). Also records how
    many sources had that site-year and the spread of their valid values.
    """
    rec = rec.copy()
    rec["prio"] = rec.source.map(CONFIG["priority"]).fillna(CONFIG["priority"]["other"])
    rec["valid"] = ~rec.censored & ~rec.flag.str.contains(INVALID_FLAGS)
    rec = rec.sort_values(["site_id", "year", "valid", "prio"], ascending=[True, True, False, True])
    g = rec.groupby(["site_id", "year"])
    n_src = g.source.nunique().rename("n_sources")
    v = rec[rec.valid].groupby(["site_id", "year"]).alt_cm
    spread = (v.max() - v.min()).rename("source_spread_cm")
    top = g.head(1).merge(n_src, on=["site_id", "year"]).merge(spread, on=["site_id", "year"], how="left")
    return top.drop(columns=["prio"])


def outlier_flags(an: pd.DataFrame) -> pd.Series:
    """
    Flags isolated spikes for review; nothing is removed.

    For each site, log(ALT) minus its centred 5-year running median (so trends and slow
    shifts are not flagged). A year is unusual when that residual is beyond
    outlier_robust_z robust SDs (1.4826 * MAD of the site's residuals) AND differs from
    the running median by more than 25 %. An unusual year is 'regional' when at least two
    other sites within 300 km depart from their own running median in the same direction
    by more than 10 % that year (a cold or warm summer); otherwise it is 'isolated'
    (possible entry or survey error, worth checking).
    Returns a Series of '', 'unusual_year_regional' or 'unusual_year_isolated'.
    """
    an = an.sort_values(["site_id", "year"]).copy()
    la = np.log(an.alt_cm)
    rm = la.groupby(an.site_id).transform(lambda x: x.rolling(5, center=True, min_periods=3).median())
    an["res"] = la - rm
    mad = an.res.abs().groupby(an.site_id).transform("median") * 1.4826
    z = an.res / mad.clip(lower=0.02)              # floor: 2 % noise on very smooth series
    unusual = (z.abs() > CONFIG["outlier_robust_z"]) & (an.res.abs() > np.log(1.25))
    out = pd.Series("", index=an.index)
    for i in an.index[unusual]:
        r = an.loc[i]
        same = an[(an.year == r.year) & (an.site_id != r.site_id)]
        d = haversine_km(r.lat, r.lon, same.lat.to_numpy(), same.lon.to_numpy())
        near = same[(d <= 300) & (np.sign(same.res) == np.sign(r.res)) & (same.res.abs() > np.log(1.10))]
        out[i] = "unusual_year_regional" if near.site_id.nunique() >= 2 else "unusual_year_isolated"
    return out


def colocated(sites: pd.DataFrame) -> pd.Series:
    """For each site: the other eligible sites within dup_distance_km (any method)."""
    out = []
    for r in sites.itertuples():
        d = haversine_km(r.lat, r.lon, sites.lat.to_numpy(), sites.lon.to_numpy())
        near = sites.site_id[(d <= CONFIG["dup_distance_km"]) & (sites.site_id != r.site_id).to_numpy()]
        out.append(";".join(near))
    return pd.Series(out, index=sites.index)


def gtnp_authors(raw_dirs) -> pd.DataFrame:
    """GTN-P grids used: name, data author and page, from the '.metadata.txt' files."""
    rows = []
    for m in files(raw_dirs, "Activelayer_*.metadata.txt"):
        txt = m.read_text(encoding="utf-8", errors="replace")
        get = lambda k: (re.search(rf"^{k}\s*:\s*(.+)$", txt, re.M) or [None, ""])[1].strip()
        rows.append(dict(site=get("Name"), calm_code=get("Ancillary Code"), data_author=get("Data Author"),
                         url=get("Url")))
    return pd.DataFrame(rows).drop_duplicates()


def canada_gt_credits(raw_dirs, valid: pd.DataFrame) -> list[str]:
    """Creator, institution and reference of each Canada-collection series that gives a
    used value (the collection asks users to cite each site's own reference too)."""
    used = valid[valid.source == "canada_gt"]
    metas = [m for m in files(raw_dirs, "site_metadata.csv") if (m.parent / "all_observations.zip").exists()]
    if used.empty or not metas:
        return []
    ids = set(used.source_file.str.split(": ").str[-1])
    meta = pd.read_csv(metas[-1], encoding="utf-8-sig")
    meta = meta[meta.site_id.isin(ids)].fillna("")
    lines = ["", "## Canada ground temperature collection: series used and their own references", ""]
    for r in meta.sort_values("site_id").itertuples():
        ref = r.references or r.metadata_link or "no reference given in the collection"
        lines.append(f"- {r.site_id}: {r.creator_name}, {r.institution}. {ref}")
    return lines


def write_credits(out: Path, inv: pd.DataFrame, raw_dirs, valid: pd.DataFrame) -> None:
    used = inv[inv.status == "used"]
    lines = ["# Data credits", "",
             "Datasets that contribute site-years to the multi-source ALT dataset. Please cite them as "
             "their landing pages ask. GTN-P data policy: inform and credit the data authors listed below.",
             ""]
    for r in used.itertuples():
        lines.append(f"- **{r.title}**. {r.reference}. Licence: {r.licence}. "
                     f"({int(r.sites_contributing)} site{'s' if r.sites_contributing != 1 else ''}, "
                     f"{int(r.site_years_used)} site-years used)")
    a = gtnp_authors(raw_dirs)
    if len(a):
        lines += ["", "## GTN-P data authors (grids read from the GTN-P database)", ""]
        for r in a.sort_values("site").itertuples():
            code = f" ({r.calm_code})" if r.calm_code else ""
            lines.append(f"- {r.site}{code}: {r.data_author or 'author not given'}, {r.url}")
    lines += canada_gt_credits(raw_dirs, valid)
    lines += gtnp_platform_credits(raw_dirs, valid)
    (out / "data_credits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_source_agreement(rec: pd.DataFrame) -> pd.DataFrame:
    """
    For every physical site read from more than one source, each year's value of each
    source against the preferred source's value (CONFIG priority). Shows, for example,
    how the GTN-P platform grids compare with PANGAEA 972777, and how close the copies of
    one borehole in two compilations are.
    """
    r = rec.assign(prio=rec.source.map(CONFIG["priority"]).fillna(CONFIG["priority"]["other"]))
    multi = r.groupby("site_id").source.nunique()
    r = r[r.site_id.isin(multi[multi > 1].index)]
    if r.empty:
        return pd.DataFrame()
    v = (r.sort_values("prio").groupby(["site_id", "year", "source"], sort=False)
         .agg(alt_cm=("alt_cm", "mean"), censored=("censored", "max"), flag=("flag", "first"),
              prio=("prio", "first")).reset_index())
    ref = v.sort_values("prio").groupby(["site_id", "year"]).head(1)
    ref = ref.rename(columns={"source": "ref_source", "alt_cm": "ref_alt_cm",
                              "censored": "ref_censored", "flag": "ref_flag"}).drop(columns="prio")
    m = v.merge(ref, on=["site_id", "year"])
    m = m[m.source != m.ref_source].drop(columns="prio")
    m["diff_cm"] = m.alt_cm - m.ref_alt_cm
    return m.sort_values(["site_id", "source", "year"])


def check_g02189(rec: pd.DataFrame, max_km: float = 20.0) -> pd.DataFrame:
    """
    NSIDC G02189 July thaw depth (0 C crossing of the July mean profile, a lower limit of
    ALT) against the Russian ALT series (ru_hms_alt, 1930-1990) of the same station
    (within max_km; both give positions to 0.1 degree): one row per station and year.
    'alt_at_least_july' should be True in nearly every year. A check of both products
    before 2000; G02189 is never used as data.
    """
    g = rec[rec.source == "g02189"]
    h = rec[rec.source == "ru_hms_alt"]
    rows = []
    if g.empty or h.empty:
        return pd.DataFrame()
    hs = h.groupby("source_site_id").agg(lat=("lat", "median"), lon=("lon", "median"),
                                         name=("site_name", "first"))
    for sid, gg in g.groupby("source_site_id"):
        d = haversine_km(gg.lat.median(), gg.lon.median(), hs.lat.to_numpy(), hs.lon.to_numpy())
        if d.min() > max_km:
            continue
        hid = hs.index[int(np.argmin(d))]
        hh = h[h.source_site_id == hid].groupby("year").alt_cm.mean()
        for yr, x in gg.groupby("year").alt_cm.mean().items():
            rows.append(dict(g02189_series=sid, ru_hms_alt_series=hid, station=hs.name[hid],
                             distance_km=round(float(d.min()), 2), year=yr, g02189_alt_cm=x,
                             ru_hms_alt_cm=hh.get(yr, np.nan),
                             g02189_flag=gg[gg.year == yr].flag.iloc[0]))
    m = pd.DataFrame(rows)
    if len(m):
        m["diff_cm"] = m.g02189_alt_cm - m.ru_hms_alt_cm
        m["alt_at_least_july"] = ((m.ru_hms_alt_cm >= m.g02189_alt_cm).astype(float)
                                .where(m.ru_hms_alt_cm.notna()))
    return m


def gtnp_platform_credits(raw_dirs, valid: pd.DataFrame) -> list[str]:
    """Citation of each GTN-P platform borehole or grid that gives a used value."""
    used = valid[valid.source.isin(["gtnp_platform", "gtnp_platform_alt"])]
    if used.empty:
        return []
    ids = set(used.source_site_id.str.extract(r"GTNP_(?:PT|ALT)_(\d+)_")[0].dropna())
    lines = ["", "## GTN-P data platform: series used and their citations", ""]
    for root in raw_dirs:
        for sub in ("pt_meta", "alt_meta"):
            for m in sorted((Path(root) / API_FOLDERS["gtnp"] / sub).glob("*.json")):
                if m.stem in ids:
                    meta = json.loads(m.read_text())
                    lines.append(f"- {meta.get('name', m.stem)}: "
                                 f"{meta.get('generic_citation') or 'no citation given on the platform'}")
    return lines


def build(raw_dirs: list[Path], out: Path, only: list[str] | None = None) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    log("== STAGE 2: read")
    parts, inventory, failed = [], [], []
    for name, fn in ADAPTERS.items():
        if only and name not in only:
            continue
        log(f"[{name}]")
        try:
            df = fn(raw_dirs)
        except Exception as e:
            # one broken reader does not stop the others, but the run ends with an error
            # and the reader is named, so an incomplete target is never taken as complete
            log(f"  {name}: FAILED ({type(e).__name__}: {e})")
            failed.append(name)
            df = pd.DataFrame()
        inventory.append(dict(source=name, series_read=df.source_site_id.nunique() if len(df) else 0,
                              site_years_read=len(df), reader_failed=name in failed))
        if len(df):
            parts.append(df)
    if not parts:
        log("\nNo reader returned any record: nothing to build.")
        (out / "build_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
        return failed or ["no_records"]
    rec = pd.concat(parts, ignore_index=True)
    out_of_scope = ~in_scope(rec)
    log(f"\n{int(out_of_scope.sum())} site-years outside the scope (south of {CONFIG['min_lat']} N or "
        f"excluded mountain regions)")
    rec = rec[~out_of_scope]
    rec.to_csv(out / "alt_multisource_all_records.csv", index=False)

    if UNREAD_TABLES:
        pd.DataFrame(UNREAD_TABLES).to_csv(out / "unread_tables.csv", index=False)
        log(f"{len(UNREAD_TABLES)} tables of the new sources not read (unread_tables.csv)")

    log("\n== STAGE 3: build\n[merge]")
    rec = assign_physical_sites(drop_short_series(rec))
    rec = site_state_flags(rec)
    agree = check_source_agreement(rec)
    agree.to_csv(out / "check_source_agreement.csv", index=False)
    pre = check_g02189(rec)
    if len(pre):
        pre.to_csv(out / "check_pre2000_g02189.csv", index=False)
        n = pre.diff_cm.notna().sum()
        log(f"  G02189 check: {pre.g02189_series.nunique()} stations, {n} years shared with "
            f"ru_hms_alt; ALT at least the July thaw depth in "
            f"{pre.alt_at_least_july.mean() * 100 if n else float('nan'):.0f}% of them")
    an = merge_site_years(rec)
    ux = outlier_flags(an).reindex(an.index)
    has = ux != ""
    an.loc[has, "flag"] = (an.loc[has, "flag"] + ";" + ux[has]).str.strip(";")
    # unusual years stay in: most are regional cold or warm summers, i.e. signal
    an["use"] = an.valid
    nval = an[an.use].groupby("site_id").year.nunique()
    keep = nval[nval >= CONFIG["min_years"]].index
    an_keep = an[an.site_id.isin(keep)].sort_values(["site_id", "year"])
    an_keep.to_csv(out / "alt_multisource_annual.csv", index=False)

    valid = an_keep[an_keep.use]
    # name, country and CALM code from the preferred source of each site
    by_pref = an_keep.assign(calm_code=an_keep.calm_code.replace("", np.nan),
                             prio=an_keep.source.map(CONFIG["priority"]).fillna(CONFIG["priority"]["other"]))
    by_pref = by_pref.sort_values(["prio", "year"])
    sites = (by_pref.groupby("site_id")
             .agg(site_name=("site_name", "first"), lat=("lat", "median"), lon=("lon", "median"),
                  country=("country", "first"), calm_code=("calm_code", "first"),
                  method=("method", lambda m: m.mode().iat[0]),
                  sources=("source", lambda s: ";".join(sorted(set(s)))),
                  first_year=("year", "min"), last_year=("year", "max"),
                  site_note=("site_note", "first"))
             .reset_index())
    vs = valid.groupby("site_id").agg(n_years_valid=("year", "nunique"), median_alt_cm=("alt_cm", "median"),
                                      sd_alt_cm=("alt_cm", "std"),
                                      first_valid_year=("year", "min"), last_valid_year=("year", "max"),
                                      n_years_before_1990=("year", lambda y: int((y < 1990).sum())))
    sites = sites.merge(vs, on="site_id", how="left")
    sites["colocated_with"] = colocated(sites)
    sites.to_csv(out / "alt_multisource_sites.csv", index=False)

    # ALDI input: valid years only. 'ALT_cm' is the site-average end-of-season thaw depth
    # (not the grid maximum that the old 'Max' column held); Lat/Long are the site's
    # median coordinates so every row of a site has the same position.
    aldi = valid.copy()
    pos = aldi.groupby("site_id")[["lat", "lon"]].transform("median")
    aldi["Lat"], aldi["Long"] = pos.lat, pos.lon
    aldi = aldi.rename(columns={"site_name": "Location", "year": "Year", "alt_cm": "ALT_cm"})
    aldi["Location"] = aldi.site_id.map(sites.set_index("site_id").site_name)
    # survey day of year where the source gives a date (probing): later surveys read
    # deeper, so a drift of the survey date over the years can look like a trend
    aldi["obs_doy"] = pd.to_datetime(aldi.obs_date, errors="coerce").dt.dayofyear
    aldi[["site_id", "Location", "Lat", "Long", "Year", "ALT_cm", "source", "method",
          "obs_doy", "flag"]].to_csv(out / "alt_multisource_for_ALDI.csv", index=False)

    # inventory: every reader, then the sources reviewed and not used
    inv = pd.DataFrame(inventory)
    used = valid.groupby("source").agg(sites_contributing=("site_id", "nunique"),
                                       site_years_used=("year", "size"))
    inv = inv.merge(used, left_on="source", right_index=True, how="left").fillna(
        {"sites_contributing": 0, "site_years_used": 0})
    in_kept = rec[rec.site_id.isin(keep)].groupby("source").site_id.nunique()
    inv["sites_with_records"] = inv.source.map(in_kept).fillna(0)
    inv["title"] = inv.source.map(lambda s: SOURCES.get(s, ("", "", ""))[0])
    inv["reference"] = inv.source.map(lambda s: SOURCES.get(s, ("", "", ""))[1])
    inv["licence"] = inv.source.map(lambda s: SOURCES.get(s, ("", "", ""))[2])
    inv["status"] = np.select(
        [inv.site_years_used > 0, inv.sites_with_records > 0, inv.site_years_read > 0],
        ["used", "read; its years are already in a preferred source for the same sites",
         "read; no site reaches the year threshold"], "not read (files missing)")
    inv.loc[inv.reader_failed.fillna(False).astype(bool), "status"] = "reader FAILED (see build_log.txt)"
    inv["reason_not_used"] = inv.source.map(READ_NOT_USED).where(inv.site_years_used == 0).fillna("")
    nu = pd.DataFrame([dict(source="", title=t, reference=u, status="not used", reason_not_used=why)
                       for t, u, why in NOT_USED])
    inv = pd.concat([inv, nu], ignore_index=True)
    cols = ["source", "title", "reference", "licence", "status", "series_read", "site_years_read",
            "sites_contributing", "site_years_used", "reason_not_used"]
    inv[cols].to_csv(out / "source_inventory.csv", index=False)
    write_credits(out, inv, raw_dirs, valid)

    log(f"\nRESULT: {len(keep)} sites with >= {CONFIG['min_years']} valid annual values, "
        f"{len(valid)} valid site-years")
    for k in (10, 15, 20, 30):
        log(f"  sites with >= {k} valid years: {int((nval >= k).sum())}")
    log("  sites by country: " + ", ".join(f"{c} {n}" for c, n in sites.country.value_counts().items()))
    log("  sites by method: " + ", ".join(f"{c} {n}" for c, n in sites.method.value_counts().items()))
    if failed:
        log(f"\nWARNING: these readers FAILED, so their sources are missing from the target: "
            f"{', '.join(failed)}")
    (out / "build_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    return failed


NEEDED = ["numpy", "pandas", "requests", "xlrd", "openpyxl", "pdfplumber", "truststore",
          "xarray", "netCDF4"]


def require(modules):
    """Stop before anything is touched if this Python lacks a package."""
    import importlib.util
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        pip = " ".join(missing)
        sys.exit(f"This Python ({sys.version.split()[0]}) is missing: {pip}. Nothing was changed.\n"
                 f"Run it with  py -3.12 {Path(sys.argv[0]).name}  or install them:\n"
                 f"    \"{sys.executable}\" -m pip install {pip}")


def publish(work: Path, final: Path) -> None:
    """The build succeeded: the current output becomes data/previous/<name> (one copy is
    kept), and the new one takes its place. Folders are renamed, not copied: on Windows a
    folder open in Explorer, Excel or a terminal cannot be renamed, so each rename is
    tried for 10 s and then the script stops with a message (nothing is half moved)."""
    import shutil

    def rename(src: Path, dst: Path) -> None:
        for _ in range(5):
            try:
                src.rename(dst)
                return
            except PermissionError:
                time.sleep(2)
        sys.exit(f"Could not rename {src} to {dst}: a program (Explorer, Excel, a terminal) is "
                 f"using it. Close it and rename by hand. The new build is in {work}.")

    if final.exists():
        prev = final.parent / "previous" / final.name
        if prev.exists():
            shutil.rmtree(prev)
        prev.parent.mkdir(parents=True, exist_ok=True)
        rename(final, prev)
    rename(work, final)


def main() -> None:
    import shutil
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", type=Path, nargs="+", default=[RAW],
                    help="raw data folder(s); downloads go to the first (default data/raw/alt_sources)")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default data/1_alt_target)")
    ap.add_argument("--min-years", type=int, default=CONFIG["min_years"])
    ap.add_argument("--only", nargs="*", help="download and read only these sources (for testing)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--download", action="store_true", help="download new or missing source files first")
    g.add_argument("--download-only", action="store_true", help="only the download stage")
    a = ap.parse_args()
    require(NEEDED)
    CONFIG["min_years"] = a.min_years
    work = a.out.with_name(a.out.name + "_building")
    if work.exists():
        shutil.rmtree(work)                  # leftover of an interrupted run
    work.mkdir(parents=True)
    if a.download or a.download_only:
        download_all(a.raw[0], work, a.only)
    if a.download_only:
        (work / "build_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
        print(f"Downloads done; report in {work}")
        return
    failed = build(a.raw, work, a.only)
    if failed:
        sys.exit(f"\nStopped: readers failed ({', '.join(failed)}). {a.out} was NOT changed; "
                 f"this build is in {work}")
    publish(work, a.out)
    print(f"\nDone. ALT target in {a.out}. Next: py -3.12 2_covariates.py")


# =============================================================================
# SOURCE_FILES: every file with a known link (reader, where it goes in the raw folder,
# link, checksum, size, licence, access, note). The download stage reads this list.
# =============================================================================
SOURCE_FILES = """\
reader,local_path,url,sha256,bytes,licence,access,note
above_field,ABoVE_field/ABoVE_Soil_ThawDepth_Moisture_Validation.csv,https://daac.ornl.gov (ABoVE_Soil_ThawDepth_Moisture_Validation),,,ORNL DAAC (cite dataset),login,ORNL DAAC needs a free NASA Earthdata login: download ABoVE_Soil_ThawDepth_Moisture_Validation*.csv into ABoVE_field/
allena,ALLena_Lena/PANGAEA_974408.zip,https://doi.pangaea.de/10.1594/PANGAEA.974408?format=zip,529f6fb049d92bc0019d87a5e44de3cd858859dbac5b2fc6e2cc2772e9d59c6f,382357,https://creativecommons.org/licenses/by/4.0/,open,"ALLena: Compilation of thaw depth measurements of the active layer in the Lena River Delta region from 1998 to 2022, Northeastern Siberia"
arc_lter,ARC_LTER_EDI/knb-lter-arc.1019.10_1990_present_TW_thaw_Kling.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-arc%2F1019%2F10%2F3b75c0fe1ec20ea5b898384885daf2c5,7ad893fcb0f2f3f52c801c6ccc4878d5b3a21a77f04a2bca97b411c06b62329c,3551,CC-BY-4.0 (EDI/LTER),open,"Tussock watershed thaw depth survey summary for 1990 to present, Arctic Long-Term Ecological Research (LTER), Toolik Research Station, Alaska."
arc_lter,ARC_LTER_EDI/knb-lter-arc.1019.10_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-arc%2F1019%2F10,f77f0ee0b271351cc8410ac08c1b7df2dfb52ca695be4b4caf1a44fef76f1b70,26205,CC-BY-4.0 (EDI/LTER),open,"Tussock watershed thaw depth survey summary for 1990 to present, Arctic Long-Term Ecological Research (LTER), Toolik Research Station, Alaska."
arc_lter,ARC_LTER_EDI/knb-lter-arc.1610.10_1993_to_currentsoilthaw.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-arc%2F1610%2F10%2F2af989579fccfa475ca5d2c5f5d45459,bfda6d3d05cd4b2e607c38be70fa29a2930f828aa38534ac26499fa3cf89dd73,101009,CC-BY-4.0 (EDI/LTER),open,"Late season thaw depth measured in the Arctic Long Term Ecological Research (ARC LTER)  moist acidic tussock experimental plots at Toolik Field station, Alaska Arctic 1993 to 2021"
arc_lter,ARC_LTER_EDI/knb-lter-arc.1610.10_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-arc%2F1610%2F10,7b5996f7bcb407f36c72a2e8cff2fdc7a60552b6acc0c80fb7fd8ae4dd034243,33404,CC-BY-4.0 (EDI/LTER),open,"Late season thaw depth measured in the Arctic Long Term Ecological Research (ARC LTER)  moist acidic tussock experimental plots at Toolik Field station, Alaska Arctic 1993 to 2021"
arc_lter,ARC_LTER_EDI/knb-lter-arc.1626.14_2003_present_Imnavait_Thaw_Kling.14.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-arc%2F1626%2F14%2Fb283c2073b3a99bde835513d2ad1d6f1,cd7ed096e0d12e29404c121bfa84f546e8c0cfc3de89fa38fab86507358f716b,4389,CC-BY-4.0 (EDI/LTER),open,"Imnavait Watershed Thaw Depth Survey Summary for 2003 to 2024, Arctic LTER, Toolik Research Station, Alaska."
arc_lter,ARC_LTER_EDI/knb-lter-arc.1626.14_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-arc%2F1626%2F14,63b3bad4ddaf958b4a922f3b6ded0980d4de9dac23ae2001571c54c920c5ae37,25504,CC-BY-4.0 (EDI/LTER),open,"Imnavait Watershed Thaw Depth Survey Summary for 2003 to 2024, Arctic LTER, Toolik Research Station, Alaska."
awi_soil_temp,Samoylov/PANGAEA_947034_soil_lv1_2019.zip,https://download.pangaea.de/dataset/947034/files/soil_lv1_2019.zip,6509b697cb898d4e725aab9769acc2a291f90eadcd651d5fc689a257937b6e37,3154118,https://creativecommons.org/licenses/by/4.0/,open,Soil data at station Samoylov (2019)
awi_soil_temp,Samoylov/PANGAEA_949198_soil_lv1_2020.dat,https://download.pangaea.de/dataset/949198/files/soil_lv1_2020.dat,16779b4aa9db6b738d498428e2cdf3ccd93cd2169f94a9b7d4af902b0bcfda28,11849192,https://creativecommons.org/licenses/by/4.0/,open,Soil data at station Samoylov (2020)
awi_soil_temp,Samoylov/PANGAEA_974420_soil_lv1_2021_cut.dat,https://download.pangaea.de/dataset/974420/files/soil_lv1_2021_cut.dat,e9af161ebd0507fa897710ea3c3505ab9fb064cce282a91cbb51526f6d1d6122,7983664,https://creativecommons.org/licenses/by/4.0/,open,Soil data at station Samoylov (2021)
awi_soil_temp,Samoylov/soil_lv2_V201908.zip,https://store.pangaea.de/Publications/Boike-etal_2019_V201908/soil_lv2_V201908.zip,c7c61def0d76046f706a938325282c0fc821f7de6bcf947be75db90c2cfcc5b2,8546543,CC-BY-4.0,open,"Soil data at station Samoylov (2002-2018, level 2, version 201908)"
awi_soil_temp,Svalbard_Bayelva/PANGAEA_982254.tab,https://doi.pangaea.de/10.1594/PANGAEA.982254?format=textfile,10b2e7dc4eab8886adfa024408aec69900992e493376df96efd258bcaaf1cfb4,45674845,https://creativecommons.org/licenses/by/4.0/,open,"Meteorological and soil data from Bayelva, Svalbard, homogenised data product (version 2: 1998-09-13 to 2024-12-31, design 2: one row per time-step)"
bnz_lter,ALASKA_other/knb-lter-bnz.11.26_11_bcefcore_active_layer_2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F11%2F26%2F2f161f57a6d6789dbb14f8396fb6f1ff,37ec6892d522d976dbf0119615632590f8260d8a5a497d8857f3b665918c955b,87517,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths at Core Floodplain Sites from 1990 to Present in the Bonanza Creek Experimental Forest near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.11.26_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F11%2F26,3831485dd6b9f6aab082c31617055a66e1f783e14bd85f2d4e98ab6a85b8c6f8,36171,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths at Core Floodplain Sites from 1990 to Present in the Bonanza Creek Experimental Forest near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.239.28_239_boundaryfire_active_layer_2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F239%2F28%2F6b9e26cbbcd1793c2c5ab129428aaa25,063735f0292cdbcdbc3c98b2ab908dda1c90e09080b9817cd7905e6ab976d805,41807,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 2004 to Present in the Boundary Fire Fireline near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.239.28_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F239%2F28,572c05ddc2b39789da8abf2ca2af9b499a91bc4479ef09eb040948a49edd7cda,33760,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 2004 to Present in the Boundary Fire Fireline near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.240.27_240_surveylinefire_active_layer_2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F240%2F27%2F3606a339c0a47211fbd153ae83b0e7ce,5a6e43c207fd81c6fe9d2a9f8205ce0a9ab079c179cbb3b8e4372f51128609b2,28208,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 2002 to Present in the Survey Line Fire near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.240.27_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F240%2F27,c1311319584781af28ab91263c6a96cb9db97ef09b4163510e47540585d2c7bf,31619,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 2002 to Present in the Survey Line Fire near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.480.26_480_EML_AK_CiPEHR_thaw_depth_data_2009-2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F480%2F26%2F078a1a000583d76fd8eb88bd54a5a4a5,6aca8bf87521f6a3720208b35c6ec51f9313d8c1dba34f80d5fd3e5c8fca1357,1016042,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake CiPEHR: Weekly thaw depth data, 2009-2025"
bnz_lter,ALASKA_other/knb-lter-bnz.480.26_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F480%2F26,7d7270dc8243452d373a83ace2e8d1114603a2c71a5cde7c0ab719ee09c709ea,39676,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake Research Watershed, Carbon in Permafrost Experimental Heating Research (CiPEHR): Weekly thaw depth data, 2009-2025"
bnz_lter,ALASKA_other/knb-lter-bnz.496.23_496_EML_AK_DryPEHR_thaw_depth_data_2011-2022.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F496%2F23%2F7384a9b678bfc0eea2999e282603362a,b9a7a078a3c37074064a0cb3aee7fb8d4563f57dc8ffa89a7fd072a9b3cd9edf,204004,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake Research Watershed, Carbon in Permafrost Experimental Heating and Drying Research (DryPEHR): Weekly thaw depth data, 2011-2022"
bnz_lter,ALASKA_other/knb-lter-bnz.496.23_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F496%2F23,c549ae140eb74451f8d3e2949c344cbbc90db0a18da7c40ed652f4d9c28c8058,38433,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake Research Watershed, Carbon in Permafrost Experimental Heating and Drying Research (DryPEHR): Weekly thaw depth data, 2011-2022"
bnz_lter,ALASKA_other/knb-lter-bnz.518.11_518_ActiveLayerDepth.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F518%2F11%2F388f647373b07d3b2d7b36e6de6a0a13,fd7913abe162a5ee85d889839ce6fc97c4a5daaf3df1c6d1494d28569678cab9,17914,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake Research Watershed, Thaw Gradient: Active Layer thickness 2004-2013."
bnz_lter,ALASKA_other/knb-lter-bnz.518.11_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F518%2F11,d4c4dd541e11c82564b26929cc61a302c6ab8b002d2f0394a0cc66215c84d503,32136,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake Research Watershed, Thaw Gradient: Active Layer thickness 2004-2013."
bnz_lter,ALASKA_other/knb-lter-bnz.519.24_519_EML_Gradient_Thaw_Depth_Data_2004-2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F519%2F24%2F5841cfcd5362e3af5fb6ba516dc34863,d77a2d9f279ffd2a835c2800132ca61c2504daaa43c4f2ce56883e5db89886c6,312018,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake Research Watershed, Thaw Gradient: Seasonal thaw depth 2004-2025"
bnz_lter,ALASKA_other/knb-lter-bnz.519.24_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F519%2F24,25b3d14d6aa366b6519cc58fed153a578569e9b81f86c159a08c333d10964eac,38041,CC-BY-4.0 (EDI/LTER),open,"Eight Mile Lake Research Watershed, Thaw Gradient: Seasonal thaw depth 2004-2025"
bnz_lter,ALASKA_other/knb-lter-bnz.605.8_605_RSN_ActiveLayerDepths_2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F605%2F8%2F1601cdfa06349f295d1de7dd757811ec,5415d098be1448289635bc840a54ecbd59e34bc96d3d9e9210d73a74f7d6b722,370265,CC-BY-4.0 (EDI/LTER),open,Bonanza Creek LTER: Active Layer Depth or Permafrost Presence for the Regional Site Network
bnz_lter,ALASKA_other/knb-lter-bnz.605.8_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F605%2F8,3f48c5e40866501835dc07714c8c92829c5b1f3a87f62cadf84814ae09766414,97339,CC-BY-4.0 (EDI/LTER),open,Bonanza Creek LTER: Active Layer Depth or Permafrost Presence for the Regional Site Network
bnz_lter,ALASKA_other/knb-lter-bnz.7.29_7_rosiecreek_active_layer_2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F7%2F29%2Fc9d57b3d5cf02cfc263b948284745369,5e4aadb532be4096bd6c4feaab5d01d6ea68384520487a1a6f47e9e386fb61a0,50600,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 1984 to Present in the Bonanza Creek Fireline near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.7.29_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F7%2F29,94bb9664f46eb9a72a3aa793986f4245948447c8f5241e2898a7e596db215f19,36719,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 1984 to Present in the Bonanza Creek Fireline near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.8.27_8_WickershamFire_active_layer_2025.csv,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fdata%2Feml%2Fknb-lter-bnz%2F8%2F27%2Fd0de335a96178fb9a20c897e812e21ea,30a110a1c8f58fa455d4152da08fbfc3e307dbb7a6cc0169343c629a915c774e,111682,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 1972 to Present in the Wickersham Fireline Sites near Fairbanks, Alaska"
bnz_lter,ALASKA_other/knb-lter-bnz.8.27_EML.xml,https://gmn.lternet.edu/mn/v2/object/https:%2F%2Fpasta.lternet.edu%2Fpackage%2Fmetadata%2Feml%2Fknb-lter-bnz%2F8%2F27,0770f1ca431cbe7f5055ddeb84461f3693b08079dcddc2e2f6ac8b8da3bd3d7d,49346,CC-BY-4.0 (EDI/LTER),open,"Bonanza Creek LTER: Annual Active Layer Depths from 1972 to Present in the Wickersham Fireline Sites near Fairbanks, Alaska"
calm_site_files,ADC_sweep/CALM_U2_Barrow_CRREL/U2_alt_1962_2024summary.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A583cbbca-a852-470c-9e9b-ddb5f6550c0e,943d80a82c10f3ce3593b475fda9dfa6bff2ec5cf7f1b6d98b1439c6603116ae,1023,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth measurements from the Circumpolar Active Layer Monitoring (CALM) project for site 'Barrow, CRREL Plots' (U2), Alaska North Slope, United States, Alaska (1962-2024)"
calm_site_files,Abisko_Tornetrask/CALM_S2_Abisko_alt_78_16.xls,https://web.archive.org/web/20180525085500id_/https://www2.gwu.edu/~calm/data/CALM_Data/Europe/Nordic/s02_abisko_area/S2_alt_abisko_78_16.xls,24c583a4476c130f6754182ef6f081b40140ed35ad6eb55698cfcea51313d6b0,70656,CALM public data (via Internet Archive snapshot 2018-05-25; live GWU server blocks xls),open,CALM S2 Abisko area ALT 1978-2016
calm_site_files,GEM_Greenland/CALM_G1_ZEROCALM1_1996-2017.xlsx,https://web.archive.org/web/20180525085500id_/https://www2.gwu.edu/~calm/data/CALM_Data/Greenland/g01_zackenberg/G1_1996-2017.xlsx,007eba0f34f3687a151043ee26052fbab4df41c52068f1647d6c5736e5c66520,158491,CALM public data (via Internet Archive snapshot 2018-05-25; live GWU server blocks xls),open,CALM G1 Zackenberg ZEROCALM-1 thaw depth 1996-2017
calm_site_files,GEM_Greenland/CALM_G2_ZEROCALM2_1996_2017.xls,https://web.archive.org/web/20180525085500id_/https://www2.gwu.edu/~calm/data/CALM_Data/Greenland/g02_zackenberg/G2_1996_2017.xls,8566e3cd46e6287e4248067dea413d98c1343eecbc4a07b0ebb58d5b5124c225,434176,CALM public data (via Internet Archive snapshot 2018-05-25; live GWU server blocks xls),open,CALM G2 Zackenberg ZEROCALM-2 thaw depth 1996-2017
calm_site_files,Svalbard_UNIS/CALM_P1_Calypsostranda_alt_2009.xls,https://web.archive.org/web/20180525085500id_/https://www2.gwu.edu/~calm/data/CALM_Data/Europe/Nordic/p01_calypsos_tranda/alt_2009.xls,ecf98dc153e78f6acc08fee421026290e1811760a30a5f99a1678ea8d5e1cf82,27648,CALM public data (via Internet Archive snapshot 2018-05-25; live GWU server blocks xls),open,CALM P1 Calypsostranda ALT
calm_site_files,Svalbard_UNIS/CALM_S1_KappLinne_alt_1972_2011.xls,https://web.archive.org/web/20180525085500id_/https://www2.gwu.edu/~calm/data/CALM_Data/Europe/Nordic/s01_kappe_linne/s01_alt_2011.xls,3a28fc1ef4cfbaf929dbea29ed89ee6446c1d6a782a2d33f6a184c49157d14c1,50688,CALM public data (via Internet Archive snapshot 2018-05-25; live GWU server blocks xls),open,CALM S1 Kapp Linne ALT 1972-2011
calm_site_files;gem;pangaea_calm;parks_canada;samoylov_calm,GTNP_CALM_PANGAEA/PANGAEA_972777.tab,https://doi.pangaea.de/10.1594/PANGAEA.972777?format=textfile,4198a0e06b76bb3d764656cc201cda417adfc2b4cfc7c0f67a186cc29ef66045,479369,https://creativecommons.org/licenses/by/4.0/,open,GTN-P CALM: 35 years of Active Layer Thickness (ALT) across latitudinal and elevational gradients in the Northern Hemisphere
canada_gt,Canada_GT_collection_2025/all_observations.zip,https://zenodo.org/records/21042525/files/all_observations.zip?download=1,9329c971d99a2be2d6f221156f611dbf75f6750172744c85672f5e7219f88702,40985959,see the Zenodo record,open,A standardized permafrost ground temperature collection for Canada 2025 (Zenodo 21042525)
canada_gt,Canada_GT_collection_2025/site_metadata.csv,https://zenodo.org/records/21042525/files/site_metadata.csv?download=1,f96bc7f200dce253746d568e3ffdc72e240d1242887e9b6e775d1c0a3588f736,564428,see the Zenodo record,open,A standardized permafrost ground temperature collection for Canada 2025 (Zenodo 21042525)
cusp,Zenodo_CUSP/CUSP_v1.2.zip,https://zenodo.org/api/records/23171354/files/v1.2.zip/content,ca4dee8e624aca480c22378f2e08c317ff403dde8947f504928786a8bc624195,2112456,CC-BY-4.0,open,CUSP - CommUnity near-Surface Permafrost dataset v1.2
gem,GEM_Greenland/AWS1-GroundTemperature_10.17897_B2K2-0Q82.zip,https://doi.org/10.17897/B2K2-0Q82,d221d7f833ee627012fce71d749f76d98e233570e86028c21ee27587a4e610f5,1944320,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gem,GEM_Greenland/AWS2-GroundTemperatureP1_10.17897_3T7W-7638.zip,https://doi.org/10.17897/3T7W-7638,42d813487c96a545bc2d2b4269bb80e8b9348d47eede30326b48ff445e07f84e,925019,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gem,GEM_Greenland/AWS2-GroundTemperatureP2_10.17897_8HPH-H716.zip,https://doi.org/10.17897/8HPH-H716,e20ea85087e3bbb66934f9246006e31902bc1f4f09952f1ace3d85d9fb0a9fc1,1272322,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gem,GEM_Greenland/ZEROCALM-1_Active_layer_10.17897_2Q7S-2876.zip,https://doi.org/10.17897/2Q7S-2876,ca6de3c84dc54f2722d2476b73c08ecdbad5a5824d751adc697c0ae33388c543,276630,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gem,GEM_Greenland/ZEROCALM-2 middle_Soil_temperature_10.17897_D2NV-GN54.zip,https://doi.org/10.17897/D2NV-GN54,64ce111ca22e4a918f7ef24aff22d96378bc9c6d259958c0392b4bb9263bf374,311473,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gem,GEM_Greenland/ZEROCALM-2 upper_Soil_temperature_10.17897_KCP6-QG90.zip,https://doi.org/10.17897/KCP6-QG90,dbc61791fbcba60f0dd160744293ac516c513b081887647453a2f430dd3d6069,449031,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gem,GEM_Greenland/ZEROCALM-2_Active_layer_10.17897_4EE8-WE92.zip,https://doi.org/10.17897/4EE8-WE92,1d57d75581c474d78ea1d0979ca255e9c0845786e55d972daef3855b7f1b4ffc,430115,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gem,GEM_Greenland/ZEROCALM-2_lower_Soil_temperature_10.17897_FX4T-2H83.zip,https://doi.org/10.17897/FX4T-2H83,62f4a24869108e270220bf7f6d6efd599aa5398378ab86263b8cc49f281d173c,429087,CC-BY-SA 4.0,login,"GEM data need a free account at data.g-e-m.dk: download the dataset zips of doi 10.17897/2Q7S-2876, 4EE8-WE92, B2K2-0Q82, 3T7W-7638, 8HPH-H716, KCP6-QG90, D2NV-GN54, FX4T-2H83 into GEM_Greenland/"
gsc_cable;gsc_thaw_tube,GSC_Mackenzie/of_9340e.zip,https://ostr-backend-prod.azure.cloud.nrcan-rncan.gc.ca/server/api/core/bitstreams/a1a9b54f-13e7-4b80-bbc4-42f0af2d99fd/content,34f41fbac254c6480433d0e6b5d290595f63a8d5d540011156c47ac3c72bb86e,66520913,Open Government Licence - Canada,open,"Permafrost monitoring from 2006 to 2022 in the Mackenzie corridor, NWT (GSC Open File 9340)"
gsc_thaw_tube,GSC_Mackenzie/of_6287.zip,https://ostr-backend-prod.azure.cloud.nrcan-rncan.gc.ca/server/api/core/bitstreams/5a84e90d-90f7-4a4a-ac6f-ed75db9cfcc5/content,faa7b1d551e249c8f48a80792418fd2ed0a5a37b8b099484f79c469add6e2f46,13335422,Open Government Licence - Canada,open,"Data for Geological Survey of Canada active layer monitoring sites in the Mackenzie Valley, N.W.T. (GSC Open File 6287)"
illisarvik,ADC_sweep/NSIDC_GGD/ggd642_thawdepth_canada/ggd642_thawdepth_400m.txt,ftp://sidads.colorado.edu/pub/DATASETS/fgdc/ggd642_thawdepth_canada/ggd642_thawdepth_400m.txt,7fc14b8408513c6d50649f981f77ceb3e7ee29063c2b884e9e49695a29542723,1527,NSIDC public (cite dataset),open,"Annual Thaw Depths in Illisarvik, Northwest Territories (GGD642)"
illisarvik,ADC_sweep/NSIDC_GGD/ggd642_thawdepth_canada/ggd642_thawdepth_600m.txt,ftp://sidads.colorado.edu/pub/DATASETS/fgdc/ggd642_thawdepth_canada/ggd642_thawdepth_600m.txt,c4e7c7404467491b793d7b1a6aed2960233c77cf481dcd9a6322212ab6fb46fd,1941,NSIDC public (cite dataset),open,"Annual Thaw Depths in Illisarvik, Northwest Territories (GGD642)"
illisarvik,ADC_sweep/NSIDC_GGD/ggd642_thawdepth_canada/ggd642_thawdepth_tundra.txt,ftp://sidads.colorado.edu/pub/DATASETS/fgdc/ggd642_thawdepth_canada/ggd642_thawdepth_tundra.txt,9afc9f2e33c8962d10d1f1293a6d801c3974edd7f1ec951bd783db4e206ecdc0,909,NSIDC public (cite dataset),open,"Annual Thaw Depths in Illisarvik, Northwest Territories (GGD642)"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/1995_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A608601ef-31e4-424c-be42-d66ceb833416,e5c60f7899d8417be851cb14de9fcc2a2a965e41c62d0cf356bd6c98f5661b98,79882,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/1996_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A0503c60b-e3af-4952-a037-31eed0378142,bc02e5e02eba11ffe64eef8a5186bd61e9bc0e9492da7017f028cf781b164e36,76316,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/1997_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A2cd50dd2-13ff-411a-ac66-c90e56521d81,54b0a97cf14bb133060ff117a75ed10b1886e36d55a0e26ff87289935f7304a8,22652,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/1999_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Aa86f5264-5262-4125-890a-1c85d1f38f81,7f75d739378540f4f30faff1cc561485cf0bdaf32a211f11649d79341c382833,11204,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2000_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A3ad0ff4e-8ec3-4ae2-a784-32f370800522,b70e5f6da2ee80509344cbf766674984931dd567a0445cb8d6b29531318fc7a4,7445,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2001_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A6ff89a77-4b6d-4794-9a3f-474dab8083c9,df07fe9e90a7fdc6d003b7fecf008570066b483a1956306aaf5a16b8bfea167d,7481,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2002_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A993079ee-bc66-4e14-86b8-24f771491015,54af6f2cf5508c18a87744f5f4a2aac9219a316e204f6c3e2512d29ca06d9a3b,7492,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2003_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Ad8fba572-c1e3-4cb1-8e78-6c918f221403,2b7e062984f2934130ea6da98f2b91190cdd537246857ee3b96ebc75815f3af9,7494,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2006_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Aa3c235c6-39e0-4eb0-ade2-f46bcc1ee260,37b2cd15cdc2b1fb7f78eb1494d42fa9ab5117a4933638b1747cd5ec3439a95c,7520,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2007_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A7001980c-2d52-4ee8-ba5c-efc4aac9888f,0db50a7c9d90cebd34eb133382ab38d731e684658db644031582d2674e10af66,7557,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2008_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A8824e9f8-52d6-43cb-8411-4fac87abf455,27304ef3a87348e9ad7258f728c84155542c8cbfdeeba7b0fa7ea8a618174112,7504,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2009_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A0700c1f4-23f4-4f96-bd95-2d1cec8a253b,28ab0f381b4c04b7d8089290603737a4e85ffdaeffaa01ff8a0756bbababcb47,7540,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2010_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Aa3bc37e4-abb7-417c-878e-1c701faa00b0,247c8a5cd3de71a6e04a2bc4faa1ef7096ac0ba516be9cd81ceed08d9daa5774,7522,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2011_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A0f67b0e6-1778-4b14-a187-4f2b2c527b34,72b3e954f8a13179333e2e232a43031e94f70307db85a410a35545e68625ed91,7747,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2012_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A25ff71b9-08b9-4d58-9e39-63e7b10efedc,351121f81675303ef6bd1a49d7c421c1f8378cec758c903cd487657d20aa72b8,7515,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2013_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A4853809d-6a32-4b38-843f-0f67c666a381,9d199a557b936fd024e31a6aa90947d6fd4edb28b92c6bd1fab5ff4969434ecf,7730,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2014_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A5ee5c050-104e-46f2-905b-2d2f35c779a3,492a188808b9709e9742d37d837a5603042fc19fdbed74a0290bf92fbff479d5,7814,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2015_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A0d94fe48-7b4a-4565-8609-3497fa67e913,98047fcddf2c5d29baf6179907103a8dba4a547928479b8d2902d1cf702d8550,7844,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2016_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A6b531e27-d996-48d8-8c2b-134c9567f037,a9eca7d4ebeba1a69a2863bd1a94953a7c452125fb70a3441a41f2d35ea4937d,7439,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2017_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Aced15b6b-19e6-4bc6-b738-dfb7bdb7a658,d9708810290ac479200ff703baf0b0f2c50167dc263871bcf223c647485336dc,7500,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2018_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Aca2cfc97-14f1-4b9c-97ce-7168ff7561a2,c406a71d0fd790827d470c2ed49d61749438341bf9049943f149fddabb7475b6,7514,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2019_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Af5f4ac2c-99a3-42ef-bbba-81f72eb70690,e270ac54854ca058cda26529faf3866980b0ad21c390e2660e2f960ee1ac1cf5,7660,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2020_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Acbf94510-5031-4fcb-a710-0b66f194a23a,b4651c268c85f06cbe99f15cb3244a0420d3f06bdced3fb849b1b3ce4cbf27b4,7509,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2021_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A8c82b735-d4c5-4110-b3aa-53b0e18e9fc3,500dcec353dfb6f6379bbe44f8eea940101eba22f4a810e59c051b6a60c8a8f5,7413,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2022_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A79a1c545-ffbf-41f6-86cb-c1d8e93fddee,ebef03899cde223797aa478a1371bd0d1b4684f099aff01a5b0a9e71e8fdcea0,7704,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2023_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A99279943-b33c-4196-b14d-87ca03d19e71,649c5c0f8983827aaecd6b5ab5bbb2eba3fc94740388ef8c8303799a6a4ec103,7703,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
itex,ADC_sweep/ITEX_Utqiagvik_Atqasuk_thaw/2024_Barrow_Atqasuk_ITEX_Thaw_v1.csv,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3Ab28c8507-40a0-4c24-bcfe-f560cdac7e9f,7b1d82e36602f7eb3b38f6ccded29df5e68e71b861078c4c4a1cef112da72a91,7660,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Thaw depth in the International Tundra Experiment (ITEX) plots at UtqiaÄ¡vik and Atqasuk, Alaska"
kwbs,Kolyma_WBS/PANGAEA_881754.tab,https://doi.pangaea.de/10.1594/PANGAEA.881754?format=textfile,a805f0fb21b80b109d4e72ccbfb0f3de6213d5960bf8e373f5404208b5106301,1361993,https://creativecommons.org/licenses/by/3.0/,open,"Thaw depth and snow height time series at different sites within Kolyma Water-Balance Station (KWBS), 1954-1997"
kwbs,Kolyma_WBS/PANGAEA_881792.tab,https://doi.pangaea.de/10.1594/PANGAEA.881792?format=textfile,ad1823503574d0e9806756b9e931163efb5b586737b5d1ffc0dadb32532be3e6,156384,https://creativecommons.org/licenses/by/3.0/,open,"Soil temperature time series at the Nizhnyaya meteorological station within Kolyma Water-Balance Station (KWBS), 1974-1981"
kytalyk,Russia_PANGAEA/PANGAEA_961879.tab,https://doi.pangaea.de/10.1594/PANGAEA.961879?format=textfile,0a87a4c504c7161d64b25d47abecf3d8e3b55e634b68dffc6d8b3b0f072cf34d,20820,https://creativecommons.org/licenses/by/4.0/,open,Thaw depth of the landscape in northeastern Siberia (S5)
metno_janssonhaugen,METNO_Janssonhaugen_shallow/P11_Janssonhaugen-1999-2025.txt,https://cryo.met.no/en/permafrost/janssonhaugen,267585284dcc2d3b9e7dbefbdce64eae9fa1a51e0ab8301821a9bce33e6f0f07,3475025,"shared by the data owner; cite Isaksen et al. 2000, 2001, 2022",email,"Sent by K. Isaksen (MET Norway) by email on 10 November 2026 as a ZIP with this file and four papers; unpack it into METNO_Janssonhaugen_shallow/. Data after the August 2025 upgrade follow later; he plans to add the series to the GTN-P platform."
ngee_barrow,NGEE_Arctic/NGEE-BRW-ABCD-sensor-locations_w-o_figs_20170720.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-6011010b2ddffef-20221216T183635146238,714281556a67871aba6ad0d1432bd57b6fc857240a985d932a95bcefd19c2a66,3774,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2012-09-01_2013-09-30_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-24858052ca470a0-20221216T183550503153,8fe94e1b6714530ff026164ea2c7ee9219245a492ad13811adaff3732e7756cb,5163348,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2013-10-01_2014-09-30_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-de711f9f19ced31-20221216T183552434371,80b66968f3b28ada62a29117accc837b44fdb370a880a2dca3bc2f5f9b25bb99,4867385,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2014-10-01_2015-09-30_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-ef0d2dbb71f50ae-20221216T183553626103,830f7733b7ba3fc28607a7b7d84709650d428d2bd32b74f4554dbe0f2d6ab1a0,4837358,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2015-10-01_2016-09-30_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-3ec3f3312fdd5fa-20221216T183554718376,453b1876e6342b1b87ea46d3e47e58100f1dbcad7a3a333f305ab89ef0826a65,4357025,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2016-10-01_2017-09-30_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-767688eb339e41e-20221216T183556790317,064e13863902c18e5a1daf67a405d503d531ae47af5c8cef512bed7a501715f7,238264,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2017-10-01_2018-10-01_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-dd31f3e5e8b2442-20221216T183557611330,7713f1c13e7c8e85326bf18d27644be0ef3adba82c6a9d2c55cbe2c9c4fb411d,235444,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2018-10-01_2019-10-01_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-53fc5ec96539a92-20221216T183558542035,8b6bed0a9978bd16b406f87de147f3f1c741c4f6f1d8545a980497f9b54836d7,4676370,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2019-10-01_2020-10-01_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-9d2b3388a264465-20221216T183559638387,c7183b9634ffe53306b688b1f5d53d0d0a0c71501448952cf270f97430fd9d76,4726929,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2020-10-01_2021-08-09_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-04395c8a52ac6de-20221216T183600843006,d66a1e5e000e990a28fbd6bba02a5d1b0cc9cd22c56e92d99bb48cf40ab52860,4062608,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2021-08-10_2022-08-09_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-575da7d505dc1d2-20221216T183601927420,1a7115584f620a59e790ec9b5bf69b300a75c81b09c9d01995fa1abe6a3c7e5d,4122659,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_A_2022_08_10_2023_08_10_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-952985ac58ffc25-20250620T191912981,ca954cb1721248f5f97d0cd84f099a740313bf84136dd6506caf9a5da46967cc,4576324,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2012-09-09_2013-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-4d0630fa3705255-20221216T183603151569,f51edc38aacc1d458459a6219200336f14bafb16fd444f6c379ad1b290097f12,5453562,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2013-10-01_2014-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-3800e8397ee7483-20221216T183604361259,7cd73777a748eb5de5974b35384cfa75dd7b11eaf6845eb8d34f604295e8240d,5154876,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2014-10-01_2015-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-85850848fe81d1e-20221216T183605559609,b894a780dfc6499be28b47672b147452a411af2a5ec38613c75877f2e5303efd,5134456,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2015-10-01_2016-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-2c58d3c10fc1315-20221216T183606682491,d75cd54b62e44d307465a3fcbbac2808d396d250cdcfe20056b3dc403f4f3993,5145572,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2016-10-01_2017-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-88ec2ca45d00720-20221216T183607580619,b3925c7b5f31d1392b38aa2c64677aed6c23b29cc20cf93d89272b70d10d1b6b,253192,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2017-10-01_2018-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-c02c28211b7f25a-20221216T183609154526,7395b2129fca8544ac8f1886c29166474cbc037a38d78bb5ce757cca1aca9c23,249497,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2018-10-01_2019-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-b7c16fa5e08bce3-20221216T183610188985,c11d5b1833acbe432b128a7204d5d523f4ce8fb72537cd9b066d552b3118e55c,4976746,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2019-10-01_2020-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-9abda88f1ddd24c-20221216T183611331734,5754a23e28dcef678a6fb4b3e695f82abc8016bd26a382d7b558a19e95c880ce,5037143,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2020-10-01_2021-08-08_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-2389ca9cdec3167-20221216T183612430656,1782dff7e10d7a84086d8c88ca6c21846713f72849a2ed00748b49f483f04611,4313222,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2021-08-09_2022-08-08_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-3566f730de973db-20221216T183613515467,44f6ec06d41e30878e36406ee6684f90fcd821d8ce06fa038aeb3872bd4868a0,4420104,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_B_2022_08_09_2023_08_09_Thermal_Transect.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-85b8b730e80e2fd-20250620T191912996,7ef0774de68b09aaf8ad743541f7884956bbfc9a58511f5ae80df3d3af99492b,4908593,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2012-09-09_2013-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-12b213f9d6beee1-20221216T183614713213,ee9aa7be12c6236310ecf67725dfb2ef3e70f0e3b7aedb317475a014336481db,5431739,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2013-10-01_2014-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-3405d13dd51848d-20221216T183615880625,87e1cb123f9375a6dcc8a8f58f54a6eb7231b083da865dd13f191be8f7a9a1ef,5131703,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2014-10-01_2015-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-ec4cc6d013aa383-20221216T183617137703,2df5cf759a7405a943f566b86b6e7f809fd2123926eb833cd481032e280a5e8b,5067986,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2015-10-01_2016-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-687390a393d3759-20221216T183618271622,110280b681cc57e541e9ae77c18862d79fa0dfbb3223b9dd8148b3b77d84117c,4518710,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2016-10-01_2017-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-4a77330c7e315ae-20221216T183619328570,eaf1579bd4848600a49297c95a2225047ce2613509e5802c2b6974f27b53f10a,252937,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2017-10-01_2018-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-1716558677c65a2-20221216T183620102277,971ca8b42a61fa52e9fdfe98545929b854f2957b9adb4c2671cb3a5d6eede950,222023,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2018-10-01_2019-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-0bc229271373a5c-20221216T183620966993,ebacc2b1efbe3cd7f966fb48720594d2b8d29af29db7e7740313a8a6efbeddcf,1695300,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2019-10-01_2020-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-ec9170fef65708e-20221216T183623095805,5804edb856e9f8a5288d1bd9681c3d5809b8948ac4a3b2dd4ed6b7df3f4f041c,4510894,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_C_2020-10-01_2021-08-08_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-ddf45fd950826fb-20221216T183624061260,bb4ed43e6be72e53699d99a408e980aabf77b341cccb4c686b86d93c4a8cf40b,1418057,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2012-09-09_2013-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-540652ec0a079d8-20221216T183624578522,b8df0cbdc217fd6a67ccecfc3b184d38902bb80ecc093186b143b06a29f7150e,360710,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2013-10-01_2014-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-749d4bb56b2f8db-20221216T183625999787,d9cff1af9f1a1e9485d948c9d4093d6b9fc40262829536c54d2259d48d6a7438,5156435,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2014-10-01_2015-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-84634c4126884cb-20221216T183627055070,dcf2c3022e307ea0378c7c03b4ab109741a9076c794fb173a17f95dbd74c50e5,5115743,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2015-10-01_2016-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-9219654fcf9392d-20221216T183628171962,9cdc7e741e4c0b30aa74ed26f2a385cdb78e61d92680be68579fb0888bf135db,5107905,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2016-10-01_2017-09-30_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-01d378847655c91-20221216T183629196790,621f06ce3030e0790fc8a10ab85a3d1c63150462e0e6c643e7d593fa6899d28d,252800,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2017-10-01_2018-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-f230c7ab3ed1db1-20221216T183630032328,c72c6e45d5ffda368450b4585a9dd929ca54f7951c9eb9974351fbfacca0cb7f,249743,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2018-10-01_2019-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-94c435345272111-20221216T183630974642,f0b18e0208e997a84ec316bc1a1bb39113590e9a112782c5f6d4faf08c56b1a5,5002829,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2019-10-01_2020-10-01_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-2dfba04043cec33-20221216T183632103759,cddf1385a53d53e0e15be5345e1fa2e7289581da00eaf88e9b459dfafb3f622e,5068023,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2020-10-01_2021-08-07_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-2e5bf59334a5c86-20221216T183633154506,8c6ed5a99535cd3b6cb4ec220ee1776a513f4b1255519a81d47a256596f92698,4337735,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2021-08-08_2022-08-07_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-6bf698b33748a3c-20221216T183634204321,546822d495156825b192a5b2edbe052e60cd057191f2c3c5df75f1c0006f65be,4395363,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
ngee_barrow,NGEE_Arctic/NGEE_BRW_D_2022_08_08_2023_08_08_Thermal_Transect_and_Borehole.csv,https://data.ess-dive.lbl.gov/catalog/d1/mn/v2/object/ess-dive-40c80cc4b7b89e2-20250620T191913013,8171b68410d6b140fe241291a0c6012c18a269379e83362a74cd47d2ccc432be,4923336,http://creativecommons.org/licenses/by/4.0/,open,"Subsurface Temperature, Moisture, Thermal Conductivity and Heat Flux, Area A, B, C, and D, Utqiagvik (Barrow), Alaska, 2012-2023"
parks_canada,ParksCanada/auyuittuq_np_tundra_active-layer_2009-2025_calm-grid_data_1.csv,https://open.canada.ca/data/dataset/b5662fca-2082-428e-a9c3-3234822bbecd/resource/18f57da8-053d-47aa-a5d7-74daf9f56d15/download/auyuittuq_np_tundra_active-layer_2009-2025_calm-grid_data_1.csv,4fa6500df027eb16985a57232fa6fe393e3e43c10b32160dc53eac44bd36b9a5,40191,Open Government Licence - Canada,open,Active Layer - Auyuittuq
parks_canada,ParksCanada/auyuittuq_np_tundra_active-layer_2010-2025_temperature-sensors_data_2.csv,https://open.canada.ca/data/dataset/b5662fca-2082-428e-a9c3-3234822bbecd/resource/fe8837b8-4944-45d7-97fc-8cde88c7e246/download/auyuittuq_np_tundra_active-layer_2010-2025_temperature-sensors_data_2.csv,2437cc1969221c9f1345920370dfada25a06d83101ff3a4f3f7e95d6b59a3255,4778592,Open Government Licence - Canada,open,Active Layer - Auyuittuq
parks_canada,ParksCanada/qausuittuq_np_tundra_active_layer_2024-2025_data_1_calm-grid.csv,https://open.canada.ca/data/dataset/ed8981ff-c8b3-42e7-b02c-1d1b0373a991/resource/29e9944a-a61d-4771-8377-8fe471cff449/download/qausuittuq_np_tundra_active_layer_2024-2025_data_1_calm-grid.csv,fde5ee5d9e8648b611c34492ed39f27cd2d73a8628717cdb87cb4ab44e6d40e0,8686,Open Government Licence - Canada,open,Active Layer - Qausuittuq
parks_canada,ParksCanada/quttinirpaaq_np_tundra_active_layer_1999-2025_data_1_calm-grid.csv,https://open.canada.ca/data/dataset/5e8d13ca-2af8-4335-b770-20b898a5b530/resource/96ac4528-5b8c-40db-a0fc-d236c2c8ddd2/download/quttinirpaaq_np_tundra_active_layer_1999-2025_data_1_calm-grid.csv,da708ee395f177dc2c3a679e65fce39fd3ed28884401007989651b6d6c590fc7,81373,Open Government Licence - Canada,open,Active Layer - Quttinirpaaq
parks_canada,ParksCanada/sirmilik_np_tundra_active_layer_2008-2025_data_1_calm-grid.csv,https://open.canada.ca/data/dataset/a29dba8f-a791-4012-8c27-084ae7b76da9/resource/7ba2529b-a617-4e92-a3e5-04e2774e6fd8/download/sirmilik_np_tundra_active_layer_2008-2025_data_1_calm-grid.csv,d3fc41b6b9f4fac096f3b3d51569e524c6c6fd48914cbb37895a734af72b9ff5,131528,Open Government Licence - Canada,open,Active Layer - Sirmilik
parks_canada,ParksCanada/sirmilik_np_tundra_active_layer_2012-2025_data_2_temperature2.csv,https://open.canada.ca/data/dataset/a29dba8f-a791-4012-8c27-084ae7b76da9/resource/e7f34a0c-d726-47de-ac0a-4442866d36fe/download/sirmilik_np_tundra_active_layer_2012-2025_data_2_temperature2.csv,18a398e0f983e3cbdaa8560ad3884d9c99793cac0d8bbc05a635f213899c30f5,7627835,Open Government Licence - Canada,open,Active Layer - Sirmilik
parks_canada,ParksCanada/ukkusiksalik_np_tundra_active-layer_2014-2024_data_1_calm-grid.csv,https://open.canada.ca/data/dataset/0a827e84-278c-4150-aeef-77eb9c9612ce/resource/dc7e21b5-017e-488f-834a-ed622bb4ec03/download/ukkusiksalik_np_tundra_active-layer_2014-2024_data_1_calm-grid.csv,35e597eeff215cbfea44dbe2ae5881c2975043b5c34ffd832043a6433ad0adbe,25342,Open Government Licence - Canada,open,Active Layer - Ukkusiksalik
roshydromet_soil,Roshydromet_soil,http://aisori-m.meteo.ru/waisori/index.xhtml?idata=4,,,RIHMI-WDC (registered users),login,"Personal AISORI-M login (http only): dataset 'Температура почвы на глубинах до 320 см', stations by WMO index, semicolon delimiter, all years and parameters; save each export ZIP unchanged in Roshydromet_soil/"
ru_hms_alt,ADC_sweep/Russian_Arctic_ALT_1930_1990_arcss160/arcss160.zip,https://arcticdata.io/metacat/d1/mn/v2/object/urn%3Auuid%3A450f1a1e-d80c-4db6-ae21-6e291c8a4a99,857d198db1dd469cc3a96f36e3cad28caa9c1b78c505c89b979c894c78b8e3e2,49684,CC-BY-4.0 / CC0 (Arctic Data Center),open,"Time Series of Active Layer Thickness in the Russian Arctic, 1930-1990"
samoylov_calm,Samoylov/PANGAEA_947069_Calm_lv1_2019.zip,https://download.pangaea.de/dataset/947069/files/Calm_lv1_2019.zip,dc37e9b382e8f2c508bbef4ad9914643ac08af5b21cf0cb79a0e3451e3f6189d,1368,https://creativecommons.org/licenses/by/4.0/,open,Calm data at station Samoylov (2019)
samoylov_calm,Samoylov/PANGAEA_949381_Calm_lv1_2020.dat,https://download.pangaea.de/dataset/949381/files/Calm_lv1_2020.dat,6b98562cdee9f83c7f4eea2404a44794562183687031deef0315974dbbbabebd,4267,https://creativecommons.org/licenses/by/4.0/,open,Calm data at station Samoylov (2020)
samoylov_calm,Samoylov/PANGAEA_961867_Calm_lv1_2021.dat,https://download.pangaea.de/dataset/961867/files/Calm_lv1_2021.dat,c06d53e82d0ac63b001e0f9e3efdfc7722b96305995f3ef54f6443763494d888,3348,https://creativecommons.org/licenses/by/4.0/,open,Calm data at station Samoylov (2021)
samoylov_calm,Samoylov/PANGAEA_961875_Calm_lv1_2022.dat,https://download.pangaea.de/dataset/961875/files/Calm_lv1_2022.dat,a74b0049f2c2652104431fb97c4981238777a7fe739cd4c247092d4335c30272,2368,https://creativecommons.org/licenses/by/4.0/,open,Calm data at station Samoylov (2022)
samoylov_calm,Samoylov/PANGAEA_973715_Calm_lv1_2024.dat,https://download.pangaea.de/dataset/973715/files/Calm_lv1_2024.dat,97a5e1406d500d5962be9de85a60db3162d3001a673eb5078b56a2d98d8714be,1965,https://creativecommons.org/licenses/by/4.0/,open,Calm data at station Samoylov (2024)
samoylov_calm,Samoylov/calm_lv1_V201908.zip,https://store.pangaea.de/Publications/Boike-etal_2019_V201908/calm_lv1_V201908.zip,53913c9834df033c7b594372627c1cd1fdb830fd0cf487859b9bf6d2bdc7ce83,25257,CC-BY-4.0,open,"Calm data at station Samoylov (2002-2018, level 1, version 201908)"
stordalen,Abisko_Tornetrask/Stordalen_Active_Layer_Water_Table_03-17.xlsx,https://zenodo.org/api/records/10420396/files/Active_Layer_Water_Table_03-17.xlsx/content,4be42209334dc2821d99af67f7460bcc1c403626af805ed3b4cc87dec56229c1,194902,CC-BY-4.0,open,Manual active layer and water table depth measurements from the autochamber sites at Stordalen Mire (2003-2017)
tarfala_pace,Tarfala_PACE/figshare_3171646_v1/PACE_100_m_thermistor_string.csv,https://ndownloader.figshare.com/files/4938394,76b2ec24ac4d1b75de5a889f1b4dc5c975260c1f330a0feabf17fbeae2b50941,1856934,CC BY 4.0,open,"PACE-1-2 bore hole temperature data 2000-2014, Tarfalaryggen (Stockholm University, figshare 3171646 v1)"
tarfala_pace,Tarfala_PACE/figshare_3171646_v1/PACE_15_m_thermistor_string.csv,https://ndownloader.figshare.com/files/4938421,0291992987294206f765a53464ff248a4e334e6b7878f6c0e9e0eefa39f29def,2509822,CC BY 4.0,open,"PACE-1-2 bore hole temperature data 2000-2014, Tarfalaryggen (Stockholm University, figshare 3171646 v1)"
tarfala_pace,Tarfala_PACE/figshare_3171646_v1/PACE_Daily_data_2001-2014.csv,https://ndownloader.figshare.com/files/4938409,25d0934cebed2642fac59f6cdfbc843fc093cbc866e84f63d3d5e22225f6b3ea,1147817,CC BY 4.0,open,"PACE-1-2 bore hole temperature data 2000-2014, Tarfalaryggen (Stockholm University, figshare 3171646 v1)"
tarfala_pace,Tarfala_PACE/figshare_3171646_v1/TRS-PACE 2000-2014 data description.pdf,https://ndownloader.figshare.com/files/4938403,7e620078ef0bdb7dae48d389096cfed7d56ab7cf9acebf00e89771d5537b4759,710097,CC BY 4.0,open,"PACE-1-2 bore hole temperature data 2000-2014, Tarfalaryggen (Stockholm University, figshare 3171646 v1)"
uaf_stations,UAF_PermafrostLab/DH1_07.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH1_07.xls,f739b0b85812a1df71142ae90e93caf51460cd44fe65b882bf8b51b72da8c177,118784,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH1_08.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH1_08.xls,63a5f3070fef7eb494030f4b7563eb4fed362767426df2fbdc4d4842111cedc7,128000,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH1_daily_2010.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH1_daily_2010.xls,04a60e8621ca26cb01320cac505d3564cdcf162d6af1ea50153e6093bc430c93,117760,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH1_daily_2011.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH1_daily_2011.xls,fcab0066e0ff75534bc31d5694c7cc9c157c48bda91dba31f973a9780897e2b8,142848,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH1_daily_2013.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH1_daily_2013.xls,e705abccf6305ad90bc3df0538cb96cc8a215dab947cfc43ec75dfbeddc60f86,137216,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH1_daily_2014.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH1_daily_2014.xls,76f07e09630fbba0253c7d4ec231a9d540819547a335c086cb87e0a5aa901f2f,123904,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_09.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_09.xls,71674ccd61f8b4f8f5aba7fd8079ddbba249e233f12fcf6e2f7df0126f6e4f13,132608,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_daily_2010.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_daily_2010.xls,2eb678e1cc89998e8477bab6a2ad04fa9c807ce7444e005afb87551461ad0c0c,155136,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_daily_2011.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_daily_2011.xls,9d6f289ae7f6c952940c0fbb28977dfe26da55c96301267106e129a69b967eaa,178688,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_daily_2013.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_daily_2013.xls,a289d8cfe25a285d19295c58bf1b672cb4c5968226e427659378dfe5aaf5f47c,169472,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_daily_2014.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_daily_2014.xls,1df044f873eac4b7486fc9b5b83e1e650f44bb2c67107508e6b233575e4a8bc6,143360,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_daily_2015.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_daily_2015.xls,cd14ef5b46aa2c92c8ffecc555c5ecadd8c96f246f37789ab1e39aec3eb27f81,142848,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_daily_2017.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_daily_2017.xls,3b1910eeb1bc97ffb2a8430acf4301663169328fb9e0710969f8b65470e402e3,149504,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH2_daily_2018.csv,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH2_daily_2018.csv,ae40eee591300a070d9eda2c432a0e8dfc1e043f6e93df33cc183e0d13017d83,46061,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH_d0102_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH_d0102_sent_to_NSIDC.xls,ef953a82dff9fe55c308874c62c0e22cd2568a0325d65f3d56fa6339b42bbc65,123392,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH_d0203_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH_d0203_sent_to_NSIDC.xls,51ae7c1aafe52ba44c156c87327aedd8f2ebac0e40b83bbadf3947364da08a90,83456,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH_d0304_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH_d0304_sent_to_NSIDC.xls,cb7f7049067d116e10d1613c5009749f8bd6b83ee9e4aeda204ac3b1861554d3,104960,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/DH_d0405_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/DH_d0405_sent_to_NSIDC.xls,217c50b3bfca4285c4c10294c7816593608bbb2eb7d717076c2a2119aa636291,128512,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/FB1_08.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_08.xls,ca614ba2ba69b11f548541f9e1647e5e6455da4c120cae48ddf3a0b6fd3fcde0,135680,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/FB1_d0102_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_d0102_sent_to_NSIDC.xls,0e1ec83d206f6b29e14a6fb095f9952c60ca717d04fca00f7eab2f22dc78a70f,118272,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/FB1_d0203_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_d0203_sent_to_NSIDC.xls,ceeab3b3830f272eb18b31967045d8a82924738e89c7a384a34f09ccda5de574,119296,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/FB1_d0304_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_d0304_sent_to_NSIDC.xls,058fe4e15b094ac68647ffb7b1dbbc8212687619e9ccf9951516bc3750000ae9,119296,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/FB1_d0405_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_d0405_sent_to_NSIDC.xls,0382afb936644bdbe5d2584f033cecd03f37afe4066d91c3b0e9d16ebe8c048e,119808,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/FB1_d0506_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_d0506_to_NSIDC.xls,8ab9d9f3831bcaffc97da72cc648191e69238c425b33d2db649d149ded92c9f6,143872,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/FB1_daily_2010.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_daily_2010.xls,85cb9eaf19e9c20bae9d96db3dbc2dda39429a40ba8784ad5a9da7ee3d7e77da,129024,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/FB1_daily_2011.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/FB1_daily_2011.xls,226d8c20c91c291a37b69d1847cb750426658b89020505c4b7cf603ca2dc59ed,134144,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/IM1_daily_2014.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/IM1_daily_2014.xls,3ceb2c8a829f5eded2b4f89ea1dfaf22e97e017db330969cf0e5f40fdbd63271,214528,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site IM1: Imnaviat 1
uaf_stations,UAF_PermafrostLab/IM1_daily_2015.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/IM1_daily_2015.xls,758a118424f2b4b53fd91534617d2abc75a13b957103f3e2e2b202bd37322d03,214016,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site IM1: Imnaviat 1
uaf_stations,UAF_PermafrostLab/IM1_daily_2016.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/IM1_daily_2016.xls,701410684be85d489787be02199c2561ff6fc688bd44610dec536df74c1ee80f,214528,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site IM1: Imnaviat 1
uaf_stations,UAF_PermafrostLab/IM1_daily_2017.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/IM1_daily_2017.xls,389a63d578c4cc8e25d01de256c3d1ab7f57ce6aa26e0e7bb5fc6796789c7af2,207360,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site IM1: Imnaviat 1
uaf_stations,UAF_PermafrostLab/SagMAT_d0102_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMAT_d0102_sent_to_NSIDC.xls,d240d5a07a3b718ba8ebf6b953a61da0a5f945bddfc85e0e2a4c8c5543af384f,112640,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG2: SagMAT (Moist Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMAT_d0203_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMAT_d0203_sent_to_NSIDC.xls,e1b86d9db60b011376b56498706d6edf555dcfa8cb8f0053b7c91b5b6d624b4a,112640,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG2: SagMAT (Moist Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMAT_d0304_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMAT_d0304_sent_to_NSIDC.xls,1e36e9755ddee9ab7e0fc66872a34ca4a34a5d34c493943317cbed756a7f414e,113664,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG2: SagMAT (Moist Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMAT_d0405_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMAT_d0405_sent_to_NSIDC.xls,f93897b7e8582b0992e7788e650208393703ca910d7ae9df98d6a2ec49587b79,108032,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG2: SagMAT (Moist Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMAT_daily_2017.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMAT_daily_2017.xls,2cbe8271d0ec83b1207ab9fa3742f448bf08a59b731b51faf73e260ae2e40612,109056,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG2: SagMAT (Moist Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMNT_d0102_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMNT_d0102_sent_to_NSIDC.xls,03e767724a8aa6b81fc5d6ffd64b44f5016108aab80393c0fb33eeb82953763a,113152,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG1: SagMNT (Moist Non-Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMNT_d0203_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMNT_d0203_sent_to_NSIDC.xls,798b3103def2422325ac473351017954a762d7d1c13f0df182d16fea5042ea0a,112640,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG1: SagMNT (Moist Non-Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMNT_d0304_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMNT_d0304_sent_to_NSIDC.xls,433b1f5528911ba9bc23ecfca3c9d066722254bb1a92ee4cfbdf74020ead9df1,113152,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG1: SagMNT (Moist Non-Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMNT_d0405_sent_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMNT_d0405_sent_to_NSIDC.xls,70976d337ac47784bb16e005c7814e84832ab6e42c0f46483b5996be23096ff6,112640,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG1: SagMNT (Moist Non-Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMNT_d0506_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMNT_d0506_to_NSIDC.xls,902fd68239c800edc43063c964bcba2f529368a51983dd797e2f6d244280f3f0,113152,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG1: SagMNT (Moist Non-Acidic Tundra)
uaf_stations,UAF_PermafrostLab/SagMNT_d0607_to_NSIDC.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/SagMNT_d0607_to_NSIDC.xls,1cf4477587b18ca19f710cb5750dbf6f63d008da556508155483989d8b3893b1,113152,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG1: SagMNT (Moist Non-Acidic Tundra)
uaf_stations,UAF_PermafrostLab/dh2_daily_2012.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/dh2_daily_2012.xls,d82d31f18cf95ce9bb784328d3fd70d350bd2374df84cfa4cf3fda18dc3fcb1e,152064,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site DH1: Deadhorse
uaf_stations,UAF_PermafrostLab/fb1_daily_2009.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/fb1_daily_2009.xls,48fb8ba69d7544216779dcb43005df4885d732538b995887d5f1129311e12bc8,128512,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/fb1_daily_2012.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/fb1_daily_2012.xls,61e7d811577497a427bb3f3249c1997e7463ba35e5b0e2399b094c4c994bbad6,134656,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site FB1: Franklin Bluffs (borehole)
uaf_stations,UAF_PermafrostLab/sagmat_daily_2012.xls,http://web.archive.org/web/2026id_/http://permafrost.gi.alaska.edu/sites/default/files/site_data/sagmat_daily_2012.xls,68c9293c47261b69c5201ea7a90e07ba0f673390b8a9c7ac27e08af1134fefcf,154112,UAF Permafrost Lab public data (cite Romanovsky et al.),open,UAF Permafrost Lab site SG2: SagMAT (Moist Acidic Tundra)
uio_boreholes,UiO_Norway_Iceland/Data Storage Paper.7z,https://ns9999k.webs.sigma2.no/10.11582_2023.00128/Data%20Storage%20Paper.7z,1871a77238184feabb44ce7e9f2979e786b984e17cd70ba84e628387d2a5255b,1577811,CC BY 4.0,open,"Permafrost data Norway Iceland (Etzelmüller, University of Oslo, NIRD 10.11582/2023.00128); unpacked by download_uio_boreholes"
umiujaq_cable,NordicanaD_Borealis/SP3_QSRW0I/data_ta1_0114_post_cleaning.xlsx,https://borealisdata.ca/api/access/datafile/885133?format=original,52dbd5042a5134e13ad346dc21602fa96ad06bbddbfb228919ebd93028a5ff18,237668,CC-BY 4.0 http://creativecommons.org/licenses/by/4.0,open,"Subsurface ground temperature data from an instrumented permafrost mound, Tasiapik Valley, Umiujaq, Nunavik, Québec, Canada (2001-2024)"
umiujaq_cable,NordicanaD_Borealis/SP3_QSRW0I/data_ta1_1424_post_cleaning.xlsx,https://borealisdata.ca/api/access/datafile/885142?format=original,feb8c1dac8890a5faa4ea3a33dc4bff5d8f6b3c408b79d2c1f2c9c55f8b0f4ff,537357,CC-BY 4.0 http://creativecommons.org/licenses/by/4.0,open,"Subsurface ground temperature data from an instrumented permafrost mound, Tasiapik Valley, Umiujaq, Nunavik, Québec, Canada (2001-2024)"
umiujaq_cable,NordicanaD_Borealis/SP3_QSRW0I/data_ta2_1424_post_cleaning.xlsx,https://borealisdata.ca/api/access/datafile/885134?format=original,02f0345873874a3823808d93710039f98b7e9dca8a50b50258d5b80bf0db6ab3,437002,CC-BY 4.0 http://creativecommons.org/licenses/by/4.0,open,"Subsurface ground temperature data from an instrumented permafrost mound, Tasiapik Valley, Umiujaq, Nunavik, Québec, Canada (2001-2024)"
umiujaq_cable,NordicanaD_Borealis/SP3_QSRW0I/data_ta3_1424_post_cleaning.xlsx,https://borealisdata.ca/api/access/datafile/885135?format=original,904199c9eabc52f9c5373ef8d4ab39c67d0eb07796ffb80046d4593b226b78fc,490635,CC-BY 4.0 http://creativecommons.org/licenses/by/4.0,open,"Subsurface ground temperature data from an instrumented permafrost mound, Tasiapik Valley, Umiujaq, Nunavik, Québec, Canada (2001-2024)"
umiujaq_cable,NordicanaD_Borealis/SP3_QSRW0I/data_ta4_0114_post_cleaning.xlsx,https://borealisdata.ca/api/access/datafile/885141?format=original,91db59b0fb8ac5d7c929a4dc1788ddc623e4b5a632a1795012354b936b6de6c4,219506,CC-BY 4.0 http://creativecommons.org/licenses/by/4.0,open,"Subsurface ground temperature data from an instrumented permafrost mound, Tasiapik Valley, Umiujaq, Nunavik, Québec, Canada (2001-2024)"
umiujaq_cable,NordicanaD_Borealis/SP3_QSRW0I/data_ta5_0114_post_cleaning.xlsx,https://borealisdata.ca/api/access/datafile/885132?format=original,679b6a8a8f6677502e536a2b415c4ab617463e2d1c1722e3a2a2b9e5b5be0df2,255417,CC-BY 4.0 http://creativecommons.org/licenses/by/4.0,open,"Subsurface ground temperature data from an instrumented permafrost mound, Tasiapik Valley, Umiujaq, Nunavik, Québec, Canada (2001-2024)"
usgs_npra,ALASKA_other/AlaskaPoints_shapefile.zip,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__c9%2Fa5%2Fe2%2Fc9a5e2968955a7b33c5bc6645b2d7db551db6bfa,8e72c9e736f26cc312712c91e85d7621c83fa75d47f0c64feb66e66bf6920abc,4842,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
usgs_npra,ALASKA_other/Awuna.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__30%2F85%2F44%2F30854459f8ebd4212230528441665e8fbde87736,00d7e682c35f17ce28a698ae3b0b658ff7e78218c69cdc980e86c131f1b371f8,3362411,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Awuna.txt"
usgs_npra,ALASKA_other/Awuna2.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__30%2Fa4%2F57%2F30a4572960e00a2cef2edcf81b66d7d44cfa33d3,fa8719ff3d7a94d659b6ca7542c1ebf9d6f7d007fd96a6e993221af7e3ea17f6,21373183,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Awuna2.txt"
usgs_npra,ALASKA_other/Camden Bay.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__97%2F15%2Ffa%2F9715fa9419eab8f9e8522387c9f8d5230d20244d,eee7e0a2c60bb662c9d7f3944e5944da8bc78a1834383036418f55a812d0cdb2,20429781,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Camden Bay.txt"
usgs_npra,ALASKA_other/Drew Point.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__6a%2F83%2F5d%2F6a835d40df5a1506641c3489970ace556d7faf1b,04f332a5b48254e655b98a0d4d5b2ca5b03d9c1feb7982f2f49fd13bb42a1d6c,23675944,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
usgs_npra,ALASKA_other/East Teshekpuk.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__4f%2Ffd%2F3f%2F4ffd3ff9c0eeac43dd78ce94c64d28d43834ce08,1673030a9756bd35c3b602f47c65eccc7c075c75b8e02bd3086fcaf2e2f6e09f,19161478,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: East Teshekpuk.txt"
usgs_npra,ALASKA_other/Fish Creek.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__b8%2F38%2F9c%2Fb8389cf0ecb6e14b9a83d5e8cb26a40d4a8765ba,f4c0728565821ebf5ad39d394fa2f7c7b9938cd7663236bd60cf7649ddae5002,24760908,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
usgs_npra,ALASKA_other/Ikpikpuk.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__11%2Fd8%2F5e%2F11d85ebb8191d10991642bffcefb71da60a7b49b,0e88080a97811b6864ae2cf097b460ddb6376acba30cf9e956c7c9c0eea30eaa,17843244,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Ikpikpuk.txt"
usgs_npra,ALASKA_other/Inigok.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__27%2F30%2F4f%2F27304fb66e5ecffedd0eb8a2f8f1a44bd4fe71ff,cba2302d19ac2786fd0aa085e1f31440974a9f22c1272208c503d710ee2bdf55,23631560,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
usgs_npra,ALASKA_other/Koluktak.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__43%2F78%2F12%2F43781241dda90b33e67e3666fdcc5fdf67ee202f,6781422757620bef0904b020730660b613e21b07ff43be8d5bcf94a5e8c059ac,22982736,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
usgs_npra,ALASKA_other/Lake 145.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__95%2F7f%2Fd7%2F957fd7761c1c93dc947d49006c884f0dd270d50b,7ec71c87d5a2127e48d415ff851b4b482ab1afedf49e031dbe04b926f25ab22e,16048017,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Lake 145.txt"
usgs_npra,ALASKA_other/Marsh Creek.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__16%2Fe2%2F3a%2F16e23a20271d066aabfc42a945f205fcdd3bf1aa,f72203c10802479df89c63cc7de7704b7cc761d563187c3250c3392f3acc652f,22772214,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Marsh Creek.txt"
usgs_npra,ALASKA_other/Niguanak.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__42%2F10%2Ff9%2F4210f917a41b62b9027296a36c74f2f1f367690d,094ff6473d072459f74a5d4853cee43d58547074d547b90fcaf9579d40e1c0c9,23414049,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
usgs_npra,ALASKA_other/Piksiksak.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__a5%2Feb%2Ff0%2Fa5ebf09c785c550342320363a547c12abdf523c3,3f971290ddd1b786d695f558cae03e900888e1f96adf5f388fd099aa172a80dd,19168340,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Piksiksak.txt"
usgs_npra,ALASKA_other/Red Sheep Creek.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__99%2F70%2F2f%2F99702f57f34f9e8cd5f3411cb6b85f14ee117dc6,c08a75640eb7d026039ab727b340f32c646220c11cd50a4eaccade91a17385d1,20104966,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: Red Sheep Creek.txt"
usgs_npra,ALASKA_other/South Meade.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__d0%2F9a%2F20%2Fd09a2058094439277e0c7ce3db0d1e2eef4017f7,61d4593bd442f0426c85b6e630f247273053d30f7b801c54401be69b4ee7ab7e,20329771,,open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021) :: South Meade.txt"
usgs_npra,ALASKA_other/Tunalik.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__b4%2Ff4%2F1a%2Fb4f41a74ce685f175546a79318429394dc4bbe4a,2ee1ce7d1fb66f1920baf870a96234d54cdd0530efe83eb7f4fae76d92848046,25318594,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
usgs_npra,ALASKA_other/Umiat.txt,https://www.sciencebase.gov/catalog/file/get/59d6a458e4b05fe04cc6b47e?f=__disk__cd%2Fb7%2F10%2Fcdb710ecd2fd6037569c188a373b4e2d11951e73,e21df4aa79828e13956baa598cdef4221e6a6806ff483bb2b6b714d738c8e7c2,23630538,Public domain (USGS),open,"Data Release associated with Data Series - DOI/GTN-P Climate and Active-Layer Data Acquired in the National Petroleum Reserve-Alaska and the Arctic National Wildlife Refuge, 1998-2019 (ver. 3.0, March 2021)"
"""


if __name__ == "__main__":
    main()
