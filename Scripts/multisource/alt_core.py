"""
Shared settings and helpers for the multi-source ALT builder.

Everything that decides how a raw measurement becomes one annual ALT value lives here:
the settings (CONFIG), the record layout every source reader returns (RECORD_COLUMNS),
the end-of-season rule for dated probing surveys, and the 0 C crossing rule for
ground-temperature profiles. The source readers are in alt_sources.py, and the merge
and outputs are in build_alt_multisource.py.

Units: cm. A "year" is the thaw season (calendar year of the end-of-season value).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS (all decisions in one place; the reason is next to each value)
# =============================================================================
CONFIG = {
    # A site needs this many valid annual values (uncensored, not flagged) to be kept.
    "min_years": 8,
    # Circumarctic: north of 50 N. Mountain permafrost of central Asia and the Alps is
    # outside the scope even if north of 50 N (Mongolia, Altai, Kazakhstan, China).
    "min_lat": 50.0,
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
    # Within-site screen on log(ALT): robust z beyond this is flagged (not removed).
    "outlier_robust_z": 4.0,
    # Source priority when the same site-year is in several sources (lower = preferred).
    # Network-curated tables first, then original data files, then compilations.
    "priority": {"calm_summary": 1, "pangaea_calm": 2, "gtnp": 3,
                 "gsc_thaw_tube": 3, "parks_canada": 3, "ru_hms_alt": 3,
                 "calm_site_files": 3, "kwbs": 3, "allena": 3, "kytalyk": 3,
                 "samoylov_calm": 3, "arc_lter": 3, "bnz_lter": 3, "itex": 3,
                 "illisarvik": 3, "stordalen": 3,
                 "usgs_npra": 4, "gsc_cable": 4, "awi_soil_temp": 4, "ngee_barrow": 4,
                 "uaf_stations": 4, "umiujaq_cable": 4,
                 "above_field": 5, "cusp": 6, "other": 7},
    # Known site problems, from earlier ALDI v5 QC: CALM code -> first affected year.
    # Pearl Creek (U19) and Bonanza Creek (U18) thaw into a talik: thaw depth stops
    # being ALT. Values from that year are flagged "talik".
    "talik_from": {"U18": 2011, "U19": 2015},
    # CALM sites whose borehole value sits at the deepest sensor most years (lower bound).
    "sensor_cap_cm": {"U30": 120.0},
}

# Flags that make a value unusable for trend analysis (it stays in the files).
# Matched as whole flags, so an informational flag such as 'deep_thaw_check_talik'
# does not count as 'talik'.
INVALID_FLAG_NAMES = ("calm_footnote", "talik", "sensor_cap", "wide_sensor_spacing",
                      "lower_bound", "treatment", "early_measurement", "no_refreeze")
INVALID_FLAGS = r"(?:^|;)(?:" + "|".join(INVALID_FLAG_NAMES) + r")(?:;|$)"

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
    Returns one row per year: year, alt_cm, censored, n_sensors, deepest_cm, bracket_cm,
    nonmonotonic (a deeper sensor is above 0 C again below the crossing), no_refreeze.
    """
    if min_season_cover is None:
        min_season_cover = CONFIG["min_summer_cover"]
    t = temps.dropna(subset=["temp_c", "depth_cm"]).copy()
    t = t[t.depth_cm > 0]
    if t.empty:
        return pd.DataFrame(columns=["year", "alt_cm", "censored", "n_sensors",
                                     "deepest_cm", "bracket_cm", "nonmonotonic", "no_refreeze"])
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
                            no_refreeze=no_refreeze(yr, d[i])))
            continue
        # the crossing lies between sensor i (above 0) and i+1 (at or below 0)
        z = d[i] + (d[i + 1] - d[i]) * (v[i] - zero_c) / (v[i] - v[i + 1])
        # bracket_cm: spacing of the two sensors around the crossing = interpolation
        # uncertainty (the true profile is not linear between sensors)
        out.append(dict(year=yr, alt_cm=z, censored=False, n_sensors=len(v),
                        deepest_cm=d[-1], bracket_cm=d[i + 1] - d[i],
                        nonmonotonic=bool(above[i + 1:].any()), no_refreeze=no_refreeze(yr, d[i])))
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
