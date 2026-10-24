"""
Multi-source circumarctic active layer thickness (ALT) dataset builder.

Reads every raw source, turns each one into ONE value per site and year (end-of-season
thaw depth, cm), joins the same physical site appearing in several sources, applies
quality flags, and keeps the sites with at least CONFIG["min_years"] valid annual values.

Run (Colab or local), from this folder:
    python build_alt_multisource.py --raw <raw folder> [<more raw folders> ...] --out <output folder>

Files:
    alt_core.py      settings (CONFIG), record layout, end-of-season and 0 C crossing rules
    alt_sources.py   one reader per source (ADAPTERS)
    this file        merge, duplicate checks, flags, outputs

Outputs (in --out):
    alt_multisource_all_records.csv every annual record from every source, before merging
    alt_multisource_annual.csv      one row per site-year (eligible sites only), all flags
    alt_multisource_sites.csv       one row per eligible site: location, sources, method,
                                    years, other eligible sites within 1 km
    alt_multisource_for_ALDI.csv    input for Scripts/ALDI_Calculation_final.ipynb:
                                    site_id, Location, Lat, Long, Year, ALT_cm (valid
                                    years only), source, method
    source_inventory.csv            per source: what was read and used, or why not used
    data_credits.md                 dataset citations, and the GTN-P data authors
    build_log.txt                   everything the run printed

Units: cm. Years are thaw seasons (calendar year of the end-of-season value).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alt_core import (CONFIG, INVALID_FLAGS, LOG_LINES, files, haversine_km, log,  # noqa: E402
                      method_family)
from alt_sources import ADAPTERS  # noqa: E402

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
    "gem": ("Greenland Ecosystem Monitoring (GEM): ZEROCALM-1/2 active layer (GeoBasis Zackenberg); ground "
            "temperatures at GeoBasis Disko AWS1, AWS2 P1/P2 and ZEROCALM-2 upper/middle/lower",
            "https://doi.org/10.17897/2Q7S-2876 ; 10.17897/4EE8-WE92 ; 10.17897/B2K2-0Q82 ; 10.17897/3T7W-7638 ; "
            "10.17897/8HPH-H716 ; 10.17897/KCP6-QG90 ; 10.17897/D2NV-GN54 ; 10.17897/FX4T-2H83",
            "CC-BY-SA 4.0 (GEM terms of use: cite each dataset DOI)"),
    "above_field": ("ABoVE: Soil thaw depth and moisture validation data (ORNL DAAC), repeat-visited plots",
                    "https://daac.ornl.gov (ABoVE_Soil_ThawDepth_Moisture_Validation)", "ORNL DAAC (cite dataset)"),
    "cusp": ("CUSP - CommUnity near-Surface Permafrost dataset v1.2 (only its sources not read elsewhere)",
             "https://doi.org/10.5281/zenodo.23171354", "CC-BY 4.0"),
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
    ("GTN-P database, new downloads", "http://gtnpdatabase.org", "server returned 502 during the build; "
     "the grids downloaded for the corridor project are used"),
    ("GEM Greenland: shallow soil temperature (GeoBasis Disko AWS2/AWS3/AWS4, Nuuk SoilEmp, SoilEmpSa, "
     "SoilVUE Heath and Fen), meteorology, snow and soil moisture", "https://data.g-e-m.dk",
     "sensors only to 20-50 cm, or 2021-2024 only (SoilVUE Fen to 100 cm); not ALT"),
    ("Yukon Permafrost Database", "https://data.geology.gov.yk.ca/permafrost",
     "only site locations could be downloaded (no temperature series)"),
    ("Nordicana D (other datasets)", "https://nordicana.cen.ulaval.ca", "server refused downloads (403); Umiujaq "
     "came from Borealis"),
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
    Two different site keys are never joined, and neither are two series of the same
    source (the source itself lists them as different plots, e.g. the Eight Mile Lake
    thaw-gradient sites or the Kolyma water-balance plots, even when they share a
    coordinate).
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
        if match is None and reps:
            lat = np.array([s["lat"] for s in reps])
            lon = np.array([s["lon"] for s in reps])
            dist = haversine_km(r.lat, r.lon, lat, lon)
            near = np.where(dist <= max(CONFIG["dup_distance_km"], CONFIG["copy_km"]))[0]
            for i in near[np.argsort(dist[near])]:
                s = reps[i]
                if (r.site_key and s["site_key"]) or r.source in s["sources"]:
                    continue
                a, b = by_key[key], by_key[s["key"]]
                common = a.index.intersection(b.index)
                same_fam = s["fam"] == r.fam
                if len(common) >= 2:
                    diff = (a[common] - b[common]).abs().median()
                    tol = max(CONFIG["dup_agree_cm"], CONFIG["dup_agree_frac"] * b.mean())
                    if same_fam and dist[i] <= CONFIG["dup_distance_km"] and diff <= tol:
                        match = s                                            # rule (b)
                    elif (same_fam and r.fam == "temperature" and len(common) >= 5
                          and dist[i] <= CONFIG["same_borehole_km"]
                          and diff <= max(CONFIG["same_borehole_agree_cm"],
                                          CONFIG["same_borehole_agree_frac"] * b.mean())):
                        match = s                                            # rule (c)
                    elif (len(common) >= 5 and diff <= CONFIG["copy_agree_cm"]
                          and a[common].corr(b[common]) >= CONFIG["copy_min_r"]):
                        match = s                                            # rule (d)
                    if match is not None:
                        log(f"    {key[0]}:{key[1]} joined to {s['site_id']} ({dist[i]:.2f} km, "
                            f"{len(common)} shared years, median difference {diff:.1f} cm)")
                        break
                if same_fam and dist[i] <= CONFIG["dup_distance_km"]:
                    n_close += 1
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


def write_credits(out: Path, inv: pd.DataFrame, raw_dirs) -> None:
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
    (out / "data_credits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(raw_dirs: list[Path], out: Path, only: list[str] | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    parts, inventory = [], []
    for name, fn in ADAPTERS.items():
        if only and name not in only:
            continue
        log(f"[{name}]")
        try:
            df = fn(raw_dirs)
        except Exception as e:                            # a broken source must not stop the run
            log(f"  {name}: FAILED ({type(e).__name__}: {e})")
            df = pd.DataFrame()
        inventory.append(dict(source=name, series_read=df.source_site_id.nunique() if len(df) else 0,
                              site_years_read=len(df)))
        if len(df):
            parts.append(df)
    rec = pd.concat(parts, ignore_index=True)
    out_of_scope = ~in_scope(rec)
    log(f"\n{int(out_of_scope.sum())} site-years outside the scope (south of {CONFIG['min_lat']} N or "
        f"excluded mountain regions)")
    rec = rec[~out_of_scope]
    rec.to_csv(out / "alt_multisource_all_records.csv", index=False)

    log("\n[merge]")
    rec = assign_physical_sites(drop_short_series(rec))
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
    aldi[["site_id", "Location", "Lat", "Long", "Year", "ALT_cm", "source", "method"]].to_csv(
        out / "alt_multisource_for_ALDI.csv", index=False)

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
         "read; no site reaches the year threshold"], "not read (files missing or reader failed)")
    inv["reason_not_used"] = inv.source.map(READ_NOT_USED).where(inv.site_years_used == 0).fillna("")
    nu = pd.DataFrame([dict(source="", title=t, reference=u, status="not used", reason_not_used=why)
                       for t, u, why in NOT_USED])
    inv = pd.concat([inv, nu], ignore_index=True)
    cols = ["source", "title", "reference", "licence", "status", "series_read", "site_years_read",
            "sites_contributing", "site_years_used", "reason_not_used"]
    inv[cols].to_csv(out / "source_inventory.csv", index=False)
    write_credits(out, inv, raw_dirs)

    log(f"\nRESULT: {len(keep)} sites with >= {CONFIG['min_years']} valid annual values, "
        f"{len(valid)} valid site-years")
    for k in (10, 15, 20, 30):
        log(f"  sites with >= {k} valid years: {int((nval >= k).sum())}")
    log("  sites by country: " + ", ".join(f"{c} {n}" for c, n in sites.country.value_counts().items()))
    log("  sites by method: " + ", ".join(f"{c} {n}" for c, n in sites.method.value_counts().items()))
    (out / "build_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, type=Path, nargs="+", help="one or more raw data folders")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-years", type=int, default=CONFIG["min_years"])
    ap.add_argument("--only", nargs="*", help="run only these readers (for testing)")
    a = ap.parse_args()
    CONFIG["min_years"] = a.min_years
    build(a.raw, a.out, a.only)
