"""
Download the new candidate ALT sources found in the 2026-11-02 review and turn the
temperature profiles into annual ALT with the build's own rule (alt_core).

    python fetch_new_sources.py --out <folder> [--only gtnp adc uaf neon zenodo nsidc]
                                [--calm-records <alt_multisource_all_records.csv>]

Run it on a machine with open internet access (the cloud sandbox and the desktop's Linux
VM cannot reach these hosts). Everything is cached under <out>/raw, so a re-run only
downloads what is missing. Each source writes:
    <out>/raw/<source>/...                      the files as served
    <out>/<source>_inventory.csv               what was found and what was kept, and why
    <out>/<source>_alt_records.csv             annual ALT from profiles (where parsed)

Sources (see alt_multisource/source_inventory.md and the 2026-11-02 review):
  gtnp    GTN-P Data Platform (https://data.gtn-p.org/api, OpenAPI at /api/openapi.json).
          Sites north of 50 N (Mongolia, Kazakhstan and China left out). Boreholes with
          has_data and a ground-temperature dataset covering >= 8 years from 1990 are
          downloaded; the profile must have >= 2 sensors at <= MAX_SHALLOW_CM. ALT
          datasets are downloaded and compared with PANGAEA 972777 (pangaea_calm records).
  adc     Arctic Data Center (DataONE API): "Network of Permafrost Observatories in North
          America and Russia: Alaska" (doi:10.18739/A2J34P, followed to its latest
          version) and the other current Romanovsky permafrost-temperature packages.
  uaf     UAF Permafrost Lab site pages (Galbraith GL1, West Dock WD1, Happy Valley HV1,
          Chandalar Shelf CS1): every data link on the page.
  neon    NEON soil temperature DP1.00041.001 at BARR, TOOL, HEAL (neonutilities).
          A plot-year is kept only if no sensor of the plot changed position during the
          year (sensor_positions) and no issue-log entry about frost heave covers it.
  zenodo  Syrdakh observatory, Yakutia (Zenodo record 14619854).
  nsidc   NSIDC G02189 Yakutian soil temperatures 1915-2000 (pre-2000 check only).
  metno   MET Norway permafrost stations on thredds.met.no (Svalbard and mainland Norway;
          most start in 2019, so they reach 8 years only from about 2027).

Parsers for formats not seen yet (GTN-P CSV, UAF/ADC spreadsheets, Syrdakh) are written
to read the common layouts (a time column plus depth columns, or a long table with a
depth column). Files they cannot read are listed in <source>_unparsed.csv with their
header, so a reader can be added.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from alt_core import CONFIG, records_from_profiles  # noqa: E402  (the build's ALT rule)

# --- settings (reason next to each) -------------------------------------------------
MIN_LAT = 50.0                 # circumarctic, as the build (sites kept from 50 N)
SKIP_COUNTRIES = {"Mongolia", "Kazakhstan", "China"}   # not circumarctic
FIRST_YEAR = 1990              # ALDI input starts in 1990
MIN_YEARS = 8                  # build threshold (>= 8 valid annual values)
MAX_SHALLOW_CM = 150           # a profile needs >= 2 sensors at or above 1.5 m to resolve ALT
MAX_FILE_MB = 300              # do not download single files larger than this
GTNP = "https://data.gtn-p.org/api"
D1 = "https://arcticdata.io/metacat/d1/mn/v2"
UAF_SITES = {"gl1": "Galbraith Lake", "wd1": "West Dock", "hv1": "Happy Valley",
             "cs1": "Chandalar Shelf"}
NEON_SITES = ["BARR", "TOOL", "HEAL"]
ZENODO_RECORD = "14619854"
NSIDC_G02189 = "https://noaadata.apps.nsidc.org/NOAA/G02189/"
# MET Norway permafrost stations (FROST -> CF-NetCDF on THREDDS; e.g. Janssonhaugen Vest
# SN99874 from 2019). The 1998-2019 Janssonhaugen 0.2-5 m series is not there (data request).
METNO_THREDDS = "https://thredds.met.no/thredds"
METNO_CATALOG = f"{METNO_THREDDS}/catalog/arcticdata/frost2netcdf-permafrost/catalog.xml"
DATA_EXT = (".csv", ".txt", ".dat", ".xls", ".xlsx", ".zip", ".tab", ".nc")

S = requests.Session()
S.headers["User-Agent"] = "ALDI-multisource-fetch/1.0 (research; contact via GitHub MOREDataset/ALDI)"


# =============================================================================
# HTTP helpers
# =============================================================================
def get(url, params=None, stream=False, tries=4):
    for k in range(tries):
        r = S.get(url, params=params, timeout=300, stream=stream)
        if r.status_code in (429, 500, 502, 503, 504) and k < tries - 1:
            time.sleep(2 ** (k + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(url)


def get_json(url, params=None):
    return get(url, params).json()


def download(url, dest: Path, params=None) -> Path:
    """Cached download; returns the path. Skips files larger than MAX_FILE_MB."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = get(url, params, stream=True)
    size = int(r.headers.get("Content-Length", 0) or 0)
    if size > MAX_FILE_MB * 1e6:
        raise ValueError(f"{url}: {size / 1e6:.0f} MB > MAX_FILE_MB")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


def log(msg):
    print(msg, flush=True)


# =============================================================================
# Generic temperature-table reader -> long table (time, depth_cm, temp_c)
# =============================================================================
_DEPTH_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*(cm|m)?\b", re.I)


def _depth_from_name(name: str):
    """Depth in cm from a column name such as '0.2', '-0.50 m', 'T_20cm', '1.5m', 'Depth 100'.
    Returns None if the name is not a depth."""
    s = str(name).strip()
    if re.search(r"(date|time|year|month|day|hour|flag|qf|qual|id|lat|lon|air|surf)", s, re.I):
        return None
    m = _DEPTH_RE.search(s.replace(",", "."))
    if not m:
        return None
    v = abs(float(m.group(1)))
    unit = (m.group(2) or "").lower()
    if unit == "cm":
        return v
    if unit == "m":
        return v * 100
    return None if v == 0 and "0" not in s else v  # unitless: decided by the caller


def to_long(df: pd.DataFrame, unit_hint: str | None = None) -> pd.DataFrame | None:
    """Wide (time + one column per depth) or long (time, depth, temperature) -> long.
    Unitless depth columns are metres when every depth <= 30, else centimetres
    (unit_hint 'm' or 'cm' overrides). Returns None when the layout is not recognised."""
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in cols if c in ("time", "datetime", "date_time", "timestamp",
                                                 "date", "startdatetime")), None)
    if tcol is None:
        tcol = next((cols[c] for c in cols if "time" in c or "date" in c), None)
    if tcol is None:
        return None
    t = pd.to_datetime(df[tcol], errors="coerce")
    if t.notna().mean() < 0.9:
        return None
    dcol = next((cols[c] for c in cols if c.startswith("depth")), None)
    vcol = next((cols[c] for c in cols if c.startswith(("temp", "value", "gt", "ground"))), None)
    if dcol is not None and vcol is not None:                     # long layout
        d = pd.to_numeric(df[dcol], errors="coerce").abs()
        unit = unit_hint or ("m" if d.max() <= 30 else "cm")
        return pd.DataFrame({"time": t, "depth_cm": d * (100 if unit == "m" else 1),
                             "temp_c": pd.to_numeric(df[vcol], errors="coerce")}).dropna()
    depth = {}
    for c in df.columns:
        if c == tcol:
            continue
        v = _depth_from_name(c)
        if v is not None and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.1:
            depth[c] = v
    if len(depth) < 2:
        return None
    unitless = [c for c in depth if not re.search(r"(cm|m)\b", str(c), re.I)]
    if unitless:
        vals = [depth[c] for c in unitless]
        unit = unit_hint or ("m" if max(vals) <= 30 else "cm")
        for c in unitless:
            depth[c] = depth[c] * (100 if unit == "m" else 1)
    long = df[[tcol] + list(depth)].melt(id_vars=tcol, var_name="col", value_name="temp_c")
    long["depth_cm"] = long["col"].map(depth)
    long["time"] = pd.to_datetime(long[tcol], errors="coerce")
    long["temp_c"] = pd.to_numeric(long["temp_c"], errors="coerce")
    long = long[["time", "depth_cm", "temp_c"]].dropna()
    long = long[long.temp_c.between(-60, 40)]                  # drop no-data codes (-999...)
    return long


def read_any_table(path: Path) -> list[tuple[str, pd.DataFrame]]:
    """All tables of a file: (name, DataFrame). Spreadsheets give one per sheet; text
    files are tried with a header row found by the first line with >= 3 separators."""
    out = []
    suf = path.suffix.lower()
    if suf in (".xls", ".xlsx"):
        for name, df in pd.read_excel(path, sheet_name=None, header=None).items():
            out.append((name, _promote_header(df)))
        return out
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    for sep in (",", "\t", ";", r"\s+"):
        hdr = next((i for i, ln in enumerate(lines[:200])
                    if len(re.split(sep, ln.strip())) >= 3 and not ln.startswith(("#", "/*"))), None)
        if hdr is None:
            continue
        try:
            df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])), sep=sep, engine="python",
                             on_bad_lines="skip")
        except (pd.errors.ParserError, ValueError):
            continue
        if df.shape[1] >= 3:
            out.append(("", df))
            break
    return out


def _promote_header(df: pd.DataFrame) -> pd.DataFrame:
    """Spreadsheet read without header: the header is the first row where most cells are
    text and the next row has a date or a number in the first column."""
    for i in range(min(40, len(df) - 1)):
        row = df.iloc[i]
        if row.notna().sum() >= 3 and row.astype(str).str.contains(r"[A-Za-z]").mean() > 0.5:
            body = df.iloc[i + 1:].copy()
            body.columns = [str(c) for c in row.fillna("").astype(str)]
            return body.reset_index(drop=True)
    return df


def profiles_to_alt(temps: pd.DataFrame, sites: pd.DataFrame, method: str,
                    key="source_site_id") -> pd.DataFrame:
    """Annual ALT with the build's rule (alt_core.records_from_profiles) for years >= 1990."""
    temps = temps[temps.time.dt.year >= FIRST_YEAR - 1]
    if temps.empty:
        return pd.DataFrame()
    return records_from_profiles(temps, sites, method=method, key=key)


def write_inventory(out: Path, name: str, rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}_inventory.csv", index=False)
    log(f"  {name}: inventory {len(df)} rows -> {name}_inventory.csv")
    return df


# =============================================================================
# 1. GTN-P Data Platform
# =============================================================================
def gtnp(out: Path, calm_records: Path | None):
    raw = out / "raw" / "gtnp"
    raw.mkdir(parents=True, exist_ok=True)

    def sites(kind):
        cache = raw / f"list_sites_{kind}.json"
        if not cache.exists():
            p = {"lat_min": MIN_LAT, "metadata": "true", f"{kind}_data": "true"}
            cache.write_text(json.dumps(get_json(f"{GTNP}/list-sites", p)))
        return json.loads(cache.read_text())

    # --- boreholes --------------------------------------------------------------------
    inv, keep = [], []
    for site in sites("borehole"):
        country = site.get("country_name") or ""
        for bh in site.get("boreholes", []):
            bid = bh["id"]
            row = {"site": site["name"], "country": country, "borehole_id": bid,
                   "borehole": bh.get("name")}
            if country in SKIP_COUNTRIES:
                inv.append({**row, "kept": False, "reason": "country not circumarctic"})
                continue
            cache = raw / "pt_meta" / f"{bid}.json"
            if not cache.exists():
                cache.parent.mkdir(exist_ok=True)
                cache.write_text(json.dumps(get_json(f"{GTNP}/pt/{bid}")))
            meta = json.loads(cache.read_text())
            loc = meta.get("location") or {}
            row.update(lat=loc.get("latitude"), lon=loc.get("longitude"),
                       has_data=meta.get("has_data"),
                       deepest_sensor_m=(meta.get("drilling") or {}).get("deepest_sensor"),
                       disturbance=json.dumps(meta.get("disturbance")),
                       citation=meta.get("generic_citation"))
            if not meta.get("has_data"):
                inv.append({**row, "kept": False, "reason": "has_data false"})
                continue
            gt = [d for d in meta.get("datasets", [])
                  if "ground" in json.dumps(d.get("variable")).lower() and not d.get("is_retired")]
            ok_ds = []
            for d in gt:
                st = d.get("statistics") or {}
                d0 = pd.to_datetime(st.get("date_start") or st.get("date_min"), errors="coerce")
                d1 = pd.to_datetime(st.get("date_end") or st.get("date_max"), errors="coerce")
                if pd.isna(d0) or pd.isna(d1):
                    d0 = pd.to_datetime(meta.get("date_start"), errors="coerce")
                    d1 = pd.to_datetime(meta.get("date_end"), errors="coerce")
                yrs = (d1.year - max(d0.year, FIRST_YEAR) + 1) if pd.notna(d0) and pd.notna(d1) else 0
                if yrs >= MIN_YEARS:
                    ok_ds.append(d["id"])
            row["gt_datasets"] = ";".join(map(str, ok_ds))
            if not ok_ds:
                inv.append({**row, "kept": False,
                            "reason": f"no ground-temperature dataset spanning >= {MIN_YEARS} years from {FIRST_YEAR}"})
                continue
            keep.append(row)
            inv.append({**row, "kept": None, "reason": "downloaded; depth check below"})

    # download and check depths
    temps_all, site_rows = [], []
    for row in keep:
        for dsid in row["gt_datasets"].split(";"):
            z = download(f"{GTNP}/data", raw / "pt_data" / f"pt_{dsid}.zip",
                         params={"pt_data": dsid, "variable": "gt", "combined": "false"})
            long = _gtnp_zip_to_long(z)
            r = next(i for i in inv if i["borehole_id"] == row["borehole_id"])
            if long is None or long.empty:
                r.update(kept=False, reason=f"dataset {dsid}: layout not recognised (see gtnp_unparsed.csv)")
                continue
            shallow = sorted(long.depth_cm[long.depth_cm <= MAX_SHALLOW_CM].unique())
            r["sensor_depths_cm"] = ";".join(f"{v:g}" for v in sorted(long.depth_cm.unique()))
            if len(shallow) < 2:
                r.update(kept=False, reason=f"fewer than 2 sensors at <= {MAX_SHALLOW_CM} cm")
                continue
            r.update(kept=True, reason="profile used")
            key = f"GTNP_PT_{row['borehole_id']}_{dsid}"
            temps_all.append(long.assign(source_site_id=key))
            site_rows.append(dict(source_site_id=key, site_name=f"{row['site']} / {row['borehole']}",
                                  lat=row["lat"], lon=row["lon"], country=row["country"],
                                  source_file=f"pt_{dsid}.zip"))
    write_inventory(out, "gtnp_boreholes", inv)
    if temps_all:
        alt = profiles_to_alt(pd.concat(temps_all), pd.DataFrame(site_rows), "borehole_temp")
        alt.to_csv(out / "gtnp_pt_alt_records.csv", index=False)
        n = alt[~alt.flag.fillna("").str.contains("lower_bound|wide_sensor|no_refreeze|inversion")]
        n = n[n.year >= FIRST_YEAR].groupby("source_site_id").size()
        log(f"  gtnp boreholes: ALT for {alt.source_site_id.nunique()} series; "
            f"{(n >= MIN_YEARS).sum()} with >= {MIN_YEARS} valid years from {FIRST_YEAR}")

    # --- active-layer datasets ----------------------------------------------------------
    inv, alt_rows = [], []
    for site in sites("activelayer"):
        country = site.get("country_name") or ""
        if country in SKIP_COUNTRIES:
            continue
        for al in site.get("activelayers", []):
            aid = al["id"]
            cache = raw / "alt_meta" / f"{aid}.json"
            if not cache.exists():
                cache.parent.mkdir(exist_ok=True)
                cache.write_text(json.dumps(get_json(f"{GTNP}/alt/{aid}")))
            meta = json.loads(cache.read_text())
            loc = meta.get("location") or {}
            for d in meta.get("datasets", []):
                var = json.dumps(d.get("variable")).lower()
                row = {"site": site["name"], "country": country, "activelayer_id": aid,
                       "name": meta.get("name"), "calm_code": meta.get("calm_code"),
                       "lat": loc.get("latitude"), "lon": loc.get("longitude"),
                       "dataset_id": d["id"], "variable": var}
                if "active layer thickness" not in var:
                    inv.append({**row, "kept": False, "reason": "not an ALT variable"})
                    continue
                code = "alt-gt" if "grid" in var else "alt-btt"
                z = download(f"{GTNP}/data", raw / "alt_data" / f"alt_{d['id']}.zip",
                             params={"alt_data": d["id"], "variable": code, "combined": "false"})
                tabs = _zip_tables(z)
                inv.append({**row, "kept": bool(tabs), "reason": "downloaded" if tabs else "empty"})
                for name, df in tabs:
                    alt_rows.append(df.assign(activelayer_id=aid, dataset_id=d["id"],
                                              calm_code=meta.get("calm_code"), file=name))
    write_inventory(out, "gtnp_alt", inv)
    if alt_rows:
        allalt = pd.concat(alt_rows, ignore_index=True)
        allalt.to_csv(out / "gtnp_alt_raw_rows.csv", index=False)
        _diff_vs_972777(out, allalt, calm_records)


def _zip_tables(z: Path):
    out = []
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.lower().endswith((".csv", ".txt")):
                txt = zf.read(n).decode("utf-8", errors="replace")
                lines = [ln for ln in txt.splitlines() if not ln.startswith("#")]
                if len(lines) > 1:
                    out.append((n, pd.read_csv(io.StringIO("\n".join(lines)), sep=None, engine="python")))
    return out


def _gtnp_zip_to_long(z: Path):
    tabs = _zip_tables(z)
    parts = []
    for name, df in tabs:
        long = to_long(df)
        if long is None:
            pd.DataFrame([{"file": f"{z.name}/{name}", "columns": "|".join(map(str, df.columns[:40])),
                           "first_row": "|".join(map(str, df.iloc[0].tolist()[:40])) if len(df) else ""}]
                         ).to_csv(z.parents[2] / "gtnp_unparsed.csv", mode="a", index=False,
                                  header=not (z.parents[2] / "gtnp_unparsed.csv").exists())
            continue
        parts.append(long)
    return pd.concat(parts) if parts else None


def _diff_vs_972777(out: Path, allalt: pd.DataFrame, calm_records: Path | None):
    """Annual ALT per GTN-P ALT dataset (deepest date per year with a value, late season
    only) against the pangaea_calm records of the build."""
    cols = {c.lower(): c for c in allalt.columns}
    tcol = next((cols[c] for c in cols if "date" in c or "time" in c), None)
    vcol = next((cols[c] for c in cols if c.startswith(("alt", "value", "thaw", "thickness"))), None)
    if tcol is None or vcol is None:
        log(f"  gtnp ALT: columns not recognised {list(allalt.columns)[:15]}; diff skipped")
        return
    a = allalt.assign(t=pd.to_datetime(allalt[tcol], errors="coerce"),
                      v=pd.to_numeric(allalt[vcol], errors="coerce")).dropna(subset=["t", "v"])
    a = a[a.t.dt.dayofyear.between(CONFIG.get("eos_doy_min", 213), CONFIG.get("eos_doy_max", 290))]
    per = (a.groupby(["dataset_id", "calm_code", a.t.dt.year.rename("year"), a.t.dt.date.rename("date")])
           .v.mean().reset_index())                        # grid mean per survey date
    ann = per.sort_values("v").groupby(["dataset_id", "calm_code", "year"]).tail(1)
    ann.to_csv(out / "gtnp_alt_annual.csv", index=False)
    if calm_records is None or not calm_records.exists():
        log("  gtnp ALT: no --calm-records given; diff against 972777 skipped")
        return
    rec = pd.read_csv(calm_records, low_memory=False)
    pc = rec[rec.source == "pangaea_calm"][["calm_code", "year", "alt_cm"]]
    m = ann.merge(pc, on=["calm_code", "year"], how="left", indicator=True)
    m["new_site_year"] = m["_merge"] == "left_only"
    m["diff_cm"] = m["v"] - m["alt_cm"]
    m.drop(columns="_merge").to_csv(out / "gtnp_alt_vs_972777.csv", index=False)
    log(f"  gtnp ALT vs 972777: {len(m)} site-years, {m.new_site_year.sum()} not in 972777, "
        f"median |diff| {m.diff_cm.abs().median():.1f} cm on shared years")


# =============================================================================
# 2. Arctic Data Center packages (UAF / TSP)
# =============================================================================
def _solr(q, fl, rows=1000):
    return get_json(f"{D1}/query/solr/", {"q": q, "fl": fl, "rows": rows, "wt": "json"})["response"]["docs"]


def _latest(pid):
    while True:
        d = _solr(f'id:"{pid}"', "id,obsoletedBy")
        if not d or not d[0].get("obsoletedBy"):
            return pid
        pid = d[0]["obsoletedBy"]


def adc(out: Path):
    raw = out / "raw" / "adc"
    pkgs = {_latest("doi:10.18739/A2J34P"): "Network of Permafrost Observatories: Alaska (latest version)"}
    # current (not obsoleted) Romanovsky permafrost-temperature packages, e.g. the yearly
    # "thermal state of permafrost" releases
    for d in _solr('origin:*Romanovsky* AND title:*ermafrost* AND -obsoletedBy:* AND formatType:METADATA',
                   "id,title,dateUploaded"):
        pkgs.setdefault(d["id"], d.get("title", ""))
    inv = []
    for pid, title in pkgs.items():
        files = _solr(f'resourceMap:"resource_map_{pid}" AND formatType:DATA', "id,fileName,size,formatId")
        for f in files:
            name = f.get("fileName") or re.sub(r"\W", "_", f["id"])
            row = {"package": pid, "title": title, "file": name, "size": f.get("size")}
            if not name.lower().endswith(DATA_EXT) or (f.get("size") or 0) > MAX_FILE_MB * 1e6:
                inv.append({**row, "downloaded": False, "reason": "not a data table or too large"})
                continue
            safe = re.sub(r"[^\w.-]", "_", pid)
            download(f"{D1}/object/{requests.utils.quote(f['id'], safe='')}", raw / safe / name)
            inv.append({**row, "downloaded": True, "reason": ""})
    write_inventory(out, "adc", inv)
    parse_folder(out, "adc", raw)


# =============================================================================
# 3. UAF Permafrost Lab site pages
# =============================================================================
def uaf(out: Path):
    raw = out / "raw" / "uaf"
    raw.mkdir(parents=True, exist_ok=True)
    inv = []
    for code, name in UAF_SITES.items():
        url = f"https://permafrost.gi.alaska.edu/site/{code}"
        html = get(url).text
        (raw / f"{code}.html").write_text(html, encoding="utf-8")
        links = sorted(set(re.findall(r'href=["\']([^"\']+)["\']', html)))
        data = [urljoin(url, h) for h in links if h.lower().split("?")[0].endswith(DATA_EXT)
                or "download" in h.lower() or "/data/" in h.lower()]
        if not data:
            inv.append({"site": code, "name": name, "url": url, "file": "",
                        "reason": "no data links in the page HTML (check the page in a browser)"})
        for u in data:
            fn = re.sub(r"[^\w.-]", "_", u.split("/")[-1].split("?")[0]) or "file"
            try:
                download(u, raw / code / fn)
                inv.append({"site": code, "name": name, "url": u, "file": fn, "reason": "downloaded"})
            except (requests.HTTPError, ValueError) as e:
                inv.append({"site": code, "name": name, "url": u, "file": fn, "reason": str(e)[:200]})
    write_inventory(out, "uaf", inv)
    parse_folder(out, "uaf", raw)


# =============================================================================
# 4. NEON soil temperature
# =============================================================================
def neon(out: Path):
    import neonutilities as nu     # pip install neonutilities
    raw = out / "raw" / "neon"
    raw.mkdir(parents=True, exist_ok=True)
    prod = get_json("https://data.neonscience.org/api/v0/products/DP1.00041.001")["data"]
    (raw / "product.json").write_text(json.dumps(prod))
    heave = [c for c in prod.get("changeLogs") or [] if "heav" in (c.get("issue") or "").lower()]
    inv, temps_all, site_rows = [], [], []
    for site in NEON_SITES:
        cache = raw / f"{site}_daily.csv"
        if not cache.exists():
            d = nu.load_by_product(dpid="DP1.00041.001", site=site, package="basic",
                                   timeindex="30", check_size=False, include_provisional=True,
                                   progress=False)
            st = d["ST_30_minute"]
            st = st[st["finalQF"] == 0]
            st["date"] = pd.to_datetime(st["startDateTime"]).dt.tz_localize(None).dt.normalize()
            daily = (st.groupby(["horizontalPosition", "verticalPosition", "date"])["soilTempMean"]
                     .max().reset_index())
            daily.to_csv(cache, index=False)
            d["sensor_positions_00041"].to_csv(raw / f"{site}_sensor_positions.csv", index=False)
            if "issueLog_00041" in d:
                d["issueLog_00041"].to_csv(raw / f"{site}_issueLog.csv", index=False)
        daily = pd.read_csv(cache, parse_dates=["date"])
        pos = pd.read_csv(raw / f"{site}_sensor_positions.csv")
        pos = pos[pos["siteID"] == site] if "siteID" in pos else pos
        hv = pos["HOR.VER"].astype(str).str.split(".", expand=True)
        pos["hor"], pos["ver"] = hv[0].astype(int), hv[1].astype(int)
        pos["start"] = pd.to_datetime(pos["positionStartDateTime"]).dt.tz_localize(None)
        pos["end"] = pd.to_datetime(pos["positionEndDateTime"], errors="coerce").dt.tz_localize(None)
        daily["hor"], daily["ver"] = daily.horizontalPosition.astype(int), daily.verticalPosition.astype(int)
        for (hor, year), g in daily.groupby(["hor", daily.date.dt.year]):
            y0, y1 = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 1, 7, 1)   # thaw + next winter
            p = pos[(pos.hor == hor) & (pos.start < y1) & (pos.end.isna() | (pos.end > y0))]
            moved = p.groupby("ver").size().gt(1).any()
            logged = any(_neon_issue_hits(c, site, hor, y0, y1) for c in heave)
            row = {"site": site, "plot": hor, "year": year, "position_change": bool(moved),
                   "heave_issue": bool(logged)}
            if moved or logged:
                inv.append({**row, "kept": False}); continue
            depth = (p.groupby("ver").zOffset.first().abs() * 100).to_dict()   # m -> cm
            gy = g[g.ver.isin(depth)].assign(depth_cm=lambda x: x.ver.map(depth))
            key = f"NEON_{site}_{hor:03d}"
            temps_all.append(pd.DataFrame({"source_site_id": key, "time": gy.date,
                                           "depth_cm": gy.depth_cm, "temp_c": gy.soilTempMean}))
            inv.append({**row, "kept": True})
            if key not in {s["source_site_id"] for s in site_rows}:
                site_rows.append({"source_site_id": key, "site_name": f"NEON {site} plot {hor}",
                                  "lat": pos.get("referenceLatitude", pd.Series([np.nan])).iloc[0],
                                  "lon": pos.get("referenceLongitude", pd.Series([np.nan])).iloc[0],
                                  "country": "United States", "source_file": "DP1.00041.001"})
    write_inventory(out, "neon", inv)
    if temps_all:
        # the refreeze check needs the following winter, which a kept plot-year may lack:
        # temperature rows of rejected years still feed only that check, never the value
        t = pd.concat(temps_all)
        alt = profiles_to_alt(t, pd.DataFrame(site_rows), "soil_temp")
        kept = {(f"NEON_{r['site']}_{r['plot']:03d}", r["year"]) for r in inv if r["kept"]}
        alt = alt[[(k, y) in kept for k, y in zip(alt.source_site_id, alt.year)]]
        alt.to_csv(out / "neon_alt_records.csv", index=False)
        log(f"  neon: ALT for {len(alt)} plot-years")


def _neon_issue_hits(c, site, hor, y0, y1):
    """True when a heave issue-log entry names this site and plot and overlaps [y0, y1)."""
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


# =============================================================================
# 5. Zenodo (Syrdakh) and NSIDC G02189
# =============================================================================
def zenodo(out: Path):
    raw = out / "raw" / "zenodo_syrdakh"
    rec = get_json(f"https://zenodo.org/api/records/{ZENODO_RECORD}")
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "record.json").write_text(json.dumps(rec))
    inv = []
    for f in rec.get("files", []):
        url = f["links"].get("self") or f["links"].get("content")
        download(url, raw / f["key"])
        inv.append({"file": f["key"], "size": f.get("size")})
        if f["key"].lower().endswith(".zip"):
            with zipfile.ZipFile(raw / f["key"]) as zf:
                zf.extractall(raw / Path(f["key"]).stem)
    write_inventory(out, "zenodo_syrdakh", inv)
    parse_folder(out, "zenodo_syrdakh", raw)


def nsidc(out: Path):
    raw = out / "raw" / "nsidc_g02189"
    raw.mkdir(parents=True, exist_ok=True)
    inv = []

    def walk(url, depth=0):
        html = get(url).text
        for h in re.findall(r'href="([^"?#]+)"', html):
            if h.startswith(("/", "..", "http")) and not h.startswith(url):
                continue
            u = urljoin(url, h)
            if h.endswith("/") and depth < 2:
                walk(u, depth + 1)
            elif u.lower().endswith(DATA_EXT + (".pdf",)):
                rel = u[len(NSIDC_G02189):]
                download(u, raw / rel)
                inv.append({"file": rel})
    walk(NSIDC_G02189)
    write_inventory(out, "nsidc_g02189", inv)


def metno(out: Path):
    """Every station of the MET Norway permafrost THREDDS catalogue, read with xarray over
    OPeNDAP (pip install xarray netCDF4), to a long daily-maximum table."""
    import xarray as xr
    raw = out / "raw" / "metno"
    raw.mkdir(parents=True, exist_ok=True)
    cat = get(METNO_CATALOG).text
    (raw / "catalog.xml").write_text(cat)
    stations = sorted(set(re.findall(r'catalogRef[^>]*xlink:href="([^"]+)/catalog\.xml"', cat)))
    inv, temps_all, site_rows = [], [], []
    for st in stations:
        name = st.strip("/").split("/")[-1]
        cache = raw / f"{name}_daily.csv"
        url = f"{METNO_THREDDS}/dodsC/arcticdata/frost2netcdf-permafrost/{name}/{name}-aggregated.ncml"
        if not cache.exists():
            try:
                ds = xr.open_dataset(url)
            except OSError as e:
                inv.append({"station": name, "kept": False, "reason": f"open failed: {e}"[:200]})
                continue
            var = next(v for v in ds.data_vars if "soil_temperature" in str(ds[v].attrs.get("standard_name", "")))
            df = ds[var].to_dataframe().reset_index()
            dcol = next(c for c in df.columns if "depth" in c.lower())
            df = df.rename(columns={var: "temp_c", dcol: "depth_m"})
            df["date"] = pd.to_datetime(df["time"]).dt.normalize()
            df.groupby(["depth_m", "date"]).temp_c.max().reset_index().assign(
                lat=float(ds.attrs.get("geospatial_lat_min", np.nan)),
                lon=float(ds.attrs.get("geospatial_lon_min", np.nan)),
                title=ds.attrs.get("title", name)).to_csv(cache, index=False)
        d = pd.read_csv(cache, parse_dates=["date"])
        years = d.date.dt.year.nunique()
        inv.append({"station": name, "title": d.title.iloc[0], "first": d.date.min().date(),
                    "last": d.date.max().date(), "years": years,
                    "depths_cm": ";".join(f"{v * 100:g}" for v in sorted(d.depth_m.abs().unique())),
                    "kept": True, "reason": ""})
        key = f"METNO_{name}"
        temps_all.append(pd.DataFrame({"source_site_id": key, "time": d.date,
                                       "depth_cm": d.depth_m.abs() * 100, "temp_c": d.temp_c}))
        site_rows.append({"source_site_id": key, "site_name": d.title.iloc[0], "lat": d.lat.iloc[0],
                          "lon": d.lon.iloc[0], "country": "Norway", "source_file": url})
    write_inventory(out, "metno", inv)
    if temps_all:
        alt = profiles_to_alt(pd.concat(temps_all), pd.DataFrame(site_rows), "soil_temp")
        alt.to_csv(out / "metno_alt_records.csv", index=False)
        log(f"  metno: ALT rows {len(alt)} at {alt.source_site_id.nunique()} stations")


# =============================================================================
# Parsing of downloaded folders with unknown layouts
# =============================================================================
def parse_folder(out: Path, name: str, folder: Path):
    """Try every table in the folder; profiles with >= 2 sensors at <= MAX_SHALLOW_CM give
    ALT records. Unread tables go to <name>_unparsed.csv with their header."""
    unparsed, temps_all, site_rows = [], [], []
    for p in sorted(folder.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in DATA_EXT or p.suffix.lower() in (".zip", ".nc"):
            continue
        try:
            tabs = read_any_table(p)
        except (ValueError, UnicodeDecodeError, ImportError) as e:
            unparsed.append({"file": str(p.relative_to(folder)), "error": str(e)[:200]})
            continue
        for sheet, df in tabs:
            long = to_long(df)
            if long is None:
                unparsed.append({"file": str(p.relative_to(folder)), "sheet": sheet,
                                 "columns": "|".join(map(str, df.columns[:30]))})
                continue
            if (long.depth_cm <= MAX_SHALLOW_CM).groupby(long.depth_cm).first().sum() < 2:
                continue
            key = f"{name}:{p.stem}:{sheet}".rstrip(":")
            temps_all.append(long.assign(source_site_id=key))
            site_rows.append({"source_site_id": key, "site_name": p.stem, "lat": np.nan,
                              "lon": np.nan, "country": "", "source_file": p.name})
    pd.DataFrame(unparsed).to_csv(out / f"{name}_unparsed.csv", index=False)
    if temps_all:
        alt = profiles_to_alt(pd.concat(temps_all), pd.DataFrame(site_rows), "soil_temp")
        alt.to_csv(out / f"{name}_alt_records.csv", index=False)
        log(f"  {name}: ALT rows {len(alt)} from {len(site_rows)} tables; {len(unparsed)} tables not read")
    else:
        log(f"  {name}: no profile table read; {len(unparsed)} listed in {name}_unparsed.csv")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=["gtnp", "adc", "uaf", "neon", "zenodo", "nsidc", "metno"])
    ap.add_argument("--calm-records", type=Path, default=None,
                    help="alt_multisource_all_records.csv (for the GTN-P ALT diff against 972777)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    steps = {"gtnp": lambda: gtnp(a.out, a.calm_records), "adc": lambda: adc(a.out),
             "uaf": lambda: uaf(a.out), "neon": lambda: neon(a.out),
             "zenodo": lambda: zenodo(a.out), "nsidc": lambda: nsidc(a.out),
             "metno": lambda: metno(a.out)}
    failed = {}
    for s in a.only:
        log(f"== {s}")
        try:
            steps[s]()
        except (requests.RequestException, KeyError, ValueError, OSError, StopIteration,
                zipfile.BadZipFile, ImportError) as e:
            # one source failing must not stop the others; the error is reported at the end
            failed[s] = f"{type(e).__name__}: {e}"
            log(f"  {s} FAILED: {failed[s]}")
    (a.out / "fetch_errors.json").write_text(json.dumps(failed, indent=1))
    log("done" + (f"; failed: {list(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
