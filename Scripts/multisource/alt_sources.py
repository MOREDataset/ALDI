"""
One reader per data source. Each reader takes the raw folder(s) and returns a table with
the columns in alt_core.RECORD_COLUMNS: one row per series (site) and thaw season.

Nothing outside this file knows about source formats. To add a source: write a reader
that returns finish(table, "<source name>"), and add it to ADAPTERS at the bottom.

The readers only reshape and convert. The rules that turn measurements into one annual
value (end-of-season survey, 0 C crossing of the temperature profile, validity range)
are shared and live in alt_core.py.
"""

from __future__ import annotations

import csv
import io
import re
import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from alt_core import (CONFIG, add_flag, as_float, clean_cell, eml_sites, empty_records,
                      end_of_season, files, finish, log, norm_calm_code, pangaea_events,
                      parse_censored, read_pangaea_tab, records_from_profiles, survey_means,
                      utm_to_latlon)


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
    Adapted from HighwayStability/pipeline/alt_dataset_pipeline.py (parse_gtnp_grid).
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
    except ImportError:
        log("    gsc: pdfplumber not installed; OF 9340 thaw-tube table skipped "
            "(pip install pdfplumber)")
        return None
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
    annual maximum-temperature profile (alt_core.alt_from_temperature_profile).
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
        # position: 'N 70o09.677' W148o27.918'' or '148.4653 W; 70.161283 N' and similar
        loc = str(x.iat[2, 0])
        dm = re.findall(r"([NW])\s*(\d+)o(\d+(?:\.\d+)?)", loc)
        if dm:
            v = {h: int(d) + float(m) / 60 for h, d, m in dm}
            pos.setdefault(code, (v.get("N"), -v.get("W")))
        else:
            n = re.search(r"(\d+\.\d+)\s*o?\s*N|N\s*(\d+\.\d+)", loc)
            w = re.search(r"-?(\d+\.\d+)\s*o?\s*W|W\s*(\d+\.\d+)", loc)
            if n and w:
                pos.setdefault(code, (float(n.group(1) or n.group(2)), -float(w.group(1) or w.group(2))))
        # header row: the first row with at least three depth labels
        lab = lambda c: re.fullmatch(r"(?:Temp_MRC_)?(-?\d+\.\d+)\s*m?", str(c).strip())
        hr = next((i for i in range(min(30, len(x))) if sum(bool(lab(c)) for c in x.iloc[i]) >= 3), None)
        if hr is None:
            continue
        body = x.iloc[hr + 1:]
        def as_dates(col):
            v = pd.to_numeric(col, errors="coerce")
            if v.notna().mean() > 0.5:                   # plain numbers (year, day, YYYYMMDD)
                return pd.Series(pd.NaT, index=col.index)
            return pd.to_datetime(col.astype(str).str[:10], format="%Y-%m-%d", errors="coerce")
        tcol = max(range(min(5, x.shape[1])), key=lambda j: as_dates(body[j]).notna().mean())
        time = as_dates(body[tcol])
        for j, c in enumerate(x.iloc[hr]):
            m = lab(c)
            if not m or j == tcol or float(m.group(1)) <= 0:
                continue
            v = pd.to_numeric(body[j], errors="coerce")
            v[v.abs() >= 99] = np.nan
            temps.append(pd.DataFrame(dict(source_site_id="UAF_" + code, time=time,
                                           depth_cm=round(float(m.group(1)) * 100, 1), temp_c=v)))
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
    "above_field": read_above_field,
    "cusp": read_cusp,
}
