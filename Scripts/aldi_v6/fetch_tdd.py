"""Thawing degree-days (TDD) for every ALDI site-year, from daily air temperature.

    python fetch_tdd.py  path/to/alt_multisource_annual.csv  out_dir  [--source power]

Writes
  out_dir/daily_t2m/<site_id>.csv   (":" in an id becomes "__") daily mean 2 m air temperature per site (cache;
                                    a site already cached is not downloaded again)
  out_dir/tdd_site_years.csv        one row per site-year of the annual file
  out_dir/tdd_all_years.csv         every calendar year per site (whole-year TDD and
                                    the preceding winter's freezing index), for model B

Source: NASA POWER daily point API (MERRA-2 based T2M, 0.5 x 0.625 deg, 1981 onward,
no key). It is used because it runs anywhere without an account. ERA5-Land from the
GEE covariate script can replace it: put one CSV per site with columns date,t2m_c into
daily_t2m/ and rerun with --source cache.

TDD window (decided here, reasons):
- probe surveys measure thaw on the survey day, so TDD runs from 1 January to the
  survey date. Without a date it runs to the site's median survey day, and without
  any date at the site to the median survey day of all dated probe surveys
  (CALM end-of-season protocol; the build keeps the deepest survey of DOY 213-290).
- thaw tubes and temperature-derived ALT record the season's maximum thaw, so TDD is
  the whole year's thawing index (1 January to 31 December).
TDD = sum over days of max(T_mean, 0) in deg C day.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M"
             "&community=RE&longitude={lon:.4f}&latitude={lat:.4f}"
             "&start={start}0101&end={end}1231&format=JSON")


def fetch_power(lat, lon, start, end, tries=4):
    url = POWER_URL.format(lat=lat, lon=lon, start=start, end=end)
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                js = json.load(r)
            t = js["properties"]["parameter"]["T2M"]
            s = pd.Series(t, dtype=float)
            s.index = pd.to_datetime(s.index, format="%Y%m%d")
            s[s <= -998] = np.nan          # POWER fill value -999
            elev = js.get("geometry", {}).get("coordinates", [None, None, None])[2]
            return s, elev
        except Exception as e:  # network errors: retry with backoff
            if k == tries - 1:
                raise
            print(f"  retry {k + 1} after {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 ** (k + 1))


def cache_name(site_id):
    """File name for a site id; ':' is not allowed in Windows file names."""
    return str(site_id).replace(":", "__").replace("/", "_") + ".csv"


def site_table(annual: pd.DataFrame) -> pd.DataFrame:
    a = annual[annual["use"] == True]  # noqa: E712
    return a.groupby("site_id")[["lat", "lon"]].first()


def tdd_for_site(t: pd.Series, years, end_doy):
    """TDD from 1 Jan to day end_doy (inclusive) of each year; NaN if > 5 % days missing."""
    out = []
    for y, e in zip(years, end_doy):
        s = t[(t.index.year == y) & (t.index.dayofyear <= e)]
        n_exp = int(e)
        if len(s) < 0.95 * n_exp or s.isna().mean() > 0.05:
            out.append(np.nan)
            continue
        # missing days (<= 5 %) filled with the mean positive part of the window
        pos = s.clip(lower=0)
        out.append(float(pos.mean() * n_exp))
    return np.array(out)


def window_sum(v: pd.Series, n_exp: int):
    """Sum over a window; NaN if more than 5 % of the n_exp days are missing."""
    v = v.dropna()
    if len(v) < 0.95 * n_exp:
        return np.nan
    return float(v.mean() * n_exp)


def all_years(site_ids, cache, start, end):
    """Every calendar year per site: thawing index of the whole year and to DOY 262,
    and the freezing index of the preceding winter (1 Oct of the previous year to
    30 Apr). Used for the last-year TDD and winter FDD terms of model B."""
    rows = []
    for sid in site_ids:
        f = cache / cache_name(sid)
        if not f.exists():
            continue
        t = pd.read_csv(f, parse_dates=["date"]).set_index("date")["t2m_c"]
        for y in range(start, end + 1):
            yr = t[t.index.year == y]
            n_year = 366 if pd.Timestamp(y, 12, 31).dayofyear == 366 else 365
            win = t[(t.index >= pd.Timestamp(y - 1, 10, 1)) & (t.index <= pd.Timestamp(y, 4, 30))]
            n_win = (pd.Timestamp(y, 4, 30) - pd.Timestamp(y - 1, 10, 1)).days + 1
            rows.append({"site_id": sid, "year": y,
                         "tdd_full_year": window_sum(yr.clip(lower=0), n_year),
                         "tdd_to_doy262": window_sum(yr[yr.index.dayofyear <= 262].clip(lower=0), 262),
                         "fdd": window_sum((-win).clip(lower=0), n_win),
                         "n_days": int(yr.notna().sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annual_file")
    ap.add_argument("out_dir")
    ap.add_argument("--source", default="power", choices=["power", "cache"])
    ap.add_argument("--start", type=int, default=1981)  # POWER starts in 1981
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args()

    out = Path(args.out_dir)
    cache = out / "daily_t2m"
    cache.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(args.annual_file, low_memory=False)
    sites = site_table(annual)

    elev = {}
    for k, (sid, row) in enumerate(sites.iterrows()):
        f = cache / cache_name(sid)
        if f.exists():
            continue
        if args.source == "cache":
            print(f"no cached temperature for {sid}", file=sys.stderr)
            continue
        print(f"[{k + 1}/{len(sites)}] {sid}  {row.lat:.3f} {row.lon:.3f}")
        s, e = fetch_power(row.lat, row.lon, args.start, args.end)
        pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "t2m_c": s.values}).to_csv(f, index=False)
        elev[sid] = e
        time.sleep(args.sleep)
    if elev:   # grid-cell elevation of the sites downloaded in this run (appended)
        ef = out / "power_grid_elevation.csv"
        new = pd.Series(elev, name="grid_elev_m").rename_axis("site_id").reset_index()
        if ef.exists():
            new = pd.concat([pd.read_csv(ef), new]).drop_duplicates("site_id", keep="last")
        new.to_csv(ef, index=False)

    # survey-day rule for probe sites
    a = annual[annual["use"] == True].copy()  # noqa: E712
    a["doy"] = pd.to_datetime(a.get("obs_date"), errors="coerce").dt.dayofyear
    probe = a["method"] == "probe"
    net_doy = float(a.loc[probe, "doy"].median()) if a.loc[probe, "doy"].notna().any() else 243.0
    site_doy = a[probe].groupby("site_id")["doy"].median()

    rows = []
    for sid, g in annual.groupby("site_id"):
        f = cache / cache_name(sid)
        if not f.exists():
            continue
        t = pd.read_csv(f, parse_dates=["date"]).set_index("date")["t2m_c"]
        g = g.drop_duplicates(["year", "method"])
        fam_full = g["method"].isin(["thaw_tube", "borehole_temp", "soil_temp", "soil_temp_monthly"])
        doy = pd.to_datetime(g.get("obs_date"), errors="coerce").dt.dayofyear
        sdoy = site_doy.get(sid, np.nan)
        end = np.where(fam_full, 365,
                       np.where(doy.notna(), doy, np.where(np.isfinite(sdoy), sdoy, net_doy)))
        rule = np.where(fam_full, "full_year",
                        np.where(doy.notna(), "survey_date",
                                 np.where(np.isfinite(sdoy), "site_median_survey_doy",
                                          "network_median_survey_doy")))
        years = g["year"].to_numpy(int)
        tdd = tdd_for_site(t, years, end.astype(int))
        tdd_full = tdd_for_site(t, years, np.full(len(years), 365))
        rows.append(pd.DataFrame({"site_id": sid, "year": years, "method": g["method"].values,
                                  "tdd_end_doy": end.astype(int), "tdd_rule": rule,
                                  "tdd": tdd, "tdd_full_year": tdd_full,
                                  "t_source": args.source}))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "tdd_site_years.csv", index=False)
    all_years(sites.index, cache, args.start, args.end).to_csv(out / "tdd_all_years.csv", index=False)
    print(f"wrote {len(res)} site-years for {res.site_id.nunique()} sites; "
          f"missing TDD: {int(res.tdd.isna().sum())}; network median probe survey DOY {net_doy:.0f}")


if __name__ == "__main__":
    main()
