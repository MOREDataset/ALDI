"""ERA5-Land inputs for model B, from the Earth Engine exports of gee_causal_features.py.

    python era5land_for_b.py  drive_export_dir  tdd_dir  [--grid]

drive_export_dir is the Google Drive folder the exports went to (ALDI_GEE_exports). It
must hold the groups t2m_daily (daily air temperature), era5 (snow and rain per ALT year)
and static (site and cell elevation). In Colab, after the usual setup cells:

    gee.export_all(sites, years=range(1989, 2026), groups=("static", "era5", "t2m_daily"),
                   drive_dir=DRIVE_DIR)

(1989 so that the previous-year TDD of 1990 exists.)

Writes
  tdd_dir/daily_t2m/<site_id>.csv   date, t2m_c: one continuous daily series per site, the
                                    format fetch_tdd.py reads with --source cache
  tdd_dir/weather_site_years.csv    site_id, year and the ERA5-Land weather terms per ALT
                                    year (Oct of the previous year to Sep), for run_aldi_v6.py
  tdd_dir/era5land_cells.csv        per site: elevation offset to the ERA5-Land cell, whether
                                    the cell was filled from neighbours (sea cell)

Then, unchanged from the POWER run:

    python fetch_tdd.py annual.csv tdd_dir --source cache
    python run_aldi_v6.py aldi_v6_config.yaml annual.csv out_dir \
        tdd_dir/tdd_site_years.csv tdd_dir/tdd_all_years.csv tdd_dir/weather_site_years.csv

Temperature is corrected to the site elevation with the summer lapse rate of
gee_causal_features.CONFIG (-6.5 K/km May-Sep, none Oct-Apr because of winter
inversions), as the causal features are. --grid keeps the plain ERA5-Land cell values
(sensitivity run); --summer-lapse -4.5 uses a gentler summer lapse rate (sensitivity:
the lapse rate is a larger uncertainty than the DEM). Sea cells filled from their land neighbours are not corrected.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "covariates"))
import gee_causal_features as gee  # noqa: E402  (does not need Earth Engine for reading)
from fetch_tdd import cache_name   # noqa: E402

WEATHER_COLS = ["snow_depth_winter", "snow_thermal_resistance_winter", "rain_summer",
                "P_summer", "snowfall_winter", "SWE_max", "snow_off_doy", "TDD", "FDD"]


def elevation_offsets(drive_dir):
    st, f = gee._read_group_static(drive_dir)
    if not len(st) or "z_cell_era5" not in st:
        print("WARNING: no static export with z_cell_era5; temperature not corrected")
        return pd.Series(dtype=float)
    st = st.drop_duplicates("site_id").set_index("site_id")
    return (st["elevation_m"] - st["z_cell_era5"]).astype(float)


def export_files(drive_dir, group):
    """{ALT year: newest file} of one export group ('t2m_daily_1990.csv', not 'era5_daily_...';
    a Drive duplicate such as 't2m_daily_1990 (1).csv' wins when newer)."""
    pat = re.compile(rf"^{re.escape(group)}_(\d{{4}})(?: \(\d+\))?\.csv$")
    files = {}
    for f in sorted(Path(drive_dir).glob(f"{group}_*.csv"), key=lambda q: q.stat().st_mtime):
        m = pat.match(f.name)
        if m:
            files[int(m.group(1))] = f
    return dict(sorted(files.items()))


def daily_series(drive_dir, dz, grid_only=False, summer_lapse=None):
    """{site_id: Series of daily mean temperature (C)} from all t2m_daily exports. The ALT
    years (Oct-Sep) do not overlap, so the years join into one series per site. Files are
    read one at a time (each has 730 day columns)."""
    files = export_files(drive_dir, "t2m_daily")
    if not files:
        sys.exit(f"no t2m_daily_*.csv in {drive_dir}")
    lapse = dict(gee.CONFIG["lapse_rate_k_per_km"])
    if summer_lapse is not None:          # sensitivity: another May-Sep lapse rate
        lapse = {m: (summer_lapse if v != 0 else 0.0) for m, v in lapse.items()}
    filled, parts = {}, {}
    for year, f in files.items():
        g = pd.read_csv(f).drop_duplicates("site_id")
        cols = sorted(c for c in g.columns if re.fullmatch(r"\d{8}_tm", str(c)))
        dates = pd.to_datetime([c[:8] for c in cols], format="%Y%m%d")
        rate = np.array([lapse[m] for m in dates.month], float)
        tm = g[cols].to_numpy(float)
        fl_col = g["era5_filled"].fillna(0).to_numpy() > 0 if "era5_filled" in g else np.zeros(len(g), bool)
        for i, sid in enumerate(g["site_id"].astype(str)):
            filled[sid] = filled.get(sid, False) or bool(fl_col[i])
            z = 0.0 if (grid_only or fl_col[i]) else float(np.nan_to_num(dz.get(sid, np.nan)))
            parts.setdefault(sid, []).append(pd.Series(tm[i] + rate * z / 1000.0, index=dates))
    series = {}
    for sid, p in parts.items():
        s = pd.concat(p).sort_index()
        series[sid] = s[~s.index.duplicated(keep="last")]
    return series, filled, list(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drive_dir")
    ap.add_argument("tdd_dir")
    ap.add_argument("--grid", action="store_true", help="no elevation correction")
    ap.add_argument("--summer-lapse", type=float, default=None,
                    help="May-Sep lapse rate in K/km for a sensitivity run (default -6.5 from CONFIG)")
    a = ap.parse_args()
    out = Path(a.tdd_dir)
    cache = out / "daily_t2m"
    cache.mkdir(parents=True, exist_ok=True)

    dz = elevation_offsets(a.drive_dir)
    series, filled, got = daily_series(a.drive_dir, dz, a.grid, a.summer_lapse)
    for sid, s in series.items():
        pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "t2m_c": s.values}).to_csv(
            cache / cache_name(sid), index=False)
    cells = pd.DataFrame({"site_id": list(series),
                          "dz_era5_m": [dz.get(s, np.nan) for s in series],
                          "era5_filled": [int(filled[s]) for s in series],
                          "corrected": [int(not a.grid and not filled[s] and np.isfinite(dz.get(s, np.nan)))
                                        for s in series]})
    cells.to_csv(out / "era5land_cells.csv", index=False)

    wf = export_files(a.drive_dir, "era5")
    wy = list(wf)
    w = pd.concat([pd.read_csv(f).assign(year=y) for y, f in wf.items()], ignore_index=True) if wf else pd.DataFrame()
    if len(w):
        keep = ["site_id", "year"] + [c for c in WEATHER_COLS if c in w]
        w = w.drop_duplicates(["site_id", "year"])[keep]
        w = w.rename(columns={"TDD": "TDD_may_sep_grid", "FDD": "FDD_grid"})
        w.to_csv(out / "weather_site_years.csv", index=False)
        print(f"weather: {len(w)} site-years, ALT years {min(wy)}-{max(wy)}")
    else:
        print("WARNING: no era5_*.csv exports; the weather variant of B cannot run")

    print(f"daily temperature: {len(series)} sites, ALT years {min(got)}-{max(got)}; "
          f"{int(cells.corrected.sum())} corrected to site elevation, "
          f"{int(cells.era5_filled.sum())} sea cells filled from neighbours")
    print(f"next: python fetch_tdd.py <annual.csv> {out} --source cache")


if __name__ == "__main__":
    main()
