"""
STEP 3: ALDI A + B, the active layer dynamics index.

    py -3.12 3_aldi.py                A and B with ERA5-Land temperature   (about 10-30 min)
    py -3.12 3_aldi.py --power        also B with NASA POWER temperature   (robustness)
    py -3.12 3_aldi.py --validate     also the simulation check            (1-2 hours)
    py -3.12 3_aldi.py --sensitivity  also the sensitivity runs            (about 1 hour)
    py -3.12 3_aldi.py --checks       also the paper checks (detection time, per-site OLS test,
                                      residual by ice and method, cm per decade; ~1 min, no refit)
    py -3.12 3_aldi.py --no-main --sensitivity --checks   only these, on the existing data/3_aldi
    py -3.12 3_aldi.py --spatial      only the spatial checks on the existing results (~1 min)

A  One hierarchical Bayesian trend model for all ALT series: each site's rate of change
   (% per decade), its 90 % interval and its class (rapid / gradual thickening, stable,
   gradual / rapid thinning, indeterminate).
B  The same trends split into the part expected from air temperature (thawing
   degree-days, TDD, Stefan's law) and a residual that air temperature does not explain,
   in seven variants (this summer's TDD only ... TDD memory, winter, snow and rain).
v5.2, the site-by-site method of the earlier version, is rerun on the same site-years
for comparison.

Reads   data/1_alt_target/alt_multisource_annual.csv   (step 1)
        data/2_covariates/  static.csv, era5_YYYY.csv, t2m_daily_YYYY.csv   (step 2)
Writes  data/3_aldi/
          aldi_v6_results.csv    one row per site: v5.2 and v6 rates and classes (A),
                                 expected and residual trends per variant (B)
          aldi_v6_summary.txt    counts, network rates, B variants, diagnostics
          aldi_v6_network.csv    network parameters (median and 95 % interval)
          aldi_v6_fig_*.png      figures
          temperature/           daily temperature per site, TDD and weather tables
        data/3_aldi_power/ (--power), data/3_aldi_validation/ (--validate),
        data/3_aldi_sensitivity/ (--sensitivity: aldi_v6_sensitivity.txt / .csv, per-site results)
        data/3_aldi_checks/ (--checks: aldi_v6_checks.txt and its tables, see PART 4c)

A new run is written to data/3_aldi_building first and replaces data/3_aldi only when it
finished without error; the run before is kept in data/previous/3_aldi.

All settings are in SETTINGS just below, each with the reason for it. The method in
detail: docs/3_aldi.md.

Python 3.12 with numpy scipy pandas pyyaml numpyro jax matplotlib.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import re
import shutil
import sys
import time
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import optimize, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ANNUAL = DATA / "1_alt_target" / "alt_multisource_annual.csv"
EE_DIR = DATA / "2_covariates"
OUT = DATA / "3_aldi"
POWER_CACHE = DATA / "raw" / "nasa_power"     # downloaded once, reused


# =========================================================================================
# SETTINGS (every value that changes a class or a model is here, with the reason)
# =========================================================================================
SETTINGS = """
sample:
  min_lat: 60.0          # same as v4 / v5
  min_year: 1990         # chosen 2026-10-25 (multi-source thread)
  max_year: 2025
  # Eligibility = v5 decision of 2026-10-06 (sensitivity: 8 obs).
  min_obs: 10            # uncensored years
  min_span_years: 12
  # Lower-bound years are used as censored values only when their other flags are
  # informational (a talik, sensor or equilibration flag still removes them).
  use_censored: true
  censored_ok_flags: [lower_bound, burned_site, unusual_year_isolated, unusual_year_regional, nonmonotonic_profile,
                      edge_max, autumn_gap, front_indeterminate]
  # One method family per series. Probe, thaw tube and temperature ALT differ in level
  # (processing_notes.md), so a family switch inside a series creates a step that a
  # trend test reads as change. Years of the minority family are dropped (20 sites).
  single_method_family: true
  method_families:
    probe: probe
    thaw_tube: thaw_tube
    borehole_temp: temperature
    soil_temp: temperature
    soil_temp_monthly: temperature

trend:
  # Two-sided 90 % interval = two one-sided 5 % tests, so "stable" below is a TOST
  # equivalence test at 5 % and "thickening/thinning" a one-sided test at 5 %.
  ci_level: 0.90
  # AR(1) correction of var(S): lag-1 autocorrelation of the detrended series,
  # bias-corrected, clipped to [0, ar1_max]. Negative values give no correction.
  ar1_max: 0.85
  # pooled (v5.2): one AR(1) coefficient for the network, REML over all series, used
  # for every site. site: each site's own lag-1 estimate (v5.1). The site estimate is
  # biased low and noisy for 10-35 values, which made the 90 % interval too narrow
  # (simulated coverage 0.80-0.87 at rho 0.5); pooled gives 0.90-0.93.
  ar1_method: pooled

classes:
  # Rates in % of the site's ALT per decade (v5 decision of 2026-10-06; sensitivity
  # stable 2.5 / 10 and rapid 5 / 15). Fixed margins, not sample percentiles, so a
  # site's class does not change when other sites are added or removed.
  stable_margin_pct_dec: 5.0
  rapid_pct_dec: 10.0

nonlinearity:
  # Continuous broken-stick (hinge) fit against a straight line; sup-F over all
  # breakpoints with >= min_segment_obs years on each side; p-value from an AR(1)
  # parametric bootstrap under the straight-line null.
  min_segment_obs: 6
  n_boot: 999
  alpha: 0.05

pooling:
  # Partial pooling of site rates: rate_i = mu + u_region + v_site + error_i, with the
  # error SE from the site's own interval. Short records shrink more toward their region.
  region_km: 250         # complete-linkage cluster diameter for region_id
  min_region_sites: 1

diagnostics:
  survey_doy_drift_days: 14   # flag when the survey date drifts by more than this over the record
  power: 0.80                 # for the minimum detectable trend

random_seed: 42

model:
  # Shared region-year anomalies (sites within 250 km see the same warm or cold summer).
  # true: modelled as a random effect (main run); false: left in each site's noise.
  region_year: true
  # Site deviations from their region: normal (main) or student (t with 4 df,
  # sensitivity for the few heavy-tailed BNZ / Mackenzie rates).
  slope_dist: normal
  chains: 4
  warmup: 1500
  draws: 1500
  target_accept: 0.95
  seed: 2026
  priors:
    mu_sd: 0.2         # network trend, log units per decade (0.2 = 22 %/dec)
    tau_sd: 0.1        # half-normal scale of region and site SDs (log/decade)
    a_sd: 0.5          # site level around its mean log ALT
    # How far above a lower bound the true ALT may lie (log units). null: no prior beyond
    # "above the bound" (flat), so a bound only says ALT >= bound: the standard censored
    # likelihood. A number s: HalfNormal(s). 0.5 was used up to 2026-11-07; it pulled the
    # true ALT toward bounds far below a site's ALT (cables whose sensors reach only 10 cm
    # at SALTSIL 2017-2022 and Parks Canada C28 2015, where ALT is 90-190 cm), which
    # turned SALTSIL's +26 %/dec into +3. Sensitivity runs keep 0.5 and 1.0.
    cens_sd: null

climate:
  # Variants of the climate-expected part. main: Stefan with this year's TDD and k per
  # site around a family mean (prior centre 0.5). kfixed: k = 0.5 everywhere (pure
  # Stefan). lag: adds last year's TDD (thaw memory). air: adds last year's TDD and the
  # winter freezing index. lag, memory, air and the weather variants need
  # tdd_all_years.csv.
  variants:
    - {name: main,   k_mode: hierarchical}
    - {name: kfixed, k_mode: fixed}
    - {name: lag,    k_mode: hierarchical, extra: [tdd_prev]}
    # memory: TDD of the two previous years as well. ALT follows TDD with a lag (the
    # 2026-11-04 review: this year's k is 0.2-0.3, but with the lags the long-run k is
    # about 0.35-0.45). Lags of three and four years (0.03-0.06) were close to the
    # coefficient of NEXT year's TDD (0.03; a placebo, since next summer cannot cause this
    # year's thaw) and did not fall off steadily, so they cannot be told from slow
    # variation that ALT and TDD share, and are left out. A lag of two years needs
    # temperature from two years before the first ALT year (1988 for 1990); without it the
    # first years of a record drop out of this variant.
    - {name: memory, k_mode: hierarchical, extra: [tdd_lag1, tdd_lag2]}
    - {name: air,    k_mode: hierarchical, extra: [tdd_prev, fdd]}
    # weather: air plus winter snow depth (insulation: deep snow keeps the ground warm in
    # winter) and summer rain (heat carried into the soil, wetter soil conducts better).
    # What is left is change that weather does not explain (ground ice, subsidence,
    # vegetation, disturbance). Needs weather_site_years.csv from era5land_for_b.py.
    - {name: weather, k_mode: hierarchical, extra: [tdd_prev, fdd, snow_depth_winter, rain_summer]}
    # weather_memory: weather with the two-year TDD memory, the most that the weather
    # record can explain.
    - {name: weather_memory, k_mode: hierarchical, extra: [tdd_lag1, tdd_lag2, fdd, snow_depth_winter, rain_summer]}
  k_fixed: 0.5
  # B runs without region-year effects: most of the shared year-to-year swing is the
  # TDD signal that k is estimated from.
  region_year: false

outputs:
  prefix: aldi_v6
  # long-record check: sites with at least this span get a second trend over only
  # their last `recent_years` years, to see whether short (recent) records are faster
  # because they are short or because they are recent
  period_check_min_span: 25
  recent_years: 15
"""

# Air temperature is corrected to the site elevation before TDD: T_site = T_cell +
# lapse * (z_site - z_cell) / 1000. Summer: the standard -6.5 K/km. Oct-Apr: none, because
# winter inversions make valley floors colder than the cell mean (same rule as step 2).
LAPSE_RATE_K_PER_KM = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: -6.5, 6: -6.5, 7: -6.5,
                       8: -6.5, 9: -6.5, 10: 0.0, 11: 0.0, 12: 0.0}


def load_settings():
    return yaml.safe_load(SETTINGS)


def _read_static(ee_dir):
    """The newest static*.csv of the Earth Engine folder (site and cell elevations)."""
    files = sorted(Path(ee_dir).glob("static*.csv"), key=lambda q: q.stat().st_mtime)
    if not files:
        return pd.DataFrame(), None
    return pd.read_csv(files[-1]), files[-1]


# =========================================================================================
# PART 1. ALDI v5.2: the data rules, the site trend test and the classes.
# v6 reruns it on the same site-years for the comparison columns.
# =========================================================================================
# ALDI v5.2: trend classes for active-layer thickness series.
#
# What changes from the v4 notebook (ALDI_Calculation_final.ipynb), and why:
#
# 1. No gap filling. Theil-Sen and Kendall's S work on irregular years, so PCHIP values
#    no longer enter the test as if they were measurements.
# 2. Lower-bound years (thaw reached the probe or the deepest sensor) stay in as
#    right-censored values (Akritas-Theil-Sen, Akritas et al. 1995). Dropping them
#    removes the deepest years of deepening sites and pulls their slopes toward zero.
# 3. Rates are in % of the site's ALT per decade with fixed margins (stable +/-5,
#    rapid 10). v4 used the 50th / 80th percentiles of the sample's slopes in cm/yr,
#    so a site's class depended on which other sites were in the sample.
# 4. "No trend" is split into "stable" (the 90 % interval lies inside +/-5 %/dec, an
#    equivalence test) and "indeterminate" (the record cannot tell). v4 put both in
#    one class, which made short records look stable or rapid.
# 5. The interval comes from inverting Kendall's S with an AR(1) variance correction
#    (Hamed and Rao 1998 with a fitted AR(1) instead of the significant sample lags,
#    which are unstable for 10-30 values). For uncensored data without autocorrelation
#    it is exactly Gilbert's (1987) rank interval for the Sen slope, and it extends to
#    censored years, which the v4 block bootstrap cannot. In simulations
#    (validate_aldi.py) both intervals hold their 90 % level when rho = 0 and both fall
#    to 80-87 % when rho = 0.5 if rho is estimated per site.
#    Since v5.2 the AR(1) coefficient is one value for the whole network, fitted by
#    REML on all series together (pooled_ar1). The per-site lag-1 estimate is biased
#    low (a fitted trend and gaps both pull it down; at n = 12 and true rho = 0.5 it
#    averages about 0.3) and very noisy (SD about 0.3 at n = 12), which is why the
#    interval was too narrow. With the pooled value the simulated coverage is 0.90-0.93
#    for rho 0 to 0.5. The per-site value is still reported (ar1_rho_site).
# 6. Reversal is tested with a broken-stick fit and an AR(1) bootstrap of the sup-F
#    statistic, so the breakpoint year is estimated, not fixed at the first and last
#    seven years, and the test has a known false-positive rate. It is a flag.
# 7. Partial pooling: site rates are shrunk toward their region with a two-level
#    random-effects model. Short noisy records shrink most, which weakens the link
#    between record length and "rapid" found in the 2026-10-25 review. The same model
#    gives the network trend with regions as clusters.
#
# Units: years in calendar years, ALT in cm, slopes in cm/yr, rates in %/decade.

EARTH_RADIUS_KM = 6371.0088

CLASSES = ["rapid_thickening", "gradual_thickening", "stable",
           "gradual_thinning", "rapid_thinning", "indeterminate"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_series(annual: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Site-year table for ALDI from alt_multisource_annual.csv.

    Returns (data, log). data has one row per site-year: site_id, lat, lon, year,
    alt_cm, censored (True = ALT is at least alt_cm), method, family, flag.
    log lists every site-year removed and why.
    """
    s = cfg["sample"]
    a = annual.copy()
    a["flag"] = a["flag"].fillna("")
    for col in ["site_id", "year", "alt_cm", "use", "lat", "lon", "method"]:
        assert col in a.columns, f"missing column {col}"

    used = a[a["use"] == True].copy()  # noqa: E712 (column holds booleans)
    assert not used.duplicated(["site_id", "year"]).any(), "two used values for one site-year"
    used["censored"] = False

    parts = [used]
    if s["use_censored"]:
        ok = set(s["censored_ok_flags"])
        lb = a[a["flag"].str.contains("lower_bound") & (a["use"] != True)].copy()  # noqa: E712
        lb = lb[lb["flag"].apply(lambda f: set(f.split(";")) <= ok)]
        have = set(zip(used["site_id"], used["year"]))
        lb = lb[[(i, y) not in have for i, y in zip(lb["site_id"], lb["year"])]]
        # Several sources can give a bound for one site-year: the largest bound holds.
        lb = lb.sort_values("alt_cm").groupby(["site_id", "year"], as_index=False).last()
        lb["censored"] = True
        parts.append(lb)

    d = pd.concat(parts, ignore_index=True)
    # One position per site (the used rows of a site can carry slightly different
    # source coordinates; take the first used row's).
    pos = used.groupby("site_id")[["lat", "lon"]].first()
    d = d[d["site_id"].isin(pos.index)].drop(columns=["lat", "lon"]).join(pos, on="site_id")

    fam = s["method_families"]
    unknown = set(d["method"]) - set(fam)
    assert not unknown, f"method without a family in the config: {unknown}"
    d["family"] = d["method"].map(fam)

    log = []
    keep = (d["lat"] >= s["min_lat"]) & d["year"].between(s["min_year"], s["max_year"])
    log.append(d.loc[~keep].assign(reason="outside latitude / year window"))
    d = d[keep]

    if s["single_method_family"]:
        main = (d[~d["censored"]].groupby(["site_id", "family"]).size()
                .rename("n").reset_index().sort_values(["site_id", "n"], ascending=[True, False])
                .drop_duplicates("site_id").set_index("site_id")["family"])
        same = d["family"].values == d["site_id"].map(main).values
        log.append(d.loc[~same].assign(reason="minority method family in the series"))
        d = d[same]

    # Record length, after the window and the method-family rule. Logged here so the
    # removed-site-years file accounts for every used row of the input.
    n_unc = d[~d["censored"]].groupby("site_id").size()
    span = d.groupby("site_id")["year"].agg(lambda v: v.max() - v.min() + 1)
    ok_sites = n_unc.index[(n_unc >= s["min_obs"]) & (span.reindex(n_unc.index) >= s["min_span_years"])]
    short = ~d["site_id"].isin(ok_sites)
    log.append(d.loc[short].assign(
        reason=f"site too short in the window (< {s['min_obs']} uncensored years or < {s['min_span_years']} yr span)"))
    d = d[~short]

    cols = ["site_id", "lat", "lon", "year", "alt_cm", "censored", "method", "family", "flag",
            "obs_date", "source"]
    d = d[[c for c in cols if c in d.columns]].sort_values(["site_id", "year"]).reset_index(drop=True)
    assert (d["alt_cm"] > 0).all()
    return d, pd.concat(log, ignore_index=True)


# ---------------------------------------------------------------------------
# Kendall's S with right-censored values, Akritas-Theil-Sen slope and interval
# ---------------------------------------------------------------------------

def _pair_signs(x: np.ndarray, r: np.ndarray, c: np.ndarray) -> np.ndarray:
    """sign(r_j - r_i) for all pairs i<j with x_j > x_i, where c marks right-censored r.

    A censored value means "at least r". The sign is known only if the censored value
    is the larger one: r_j censored and r_j >= r_i gives +1; r_i censored and
    r_j <= r_i gives -1. Other pairs with a censored member give 0 (unknown).
    """
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    assert (dx > 0).all(), "years must be unique and sorted"
    diff = r[j] - r[i]
    diff = np.where(np.abs(diff) < 1e-9, 0.0, diff)   # floating-point ties
    sgn = np.sign(diff)
    ci, cj = c[i], c[j]
    sgn = np.where(ci & cj, 0.0, sgn)
    sgn = np.where(cj & ~ci, np.where(diff >= 0, 1.0, 0.0), sgn)
    sgn = np.where(ci & ~cj, np.where(diff <= 0, -1.0, 0.0), sgn)
    return sgn


def kendall_s(x, y, c, beta):
    return _pair_signs(x, y - beta * x, c).sum()


def _candidate_slopes(x, y):
    i, j = np.triu_indices(len(x), k=1)
    # rounded so that equal slopes computed from different pairs count once
    return np.unique(np.round((y[j] - y[i]) / (x[j] - x[i]), 9))


def _s_between(x, y, c, cand):
    """S evaluated between consecutive candidate slopes (and beyond both ends)."""
    mids = np.concatenate([[cand[0] - 1.0], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1.0]])
    s_mid = np.array([kendall_s(x, y, c, b) for b in mids])
    assert np.all(np.diff(s_mid) <= 1e-9), "S(beta) must be non-increasing"
    return s_mid


def _solve_s(s_mid, cand, target, side):
    """Root of the non-increasing step function S(beta) at level `target`.

    S is constant between candidate slopes; s_mid[k] is its value just below cand[k]
    and s_mid[k+1] just above.
    side='lo': smallest candidate b with S(b+) <= target.
    side='hi': largest candidate b with S(b-) >= target.
    """
    if side == "lo":
        idx = np.nonzero(s_mid[1:] <= target)[0]
        return cand[idx[0]] if len(idx) else np.nan
    idx = np.nonzero(s_mid[:-1] >= target)[0]
    return cand[idx[-1]] if len(idx) else np.nan


def var_s_null(y, c, n):
    """Variance of S without autocorrelation, with the tie correction (uncensored ties)."""
    _, counts = np.unique(y[~c], return_counts=True)
    ties = np.sum(counts * (counts - 1) * (2 * counts + 5))
    return (n * (n - 1) * (2 * n + 5) - ties) / 18.0


def ar1_variance_factor(n, rho):
    """Hamed-Rao (1998) factor n/n* for an AR(1) series.

    The rank autocorrelation at lag k of a Gaussian AR(1) is (6/pi) asin(rho^k / 2).
    """
    if rho <= 0:
        return 1.0
    k = np.arange(1, n)
    rho_s = (6 / np.pi) * np.arcsin(rho ** k / 2)
    w = (n - k) * (n - k - 1) * (n - k - 2)
    return 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * np.sum(w * rho_s)


def lag1_autocorrelation(x, resid, ar1_max):
    """Lag-1 autocorrelation from consecutive-year pairs, bias-corrected.

    Bias correction (n rho + 1)/(n - 4) (Kendall 1954; used for trend tests by
    Wang and Swail 2001). Clipped to [0, ar1_max]; returns (corrected, raw, n_pairs).
    """
    nxt = np.nonzero(np.diff(x) == 1)[0]
    if len(nxt) < 5:
        return 0.0, np.nan, len(nxt)
    e = resid - resid.mean()
    n = len(x)
    # mean lag-1 product over the consecutive pairs / mean square over all years
    # (dividing a sum over the pairs by a sum over all years shrinks rho when years
    # are missing)
    raw = (np.sum(e[nxt] * e[nxt + 1]) / len(nxt)) / (np.sum(e ** 2) / n)
    corr = (n * raw + 1) / (n - 4)
    return float(np.clip(corr, 0.0, ar1_max)), float(raw), len(nxt)


AR1_GRID = np.round(np.arange(-0.5, 0.951, 0.01), 2)


def ar1_reml_loglik(x, y, grid=AR1_GRID):
    """REML log-likelihood of y = a + b x + e, e AR(1) in calendar years, over a rho grid.

    Missing years are handled exactly: corr(e_i, e_j) = rho^|x_i - x_j|. The intercept,
    slope and noise variance are profiled out (restricted likelihood), which removes
    most of the downward bias that a fitted trend gives the lag-1 estimate.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    X = np.column_stack([np.ones(n), x - x.mean()])
    lag = np.abs(x[:, None] - x[None, :])
    out = np.empty(len(grid))
    for k, r in enumerate(grid):
        R = np.sign(r) ** lag * np.abs(r) ** lag
        L = np.linalg.cholesky(R)
        Xs = np.linalg.solve(L, X)
        ys = np.linalg.solve(L, y)
        b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        e = ys - Xs @ b
        s2 = e @ e / (n - 2)
        out[k] = -0.5 * ((n - 2) * np.log(s2) + 2 * np.log(np.diag(L)).sum()
                         + np.linalg.slogdet(Xs.T @ Xs)[1])
    return out


def pooled_ar1(series, ar1_max, grid=AR1_GRID):
    """One AR(1) coefficient for many series: maximise the summed REML log-likelihood.

    series: iterable of (years, alt) arrays (uncensored years). Each series keeps its own
    level, slope and noise variance. Returns (rho clipped to [0, ar1_max], rho unclipped).
    """
    ll = sum(ar1_reml_loglik(x, y, grid) for x, y in series)
    r = float(grid[int(np.argmax(ll))])
    return float(np.clip(r, 0.0, ar1_max)), r


@dataclass
class TrendResult:
    slope: float          # cm/yr
    lo: float             # cm/yr
    hi: float             # cm/yr
    intercept_ref: float  # fitted ALT (cm) at the median year
    p_value: float        # Kendall test of zero slope, AR(1)-corrected, two-sided
    rho: float            # AR(1) coefficient used for the variance correction
    rho_raw: float        # this site's own lag-1 estimate before bias correction
    rho_site: float       # this site's own bias-corrected, clipped estimate
    vif: float
    n: int
    n_censored: int
    resid_sd: float       # robust SD of residuals (cm), uncensored values


def ats_trend(x, y, c, ci_level, ar1_max, rho_fixed=None) -> TrendResult:
    """Akritas-Theil-Sen slope and interval. rho_fixed: AR(1) coefficient to use
    (the pooled network value); None uses this site's own lag-1 estimate."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    c = np.asarray(c, bool)
    n = len(x)
    cand = _candidate_slopes(x, y)
    s_mid = _s_between(x, y, c, cand)
    lo0 = _solve_s(s_mid, cand, 0.0, "lo")
    hi0 = _solve_s(s_mid, cand, 0.0, "hi")
    slope = 0.5 * (lo0 + hi0)

    xm = np.median(x)
    unc = ~c
    intercept_ref = float(np.median(y[unc] - slope * (x[unc] - xm)))
    resid = y - (intercept_ref + slope * (x - xm))
    # autocorrelation from uncensored years only (censored residuals are bounds)
    rho_site, rho_raw, _ = lag1_autocorrelation(x[unc], resid[unc], ar1_max)
    rho = rho_site if rho_fixed is None else float(rho_fixed)
    vif = ar1_variance_factor(n, rho)
    var_s = var_s_null(y, c, n) * vif

    z = stats.norm.ppf(0.5 + ci_level / 2)
    crit = z * np.sqrt(var_s)
    lo = _solve_s(s_mid, cand, crit, "lo")
    hi = _solve_s(s_mid, cand, -crit, "hi")

    s0 = kendall_s(x, y, c, 0.0)
    zs = (s0 - np.sign(s0)) / np.sqrt(var_s)
    p = 2 * stats.norm.sf(abs(zs))
    mad = np.median(np.abs(resid[unc] - np.median(resid[unc])))
    return TrendResult(slope, lo, hi, intercept_ref, float(p), rho, rho_raw, rho_site, vif, n,
                       int(c.sum()), 1.4826 * mad)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

def classify(est, lo, hi, stable_margin, rapid):
    """Class from a rate and its interval (all in %/decade).

    stable:        the interval lies inside (-margin, +margin) (equivalence test)
    thickening:    lower end > 0; rapid when the estimate >= rapid
    thinning:      upper end < 0; rapid when the estimate <= -rapid
    indeterminate: none of these (the interval spans zero and a margin)
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "indeterminate"
    assert lo <= est <= hi, (lo, est, hi)
    if lo > -stable_margin and hi < stable_margin:
        return "stable"
    if lo > 0:
        return "rapid_thickening" if est >= rapid else "gradual_thickening"
    if hi < 0:
        return "rapid_thinning" if est <= -rapid else "gradual_thinning"
    return "indeterminate"


def band_probabilities(mean, sd, stable_margin, rapid):
    """Posterior probability of each rate band (normal posterior)."""
    cdf = lambda v: stats.norm.cdf((v - mean) / sd)  # noqa: E731
    return {
        "p_rapid_thickening": 1 - cdf(rapid),
        "p_gradual_thickening": cdf(rapid) - cdf(stable_margin),
        "p_stable": cdf(stable_margin) - cdf(-stable_margin),
        "p_gradual_thinning": cdf(-stable_margin) - cdf(-rapid),
        "p_rapid_thinning": cdf(-rapid),
        "p_thickening": 1 - cdf(0.0),
    }


# ---------------------------------------------------------------------------
# Nonlinearity (broken stick)
# ---------------------------------------------------------------------------

def _hinge_sup_f(x, Y, taus):
    """sup over taus of F(line vs continuous hinge). Y: (n,) or (n, B)."""
    n = len(x)
    Y = Y.reshape(n, -1)
    X0 = np.column_stack([np.ones(n), x - x.mean()])
    rss0 = np.sum((Y - X0 @ np.linalg.lstsq(X0, Y, rcond=None)[0]) ** 2, axis=0)
    best_f = np.full(Y.shape[1], -np.inf)
    best_tau = np.full(Y.shape[1], np.nan)
    for t in taus:
        X1 = np.column_stack([X0, np.maximum(x - t, 0.0)])
        rss1 = np.sum((Y - X1 @ np.linalg.lstsq(X1, Y, rcond=None)[0]) ** 2, axis=0)
        f = (rss0 - rss1) / (rss1 / (n - 3))
        better = f > best_f
        best_f[better] = f[better]
        best_tau[better] = t
    return best_f, best_tau


def breakpoint_test(x, y, min_seg, n_boot, rho, rng):
    """Broken-stick test on uncensored values. Returns dict with p-value and slopes."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xs = np.sort(x)
    taus = [t for t in xs[min_seg - 1: n - min_seg + 1]
            if (x <= t).sum() >= min_seg and (x >= t).sum() >= min_seg]
    if len(taus) == 0 or n < 2 * min_seg:
        return {"nl_p": np.nan, "nl_break_year": np.nan, "nl_slope_before": np.nan,
                "nl_slope_after": np.nan}
    f_obs, tau = _hinge_sup_f(x, y, taus)
    tau = float(tau[0])
    X1 = np.column_stack([np.ones(n), x - x.mean(), np.maximum(x - tau, 0.0)])
    b = np.linalg.lstsq(X1, y, rcond=None)[0]

    # Null: straight line + AR(1) noise on the actual (possibly gapped) years.
    X0 = np.column_stack([np.ones(n), x - x.mean()])
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    sd = np.std(y - X0 @ b0, ddof=2)
    years = np.arange(xs[0], xs[-1] + 1)
    e = np.empty((len(years), n_boot))
    e[0] = rng.standard_normal(n_boot)
    innov = rng.standard_normal((len(years), n_boot)) * np.sqrt(1 - rho ** 2)
    for k in range(1, len(years)):
        e[k] = rho * e[k - 1] + innov[k]
    e = e[np.searchsorted(years, x)] * sd
    f_null, _ = _hinge_sup_f(x, (X0 @ b0)[:, None] + e, taus)
    p = (1 + np.sum(f_null >= f_obs[0])) / (n_boot + 1)
    return {"nl_p": float(p), "nl_break_year": tau, "nl_slope_before": float(b[1]),
            "nl_slope_after": float(b[1] + b[2])}


# ---------------------------------------------------------------------------
# Regions and partial pooling
# ---------------------------------------------------------------------------

def haversine_matrix(lat, lon):
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    dlat, dlon = la - la.T, lo - lo.T
    h = np.sin(dlat / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def assign_regions(lat, lon, diameter_km):
    """Complete-linkage clusters: no two sites of one region are > diameter_km apart."""
    if len(lat) == 1:
        return np.array([1])
    d = haversine_matrix(np.asarray(lat), np.asarray(lon))
    np.fill_diagonal(d, 0.0)
    z = linkage(squareform(d, checks=False), method="complete")
    return fcluster(z, t=diameter_km, criterion="distance")


def _neg_loglik(log_tau, y, se2, groups):
    tr2, ts2 = np.exp(2 * log_tau)
    # GLS mean for given variances, then the profile likelihood
    blocks = []
    for g in groups:
        V = np.diag(ts2 + se2[g]) + tr2
        blocks.append((g, np.linalg.inv(V), np.linalg.slogdet(V)[1]))
    w = sum(Vi.sum() for _, Vi, _ in blocks)
    mu = sum(Vi.sum(axis=0) @ y[g] for g, Vi, _ in blocks) / w
    nll = 0.0
    for g, Vi, logdet in blocks:
        r = y[g] - mu
        nll += 0.5 * (logdet + r @ Vi @ r)
    # REML term
    nll += 0.5 * np.log(w)
    return nll, mu, 1.0 / w


def pool_rates(y, se, region):
    """Two-level random-effects model fitted by REML; returns network and site posteriors.

    rate_i = mu + u_r + v_i + e_i, u_r ~ N(0, tau_r^2), v_i ~ N(0, tau_s^2), e_i ~ N(0, se_i^2).
    """
    y = np.asarray(y, float)
    se2 = np.asarray(se, float) ** 2
    assert np.all(se2 > 0) and np.all(np.isfinite(y))
    groups = [np.nonzero(region == r)[0] for r in np.unique(region)]
    start = np.log([np.std(y) / 2 + 1e-3] * 2)
    res = optimize.minimize(lambda t: _neg_loglik(t, y, se2, groups)[0], start,
                            method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-8,
                                                           "maxiter": 4000})
    assert res.success, res.message
    _, mu, var_mu = _neg_loglik(res.x, y, se2, groups)
    tr2, ts2 = np.exp(2 * res.x)

    post_mean = np.empty_like(y)
    post_sd = np.empty_like(y)
    region_mean = np.empty_like(y)
    for g in groups:
        k = len(g)
        T = ts2 * np.eye(k) + tr2
        Dinv = np.diag(1 / se2[g])
        P = np.linalg.inv(np.linalg.inv(T) + Dinv)     # posterior cov given mu
        A = P @ Dinv                                    # weight on the site's own value
        one = np.ones(k)
        post_mean[g] = A @ y[g] + (np.eye(k) - A) @ one * mu
        m = (np.eye(k) - A) @ one
        post_sd[g] = np.sqrt(np.diag(P) + m ** 2 * var_mu)
        # regional mean mu + u_r: BLUP of u_r
        V = np.diag(ts2 + se2[g]) + tr2
        region_mean[g] = mu + tr2 * one @ np.linalg.solve(V, y[g] - mu)
    return {"mu": mu, "se_mu": np.sqrt(var_mu), "tau_region": np.sqrt(tr2),
            "tau_site": np.sqrt(ts2), "post_mean": post_mean, "post_sd": post_sd,
            "region_mean": region_mean, "n_regions": len(groups)}


# ---------------------------------------------------------------------------
# One site
# ---------------------------------------------------------------------------

def site_seed(seed, site_id):
    return seed + zlib.crc32(str(site_id).encode())


def analyse_site(g: pd.DataFrame, cfg: dict, rho_fixed=None) -> dict:
    g = g.sort_values("year")
    x = g["year"].to_numpy(float)
    y = g["alt_cm"].to_numpy(float)
    c = g["censored"].to_numpy(bool)
    tc, cc = cfg["trend"], cfg["classes"]
    tr = ats_trend(x, y, c, tc["ci_level"], tc["ar1_max"], rho_fixed)
    assert tr.intercept_ref > 0, "fitted ALT must be positive"
    to_rate = 1000.0 / tr.intercept_ref   # cm/yr -> % of ALT per decade
    rate, rlo, rhi = tr.slope * to_rate, tr.lo * to_rate, tr.hi * to_rate
    z = stats.norm.ppf(0.5 + tc["ci_level"] / 2)
    se_rate = (rhi - rlo) / (2 * z)
    mdt = (stats.norm.ppf(0.5 + tc["ci_level"] / 2) + stats.norm.ppf(cfg["diagnostics"]["power"])) * se_rate

    nl = cfg["nonlinearity"]
    rng = np.random.default_rng(site_seed(cfg["random_seed"], g["site_id"].iloc[0]))
    bp = breakpoint_test(x[~c], y[~c], nl["min_segment_obs"], nl["n_boot"], tr.rho, rng)
    for k in ["nl_slope_before", "nl_slope_after"]:
        bp[k.replace("slope", "rate")] = bp[k] * to_rate
    nonlinear = bool(np.isfinite(bp["nl_p"]) and bp["nl_p"] < nl["alpha"])
    # A reversal needs both segments beyond the stable margin with opposite signs;
    # otherwise the break is a change of rate (onset, stop, faster or slower).
    m = cc["stable_margin_pct_dec"]
    reversal = bool(nonlinear and abs(bp["nl_rate_before"]) > m and abs(bp["nl_rate_after"]) > m
                    and np.sign(bp["nl_rate_before"]) != np.sign(bp["nl_rate_after"]))

    # survey date drift (only where dates are given)
    drift = np.nan
    if "obs_date" in g.columns:
        doy = pd.to_datetime(g["obs_date"], errors="coerce").dt.dayofyear
        ok = doy.notna().to_numpy()
        if ok.sum() >= 5:
            xd, dd = x[ok], doy.to_numpy(float)[ok]
            i, j = np.triu_indices(len(xd), k=1)
            drift = float(np.median((dd[j] - dd[i]) / (xd[j] - xd[i])) * (xd.max() - xd.min()))

    flags = set(";".join(g["flag"]).split(";")) - {""}
    return {
        "site_id": g["site_id"].iloc[0], "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0],
        "family": g["family"].iloc[0], "methods": ";".join(sorted(set(g["method"]))),
        "year_min": int(x.min()), "year_max": int(x.max()), "span_years": int(x.max() - x.min() + 1),
        "n_obs": int((~c).sum()), "n_censored": tr.n_censored,
        "alt_ref_cm": tr.intercept_ref, "slope_cm_yr": tr.slope,
        "slope_lo_cm_yr": tr.lo, "slope_hi_cm_yr": tr.hi,
        "rate_pct_dec": rate, "rate_lo": rlo, "rate_hi": rhi, "rate_se": se_rate,
        "p_trend": tr.p_value, "ar1_rho": tr.rho, "ar1_rho_site": tr.rho_site, "ar1_rho_raw": tr.rho_raw, "var_factor": tr.vif,
        "resid_sd_cm": tr.resid_sd, "mdt_pct_dec": mdt,
        "class_site": classify(rate, rlo, rhi, cc["stable_margin_pct_dec"], cc["rapid_pct_dec"]),
        **bp, "flag_nonlinear": nonlinear, "flag_reversal": bool(reversal),
        "survey_doy_drift_days": drift,
        "flag_doy_drift": bool(np.isfinite(drift) and abs(drift) > cfg["diagnostics"]["survey_doy_drift_days"]),
        "flag_burned": "burned_site" in flags,
        "data_flags": ";".join(sorted(flags)),
    }


def bh_qvalues(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def run(data: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    s, tc = cfg["sample"], cfg["trend"]
    sites = []
    for sid, g in data.groupby("site_id"):
        n_obs = int((~g["censored"]).sum())
        span = int(g["year"].max() - g["year"].min() + 1)
        if n_obs >= s["min_obs"] and span >= s["min_span_years"]:
            sites.append(g.sort_values("year"))
    method = tc.get("ar1_method", "site")
    assert method in ("pooled", "site"), method
    rho_pool, rho_pool_raw = pooled_ar1(
        [(g.loc[~g["censored"], "year"].to_numpy(float), g.loc[~g["censored"], "alt_cm"].to_numpy(float))
         for g in sites], tc["ar1_max"])
    rho_fixed = rho_pool if method == "pooled" else None
    rows = [analyse_site(g, cfg, rho_fixed) for g in sites]
    res = pd.DataFrame(rows)
    assert len(res) > 0
    res["q_trend"] = bh_qvalues(res["p_trend"])

    res["region_id"] = assign_regions(res["lat"], res["lon"], cfg["pooling"]["region_km"])
    pool = pool_rates(res["rate_pct_dec"], res["rate_se"], res["region_id"].to_numpy())
    z = stats.norm.ppf(0.5 + cfg["trend"]["ci_level"] / 2)
    cc = cfg["classes"]
    res["rate_pooled"] = pool["post_mean"]
    res["rate_pooled_sd"] = pool["post_sd"]
    res["region_rate"] = pool["region_mean"]
    res["class_pooled"] = [
        classify(m, m - z * sd, m + z * sd, cc["stable_margin_pct_dec"], cc["rapid_pct_dec"])
        for m, sd in zip(pool["post_mean"], pool["post_sd"])]
    probs = pd.DataFrame([band_probabilities(m, sd, cc["stable_margin_pct_dec"], cc["rapid_pct_dec"])
                          for m, sd in zip(pool["post_mean"], pool["post_sd"])])
    res = pd.concat([res, probs], axis=1)
    network = {k: pool[k] for k in ["mu", "se_mu", "tau_region", "tau_site", "n_regions"]}
    network["n_sites"] = len(res)
    network["ar1_method"] = method
    network["ar1_rho_pooled"] = rho_pool
    network["ar1_rho_pooled_unclipped"] = rho_pool_raw
    return res, network


# =========================================================================================
# PART 2. ALDI v6 models (A: hierarchical trends; B: climate-adjusted)
# =========================================================================================
# ALDI v6: one hierarchical Bayesian model for all ALT series (A), and a climate-
# adjusted version that splits each site's trend into a part expected from thawing
# degree-days and a residual part (B).
#
# Why (review of 2026-11-03, doc "ALDI paper: publishability and stronger analyses"):
# v5.2 classifies each site from its own 10-35 values, so most short records are
# "indeterminate" and class shares depend on record length. v5.2 already pools in a
# second step (site rate and SE -> REML random effects), but the site SE there treats
# rho as known and the censored years only through ranks.
#
# A. Hierarchical trend model (one step, all uncertainty propagated)
#    log ALT_it = a_i + b_i * t_it + e_it          t in decades from the site's mean year
#    b_i  = beta_region(i) + tau_site * z_i         sites shrink toward their region
#    beta_r = mu + tau_region * z_r                 regions shrink toward the network
#    e_it  AR(1) in calendar years, rho per method family, exact for gaps (rho^gap)
#    sd of e_it per site, log-normal around a family mean (probe, tube and temperature
#    ALT have different noise)
#    Lower-bound years (thaw reached the probe or deepest sensor) are latent values
#    above the bound, so they inform the trend with the right direction and no made-up
#    value.
#    Rates in % per decade = 100 (exp(b) - 1). Classes use the v5.2 rules applied to
#    the 90 % credible interval, and every band probability is reported.
#
# B. Climate-adjusted model (Stefan: ALT = E sqrt(TDD), so log ALT = log E + k log TDD)
#    log ALT_it = a_i + k_i * x_it + d_i * t_it + e_it
#    x_it = log TDD_it minus the site mean of log TDD
#    k_i  = k_family + tau_k * z_i,  k_family ~ N(0.5, 0.25) for probe, tube, temperature
#    d_i  hierarchical by region as b_i in A.
#    Because d_i takes the linear trend, k_i is identified by the year-to-year swings of
#    TDD around their trend (the within-site sensitivity), not by two series that both
#    rise. For each site:
#        climate-expected trend = k_i * (OLS slope of x_it on t over the site's years)
#        residual trend         = d_i
#        total                  = expected + residual
#    A residual > 0 means the ground thaws deeper over time than the TDD record explains
#    (ground ice loss and subsidence, talik, vegetation or snow change, or TDD error);
#    < 0 means it lags the TDD record.
#    Option k_mode: fixed sets k_i = 0.5 for every site (pure Stefan), as a sensitivity.
#
# Caveat built into the design: error in reanalysis TDD attenuates k, which moves part
# of the climate trend into the residual. validate_aldi_v6.py measures this with
# simulated TDD error.

FAMILIES = ["probe", "thaw_tube", "temperature"]


# ---------------------------------------------------------------------------
# Data arrays
# ---------------------------------------------------------------------------

def eligible_sites(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Same sites as v5.2 run(): >= min_obs uncensored years over >= min_span_years."""
    s = cfg["sample"]
    keep = []
    for sid, g in data.groupby("site_id"):
        n_obs = int((~g["censored"]).sum())
        span = int(g["year"].max() - g["year"].min() + 1)
        if n_obs >= s["min_obs"] and span >= s["min_span_years"]:
            keep.append(sid)
    return data[data["site_id"].isin(keep)].sort_values(["site_id", "year"]).reset_index(drop=True)


# How each extra term enters model B. "log": log of the value, centred per site (a
# degree-day sum acts multiplicatively, like TDD). "lin": the value minus the site mean,
# divided by one network-wide SD of these anomalies (snow depth or rain can be near 0 in
# a dry year, so a log would blow up); its coefficient is the change in log ALT per SD.
EXTRA_TRANSFORM = {"tdd_prev": "log", "fdd": "log"}

# TDD of earlier years (thaw memory): tdd_lag1 = last calendar year's thawing index (the
# same as tdd_prev), tdd_lag2 = two years before, and so on. Their coefficients add to
# this year's k: k + sum of the lag coefficients is the long-run sensitivity, the change
# in log ALT after TDD has stayed higher for several years.
LAG_TERM = re.compile(r"^tdd_lag(\d+)$")


def lag_years(name):
    """Years back for a TDD lag term (tdd_prev counts as tdd_lag1); None for other terms."""
    if name == "tdd_prev":
        return 1
    m = LAG_TERM.match(name)
    return int(m.group(1)) if m else None


def extra_transform(name):
    """'log' for degree-day sums (TDD lags, FDD), 'lin' for the weather terms."""
    return "log" if (lag_years(name) or EXTRA_TRANSFORM.get(name) == "log") else "lin"

# Weather terms read from weather_site_years.csv (era5land_for_b.py). Soil moisture,
# NDVI and NDWI are left out on purpose: they respond to the thaw itself (mediators), so
# adjusting for them would remove part of the change that B is meant to measure.
WEATHER_TERMS = ("snow_depth_winter", "snow_thermal_resistance_winter", "rain_summer",
                 "P_summer", "snowfall_winter", "SWE_max")


def extra_climate(tdd_all: pd.DataFrame | None, names, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Extra climate terms per site-year.

    From tdd_all_years.csv:
      tdd_prev: previous calendar year's thawing index (thaw memory: a warm summer melts
                ground ice and wets the soil, which deepens the next thaw too)
      tdd_lag1, tdd_lag2, ...: thawing index of 1, 2, ... calendar years before
                (tdd_lag1 = tdd_prev); a lag of L years needs temperature from L years
                before the first ALT year
      fdd:      freezing index of the preceding winter (1 Oct - 30 Apr)
    From weather_site_years.csv (ERA5-Land, ALT year = Oct of the previous year to Sep):
      snow_depth_winter (Nov-Apr mean), snow_thermal_resistance_winter, rain_summer (JJA),
      and the other WEATHER_TERMS
    """
    parts = []
    if tdd_all is not None:
        t = tdd_all.copy()
        out = t[["site_id", "year"]].copy()
        for c in names:
            lag = lag_years(c)
            if lag:
                prev = t[["site_id", "year", "tdd_full_year"]].copy()
                prev["year"] += lag
                out = out.merge(prev.rename(columns={"tdd_full_year": c}), on=["site_id", "year"], how="left")
        if "fdd" in names:
            out = out.merge(t[["site_id", "year", "fdd"]], on=["site_id", "year"], how="left")
        parts.append(out)
    w_names = [c for c in names if c in WEATHER_TERMS]
    if w_names:
        assert weather is not None, f"{w_names} need weather_site_years.csv"
        parts.append(weather[["site_id", "year"] + w_names].drop_duplicates(["site_id", "year"]))
    missing = [c for c in names if not any(c in p.columns for p in parts)]
    assert not missing, f"unknown extra terms {missing}"
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["site_id", "year"], how="outer")
    return out


def build_arrays(d: pd.DataFrame, cfg: dict, tdd: pd.DataFrame | None = None,
                 extra: pd.DataFrame | None = None, extra_names=()) -> dict:
    """Model inputs. d: output of eligible_sites (one row per site-year).

    tdd: table site_id, year, method, tdd (deg C day). Site-years without TDD are
    dropped for model B only (reported in the summary).
    extra: output of extra_climate; extra_names: its columns to use (centred per
    site, transformed as in extra_transform).
    """
    d = d.copy()
    extra_names = list(extra_names)
    n_extra_dropped = 0
    if extra is not None and extra_names:
        d = d.merge(extra[["site_id", "year"] + extra_names], on=["site_id", "year"], how="left")
        n0 = len(d)
        for c in extra_names:
            ok = np.isfinite(d[c])
            if extra_transform(c) == "log":
                ok &= d[c] > 0
            d = d[ok]
        n_extra_dropped = n0 - len(d)
    if tdd is not None:
        t = tdd[["site_id", "year", "method", "tdd"]].drop_duplicates(["site_id", "year", "method"])
        d = d.merge(t, on=["site_id", "year", "method"], how="left")
        n0 = len(d)
        d = d[np.isfinite(d["tdd"]) & (d["tdd"] > 0)].copy()
        dropped = n0 - len(d)
        # a site must still pass the record-length rule with TDD available
        d = eligible_sites(d, cfg)
    else:
        dropped = 0
    sites = d.groupby("site_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                     family=("family", "first")).reset_index()
    sites["region_id"] = assign_regions(sites["lat"], sites["lon"], cfg["pooling"]["region_km"])
    sid_index = {s: i for i, s in enumerate(sites["site_id"])}
    reg_codes, reg_idx = np.unique(sites["region_id"], return_inverse=True)
    fam_idx_site = sites["family"].map({f: i for i, f in enumerate(FAMILIES)}).to_numpy()
    assert not np.isnan(fam_idx_site.astype(float)).any(), "unknown method family"

    s_idx = d["site_id"].map(sid_index).to_numpy()
    year = d["year"].to_numpy(float)
    mean_year = d.groupby("site_id")["year"].transform("mean").to_numpy(float)
    t = (year - mean_year) / 10.0                       # decades
    y = np.log(d["alt_cm"].to_numpy(float))
    cens = d["censored"].to_numpy(bool)

    # previous observation of the same site and the gap in years (AR(1) with gaps)
    prev = np.full(len(d), -1)
    gap = np.zeros(len(d))
    for i in range(1, len(d)):
        if s_idx[i] == s_idx[i - 1]:
            prev[i] = i - 1
            gap[i] = year[i] - year[i - 1]
    assert (gap[prev >= 0] >= 1).all()

    # site mean of y over uncensored years (centre of the intercept prior)
    ybar = pd.Series(np.where(cens, np.nan, y)).groupby(s_idx).mean().to_numpy()

    # region-year cells observed by >= 2 sites: shared year-to-year anomaly (weather
    # shocks common to nearby sites). Cells with one site get index -1 (no effect).
    r_obs = reg_idx[s_idx]
    ry = pd.Series(list(zip(r_obs, d["year"].to_numpy(int))))
    counts = ry.map(ry.value_counts())
    ry_codes = {c: i for i, c in enumerate(sorted(set(ry[counts >= 2])))}
    ry_idx = np.array([ry_codes.get(c, -1) for c in ry])

    out = dict(d=d, sites=sites, n_sites=len(sites), n_regions=len(reg_codes),
               ry_idx=ry_idx, n_ry=len(ry_codes),
               s_idx=s_idx, r_idx_site=reg_idx, f_idx_site=fam_idx_site,
               t=t, y=y, cens=cens, prev=prev, gap=gap, ybar=ybar, tdd_dropped=dropped,
               extra_dropped=n_extra_dropped)
    if tdd is not None:
        lt = np.log(d["tdd"].to_numpy(float))
        x = lt - pd.Series(lt).groupby(s_idx).transform("mean").to_numpy()
        out["x"] = x
        # OLS slope of x on t per site (per decade): the climate forcing trend
        df = pd.DataFrame({"s": s_idx, "x": x, "t": t})
        sx = df.groupby("s").apply(lambda g: np.polyfit(g.t, g.x, 1)[0] if g.t.std() > 0 else 0.0,
                                   include_groups=False)
        out["sx"] = sx.reindex(range(len(sites))).to_numpy()
        out["tdd_trend_pct_dec"] = 100 * (np.exp(out["sx"]) - 1)
        out["x_sd"] = df.groupby("s")["x"].std().reindex(range(len(sites))).to_numpy()
        xe, sxe = [], []
        for c in extra_names:
            v = d[c].to_numpy(float)
            log_c = extra_transform(c) == "log"
            lc = np.log(v) if log_c else v
            xc = lc - pd.Series(lc).groupby(s_idx).transform("mean").to_numpy()
            if not log_c:
                sd_c = float(np.std(xc))
                xc = xc / sd_c if sd_c > 0 else xc
                out[f"sd_{c}"] = sd_c
            dfc = pd.DataFrame({"s": s_idx, "x": xc, "t": t})
            sxc = dfc.groupby("s").apply(lambda g: np.polyfit(g.t, g.x, 1)[0] if g.t.std() > 0 else 0.0,
                                         include_groups=False)
            xe.append(xc)
            sxe.append(sxc.reindex(range(len(sites))).to_numpy())
        out["extra_names"] = extra_names
        out["X_extra"] = np.column_stack(xe) if xe else np.zeros((len(d), 0))
        out["SX_extra"] = np.column_stack(sxe) if sxe else np.zeros((len(sites), 0))
    return out


# ---------------------------------------------------------------------------
# NumPyro model
# ---------------------------------------------------------------------------

def model(arr, mode="A", k_mode="hierarchical", slope_dist="normal", priors=None,
          region_year=True):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    p = priors or {}
    ns, nr = arr["n_sites"], arr["n_regions"]
    nf = len(FAMILIES)
    s_idx = jnp.asarray(arr["s_idx"])
    r_site = jnp.asarray(arr["r_idx_site"])
    f_site = jnp.asarray(arr["f_idx_site"])
    t = jnp.asarray(arr["t"])
    y_obs = jnp.asarray(arr["y"])
    cens = np.asarray(arr["cens"])
    prev = np.asarray(arr["prev"])
    gap = jnp.asarray(arr["gap"])

    # trend hierarchy (log units per decade; 0.1 ~ 10.5 %/dec)
    mu = numpyro.sample("mu", dist.Normal(0.0, p.get("mu_sd", 0.2)))
    tau_r = numpyro.sample("tau_region", dist.HalfNormal(p.get("tau_sd", 0.1)))
    tau_s = numpyro.sample("tau_site", dist.HalfNormal(p.get("tau_sd", 0.1)))
    z_r = numpyro.sample("z_region", dist.Normal(0, 1).expand([nr]))
    if slope_dist == "student":
        # t(4) as a normal scale mixture (samples better than a direct Student-t)
        lam = numpyro.sample("lambda_site", dist.Gamma(2.0, 2.0).expand([ns]))
        z_s = numpyro.sample("z_site", dist.Normal(0, 1).expand([ns])) / jnp.sqrt(lam)
    else:
        z_s = numpyro.sample("z_site", dist.Normal(0, 1).expand([ns]))
    beta_r = numpyro.deterministic("beta_region", mu + tau_r * z_r)
    slope = numpyro.deterministic("slope_site", beta_r[r_site] + tau_s * z_s)

    za = numpyro.sample("z_a", dist.Normal(0, 1).expand([ns]))
    a = numpyro.deterministic("a_site", jnp.asarray(arr["ybar"]) + p.get("a_sd", 0.5) * za)

    # noise: per family AR(1) and log-normal site SDs around a family mean
    rho_f = numpyro.sample("rho_family", dist.Beta(2.0, 2.0).expand([nf])) * 0.95
    m_f = numpyro.sample("log_sigma_family", dist.Normal(np.log(0.08), 0.7).expand([nf]))
    sd_ls = numpyro.sample("sd_log_sigma", dist.HalfNormal(0.5))
    zl = numpyro.sample("z_log_sigma", dist.Normal(0, 1).expand([ns]))
    sigma = numpyro.deterministic("sigma_site", jnp.exp(m_f[f_site] + sd_ls * zl))

    mean = a[s_idx] + slope[s_idx] * t
    if mode == "B":
        x = jnp.asarray(arr["x"])
        if k_mode == "fixed":
            k = jnp.full(ns, p.get("k_fixed", 0.5))
        else:
            # one mean per method family (3 values: too few for a variance), prior
            # centred on Stefan's 0.5 and wide enough to reach 0 or 1
            k_f = numpyro.sample("k_family", dist.Normal(0.5, 0.25).expand([nf]))
            k0 = numpyro.deterministic("k0", k_f.mean())
            tau_k = numpyro.sample("tau_k", dist.HalfNormal(0.2))
            zk = numpyro.sample("z_k", dist.Normal(0, 1).expand([ns]))
            k = k_f[f_site] + tau_k * zk
        k = numpyro.deterministic("k_site", k)
        mean = mean + k[s_idx] * x
        ne = arr["X_extra"].shape[1] if "X_extra" in arr else 0
        if ne:
            # extra terms: one coefficient per method family, prior N(0, 0.25)
            ke = numpyro.sample("k_extra_family", dist.Normal(0.0, 0.25).expand([nf, ne]))
            ke_site = numpyro.deterministic("k_extra_site", ke[f_site])          # (ns, ne)
            mean = mean + jnp.sum(ke_site[s_idx] * jnp.asarray(arr["X_extra"]), axis=1)

    if region_year and arr["n_ry"] > 0:
        sd_ry = numpyro.sample("sd_region_year", dist.HalfNormal(0.1))
        z_ry = numpyro.sample("z_region_year", dist.Normal(0, 1).expand([arr["n_ry"]]))
        g = jnp.concatenate([sd_ry * z_ry, jnp.zeros(1)])   # index -1 -> 0
        mean = mean + g[jnp.asarray(arr["ry_idx"])]

    # censored years: latent value above the bound (log units)
    y = y_obs
    ci = np.nonzero(cens)[0]
    if len(ci):
        if p.get("cens_sd") is None:     # flat above the bound (censored likelihood)
            off = numpyro.sample("cens_offset", dist.ImproperUniform(dist.constraints.positive, (len(ci),), ()))
        else:
            off = numpyro.sample("cens_offset", dist.HalfNormal(p["cens_sd"]).expand([len(ci)]))
        y = y.at[ci].set(y_obs[ci] + off)

    e = y - mean
    rho_obs = rho_f[f_site][s_idx]
    sig_obs = sigma[s_idx]
    has_prev = prev >= 0
    first = np.nonzero(~has_prev)[0]
    nxt = np.nonzero(has_prev)[0]
    pr = prev[nxt]
    rg = rho_obs[nxt] ** gap[nxt]
    numpyro.sample("e_first", dist.Normal(0.0, sig_obs[first]), obs=e[first])
    numpyro.sample("e_next", dist.Normal(rg * e[pr], sig_obs[nxt] * jnp.sqrt(1 - rg ** 2)),
                   obs=e[nxt])


def fit(arr, mode="A", k_mode="hierarchical", slope_dist="normal", chains=4, warmup=1000,
        draws=1000, seed=0, priors=None, progress=False, target_accept=0.9, region_year=True):
    import jax
    from numpyro.infer import MCMC, NUTS

    kernel = NUTS(model, target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=chains,
                chain_method="parallel" if jax.local_device_count() >= chains else "sequential",
                progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed), arr, mode=mode, k_mode=k_mode, slope_dist=slope_dist,
             priors=priors, region_year=region_year, extra_fields=("diverging",))
    return mcmc


def diagnostics(mcmc, names=("mu", "tau_region", "tau_site", "rho_family", "log_sigma_family",
                              "slope_site", "k0", "k_site", "sigma_site")):
    from numpyro.diagnostics import summary
    s = summary(mcmc.get_samples(group_by_chain=True), prob=0.9)
    rhat, ess = [], []
    for k in names:
        if k in s:
            r = np.atleast_1d(s[k]["r_hat"])
            if np.isfinite(r).any():          # constant quantities (k fixed) have no r_hat
                rhat.append(np.nanmax(r))
                ess.append(np.nanmin(np.atleast_1d(s[k]["n_eff"])))
    div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return {"max_rhat": float(np.nanmax(rhat)), "min_ess": float(np.nanmin(ess)), "divergences": div}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def pct(b):
    return 100.0 * (np.exp(b) - 1.0)


def interval(draws, level):
    q = (1 - level) / 2
    return np.quantile(draws, q, axis=0), np.quantile(draws, 1 - q, axis=0)


def band_probs(rate_draws, margin, rapid, prefix="p_"):
    r = rate_draws
    return {
        f"{prefix}rapid_thickening": (r >= rapid).mean(0),
        f"{prefix}gradual_thickening": ((r >= margin) & (r < rapid)).mean(0),
        f"{prefix}stable": ((r > -margin) & (r < margin)).mean(0),
        f"{prefix}gradual_thinning": ((r <= -margin) & (r > -rapid)).mean(0),
        f"{prefix}rapid_thinning": (r <= -rapid).mean(0),
        f"{prefix}thickening": (r > 0).mean(0),
    }


def classes_from_draws(rate_draws, level, margin, rapid):
    lo, hi = interval(rate_draws, level)
    med = np.median(rate_draws, 0)
    return [classify(m, min(l, m), max(h, m), margin, rapid) for m, l, h in zip(med, lo, hi)], med, lo, hi


RESID_CLASS = {"stable": "consistent_with_TDD", "rapid_thickening": "deeper_than_TDD",
               "gradual_thickening": "deeper_than_TDD", "rapid_thinning": "shallower_than_TDD",
               "gradual_thinning": "shallower_than_TDD", "indeterminate": "not_resolved"}


def summarise_A(mcmc, arr, cfg):
    smp = mcmc.get_samples()
    lvl = cfg["trend"]["ci_level"]
    m, rp = cfg["classes"]["stable_margin_pct_dec"], cfg["classes"]["rapid_pct_dec"]
    rate = pct(np.asarray(smp["slope_site"]))
    cls, med, lo, hi = classes_from_draws(rate, lvl, m, rp)
    s = arr["sites"].copy()
    s["rate_v6"] = med
    s["rate_v6_lo"], s["rate_v6_hi"] = lo, hi
    s["rate_v6_sd"] = rate.std(0)
    s["class_v6"] = cls
    for k, v in band_probs(rate, m, rp, prefix="p6_").items():
        s[k] = v
    s["sigma_pct"] = 100 * np.asarray(smp["sigma_site"]).mean(0)
    reg = pct(np.asarray(smp["beta_region"]))
    s["region_rate_v6"] = np.median(reg, 0)[arr["r_idx_site"]]
    net = {
        "mu": pct(np.asarray(smp["mu"])),
        "tau_region": 100 * np.asarray(smp["tau_region"]),
        "tau_site": 100 * np.asarray(smp["tau_site"]),
        "rho_family": np.asarray(smp["rho_family"]) * 0.95,
        "sigma_family": 100 * np.exp(np.asarray(smp["log_sigma_family"])),
    }
    return s, net, rate


def summarise_B(mcmc, arr, cfg, k_mode):
    smp = mcmc.get_samples()
    lvl = cfg["trend"]["ci_level"]
    m, rp = cfg["classes"]["stable_margin_pct_dec"], cfg["classes"]["rapid_pct_dec"]
    d = np.asarray(smp["slope_site"])                 # residual trend (log/decade)
    k = np.asarray(smp["k_site"])
    exp_b = k * arr["sx"][None, :]
    if arr.get("extra_names"):
        ke = np.asarray(smp["k_extra_site"])                                   # (draws, ns, ne)
        exp_b = exp_b + np.sum(ke * arr["SX_extra"][None, :, :], axis=2)
    resid = pct(d)
    expected = pct(exp_b)
    total = pct(d + exp_b)
    s = arr["sites"].copy()
    s["tdd_trend_pct_dec"] = arr["tdd_trend_pct_dec"]
    s["x_sd"] = arr["x_sd"]
    s["k_site"] = np.median(k, 0)
    s["k_lo"], s["k_hi"] = interval(k, lvl)
    for name, dr in [("expected", expected), ("resid", resid), ("total", total)]:
        lo, hi = interval(dr, lvl)
        s[f"{name}_rate"] = np.median(dr, 0)
        s[f"{name}_lo"], s[f"{name}_hi"] = lo, hi
        s[f"{name}_sd"] = dr.std(0)
    rc, *_ = classes_from_draws(resid, lvl, m, rp)
    s["class_resid_raw"] = rc
    s["class_resid"] = [RESID_CLASS[c] for c in rc]
    s["p_resid_pos"] = (resid > 0).mean(0)
    s["p_resid_within_margin"] = ((resid > -m) & (resid < m)).mean(0)
    s["share_expected"] = np.median(np.where(np.abs(total) > 1e-9, expected / total, np.nan), 0)
    s["sigma_pct_B"] = 100 * np.asarray(smp["sigma_site"]).mean(0)
    s["k_mode"] = k_mode
    net = {"mu_resid": pct(np.asarray(smp["mu"])),
           "expected_mean": expected.mean(1), "total_mean": total.mean(1),
           "resid_mean": resid.mean(1),
           "rho_family": np.asarray(smp["rho_family"]) * 0.95,
           "sigma_family": 100 * np.exp(np.asarray(smp["log_sigma_family"]))}
    lag_cols = []
    for j, c in enumerate(arr.get("extra_names", [])):
        if extra_transform(c) == "log":
            s[f"trend_{c}_pct_dec"] = pct(arr["SX_extra"][:, j])
        else:                       # network SDs of the yearly anomaly per decade
            s[f"trend_{c}_sd_dec"] = arr["SX_extra"][:, j]
        net[f"k_{c}_family"] = np.asarray(smp["k_extra_family"])[:, :, j]
        if lag_years(c):
            lag_cols.append(j)
    if k_mode != "fixed":
        net["k0"] = np.asarray(smp["k0"])
        net["k_family"] = np.asarray(smp["k_family"])
        net["tau_k"] = np.asarray(smp["tau_k"])
        if lag_cols:
            # long-run TDD sensitivity: this year's k plus the coefficients of the lags
            ke = np.asarray(smp["k_extra_family"])[:, :, lag_cols].sum(axis=2)
            net["k_long_family"] = net["k_family"] + ke
            s["k_long_site"] = np.median(k + np.asarray(smp["k_extra_site"])[:, :, lag_cols].sum(axis=2), 0)
    return s, net


def resid_var_site(mcmc, arr, mode="A", max_draws=1000):
    """Posterior mean, per site, of the variance of y minus the model mean (site level,
    trend, climate terms and region-year effects; not the AR(1) part) over the uncensored
    years. A and B on the same site-years give the share of the year-to-year variance
    that the climate terms explain: 1 - var_B / var_A."""
    smp = mcmc.get_samples()
    nd = len(np.asarray(smp["mu"]))
    idx = np.unique(np.linspace(0, nd - 1, min(nd, max_draws)).astype(int))
    s = arr["s_idx"]
    mean = np.asarray(smp["a_site"])[idx][:, s] + np.asarray(smp["slope_site"])[idx][:, s] * arr["t"][None, :]
    if mode == "B":
        mean = mean + np.asarray(smp["k_site"])[idx][:, s] * arr["x"][None, :]
        if arr.get("extra_names"):
            ke = np.asarray(smp["k_extra_site"])[idx]                           # (draws, ns, ne)
            mean = mean + np.einsum("dne,ne->dn", ke[:, s, :], arr["X_extra"])
    if "z_region_year" in smp:
        g = np.asarray(smp["sd_region_year"])[idx][:, None] * np.asarray(smp["z_region_year"])[idx]
        g = np.concatenate([g, np.zeros((len(idx), 1))], axis=1)
        mean = mean + g[:, arr["ry_idx"]]
    unc = ~arr["cens"]
    e = (arr["y"][None, :] - mean)[:, unc]
    su = s[unc]
    out = np.empty(arr["n_sites"])
    for i in range(arr["n_sites"]):
        ei = e[:, su == i]
        out[i] = np.mean(np.var(ei, axis=1))
    return out


def ci_str(v, level=0.95, fmt="{:.2f}"):
    lo, hi = np.quantile(v, [(1 - level) / 2, 1 - (1 - level) / 2])
    return f"{fmt.format(np.median(v))} ({fmt.format(lo)} to {fmt.format(hi)})"


# =========================================================================================
# PART 3. Air temperature for B: daily series per site and TDD per site-year
# =========================================================================================
# Daily series: ERA5-Land from the Earth Engine files of step 2 (2_covariates.py),
# corrected to the site elevation in summer; or NASA POWER (--power).
#
# TDD window (decided here, reasons):
# - probe surveys measure thaw on the survey day, so TDD runs from 1 January to the
#   survey date. Without a date it runs to the site's median survey day, and without
#   any date at the site to the median survey day of all dated probe surveys
#   (CALM end-of-season protocol; the build keeps the deepest survey of DOY 213-290).
# - thaw tubes and temperature-derived ALT record the season's maximum thaw, so TDD is
#   the whole year's thawing index (1 January to 31 December).
# TDD = sum over days of max(T_mean, 0) in deg C day.

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


def tdd_for_site(t: pd.Series, years, end_doy, tail_from_doy=274):
    """TDD from 1 Jan to day end_doy (inclusive) of each year; NaN if > 5 % days missing.

    One exception, at the end of the series: when the daily series stops inside the window
    on or after 30 Sep (day tail_from_doy - 1), the days after its last day are filled with
    the site's mean max(T, 0) of the same calendar days in the other years. The ERA5-Land
    exports end on 30 Sep of the last ALT year, and the days after mid-September add a
    median 1-2 % of the whole-year TDD, so the last year of a thaw-tube, temperature-ALT
    or late-survey record is kept instead of dropped (an end year weighs most in a trend).
    Returns (tdd, tail_filled).
    """
    clim = t.clip(lower=0).groupby(t.index.dayofyear).mean()   # mean positive T per calendar day
    last = t.dropna().index.max()
    out, filled = [], []
    for y, e in zip(years, end_doy):
        e = int(e)
        s = t[(t.index.year == y) & (t.index.dayofyear <= e)]
        n_exp, tail = e, 0.0
        is_tail = bool(last.year == y and tail_from_doy - 1 <= last.dayofyear < e)
        if is_tail:
            n_exp = last.dayofyear
            tail = float(clim.reindex(range(n_exp + 1, e + 1)).fillna(0.0).sum())
        if len(s) < 0.95 * n_exp or s.isna().mean() > 0.05:
            out.append(np.nan)
            filled.append(False)
            continue
        # missing days (<= 5 %) filled with the mean positive part of the window
        pos = s.clip(lower=0)
        out.append(float(pos.mean() * n_exp + tail))
        filled.append(is_tail)
    return np.array(out), np.array(filled)


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


def tdd_tables(argv=None):
    """TDD per site-year and per calendar year from the daily series in out_dir/daily_t2m
    (ERA5-Land with --source cache, else downloaded from NASA POWER)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("annual_file")
    ap.add_argument("out_dir")
    ap.add_argument("--source", default="power", choices=["power", "cache"])
    ap.add_argument("--start", type=int, default=1981)  # POWER starts in 1981
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    args = ap.parse_args(argv)

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
        tdd, tail = tdd_for_site(t, years, end.astype(int))
        tdd_full, _ = tdd_for_site(t, years, np.full(len(years), 365))
        rule = np.where(tail, np.char.add(rule.astype(str), "+tail_filled"), rule)
        rows.append(pd.DataFrame({"site_id": sid, "year": years, "method": g["method"].values,
                                  "tdd_end_doy": end.astype(int), "tdd_rule": rule,
                                  "tdd": tdd, "tdd_full_year": tdd_full,
                                  "t_source": args.source}))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out / "tdd_site_years.csv", index=False)
    ay = all_years(sites.index, cache, args.start, args.end)
    ay.to_csv(out / "tdd_all_years.csv", index=False)
    print(f"wrote {len(res)} site-years for {res.site_id.nunique()} sites; "
          f"missing TDD: {int(res.tdd.isna().sum())}; network median probe survey DOY {net_doy:.0f}")
    # Where the daily series is too short. The ERA5-Land exports cover ALT years (Oct-Sep),
    # so the last exported year ends on 30 Sep: whole-year TDD (thaw tubes, temperature ALT)
    # and surveys after 30 Sep of that year are tail-filled (see tdd_for_site) until the
    # next ALT year is exported. Lag terms of model B need the years before the first ALT
    # year in the same way (export them, or those first years drop out of the lag variants).
    n_tail = int(res.tdd_rule.str.endswith("+tail_filled").sum())
    if n_tail:
        print(f"{n_tail} site-years of the last year: days after the end of the daily series filled "
              f"with the site's mean of those days (tdd_rule '...+tail_filled')")
    miss = res[res.tdd.isna()]
    if len(miss):
        print("site-years without TDD, by year and window rule (the daily series does not cover the window):")
        print(miss.groupby(["year", "tdd_rule"]).size().rename("n").to_string())
    full = ay.dropna(subset=["tdd_full_year"])
    if len(full):
        print(f"whole-year TDD (lag terms) available for calendar years {int(full.year.min())}-{int(full.year.max())}; "
              f"a lag of L years is missing for ALT years before {int(full.year.min())} + L")


WEATHER_COLS = ["snow_depth_winter", "snow_thermal_resistance_winter", "rain_summer",
                "P_summer", "snowfall_winter", "SWE_max", "snow_off_doy", "TDD", "FDD"]


def elevation_offsets(drive_dir):
    st, f = _read_static(drive_dir)
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
    lapse = dict(LAPSE_RATE_K_PER_KM)
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


def era5_daily_series(argv=None):
    """Daily site temperature and the weather table from the Earth Engine files of step 2."""
    ap = argparse.ArgumentParser()
    ap.add_argument("drive_dir")
    ap.add_argument("tdd_dir")
    ap.add_argument("--grid", action="store_true", help="no elevation correction")
    ap.add_argument("--summer-lapse", type=float, default=None,
                    help="May-Sep lapse rate in K/km for a sensitivity run (default -6.5 from CONFIG)")
    a = ap.parse_args(argv)
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


# =========================================================================================
# PART 4. The run: v5.2 reference, A, B variants, outputs
# =========================================================================================
ORDER = CLASSES
RESID_ORDER = ["deeper_than_TDD", "consistent_with_TDD", "shallower_than_TDD", "not_resolved"]



def span_bins(span):
    return pd.cut(span, [0, 15, 20, 25, 60], labels=["12-15", "16-20", "21-25", "26+"])


def counts(series, order):
    vc = series.value_counts()
    return {k: int(vc.get(k, 0)) for k in order}


def region_bootstrap(values, regions, n_boot=4000, seed=0):
    """95 % interval of the mean over sites when whole regions are resampled. The model
    intervals of a mean over sites treat these sites as the population; resampling regions
    adds the uncertainty of which places happen to be monitored, and keeps the
    correlation of sites within a region."""
    v = pd.Series(np.asarray(values, float), index=np.asarray(regions))
    v = v[np.isfinite(v.to_numpy())]
    groups = [g.to_numpy() for _, g in v.groupby(level=0)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        means[b] = np.concatenate([groups[i] for i in pick]).mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# -----------------------------------------------------------------------------------------
# Spatial checks (added 2026-11-07, geodata review). They use the fitted rates only, so
# `--spatial` can rerun them on an existing aldi_v6_results.csv in a few seconds.
# -----------------------------------------------------------------------------------------
PF_MAP = DATA / "raw" / "permafrost_map" / "permaice.shp"   # IPA map (Brown et al. 2002)
LOCATION_KM = 1.0              # series closer than this are one place (e.g. tube + grid)
DECLUSTER_KM = (100, 250, 500)  # equal-area grid cells for the declustered network mean
SPATIAL_RATES = ["rate_v6", "expected_rate", "resid_rate",
                 "expected_rate_weather_memory", "resid_rate_weather_memory"]


def laea_north_km(lat, lon, radius_km=EARTH_RADIUS_KM):
    """Lambert azimuthal equal-area projection centred on the North Pole (sphere), in km.
    Equal-area, so grid cells of one size cover the same ground at every latitude."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    rho = 2 * radius_km * np.sin((np.pi / 2 - lat) / 2)
    return rho * np.sin(lon), -rho * np.cos(lon)


def location_ids(lat, lon, km=LOCATION_KM):
    """Single-linkage groups of series closer than km: one physical place measured by
    several instruments or sources (a CALM thaw tube and its probing grid, a grid and the
    borehole beside it). The build keeps them apart because the methods read different
    levels; their trends are not independent evidence about the place."""
    if len(lat) == 1:
        return np.array([1])
    d = haversine_matrix(np.asarray(lat, float), np.asarray(lon, float))
    np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(d, checks=False), method="single"), t=km, criterion="distance")


def permafrost_zone(lat, lon, shapefile=PF_MAP, max_snap_km=20.0):
    """Permafrost extent (C/D/S/I) and ground-ice content (h/m/l) of the IPA map polygon
    each site is in; a site in a lake or on the coast takes the nearest permafrost polygon
    within max_snap_km (same rule as 2_covariates.permafrost_classes). Returns None when
    geopandas or the map is not available (the map is an independent source of ground-ice
    information, so it can be used to stratify the residual trend without circularity)."""
    try:
        import geopandas as gpd
    except ImportError:
        return None
    if not Path(shapefile).exists():
        return None
    shp = gpd.read_file(shapefile).to_crs("EPSG:6931")
    code = shp["COMBO"].astype(str).str.lower()
    poly = shp[(code.str.len() >= 3) & code.str[0].isin(list("cdsi"))][["COMBO", "geometry"]]
    pts = gpd.GeoDataFrame({"i": np.arange(len(lat))}, geometry=gpd.points_from_xy(lon, lat),
                           crs="EPSG:4326").to_crs("EPSG:6931")
    hit = gpd.sjoin(pts, poly, how="left", predicate="within").drop_duplicates("i").set_index("i")["COMBO"]
    miss = hit.isna().to_numpy()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], poly, how="left", max_distance=max_snap_km * 1000)
        near = near.drop_duplicates("i").set_index("i")["COMBO"]
        hit = hit.fillna(near)
    c = hit.reindex(np.arange(len(lat))).fillna("").astype(str).str.lower()
    extent = c.str[0].map({"c": "continuous", "d": "discontinuous", "s": "sporadic", "i": "isolated"})
    ice = c.str[1].map({"h": "high", "m": "medium", "l": "low"})
    return pd.DataFrame({"pf_extent": extent.fillna("unknown").to_numpy(),
                         "pf_ice": ice.fillna("unknown").to_numpy()})


def _region_fe_contrast(y, group, region, base, n_boot=2000, seed=0):
    """Difference of each group from `base` WITHIN regions (OLS with region fixed effects),
    with a 95 % interval from resampling whole regions. Regions holding one group only
    add nothing, so the contrast is not the difference between, say, Fairbanks and
    the Mackenzie Delta."""
    df = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(group), "r": np.asarray(region)}).dropna()
    levels = [g for g in sorted(df.g.unique()) if g != base]

    def identified(dd):
        """Levels that share at least one region with the base level (only those regions
        inform a within-region difference)."""
        with_base = set(dd.r[dd.g == base])
        return {g for g in levels if ((dd.g == g) & dd.r.isin(with_base)).any()}

    def est(dd):
        # region fixed effects by the within transformation (demean inside each region)
        ok = identified(dd)
        X = pd.DataFrame({g: (dd.g == g).astype(float) for g in levels}, index=dd.index)
        Xw = (X - X.groupby(dd.r).transform("mean")).to_numpy()
        yw = (dd.y - dd.y.groupby(dd.r).transform("mean")).to_numpy()
        b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
        return {g: (b[k] if g in ok else np.nan) for k, g in enumerate(levels)}

    point = est(df)
    n_shared = {g: df[df.g == g].r.isin(set(df.r[df.g == base])).groupby(df[df.g == g].r).any().sum()
                for g in levels}
    regions = df.r.unique()
    rng = np.random.default_rng(seed)
    boots = {g: [] for g in levels}
    by_r = {r: g for r, g in df.groupby("r")}
    for _ in range(n_boot):
        pick = rng.choice(regions, len(regions))
        dd = pd.concat([by_r[r].assign(r=f"{r}_{k}") for k, r in enumerate(pick)])
        if dd.g.nunique() < 2 or base not in set(dd.g):
            continue
        e = est(dd)
        for g in levels:
            boots[g].append(e[g])
    out = {}
    for g in levels:
        v = np.asarray(boots[g], float)
        v = v[np.isfinite(v)]
        out[g] = (point[g], *(np.quantile(v, [0.025, 0.975]) if len(v) > 50 else (np.nan, np.nan)),
                  int(n_shared[g]))
    return out


def spatial_checks(res, shapefile=PF_MAP):
    """Adds location_id, pf_extent and pf_ice to the results and returns the text block of
    the spatial checks:
    1. places: series within 1 km of each other count once in a location mean;
    2. network means that do not let dense clusters dominate (equal-area grid declustering);
    3. spatial correlation of site rates with distance, and of their deviations from the
       region rate across region borders (does the 250 km region capture the dependence?);
    4. rates by permafrost extent and ground-ice content (IPA map), within regions."""
    res = res.copy()
    rates = [c for c in SPATIAL_RATES if c in res]
    L = ["\nSPATIAL CHECKS"]

    # 1. locations
    res["location_id"] = location_ids(res.lat, res.lon)
    nloc = res.location_id.nunique()
    multi = res.groupby("location_id").site_id.transform("size") > 1
    disagree = int((res[multi].groupby("location_id").class_v6.nunique() > 1).sum())
    L.append(f"locations (series within {LOCATION_KM:g} km joined): {nloc} places for {len(res)} series; "
             f"{res[multi].location_id.nunique()} places hold {int(multi.sum())} series; "
             f"at {disagree} of them the series fall in different v6 classes")
    pairs = [(i, j) for _, g in res[multi].groupby("location_id") for k, i in enumerate(g.index) for j in g.index[k + 1:]]
    if len(pairs) > 5:
        a = np.array([[res.at[i, "rate_v6"], res.at[j, "rate_v6"]] for i, j in pairs])
        L.append(f"  same-place pairs: {len(pairs)}; correlation of their v6 rates {np.corrcoef(a.T)[0, 1]:.2f}, "
                 f"median absolute difference {np.median(np.abs(a[:, 0] - a[:, 1])):.1f} %/dec "
                 f"(method and micro-site differences at one place)")

    # 2. network means
    x, y = laea_north_km(res.lat, res.lon)
    rows = {"mean over series": res[rates].mean(),
            "mean over places (1 km)": res.groupby("location_id")[rates].mean().mean(),
            "mean over regions (250 km)": res.groupby("region_id")[rates].mean().mean()}
    for cs in DECLUSTER_KM:
        cell = pd.Series(np.floor(x / cs).astype(int).astype(str)) + "_" + np.floor(y / cs).astype(int).astype(str)
        w = 1.0 / cell.map(cell.value_counts()).to_numpy()
        rows[f"declustered, {cs} km cells ({cell.nunique()})"] = pd.Series(
            {c: np.average(res[c].to_numpy(float)[np.isfinite(res[c])], weights=w[np.isfinite(res[c])]) for c in rates})
    L.append("\nnetwork means, %/dec (declustered = each equal-area cell counts once; the network is not a"
             " random sample, so the mean over series over-weights dense clusters):")
    L.append(pd.DataFrame(rows).T.round(2).to_string())

    # 3. spatial correlation
    d = haversine_matrix(res.lat.to_numpy(float), res.lon.to_numpy(float))
    same_r = res.region_id.to_numpy()[:, None] == res.region_id.to_numpy()[None, :]
    dev = (res.rate_v6 - res.region_rate_v6).to_numpy(float)
    L.append("\ncorrelation of site rates by distance (all pairs) and of deviations from the region rate"
             " (pairs in different regions):")
    for lo_, hi_ in [(0, 100), (100, 250), (250, 500), (500, 1000), (1000, 20000)]:
        band = np.triu((d > lo_) & (d <= hi_), 1)
        i, j = np.nonzero(band)
        r_all = np.corrcoef(res.rate_v6.to_numpy()[i], res.rate_v6.to_numpy()[j])[0, 1] if len(i) > 10 else np.nan
        i2, j2 = np.nonzero(band & ~same_r)
        r_dev = np.corrcoef(dev[i2], dev[j2])[0, 1] if len(i2) > 10 else np.nan
        L.append(f"  {lo_:>5}-{hi_:<5} km  rates {r_all:5.2f} ({len(i)} pairs)   deviations across regions "
                 f"{r_dev:5.2f} ({len(i2)} pairs)")

    # 4. permafrost map
    pz = permafrost_zone(res.lat.to_numpy(float), res.lon.to_numpy(float), shapefile)
    if pz is None:
        L.append("\npermafrost zone: skipped (needs geopandas and data/raw/permafrost_map/permaice.shp)")
        res["pf_extent"], res["pf_ice"] = np.nan, np.nan
        return res, L
    res["pf_extent"], res["pf_ice"] = pz.pf_extent.to_numpy(), pz.pf_ice.to_numpy()
    for col, base in [("pf_extent", "continuous"), ("pf_ice", "high")]:
        L.append(f"\nby {col} (IPA map; median %/dec, n):")
        L.append(res.groupby(col)[rates].median().round(1).assign(n=res.groupby(col).size()).to_string())
        for c in [c for c in ["rate_v6", "resid_rate", "resid_rate_weather_memory"] if c in res]:
            con = _region_fe_contrast(res[c], res[col], res.region_id, base)
            L.append(f"  {c}: difference from {base} within regions: " + "; ".join(
                f"{g} {v[0]:+.1f} ({v[1]:+.1f} to {v[2]:+.1f}; {v[3]} regions)"
                for g, v in con.items() if np.isfinite(v[0])))
    return res, L


def period_check(d, cfg, rho):
    """Long records: trend over the full record vs over the last `recent_years` years."""
    oc = cfg["outputs"]
    rows = []
    for sid, g in d.groupby("site_id"):
        span = g.year.max() - g.year.min() + 1
        if span < oc["period_check_min_span"]:
            continue
        rec = g[g.year > g.year.max() - oc["recent_years"]]
        if (~rec.censored).sum() < 10:
            continue
        full = ats_trend(g.year, g.alt_cm, g.censored, 0.9, 0.85, rho)
        part = ats_trend(rec.year, rec.alt_cm, rec.censored, 0.9, 0.85, rho)
        rows.append({"site_id": sid, "rate_full": 1000 * full.slope / full.intercept_ref,
                     "rate_recent": 1000 * part.slope / part.intercept_ref,
                     "first_recent_year": int(rec.year.min())})
    return pd.DataFrame(rows)


def figures(res, out, prefix, have_b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. site rates: v5.2 site estimate vs v6, with intervals, coloured by record span
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(res.rate_pct_dec, res.rate_v6, c=res.span_years, cmap="viridis", s=18, zorder=3)
    ax.errorbar(res.rate_pct_dec, res.rate_v6, yerr=[res.rate_v6 - res.rate_v6_lo, res.rate_v6_hi - res.rate_v6],
                fmt="none", ecolor="0.75", lw=0.6, zorder=2)
    lim = np.nanpercentile(np.r_[res.rate_pct_dec, res.rate_v6], [1, 99]) * [1.1, 1.1]
    ax.plot(lim, lim, color="0.4", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("v5.2 site rate (Theil-Sen, % per decade)")
    ax.set_ylabel("v6 hierarchical rate (median, 90 % interval)")
    fig.colorbar(sc, ax=ax, label="record span (years)")
    ax.set_title("Short records shrink toward their region")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_fig_rates_v52_vs_v6.png", dpi=200)
    plt.close(fig)

    # 2. class shares by record span: v5.2 site, v5.2 pooled, v6
    cols = [("class_site", "v5.2 site"), ("class_pooled", "v5.2 pooled"), ("class_v6", "v6")]
    colors = {"rapid_thickening": "#b2182b", "gradual_thickening": "#ef8a62", "stable": "#999999",
              "gradual_thinning": "#67a9cf", "rapid_thinning": "#2166ac", "indeterminate": "#e7e7e7"}
    bins = span_bins(res.span_years)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    for ax, (c, lab) in zip(axes, cols):
        tab = (pd.crosstab(bins, res[c], normalize="index") * 100).reindex(columns=ORDER, fill_value=0)
        left = np.zeros(len(tab))
        for k in ORDER:
            ax.barh(tab.index.astype(str), tab[k], left=left, color=colors[k], edgecolor="white", lw=0.5,
                    label=k.replace("_", " "))
            left += tab[k].to_numpy()
        ax.set_title(lab)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of sites")
    axes[0].set_ylabel("record span (years)")
    axes[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_fig_classes_by_span.png", dpi=200)
    plt.close(fig)

    if not have_b:
        return
    # 3. two-axis index: climate-expected vs residual trend
    b = res.dropna(subset=["expected_rate"])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rc = {"deeper_than_TDD": "#b2182b", "consistent_with_TDD": "#4d4d4d",
          "shallower_than_TDD": "#2166ac", "not_resolved": "#bdbdbd"}
    for k, col in rc.items():
        m = b.class_resid == k
        ax.errorbar(b.expected_rate[m], b.resid_rate[m],
                    yerr=[b.resid_rate[m] - b.resid_lo[m], b.resid_hi[m] - b.resid_rate[m]],
                    fmt="o", ms=3.5, color=col, ecolor=col, elinewidth=0.5, alpha=0.85,
                    label=f"{k.replace('_', ' ')} ({int(m.sum())})")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_xlabel("climate-expected trend, k x TDD trend (% per decade)")
    ax.set_ylabel("residual trend (% per decade, 90 % interval)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Climate-adjusted ALDI")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_fig_expected_vs_residual.png", dpi=200)
    plt.close(fig)


def run_aldi(cfg, annual_file, out, tdd_file=None, tdd_all_file=None, weather_file=None):
    """v5.2 (reference), A (three fits) and B (every climate variant); writes the outputs."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mc_cfg, pri = cfg["model"], cfg["model"]["priors"]
    pre = cfg["outputs"]["prefix"]
    fit_kw = dict(chains=mc_cfg["chains"], warmup=mc_cfg["warmup"], draws=mc_cfg["draws"],
                  target_accept=mc_cfg["target_accept"], seed=mc_cfg["seed"])
    import numpyro
    numpyro.set_host_device_count(mc_cfg["chains"])
    import jax
    import numpyro as npy

    log = [f"ALDI v6  numpyro {npy.__version__}  jax {jax.__version__}  numpy {np.__version__}  pandas {pd.__version__}"]
    annual = pd.read_csv(annual_file, low_memory=False)
    data, removed = load_series(annual, cfg)
    d = eligible_sites(data, cfg)

    # --- v5.2 on the same data (reference) ---
    t0 = time.time()
    res52, net52 = run(data, cfg)
    log.append(f"v5.2 rerun: {len(res52)} sites, network {net52['mu']:.2f} %/dec ({time.time() - t0:.0f} s)")

    # --- A: main and sensitivity fits ---
    arr = build_arrays(d, cfg)
    runs = {}
    for name, ry, sd in [("main", mc_cfg["region_year"], mc_cfg["slope_dist"]),
                         ("no_region_year" if mc_cfg["region_year"] else "region_year", not mc_cfg["region_year"], mc_cfg["slope_dist"]),
                         ("student_slopes", mc_cfg["region_year"], "student")]:
        t0 = time.time()
        mc = fit(arr, "A", slope_dist=sd, region_year=ry, priors=pri, **fit_kw)
        s, net, rate = summarise_A(mc, arr, cfg)
        runs[name] = dict(mc=mc, s=s, net=net, rate=rate, diag=diagnostics(mc), sec=time.time() - t0,
                          sd_ry=(100 * np.asarray(mc.get_samples()["sd_region_year"])
                                 if "sd_region_year" in mc.get_samples() else None))
        log.append(f"A {name}: {runs[name]['diag']}  ({runs[name]['sec']:.0f} s)")
    s = runs["main"]["s"]
    res = res52.merge(s.drop(columns=["lat", "lon", "family", "region_id"]), on="site_id", how="left",
                      validate="one_to_one")
    assert res["class_v6"].notna().all(), "v5.2 and v6 site sets differ"
    for name in runs:
        if name != "main":
            res[f"class_v6_{name}"] = res.site_id.map(runs[name]["s"].set_index("site_id")["class_v6"])
            res[f"rate_v6_{name}"] = res.site_id.map(runs[name]["s"].set_index("site_id")["rate_v6"])

    # --- B ---
    have_b = tdd_file is not None
    if have_b:
        tdd = pd.read_csv(tdd_file)
        arrB = build_arrays(d, cfg, tdd)
        # A on exactly B's site-years, so A and B totals compare like with like
        arrAB = build_arrays(eligible_sites(arrB["d"].drop(columns=["tdd"]), cfg), cfg)
        mcAB = fit(arrAB, "A", region_year=cfg["climate"]["region_year"], priors=pri, **fit_kw)
        sAB, _, _ = summarise_A(mcAB, arrAB, cfg)
        bruns = {}
        tdd_all = pd.read_csv(tdd_all_file) if tdd_all_file else None
        weather = pd.read_csv(weather_file) if weather_file else None
        for v in cfg["climate"]["variants"]:
            ex_names = v.get("extra") or []
            if any(lag_years(c) or c == "fdd" for c in ex_names) and tdd_all is None:
                log.append(f"B {v['name']}: skipped (needs tdd_all_years.csv)")
                continue
            if any(c in WEATHER_TERMS for c in ex_names) and weather is None:
                log.append(f"B {v['name']}: skipped (needs weather_site_years.csv)")
                continue
            t0 = time.time()
            p2 = dict(pri, k_fixed=cfg["climate"]["k_fixed"])
            if v.get("extra"):
                ex = extra_climate(tdd_all, v["extra"], weather)
                arrV = build_arrays(d, cfg, tdd, ex, v["extra"])
            else:
                arrV = arrB
            mcB = fit(arrV, "B", k_mode=v["k_mode"], region_year=cfg["climate"]["region_year"], priors=p2, **fit_kw)
            sB, netB = summarise_B(mcB, arrV, cfg, v["k_mode"])
            bruns[v["name"]] = dict(s=sB, net=netB, diag=diagnostics(mcB), sec=time.time() - t0,
                                    n_sites=arrV["n_sites"], n_obs=len(arrV["y"]))
            if v["name"] == "main":
                resvar_B = resid_var_site(mcB, arrV, "B")
            lost = len(arrB["y"]) - len(arrV["y"])     # e.g. first years without a TDD lag
            log.append(f"B {v['name']}: {bruns[v['name']]['diag']}  ({bruns[v['name']]['sec']:.0f} s; "
                       f"{arrV['n_sites']} sites, {len(arrV['y'])} site-years"
                       + (f", {lost} fewer than main: no value for an extra term" if lost else "") + ")")
        sB = bruns["main"]["s"].copy()
        # share of the year-to-year variance explained by TDD: residual variance of the mean
        # structure without (A, same site-years) and with TDD (B). The ratio of the fitted
        # sigma_site (used up to 2026-11-04) mixed in the change of the AR(1) coefficient:
        # sigma is the stationary SD, which grows when rho grows, so it gave negative
        # shares for the temperature family.
        resvar_A = pd.Series(resid_var_site(mcAB, arrAB, "A"), index=arrAB["sites"]["site_id"])
        sB["tdd_var_explained"] = 1 - resvar_B / resvar_A.reindex(sB.site_id).to_numpy()
        sB["rate_A_same_years"] = sAB.set_index("site_id").reindex(sB.site_id).rate_v6.to_numpy()
        keepB = ["site_id", "tdd_trend_pct_dec", "x_sd", "k_site", "k_lo", "k_hi", "expected_rate", "expected_lo",
                 "expected_hi", "resid_rate", "resid_lo", "resid_hi", "resid_sd", "total_rate", "total_lo", "total_hi",
                 "class_resid", "p_resid_pos", "p_resid_within_margin", "tdd_var_explained", "rate_A_same_years"]
        res = res.merge(sB[keepB], on="site_id", how="left")
        for vn, r in bruns.items():
            if vn == "main":
                continue
            fx = r["s"].set_index("site_id")
            res[f"expected_rate_{vn}"] = res.site_id.map(fx["expected_rate"])
            res[f"resid_rate_{vn}"] = res.site_id.map(fx["resid_rate"])
            res[f"class_resid_{vn}"] = res.site_id.map(fx["class_resid"])
            if "k_long_site" in fx:
                res[f"k_long_{vn}"] = res.site_id.map(fx["k_long_site"])
        res["two_axis"] = np.where(res.class_resid.isna(), np.nan,
                                   res.class_v6.astype(str) + " | " + res.class_resid.astype(str))

    res, spatial_lines = spatial_checks(res)
    res.to_csv(out / f"{pre}_results.csv", index=False)

    # --- network table ---
    def q(v):
        v = np.asarray(v)
        return pd.Series({"median": np.median(v), "lo95": np.quantile(v, 0.025), "hi95": np.quantile(v, 0.975)})
    netrows = {}
    for name, r in runs.items():
        netrows[f"A_{name}_network_rate_pct_dec"] = q(r["net"]["mu"])
        netrows[f"A_{name}_mean_site_rate_pct_dec"] = q(r["rate"].mean(1))
        netrows[f"A_{name}_tau_region_pct"] = q(r["net"]["tau_region"])
        netrows[f"A_{name}_tau_site_pct"] = q(r["net"]["tau_site"])
        for i, f in enumerate(FAMILIES):
            netrows[f"A_{name}_rho_{f}"] = q(r["net"]["rho_family"][:, i])
            netrows[f"A_{name}_sigma_{f}_pct"] = q(r["net"]["sigma_family"][:, i])
        if r["sd_ry"] is not None:
            netrows[f"A_{name}_sd_region_year_pct"] = q(r["sd_ry"])
    if have_b:
        for km, r in bruns.items():
            n = r["net"]
            netrows[f"B_{km}_mean_expected_pct_dec"] = q(n["expected_mean"])
            netrows[f"B_{km}_mean_residual_pct_dec"] = q(n["resid_mean"])
            netrows[f"B_{km}_mean_total_pct_dec"] = q(n["total_mean"])
            netrows[f"B_{km}_network_residual_pct_dec"] = q(n["mu_resid"])
            for i, f in enumerate(FAMILIES):
                netrows[f"B_{km}_rho_{f}"] = q(n["rho_family"][:, i])
            for key in n:
                if key.startswith("k_") and key.endswith("_family") and key != "k_family":
                    for i, f in enumerate(FAMILIES):
                        netrows[f"B_{km}_{key[:-7]}_{f}"] = q(n[key][:, i])
            if "k0" in n:
                netrows[f"B_{km}_k0"] = q(n["k0"])
                netrows[f"B_{km}_tau_k"] = q(n["tau_k"])
                for i, f in enumerate(FAMILIES):
                    netrows[f"B_{km}_k_{f}"] = q(n["k_family"][:, i])
    nett = pd.DataFrame(netrows).T
    nett.to_csv(out / f"{pre}_network.csv")

    # --- summary text ---
    L = log + [""]
    L.append(f"sites {len(res)}; site-years {len(arr['y'])} ({int(arr['cens'].sum())} censored); "
             f"regions {arr['n_regions']}; region-year cells with >= 2 sites {arr['n_ry']}")
    L.append("\nclass counts (same sites):")
    cc = pd.DataFrame({"v5.2_site": counts(res.class_site, ORDER), "v5.2_pooled": counts(res.class_pooled, ORDER),
                       "v6_main": counts(res.class_v6, ORDER)})
    for name in runs:
        if name != "main":
            cc[f"v6_{name}"] = pd.Series(counts(res[f"class_v6_{name}"], ORDER))
    L.append(cc.to_string())
    L.append("\nv5.2 site class (rows) vs v6 class (columns):")
    L.append(pd.crosstab(res.class_site, res.class_v6).reindex(index=ORDER, columns=ORDER, fill_value=0).to_string())
    bins = span_bins(res.span_years)
    for c in ["class_site", "class_pooled", "class_v6"]:
        L.append(f"\n{c} by record span (row %):")
        L.append((pd.crosstab(bins, res[c], normalize="index") * 100).round(0).reindex(columns=ORDER, fill_value=0).to_string())
    L.append("\nnetwork parameters (median, 95 % interval):")
    L.append(nett.round(3).to_string())
    L.append(f"\nmedian 90 % interval width, % per decade: v5.2 site {np.median(res.rate_hi - res.rate_lo):.1f}, "
             f"v5.2 pooled {np.median(2 * 1.6449 * res.rate_pooled_sd):.1f}, v6 {np.median(res.rate_v6_hi - res.rate_v6_lo):.1f}")
    L.append("\nwho is in each record-span group (composition):")
    comp = res.assign(span_bin=bins).groupby("span_bin", observed=True).agg(
        n=("site_id", "size"), probe=("family", lambda f: int((f == "probe").sum())),
        thaw_tube=("family", lambda f: int((f == "thaw_tube").sum())),
        temperature=("family", lambda f: int((f == "temperature").sum())),
        median_first_year=("year_min", "median"), median_last_year=("year_max", "median"),
        median_rate_v6=("rate_v6", "median"))
    L.append(comp.round(1).to_string())
    lg = res.assign(long=res.span_years >= 26).groupby(["region_id", "long"]).rate_v6.median().unstack().dropna()
    if len(lg):
        L.append(f"regions with both short (< 26 yr) and long records: {len(lg)}; median within-region "
                 f"difference short minus long {np.median(lg[False] - lg[True]):.1f} %/dec")
    pc = period_check(d, cfg, net52["ar1_rho_pooled"])
    if len(pc):
        L.append(f"\nperiod check ({len(pc)} sites with span >= {cfg['outputs']['period_check_min_span']} yr): "
                 f"median rate full record {pc.rate_full.median():.1f}, last {cfg['outputs']['recent_years']} yr "
                 f"{pc.rate_recent.median():.1f} %/dec; recent faster at {int((pc.rate_recent > pc.rate_full).sum())} sites")
        pc.to_csv(out / f"{pre}_period_check.csv", index=False)
    if have_b:
        b = res.dropna(subset=["class_resid"])
        L.append(f"\nB (climate-adjusted): {len(b)} sites; site-years without TDD dropped: {arrB['tdd_dropped']}")
        L.append("residual class by B variant (main = TDD with hierarchical k):")
        vt = {"main": counts(b.class_resid, RESID_ORDER)}
        for vn in bruns:
            if vn != "main":
                vt[vn] = counts(res[f"class_resid_{vn}"].dropna(), RESID_ORDER)
        L.append(pd.DataFrame(vt).to_string())
        L.append("\nmean site trends by B variant, % per decade (expected | residual):")
        for vn, r in bruns.items():
            L.append(f"  {vn:14s} expected {np.median(r['net']['expected_mean']):5.2f}  residual "
                     f"{np.median(r['net']['resid_mean']):5.2f}  total {np.median(r['net']['total_mean']):5.2f}  "
                     f"({r['n_sites']} sites, {r['n_obs']} site-years)")
        L.append("  (variants with TDD lags or FDD lose the first year(s) of a record when the temperature"
                 " series starts too late for the lag; compare such a variant with main on the same years)")
        kl = {vn: r["net"]["k_long_family"] for vn, r in bruns.items() if "k_long_family" in r["net"]}
        if kl:
            L.append("\nlong-run TDD sensitivity (this year's k + lag coefficients), median by family "
                     + " / ".join(FAMILIES) + ":")
            for vn, v in kl.items():
                L.append(f"  {vn:14s} " + " / ".join(f"{np.median(v[:, i]):.2f}" for i in range(len(FAMILIES)))
                         + f"   (this year's k alone: "
                         + " / ".join(f"{np.median(bruns[vn]['net']['k_family'][:, i]):.2f}" for i in range(len(FAMILIES))) + ")")
        L.append("\nmean over sites with a region bootstrap 95 % interval (resamples whole regions, so it adds"
                 " which places are monitored; the model intervals above treat these sites as the population):")
        rb = res.set_index("site_id")
        a_lo, a_hi = region_bootstrap(rb["rate_v6"], rb["region_id"])
        L.append(f"  A rate_v6 ({len(rb)} sites): mean {rb['rate_v6'].mean():.2f} ({a_lo:.2f} to {a_hi:.2f})")
        for vn, r in bruns.items():
            sv = r["s"].set_index("site_id")
            reg = rb["region_id"].reindex(sv.index)
            parts = []
            for c in ["expected_rate", "resid_rate", "total_rate"]:
                lo_, hi_ = region_bootstrap(sv[c], reg)
                parts.append(f"{c.split('_')[0]} {sv[c].mean():.2f} ({lo_:.2f} to {hi_:.2f})")
            L.append(f"  B {vn:12s} " + "; ".join(parts))
        L.append("\nv6 class (rows) x residual class (columns):")
        L.append(pd.crosstab(b.class_v6, b.class_resid).reindex(index=ORDER, columns=RESID_ORDER, fill_value=0).to_string())
        L.append(f"\nmedian TDD trend {b.tdd_trend_pct_dec.median():.1f} %/dec; median k {b.k_site.median():.2f}; "
                 f"median share of the year-to-year ALT variance explained by TDD {100 * b.tdd_var_explained.median():.0f} % "
                 f"(below 0 at {int((b.tdd_var_explained < 0).sum())} sites: their k is pulled toward the family mean although "
                 f"their ALT barely follows TDD)")
        L.append(f"median 90 % interval width: total (A, same years) vs residual: "
                 f"{np.median(res.rate_v6_hi - res.rate_v6_lo):.1f} vs {np.median(b.resid_hi - b.resid_lo):.1f} %/dec")
        L.append("\nby method family (median):")
        L.append(b.groupby("family")[["k_site", "tdd_trend_pct_dec", "expected_rate", "resid_rate", "rate_v6",
                                      "tdd_var_explained"]].median().round(2).to_string())
        L.append("\nresidual class by family:")
        L.append(pd.crosstab(b.family, b.class_resid).reindex(columns=RESID_ORDER, fill_value=0).to_string())
    L += spatial_lines
    txt = "\n".join(L)
    (out / f"{pre}_summary.txt").write_text(txt)
    (out / f"{pre}_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    figures(res, out, pre, have_b)
    print(txt)



# =========================================================================================
# PART 4b. Sensitivity runs (--sensitivity)
# =========================================================================================
# Each run changes ONE choice and refits A (main) and B (main and weather_memory) with
# it. The "main" row repeats the same three fits with the main settings, so every row
# compares like with like. Class margins need no refit: they are read from the main
# fit's draws. Written to data/3_aldi_sensitivity:
#   aldi_v6_sensitivity.csv        one row per run: sample size, class counts, network
#                                  rates (A) and expected / residual means (B)
#   aldi_v6_sensitivity_sites.csv  one row per site and run: rate, class, residual
#   aldi_v6_sensitivity.txt        the table with the change from main and the reasons
SENSITIVITY = [
    dict(name="main", why="main settings (reference row; same fits as every other row)"),
    dict(name="cens_sd_0.5", set={"model.priors.cens_sd": 0.5},
         why="lower-bound years: true ALT a HalfNormal(0.5) above the bound (log units; the choice up "
             "to 2026-11-07) instead of anywhere above it"),
    dict(name="cens_sd_1.0", set={"model.priors.cens_sd": 1.0},
         why="lower-bound years: true ALT a HalfNormal(1.0) above the bound"),
    dict(name="no_censored", set={"sample.use_censored": False},
         why="lower-bound years left out instead of used as censored values"),
    dict(name="undated_doy_243", undated_doy=243, skip_A=True,
         why="probe surveys without a date (no date at the site either): TDD to 31 Aug (DOY 243) "
             "instead of the network median survey day (DOY 262); changes B only"),
    dict(name="drop_site_quality", drop="site_quality",
         why="without sites whose latitude or longitude is given to 0.1 deg or coarser, whose elevation differs "
             "from the ERA5-Land cell by more than 200 m, or whose cell is sea (temperature filled from "
             "neighbours)"),
    dict(name="drop_disturbed", drop="disturbed",
         why="without the fire-disturbed Bonanza Creek plots (burned, fire line, safety zone)"),
    dict(name="one_per_place", drop="one_per_place",
         why="one series per place (series within 1 km: keep the one with most years), so a "
             "thaw tube and its grid do not count twice"),
    dict(name="window_2000_2024", set={"sample.min_year": 2000, "sample.max_year": 2024},
         why="common window 2000-2024 (the Streletskiy et al. 2026 period)"),
    dict(name="min_obs_8", set={"sample.min_obs": 8},
         why="shorter records allowed (8 instead of 10 uncensored years)"),
    dict(name="region_150km", set={"pooling.region_km": 150}, why="smaller regions (150 km)"),
    dict(name="region_500km", set={"pooling.region_km": 500}, why="larger regions (500 km)"),
    # Thaw subsidence (added 2026-11-09). A probe, and a thaw tube read against the ground
    # surface (the GSC Mackenzie tubes here), measure ALT from a surface that sinks where
    # ground ice melts, so their trend misses part of the deepening of the permafrost
    # table. These runs add a published subsidence rate to the probe and tube ALT of the
    # series on medium- or high-ice ground (IPA map), counted from the first year of each
    # record. A bound, not a correction. Rates in undisturbed tundra: 0.4-1.0 cm/yr at
    # Utqiagvik 2003-2015 (Streletskiy et al. 2017), 0.4 cm/yr median in the Mackenzie
    # Valley (Smith et al. 2024), 12-15 cm in 18 years on the North Slope (Nyland et al. 2021).
    dict(name="subsidence_0.2", subsidence_cm_yr=0.2,
         why="ALT + 0.2 cm/yr of subsidence since the first year, probe and tube series on medium- or "
             "high-ice ground (low end of published rates)"),
    dict(name="subsidence_0.5", subsidence_cm_yr=0.5,
         why="as above with 0.5 cm/yr (typical published rate)"),
    dict(name="subsidence_1.0", subsidence_cm_yr=1.0,
         why="as above with 1.0 cm/yr (high end: Utqiagvik, warm Interior Alaska)"),
]
SENS_MARGINS = [(2.5, 10.0), (5.0, 15.0)]     # (stable margin, rapid threshold), %/dec
DISTURBED_NOTE = re.compile(r"burned|fire line|safety region", re.I)
DISTURBED_NOT = re.compile(r"control|did not burn", re.I)


def site_flags(d, annual, temp):
    """Per-site quality and disturbance flags for the drop rules of the sensitivity runs."""
    s = d.groupby("site_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                 n_obs=("censored", lambda c: int((~c).sum())),
                                 span=("year", lambda y: int(y.max() - y.min() + 1)),
                                 family=("family", "first")).reset_index()
    # latitude or longitude given to 0.1 deg (about 5-11 km) or coarser: the ERA5-Land cell
    # and its elevation offset may then be the wrong ones
    s["coarse_coords"] = (np.isclose(s.lat, s.lat.round(1), atol=1e-6)
                          | np.isclose(s.lon, s.lon.round(1), atol=1e-6))
    cells = Path(temp) / "era5land_cells.csv"
    if cells.exists():
        c = pd.read_csv(cells).set_index("site_id")
        s["dz_era5_m"] = s.site_id.map(c["dz_era5_m"])
        s["sea_filled"] = s.site_id.map(c["era5_filled"]).fillna(0).astype(int) == 1
    else:
        s["dz_era5_m"], s["sea_filled"] = np.nan, False
    s["high_dz"] = s.dz_era5_m.abs() > 200
    a = annual[annual.site_id.isin(s.site_id)]
    burned = set(a.loc[a.flag.fillna("").str.contains("burned_site"), "site_id"])
    note = a.groupby("site_id").site_note.first().fillna("")
    noted = set(note.index[note.str.contains(DISTURBED_NOTE) & ~note.str.contains(DISTURBED_NOT)])
    s["disturbed"] = s.site_id.isin(burned | noted)
    # one series per place: most uncensored years, then the longest span, then probe
    # (the CALM standard) before thaw tube before temperature
    s["location_id"] = location_ids(s.lat, s.lon)
    order = s.assign(f=s.family.map({f: i for i, f in enumerate(FAMILIES)})).sort_values(
        ["location_id", "n_obs", "span", "f"], ascending=[True, False, False, True])
    s["place_rep"] = s.site_id.isin(order.drop_duplicates("location_id").site_id)
    return s


def sens_drop(flags, rule):
    if rule == "site_quality":
        return set(flags.site_id[flags.coarse_coords | flags.high_dz | flags.sea_filled])
    if rule == "disturbed":
        return set(flags.site_id[flags.disturbed])
    if rule == "one_per_place":
        return set(flags.site_id[~flags.place_rep])
    raise ValueError(rule)


def site_ice(flags):
    """Ground-ice class (IPA map: high, medium, low, unknown) per site_id, or None when
    geopandas or the map is missing."""
    pz = permafrost_zone(flags.lat.to_numpy(float), flags.lon.to_numpy(float))
    return None if pz is None else pd.Series(pz.pf_ice.to_numpy(), index=flags.site_id.to_numpy())


def add_subsidence(d, ice, s_cm_yr):
    """ALT plus s * (years since the first year of the record) for probe and thaw-tube
    series on medium- or high-ice ground: the thaw depth below the first year's surface
    if that surface sank by s cm a year. Returns the table and the number of series changed."""
    d = d.copy()
    on = d.family.isin(["probe", "thaw_tube"]) & d.site_id.map(ice).isin(["medium", "high"])
    first = d.groupby("site_id").year.transform("min")
    d.loc[on, "alt_cm"] = d.loc[on, "alt_cm"] + s_cm_yr * (d.loc[on, "year"] - first[on])
    return d, int(d.loc[on, "site_id"].nunique())


def retime_undated(tdd, cache, doy):
    """TDD of the probe site-years that use the network median survey day, recomputed to
    1 Jan - day `doy`."""
    t = tdd.copy()
    m = t.tdd_rule.astype(str).str.startswith("network_median_survey_doy")
    for sid, g in t[m].groupby("site_id"):
        f = Path(cache) / cache_name(sid)
        if not f.exists():
            continue
        ser = pd.read_csv(f, parse_dates=["date"]).set_index("date")["t2m_c"]
        v, _ = tdd_for_site(ser, g.year.to_numpy(int), np.full(len(g), doy))
        t.loc[g.index, "tdd"] = v
        t.loc[g.index, "tdd_end_doy"] = doy
    return t, int(m.sum())


def _set(cfg, key, value):
    node = cfg
    *path, last = key.split(".")
    for p in path:
        node = node[p]
    assert last in node, f"unknown setting {key}"
    node[last] = value


def declustered_mean(lat, lon, v, km=250):
    x, y = laea_north_km(lat, lon)
    cell = pd.Series(np.floor(x / km).astype(int).astype(str)) + "_" + np.floor(y / km).astype(int).astype(str)
    w = 1.0 / cell.map(cell.value_counts()).to_numpy()
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    return float(np.average(v[ok], weights=w[ok]))


def sensitivity_runs(cfg, annual_file, temp, out, fit_kw):
    """The runs in SENSITIVITY; see the comment above it."""
    out, temp = Path(out), Path(temp)
    annual = pd.read_csv(annual_file, low_memory=False)
    tdd = pd.read_csv(temp / "tdd_site_years.csv")
    tdd_all = pd.read_csv(temp / "tdd_all_years.csv")
    weather = pd.read_csv(temp / "weather_site_years.csv") if (temp / "weather_site_years.csv").exists() else None
    wm = next(v for v in cfg["climate"]["variants"] if v["name"] == "weather_memory")
    rows, site_rows, log, main_A = [], [], [], None
    data0, _ = load_series(annual, cfg)
    flags = site_flags(eligible_sites(data0, cfg), annual, temp)
    flags.to_csv(out / "aldi_v6_sensitivity_flags.csv", index=False)
    ice = None
    for sc in SENSITIVITY:
        t0 = time.time()
        c = copy.deepcopy(cfg)
        for k, v in sc.get("set", {}).items():
            _set(c, k, v)
        pri = c["model"]["priors"]
        data, _ = load_series(annual, c)
        d = eligible_sites(data, c)
        if sc.get("drop"):
            d = eligible_sites(d[~d.site_id.isin(sens_drop(flags, sc["drop"]))], c)
        n_subsided = 0
        if sc.get("subsidence_cm_yr"):
            ice = site_ice(flags) if ice is None else ice
            if ice is None:
                log.append(f"{sc['name']}: skipped (needs geopandas and {PF_MAP})")
                print(log[-1], flush=True)
                continue
            d, n_subsided = add_subsidence(d, ice, sc["subsidence_cm_yr"])
        t, n_retimed = (retime_undated(tdd, temp / "daily_t2m", sc["undated_doy"])
                        if sc.get("undated_doy") else (tdd, 0))
        row = {"run": sc["name"]}
        if n_subsided:
            row["n_subsidence_series"] = n_subsided
        diags = []
        # A
        if sc.get("skip_A") and main_A is not None:
            sA, netA = main_A
        else:
            arr = build_arrays(d, c)
            mc = fit(arr, "A", region_year=c["model"]["region_year"], slope_dist=c["model"]["slope_dist"],
                     priors=pri, **fit_kw)
            sA, netA, rateA = summarise_A(mc, arr, c)
            diags.append(diagnostics(mc))
            row.update(n_sites=arr["n_sites"], n_site_years=len(arr["y"]), n_censored=int(arr["cens"].sum()),
                       n_regions=arr["n_regions"])
            if sc["name"] == "main":
                main_A = (sA, netA)
                main_rows = row.copy()
                for m_, r_ in SENS_MARGINS:
                    cls, *_ = classes_from_draws(rateA, c["trend"]["ci_level"], m_, r_)
                    mrow = {"run": f"margins_{m_:g}_{r_:g}", "n_sites": row["n_sites"],
                            "n_site_years": row["n_site_years"]}
                    mrow.update({f"A_{k}": v for k, v in counts(pd.Series(cls), ORDER).items()})
                    rows.append(mrow)
            del mc
            free_jit()
        if sc.get("skip_A"):
            row.update({k: main_rows[k] for k in ["n_sites", "n_site_years", "n_censored", "n_regions"]})
        row.update({f"A_{k}": v for k, v in counts(sA.class_v6, ORDER).items()})
        mu = netA["mu"]
        row.update(A_network_rate=float(np.median(mu)), A_network_lo95=float(np.quantile(mu, 0.025)),
                   A_network_hi95=float(np.quantile(mu, 0.975)), A_mean_site_rate=float(sA.rate_v6.mean()),
                   A_declustered_250km=declustered_mean(sA.lat, sA.lon, sA.rate_v6),
                   A_mean_over_regions=float(sA.groupby("region_id").rate_v6.mean().mean()))
        # B main and B weather_memory
        arrB = build_arrays(d, c, t)
        p2 = dict(pri, k_fixed=c["climate"]["k_fixed"])
        mcB = fit(arrB, "B", k_mode="hierarchical", region_year=c["climate"]["region_year"], priors=p2, **fit_kw)
        sB, netB = summarise_B(mcB, arrB, c, "hierarchical")
        diags.append(diagnostics(mcB))
        del mcB
        free_jit()
        ex = extra_climate(tdd_all, wm["extra"], weather) if weather is not None else None
        if ex is not None:
            arrW = build_arrays(d, c, t, ex, wm["extra"])
            mcW = fit(arrW, "B", k_mode=wm["k_mode"], region_year=c["climate"]["region_year"], priors=p2, **fit_kw)
            sW, netW = summarise_B(mcW, arrW, c, wm["k_mode"])
            diags.append(diagnostics(mcW))
            del mcW
            free_jit()
        rc = counts(sB.class_resid, RESID_ORDER)
        row.update(B_n_sites=arrB["n_sites"], B_n_site_years=len(arrB["y"]), B_undated_retimed=n_retimed,
                   B_mean_expected=float(np.median(netB["expected_mean"])),
                   B_mean_residual=float(np.median(netB["resid_mean"])),
                   B_mean_residual_declustered_250km=declustered_mean(sB.lat, sB.lon, sB.resid_rate),
                   B_median_k=float(sB.k_site.median()),
                   **{f"B_{k}": v for k, v in rc.items()})
        if ex is not None:
            row.update(Bwm_n_sites=arrW["n_sites"], Bwm_mean_expected=float(np.median(netW["expected_mean"])),
                       Bwm_mean_residual=float(np.median(netW["resid_mean"])),
                       Bwm_deeper_than_TDD=int((sW.class_resid == "deeper_than_TDD").sum()))
        row.update(max_rhat=max(x["max_rhat"] for x in diags), divergences=sum(x["divergences"] for x in diags),
                   seconds=round(time.time() - t0))
        rows.append(row)
        st = sA[["site_id", "rate_v6", "rate_v6_lo", "rate_v6_hi", "class_v6"]].merge(
            sB[["site_id", "expected_rate", "resid_rate", "class_resid"]], on="site_id", how="outer")
        if ex is not None:
            st = st.merge(sW[["site_id", "expected_rate", "resid_rate", "class_resid"]].rename(
                columns=lambda x: x if x == "site_id" else f"{x}_weather_memory"), on="site_id", how="outer")
        site_rows.append(st.assign(run=sc["name"]))
        log.append(f"{sc['name']}: {row['n_sites']} sites, B {row['B_n_sites']}; max R-hat {row['max_rhat']:.3f}, "
                   f"divergences {row['divergences']} ({row['seconds']} s)")
        print(log[-1], flush=True)

    tab = pd.DataFrame(rows).set_index("run")
    for col in tab:
        if (col.startswith(("n_", "B_n_", "Bwm_n_", "A_", "B_", "Bwm_")) and col.split("_", 1)[1] in
                ORDER + RESID_ORDER + ["sites", "site_years", "censored", "regions", "n_sites", "n_site_years",
                                       "undated_retimed", "deeper_than_TDD"]) or col in (
                "divergences", "seconds", "n_subsidence_series"):
            tab[col] = tab[col].astype("Int64")
    tab = tab.loc[[s["name"] for s in SENSITIVITY if s["name"] in tab.index]
                  + [f"margins_{m:g}_{r:g}" for m, r in SENS_MARGINS]]
    tab.to_csv(out / "aldi_v6_sensitivity.csv")
    sites = pd.concat(site_rows, ignore_index=True)
    sites.to_csv(out / "aldi_v6_sensitivity_sites.csv", index=False)

    # change of class at the sites a run shares with main
    m0 = sites[sites.run == "main"].set_index("site_id")
    changed = {}
    for r, g in sites[sites.run != "main"].groupby("run"):
        g = g.set_index("site_id")
        shared = g.index.intersection(m0.index)
        changed[r] = (int((g.loc[shared, "class_v6"] != m0.loc[shared, "class_v6"]).sum()),
                      int((g.loc[shared, "class_resid"].astype(str) != m0.loc[shared, "class_resid"].astype(str)).sum()),
                      len(shared))
    L = ["SENSITIVITY RUNS (each changes one choice; A main + B main + B weather_memory refitted)", ""]
    L += log + [""]
    L.append("flags used by the drop rules (eligible sites of the main sample):")
    L.append(f"  coarse coordinates (lat or lon to 0.1 deg) {int(flags.coarse_coords.sum())}; |site - ERA5-Land cell "
             f"elevation| > 200 m {int(flags.high_dz.sum())}; sea cell filled {int(flags.sea_filled.sum())}; "
             f"any of the three {int((flags.coarse_coords | flags.high_dz | flags.sea_filled).sum())}")
    L.append(f"  fire-disturbed plots {int(flags.disturbed.sum())}: " + ", ".join(flags.site_id[flags.disturbed]))
    L.append(f"  places (series within 1 km joined) {flags.location_id.nunique()}; series that are not their "
             f"place's representative {int((~flags.place_rep).sum())}")
    L.append("\nA: class counts and network rates (%/dec; network = model mean mu with its 95 % interval; "
             "declustered = 250 km equal-area cells count once):")
    cols_a = ["n_sites", "n_site_years"] + [f"A_{k}" for k in ORDER] + [
        "A_network_rate", "A_network_lo95", "A_network_hi95", "A_mean_site_rate", "A_declustered_250km",
        "A_mean_over_regions"]
    L.append(tab[[c for c in cols_a if c in tab]].rename(columns=lambda x: x.replace("A_", "")).round(2).to_string())
    L.append("\nB: means over sites, %/dec (main = this summer's TDD; wm = weather_memory) and residual classes:")
    cols_b = ["B_n_sites", "B_mean_expected", "B_mean_residual", "B_mean_residual_declustered_250km", "B_median_k"] + [
        f"B_{k}" for k in RESID_ORDER] + ["Bwm_mean_expected", "Bwm_mean_residual", "Bwm_deeper_than_TDD"]
    L.append(tab.drop(index=[i for i in tab.index if i.startswith("margins")])[[c for c in cols_b if c in tab]]
             .round(2).to_string())
    L.append("\nchange from main (shared sites): A mean site rate, B mean residual (main / wm), class changes:")
    base = tab.loc["main"]
    for r in tab.index:
        if r == "main" or r.startswith("margins"):
            continue
        n_a, n_b, n_sh = changed.get(r, (0, 0, 0))
        L.append(f"  {r:18s} A {tab.at[r, 'A_mean_site_rate'] - base.A_mean_site_rate:+5.2f}  "
                 f"B {tab.at[r, 'B_mean_residual'] - base.B_mean_residual:+5.2f} / "
                 f"{tab.at[r, 'Bwm_mean_residual'] - base.Bwm_mean_residual:+5.2f}   "
                 f"A class changed {n_a}, residual class changed {n_b} of {n_sh}")
    L.append("\nwhy each run:")
    for s in SENSITIVITY:
        L.append(f"  {s['name']:18s} {s['why']}")
    for m, r in SENS_MARGINS:
        L.append(f"  {'margins_%g_%g' % (m, r):18s} class margins stable +/-{m:g}, rapid {r:g} %/dec "
                 f"(from the main fit's draws, no refit)")
    txt = "\n".join(L)
    (out / "aldi_v6_sensitivity.txt").write_text(txt)
    sensitivity_figure(tab.drop(index=[i for i in tab.index if i.startswith("margins")]), out)
    print(txt)
    return tab


def sensitivity_figure(tab, out):
    """Network rates (A) and expected / residual means (B) for every sensitivity run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yy = np.arange(len(tab))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(tab) + 2.2), sharey=True)
    ax = axes[0]
    ax.errorbar(tab.A_network_rate, yy, xerr=[tab.A_network_rate - tab.A_network_lo95,
                                              tab.A_network_hi95 - tab.A_network_rate],
                fmt="o", color="#2166ac", ms=5, lw=1, label="network mean (model, 95 %)")
    ax.plot(tab.A_mean_site_rate, yy, "s", mfc="none", color="#4d4d4d", ms=5, label="mean over series")
    ax.plot(tab.A_declustered_250km, yy, "x", color="#b2182b", ms=6, label="declustered, 250 km cells")
    ax.axvline(tab.loc["main", "A_network_rate"], color="#2166ac", lw=0.6, ls="--")
    ax.set_xlabel("A: ALT trend (% per decade)")
    ax.set_title("Trend (A)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax = axes[1]
    ax.plot(tab.B_mean_expected, yy + 0.15, "o", color="#4d4d4d", ms=5, label="expected, TDD only")
    ax.plot(tab.B_mean_residual, yy + 0.15, "o", color="#b2182b", ms=5, label="residual, TDD only")
    if "Bwm_mean_residual" in tab:
        ax.plot(tab.Bwm_mean_expected, yy - 0.15, "s", mfc="none", color="#4d4d4d", ms=5,
                label="expected, weather + memory")
        ax.plot(tab.Bwm_mean_residual, yy - 0.15, "s", mfc="none", color="#b2182b", ms=5,
                label="residual, weather + memory")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel("B: mean over sites (% per decade)")
    ax.set_title("Climate-adjusted (B)")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels(tab.index)
    for a_ in axes:
        a_.grid(axis="x", color="0.9", lw=0.6)
    fig.tight_layout()
    fig.savefig(out / "aldi_v6_fig_sensitivity.png", dpi=200)
    plt.close(fig)


# =========================================================================================
# PART 4c. Paper checks (--checks): from the outputs above, no refit (about a minute)
# =========================================================================================
# Added 2026-11-09 (novelty review). Uses data/3_aldi (results, network table, summary,
# TDD table) and, when present, data/3_aldi_sensitivity. Written to data/3_aldi_checks:
#   aldi_v6_checks_sites.csv        one row per site: per-site OLS on the same series (the
#                                   standard test of CALM syntheses), the chance that a
#                                   record of this length and noise resolves its trend,
#                                   shrinkage toward the region, rate in cm per decade
#   aldi_v6_checks_by_span.csv      observed and expected shares by record length
#   aldi_v6_checks_detection.csv    years of record a site needs, per method family
#   aldi_v6_checks_ice_method.csv   residual trend by ground-ice class and method family
#   aldi_v6_checks_cm.csv           rates in cm per decade beside published numbers
#   aldi_v6_checks_diagnostics.csv  MCMC diagnostics of the fits of the main run
#   aldi_v6_checks.txt              the numbers for the text
#
# Detection time. A record's trend is classed when its 90 % interval clears zero (or,
# for "stable", lies inside +-5 %/decade). With the noise of the fitted model (AR(1)
# with the family's rho and the site's sigma) the SD of a trend from given years is
# exact (w' C w for the OLS weights w), so the chance of each class follows from the
# true rate. Two noise levels bracket what a site needs: IN THE NETWORK, the shared
# region-year anomalies are learned from neighbours and removed (the hierarchical model;
# its pooling toward the region narrows the interval further, so the expected share of
# unresolved sites is an upper bound for it); ALONE, they stay in the record's noise (a
# per-site test). For long consecutive records
# this is Weatherhead et al. (1998, JGR 103:17149), written for annual values:
#   n* = [C * sqrt(12) * sigma * sqrt((1 + rho) / (1 - rho)) / |w0|]^(2/3),  w0 = ln(1 + p/100) / 10
# (their formula is for monthly values; one value a year adds the sqrt(12)); C = 1.645 +
# 1.282 = 2.93 to class a site thickening with 90 % probability, 3.29 to show "stable".
# The years below are found from the exact SD, not from this approximation.
#
# Standard test. Per-site OLS of log ALT on year, p from the t (= F) test, all years
# (lower-bound years at their bound, as a per-site analysis would use them): the test of
# Streletskiy et al. (2026, Commun. Earth Environ. 7:671), who fit ALT in cm (the cm
# version is in the rates-in-cm table, also on their 2000-2024 rule). Its expected behaviour on
# these records (power, false positives without a trend, sign errors and exaggeration
# of the "significant" rates, Gelman & Carlin 2014) follows from the same noise model,
# with the hierarchical rate as the true rate.
CHECK_SPANS = ["12-15", "16-20", "21-25", "26+"]


def noise_cov(years, sigma, rho, sd_shared=0.0):
    """Noise covariance at the given years: AR(1) with stationary SD sigma (a gap of g
    years gives rho**g) plus an independent shared-year term of SD sd_shared."""
    lag = np.abs(np.subtract.outer(years, years))
    return sigma ** 2 * rho ** lag + sd_shared ** 2 * np.eye(len(years))


def ols_slope_sd(years, cov):
    """SD of the OLS slope (log units per decade) for noise with covariance cov."""
    t = (np.asarray(years, float) - np.mean(years)) / 10.0
    w = t / np.sum(t ** 2)
    return float(np.sqrt(w @ cov @ w))


def class_chances(b, s, z, margin):
    """Chances of thickening, stable, thinning and unresolved for an estimate normal
    around the true slope b (log units per decade) with SD s and interval +- z*s, with
    the rules of classify (stable first, then the sign)."""
    b, s = np.asarray(b, float), np.asarray(s, float)
    lm, um = np.log(1 - margin / 100), np.log(1 + margin / 100)
    F = lambda v: stats.norm.cdf((v - b) / s)  # noqa: E731
    stable = np.clip(F(um - z * s) - F(lm + z * s), 0, None)
    up = 1 - F(z * s) - np.clip(F(um - z * s) - F(z * s), 0, None)
    down = F(-z * s) - np.clip(F(-z * s) - F(lm + z * s), 0, None)
    return up, stable, down, 1 - up - stable - down


def years_needed(sigma, rho, sd_shared, b, z, power, margin, n_max=150):
    """Fewest consecutive annual values for which a site with true slope b is classed
    thickening (b > 0) or stable (b = 0) with at least the given chance."""
    for n in range(5, n_max + 1):
        yrs = np.arange(n, dtype=float)
        s = ols_slope_sd(yrs, noise_cov(yrs, sigma, rho, sd_shared))
        up, st, _, _ = class_chances(b, s, z, margin)
        if (st if b == 0 else up) >= power:
            return n
    return np.nan


def retrodesign(b, sd_true, se_nominal, df):
    """Per-site OLS test at p < 0.05 when the estimate is normal around b with SD sd_true
    but the test uses se_nominal (Gelman & Carlin 2014): power, chance of a wrong sign
    among significant results (Type S), E|estimate| / |b| among significant results
    (Type M, the exaggeration) and the chance of a significant result when b = 0."""
    c = stats.t.ppf(0.975, df) * se_nominal
    a_hi, a_lo = (c - b) / sd_true, (-c - b) / sd_true
    p_hi, p_lo = 1 - stats.norm.cdf(a_hi), stats.norm.cdf(a_lo)
    power = p_hi + p_lo
    type_s = np.where(b >= 0, p_lo, p_hi) / power
    # E[|est|; |est| > c] from the moments of the truncated normal
    e_abs = (b * p_hi + sd_true * stats.norm.pdf(a_hi)) + (-b * p_lo + sd_true * stats.norm.pdf(a_lo))
    type_m = e_abs / power / np.abs(b)
    false_pos = 2 * stats.norm.cdf(-c / sd_true)
    return power, type_s, type_m, false_pos


def mcmc_table(summary_text):
    """The diagnostics lines of aldi_v6_summary.txt as a table."""
    rows = []
    pat = re.compile(r"^(A|B) (\S+): \{'max_rhat': ([\d.naninf]+), 'min_ess': ([\d.naninf]+), "
                     r"'divergences': (\d+)\}")
    for line in summary_text.splitlines():
        m = pat.match(line.strip())
        if m:
            rows.append({"model": m.group(1), "fit": m.group(2), "max_rhat": float(m.group(3)),
                         "min_ess": float(m.group(4)), "divergences": int(m.group(5))})
    return pd.DataFrame(rows)


def paper_checks(cfg, annual_file, main_out, out, sens_dir=None):
    """The checks described above PART 4c."""
    main_out, out = Path(main_out), Path(out)
    pre = cfg["outputs"]["prefix"]
    res = pd.read_csv(main_out / f"{pre}_results.csv")
    net = pd.read_csv(main_out / f"{pre}_network.csv", index_col=0)
    summary = (main_out / f"{pre}_summary.txt").read_text()
    z = stats.norm.ppf(0.5 + cfg["trend"]["ci_level"] / 2)
    margin, rapid = cfg["classes"]["stable_margin_pct_dec"], cfg["classes"]["rapid_pct_dec"]
    nv = lambda k: float(net.loc[k, "median"]) if k in net.index else np.nan  # noqa: E731
    rho_f = {f: nv(f"A_main_rho_{f}") for f in FAMILIES}
    sig_f = {f: nv(f"A_main_sigma_{f}_pct") / 100 for f in FAMILIES}
    sd_ry = nv("A_main_sd_region_year_pct") / 100
    sd_ry = 0.0 if not np.isfinite(sd_ry) else sd_ry
    mu = nv("A_main_network_rate_pct_dec")
    L = ["PAPER CHECKS (from data/3_aldi; nothing refitted)", ""]

    # --- the analysed series (same rules as the main run) ---
    annual = pd.read_csv(annual_file, low_memory=False)
    data, _ = load_series(annual, cfg)
    d = eligible_sites(data, cfg)
    d = d[d.site_id.isin(set(res.site_id))]
    r = res.set_index("site_id")
    rows = []
    for sid, g in d.groupby("site_id"):
        g = g.sort_values("year")
        yrs = g.year.to_numpy(float)
        y = np.log(g.alt_cm.to_numpy(float))
        n = len(yrs)
        tdec = (yrs - yrs.mean()) / 10.0
        sxx = float(np.sum(tdec ** 2))
        b_ols, a_ols = np.polyfit(tdec, y, 1)
        res_sd = float(np.sqrt(np.sum((y - a_ols - b_ols * tdec) ** 2) / (n - 2)))
        se = res_sd / np.sqrt(sxx)
        p = 2 * stats.t.sf(abs(b_ols / se), n - 2)
        q90 = stats.t.ppf(0.5 + cfg["trend"]["ci_level"] / 2, n - 2)
        lo, hi = pct(b_ols - q90 * se), pct(b_ols + q90 * se)
        # same test on ALT in cm (the unit of the syntheses), for the cm/yr comparison
        bc, ac = np.polyfit(yrs, g.alt_cm.to_numpy(float), 1)
        sdc = np.sqrt(np.sum((g.alt_cm.to_numpy(float) - ac - bc * yrs) ** 2) / (n - 2))
        p_cm = 2 * stats.t.sf(abs(bc / (sdc / np.sqrt(np.sum((yrs - yrs.mean()) ** 2)))), n - 2)
        s_i = r.at[sid, "sigma_pct"] / 100 if np.isfinite(r.at[sid, "sigma_pct"]) else sig_f[r.at[sid, "family"]]
        rho = rho_f[r.at[sid, "family"]]
        b_true = np.log(1 + r.at[sid, "rate_v6"] / 100)
        s_net = ols_slope_sd(yrs, noise_cov(yrs, s_i, rho))
        s_alone = ols_slope_sd(yrs, noise_cov(yrs, s_i, rho, sd_ry))
        up_n, st_n, dn_n, un_n = class_chances(b_true, s_net, z, margin)
        up_a, st_a, dn_a, un_a = class_chances(b_true, s_alone, z, margin)
        se_nom = np.sqrt((s_i ** 2 + sd_ry ** 2) / sxx)      # what OLS reports, on average
        with np.errstate(divide="ignore", invalid="ignore"):
            power, type_s, type_m, fp = retrodesign(b_true, s_alone, se_nom, n - 2)
        rows.append({"site_id": sid, "n_years": n, "ols_rate": pct(b_ols), "ols_lo": lo, "ols_hi": hi,
                     "ols_p": p, "ols_p_cm": p_cm, "ols_slope_cm_yr": bc,
                     "class_ols": classify(pct(b_ols), lo, hi, margin, rapid),
                     "p_resolved_network": 1 - un_n, "p_resolved_alone": 1 - un_a,
                     "p_stable_network": st_n, "p_stable_alone": st_a,
                     "trend_sd_network": 100 * s_net, "trend_sd_alone": 100 * s_alone,
                     "ols_power": power, "ols_type_s": type_s, "ols_type_m": type_m,
                     "ols_false_pos": fp})
    ck = pd.DataFrame(rows)
    ck["ols_q"] = bh_qvalues(ck.ols_p.to_numpy())
    ck = res[["site_id", "family", "region_id", "span_years", "rate_v6", "rate_v6_sd", "region_rate_v6",
              "class_v6", "class_site", "alt_ref_cm", "pf_ice", "resid_rate"]].merge(ck, on="site_id", how="inner")
    ck["span_bin"] = span_bins(ck.span_years).astype(str)
    # shrinkage toward the region: 0 = the site's own OLS rate, 1 = the region rate
    gap = ck.ols_rate - ck.region_rate_v6
    ck["shrinkage"] = np.where(np.abs(gap) >= 2, (ck.ols_rate - ck.rate_v6) / gap, np.nan)
    ck["rate_v6_cm_dec"] = ck.rate_v6 * ck.alt_ref_cm / 100
    ck.to_csv(out / "aldi_v6_checks_sites.csv", index=False)

    # --- by record length ---
    sig = ck.ols_p < 0.05
    big = np.abs(ck.rate_v6) >= 1                     # Type M is undefined at a true rate of 0
    pw = lambda col, k: float(np.sum((ck[col] * ck.ols_power)[k]) / np.sum(ck.ols_power[k]))  # noqa: E731
    span_rows = []
    for sb in CHECK_SPANS + ["all"]:
        k = (ck.span_bin == sb) if sb != "all" else np.ones(len(ck), bool)
        kb = k & big
        span_rows.append({
            "span": sb, "sites": int(k.sum()),
            "unresolved_hierarchical_pct": 100 * (ck.class_v6[k] == "indeterminate").mean(),
            "unresolved_per_site_pct": 100 * (ck.class_site[k] == "indeterminate").mean(),
            "expected_unresolved_network_pct": 100 * (1 - ck.p_resolved_network[k]).mean(),
            "expected_unresolved_alone_pct": 100 * (1 - ck.p_resolved_alone[k]).mean(),
            "stable_hierarchical_pct": 100 * (ck.class_v6[k] == "stable").mean(),
            "expected_stable_network_pct": 100 * ck.p_stable_network[k].mean(),
            "ols_significant_up_pct": 100 * (sig & (ck.ols_rate > 0))[k].mean(),
            "ols_significant_down_pct": 100 * (sig & (ck.ols_rate < 0))[k].mean(),
            "ols_significant_fdr_pct": 100 * (ck.ols_q[k] < 0.05).mean(),
            "ols_expected_significant_pct": 100 * ck.ols_power[k].mean(),
            "ols_false_pos_if_no_trend_pct": 100 * ck.ols_false_pos[k].mean(),
            "ols_wrong_sign_among_significant_pct": 100 * pw("ols_type_s", kb),
            "ols_exaggeration_among_significant": pw("ols_type_m", kb),
            "median_shrinkage": float(np.nanmedian(ck.shrinkage[k])),
            "median_trend_sd_network": float(ck.trend_sd_network[k].median()),
            "median_trend_sd_alone": float(ck.trend_sd_alone[k].median())})
    sp = pd.DataFrame(span_rows)
    sp.to_csv(out / "aldi_v6_checks_by_span.csv", index=False)
    L.append("BY RECORD LENGTH (% of sites; expected = from record length and the fitted noise, with the "
             "hierarchical rate as the true rate):")
    L.append(sp.round(1).to_string(index=False))
    L.append(f"(OLS significant at p < 0.05 on ALT in cm instead of log ALT: {100 * (ck.ols_p_cm < 0.05).mean():.0f} % "
             f"of sites; on log ALT {100 * sig.mean():.0f} %)")

    # --- detection time per family ---
    det = []
    b_net = np.log(1 + mu / 100)
    for f in FAMILIES:
        for where, sh in [("network", 0.0), ("alone", sd_ry)]:
            lr = sig_f[f] ** 2 * (1 + rho_f[f]) / (1 - rho_f[f]) + sh ** 2   # long-run variance
            det.append({"family": f, "noise": where, "sigma_pct": 100 * sig_f[f], "rho": rho_f[f],
                        "sd_region_year_pct": 100 * sh,
                        "longrun_sd_pct": 100 * np.sqrt(lr),
                        "years_thickening_90pct": years_needed(sig_f[f], rho_f[f], sh, b_net, z, 0.9, margin),
                        "years_thickening_50pct": years_needed(sig_f[f], rho_f[f], sh, b_net, z, 0.5, margin),
                        "years_stable_90pct": years_needed(sig_f[f], rho_f[f], sh, 0.0, z, 0.9, margin),
                        "weatherhead_thickening_90pct": float((2.93 * np.sqrt(12 * lr) / (b_net / 10)) ** (2 / 3)),
                        "sites": int((res.family == f).sum())})
    det = pd.DataFrame(det)
    det.to_csv(out / "aldi_v6_checks_detection.csv", index=False)
    L.append(f"\nYEARS OF RECORD A SITE NEEDS (consecutive years; thickening at the network rate {mu:.1f} %/dec, "
             f"stable = a trendless site shown inside +-{margin:g} %/dec; network = shared region-year anomalies "
             f"removed, alone = left in; region-year SD {100 * sd_ry:.1f} %):")
    L.append(det.round(2).to_string(index=False))
    for f in FAMILIES:
        L.append(f"  share of the year-to-year noise that is shared within a region, {f}: "
                 f"{100 * sd_ry ** 2 / (sd_ry ** 2 + sig_f[f] ** 2):.0f} %")

    # --- standard test on the real series ---
    n = len(ck)
    L.append(f"\nSTANDARD PER-SITE TEST ON THE {n} SERIES (OLS on log ALT, t/F test):")
    L.append(f"  p < 0.05 increasing {int((sig & (ck.ols_rate > 0)).sum())} ({100 * (sig & (ck.ols_rate > 0)).mean():.0f} %), "
             f"decreasing {int((sig & (ck.ols_rate < 0)).sum())}, not significant {int((~sig).sum())} "
             f"({100 * (~sig).mean():.0f} %)")
    L.append(f"  after the false-discovery-rate correction (Benjamini-Hochberg, q < 0.05): "
             f"{int((ck.ols_q < 0.05).sum())} ({100 * (ck.ols_q < 0.05).mean():.0f} %)")
    L.append(f"  of the 'not significant' sites the hierarchical model classes {int(((~sig) & ck.class_v6.str.contains('thickening')).sum())} "
             f"thickening, {int(((~sig) & (ck.class_v6 == 'stable')).sum())} stable, "
             f"{int(((~sig) & (ck.class_v6 == 'indeterminate')).sum())} not yet resolved")
    L.append("  OLS classes with the same 90 % rules: " + ", ".join(
        f"{c} {int((ck.class_ols == c).sum())}" for c in ORDER))
    a = sp[sp.span == "all"].iloc[0]
    L.append(f"  expected from the noise model: {a.ols_expected_significant_pct:.0f} % significant; "
             f"{a.ols_false_pos_if_no_trend_pct:.0f} % significant if there were no trend at all (nominal 5 %); "
             f"significant rates overstate the true rate by a factor {a.ols_exaggeration_among_significant:.2f} "
             f"(12-15 yr: {sp.ols_exaggeration_among_significant.iloc[0]:.2f}); wrong sign among significant "
             f"{a.ols_wrong_sign_among_significant_pct:.1f} %")

    # --- shrinkage and the regional prior alone ---
    tau = nv("A_main_tau_site_pct") / 100
    tr = nv("A_main_tau_region_pct") / 100
    L.append("\nPOOLING: median shrinkage of the site rate toward its region (0 = own OLS rate, 1 = region rate) "
             + ", ".join(f"{row.span} {row.median_shrinkage:.2f}" for row in sp.itertuples()))
    L.append(f"  between-site spread of trends: regions {100 * tr:.1f}, sites within a region {100 * tau:.1f} %/dec "
             f"(log units x 100); share that is regional {100 * tr ** 2 / (tr ** 2 + tau ** 2):.0f} %")
    reg = res.groupby("region_id").agg(region_rate=("region_rate_v6", "first"), n=("site_id", "size"))
    bp = np.log(1 + reg.region_rate / 100)
    prior_cls = [classify(pct(m_), pct(m_ - z * tau), pct(m_ + z * tau), margin, rapid) for m_ in bp]
    pc = pd.Series(prior_cls).value_counts()
    L.append(f"  class of a site with NO data, from its region's rate and the between-site spread "
             f"(tau_site {100 * tau:.1f} %; region uncertainty left out, so this overstates the prior): "
             + ", ".join(f"{k} {v}" for k, v in pc.items()) + f" of {len(reg)} regions")

    # --- expected class counts and probabilities ---
    pcols = [f"p6_{c}" for c in ORDER if f"p6_{c}" in res]
    L.append("\nPOSTERIOR EXPECTED CLASS COUNTS (sum of the band probabilities; bands, not interval rules): "
             + ", ".join(f"{c[3:]} {res[c].sum():.0f}" for c in pcols))
    if {"p6_rapid_thickening", "p6_gradual_thickening"} <= set(res):
        p5 = res.p6_rapid_thickening + res.p6_gradual_thickening
        L.append(f"  P(rate > {margin:g} %/dec) > 0.9 at {int((p5 > 0.9).sum())} sites, > 0.5 at {int((p5 > 0.5).sum())}; "
                 f"P(thickening) > 0.9 at {int((res.p6_thickening > 0.9).sum())}")

    # --- TDD: share of the variance vs share of the trend ---
    tf = main_out / "temperature" / "tdd_site_years.csv"
    if tf.exists():
        t = pd.read_csv(tf)[["site_id", "year", "method", "tdd"]].drop_duplicates(["site_id", "year", "method"])
        dt = d.merge(t, on=["site_id", "year", "method"], how="inner")
        dt = dt[dt.tdd > 0]
        yv, xv = dt.alt_cm.to_numpy(float), np.sqrt(dt.tdd.to_numpy(float))
        g_ = dt.site_id.to_numpy()
        yw = yv - pd.Series(yv).groupby(g_).transform("mean").to_numpy()
        xw = xv - pd.Series(xv).groupby(g_).transform("mean").to_numpy()
        beta = float(np.sum(xw * yw) / np.sum(xw ** 2))
        ssr = float(np.sum((yw - beta * xw) ** 2))
        r2_fe = 1 - ssr / float(np.sum((yv - yv.mean()) ** 2))
        r2_within = 1 - ssr / float(np.sum(yw ** 2))
        # the same within each site after removing its linear trend: year-to-year only
        def detr(v):
            return pd.Series(v).groupby(g_).transform(
                lambda s: s - np.polyval(np.polyfit(dt.year.to_numpy(float)[s.index], s.to_numpy(), 1),
                                         dt.year.to_numpy(float)[s.index])).to_numpy()
        yd, xd = detr(yv), detr(xv)
        bd = float(np.sum(xd * yd) / np.sum(xd ** 2))
        r2_yy = 1 - float(np.sum((yd - bd * xd) ** 2)) / float(np.sum(yd ** 2))
        e_, t_ = nv("B_main_mean_expected_pct_dec"), nv("B_main_mean_total_pct_dec")
        L.append(f"\nTDD AND ALT ({len(dt)} site-years; ALT in cm on sqrt(TDD), as Streletskiy et al. 2026):")
        L.append(f"  with site fixed effects R2 = {100 * r2_fe:.0f} % (mostly the differences between sites); "
                 f"within sites {100 * r2_within:.0f} %; year-to-year (each site detrended) {100 * r2_yy:.0f} %; "
                 f"{beta:.2f} cm per unit sqrt(TDD)")
        L.append(f"  share of the TREND that TDD explains (model B main): {e_:.1f} of {t_:.1f} %/dec = "
                 f"{100 * e_ / t_:.0f} %")
    for vn in ["main", "kfixed"]:
        if f"B_{vn}_mean_expected_pct_dec" in net.index:
            L.append(f"  B {vn}: expected {nv(f'B_{vn}_mean_expected_pct_dec'):.2f}, residual "
                     f"{nv(f'B_{vn}_mean_residual_pct_dec'):.2f} %/dec"
                     + (" (k = 0.5, Stefan: the most the expected part can take if the fitted k is attenuated"
                        " by errors in TDD)" if vn == "kfixed" else ""))

    # --- residual by ground ice and method ---
    rows = []
    b = res.dropna(subset=["resid_rate"])
    for f in FAMILIES + ["all"]:
        bf = b if f == "all" else b[b.family == f]
        for ice in ["low", "medium", "high", "unknown"]:
            k = bf.pf_ice.astype(str) == ice
            if not k.any():
                continue
            rows.append({"family": f, "pf_ice": ice, "sites": int(k.sum()),
                         "mean_resid": float(bf.resid_rate[k].mean()) if k.any() else np.nan,
                         "median_resid": float(bf.resid_rate[k].median()) if k.any() else np.nan})
        if bf.pf_ice.isin(["low"]).any() and bf.pf_ice.isin(["high"]).any():
            con = _region_fe_contrast(bf.resid_rate, bf.pf_ice.astype(str), bf.region_id, "high")
            if "low" in con and np.isfinite(con["low"][0]):
                rows.append({"family": f, "pf_ice": "low_minus_high_within_regions", "sites": con["low"][3],
                             "mean_resid": con["low"][0], "lo95": con["low"][1], "hi95": con["low"][2]})
    im = pd.DataFrame(rows)
    im.to_csv(out / "aldi_v6_checks_ice_method.csv", index=False)
    L.append("\nRESIDUAL TREND BY GROUND ICE (IPA) AND METHOD (%/dec; contrast rows: sites = shared regions):")
    L.append(im.round(2).to_string(index=False))

    # --- cm per decade beside published numbers ---
    cm = [{"what": "ALDI hierarchical, mean of site rates", "value": float(ck.rate_v6_cm_dec.mean()),
           "unit": "cm/decade", "sites": len(ck)},
          {"what": "ALDI hierarchical, median of site rates", "value": float(ck.rate_v6_cm_dec.median()),
           "unit": "cm/decade", "sites": len(ck)},
          {"what": "ALDI per-site OLS on cm, sites with a significant increase (as Streletskiy et al. 2026)",
           "value": float(ck.ols_slope_cm_yr[(ck.ols_p_cm < 0.05) & (ck.ols_slope_cm_yr > 0)].mean()),
           "unit": "cm/yr", "sites": int(((ck.ols_p_cm < 0.05) & (ck.ols_slope_cm_yr > 0)).sum())},
          {"what": "ALDI per-site OLS on cm, all increasing sites (as Lin et al. 2026)",
           "value": float(ck.ols_slope_cm_yr[ck.ols_slope_cm_yr > 0].mean()), "unit": "cm/yr",
           "sites": int((ck.ols_slope_cm_yr > 0).sum())},
          {"what": "ALDI share of sites with a significant increase, OLS on cm",
           "value": float(100 * ((ck.ols_p_cm < 0.05) & (ck.ols_slope_cm_yr > 0)).mean()), "unit": "%",
           "sites": len(ck)}]
    # The same test on the rule of Streletskiy et al. (2026): years 2000-2024 only, at least ten years
    # in that window (here ten years that are not lower bounds, north of 60 N as everywhere in ALDI),
    # a linear trend of ALT in cm per site, F test (= t test of the slope) at p < 0.05. Lower-bound
    # years enter at their bound, as in the whole-record test above.
    sw = dict(cfg, sample=dict(cfg["sample"], min_year=2000, max_year=2024, min_span_years=10))
    dw, _ = load_series(annual, sw)
    slope_w, p_w = [], []
    for _, g in dw.groupby("site_id"):
        yrs, a = g.year.to_numpy(float), g.alt_cm.to_numpy(float)
        bw, aw = np.polyfit(yrs, a, 1)
        sdw = np.sqrt(np.sum((a - aw - bw * yrs) ** 2) / (len(yrs) - 2))
        slope_w.append(bw)
        p_w.append(2 * stats.t.sf(abs(bw / (sdw / np.sqrt(np.sum((yrs - yrs.mean()) ** 2)))), len(yrs) - 2))
    if slope_w:
        slope_w, p_w = np.array(slope_w), np.array(p_w)
        up_w = (p_w < 0.05) & (slope_w > 0)
        cm += [{"what": "ALDI per-site OLS on cm, 2000-2024 and >= 10 years (rule of Streletskiy et al. 2026), "
                        "share with a significant increase", "value": float(100 * up_w.mean()), "unit": "%",
                "sites": len(slope_w)},
               {"what": "ALDI per-site OLS on cm, 2000-2024 and >= 10 years, sites with a significant increase",
                "value": float(slope_w[up_w].mean()) if up_w.any() else np.nan, "unit": "cm/yr",
                "sites": int(up_w.sum())}]
    sens_sites = Path(sens_dir) / "aldi_v6_sensitivity_sites.csv" if sens_dir else None
    if sens_sites is not None and sens_sites.exists():
        ss = pd.read_csv(sens_sites)
        w = ss[ss.run == "window_2000_2024"].merge(res[["site_id", "alt_ref_cm"]], on="site_id")
        if len(w):
            cm.append({"what": "ALDI hierarchical, 2000-2024 window run, mean of site rates",
                       "value": float((w.rate_v6 * w.alt_ref_cm / 100).mean()), "unit": "cm/decade",
                       "sites": len(w)})
    for reg_id, g in ck.groupby("region_id"):
        if len(g) >= 4:
            cm.append({"what": f"ALDI region {reg_id} ({g.region_rate_v6.iloc[0]:.1f} %/dec)",
                       "value": float(g.region_rate_v6.iloc[0] * g.alt_ref_cm.mean() / 100),
                       "unit": "cm/decade", "sites": len(g)})
    # Streletskiy et al. (2026, Commun. Earth Environ. 7:671, doi:10.1038/s43247-026-03824-1): 156 CALM sites
    # with at least ten years of ALT in 2000-2024, 137 of them in the Northern Hemisphere. Arctic: 55.2 %
    # (53 sites) with a significant increase, by 0.8 +/- 0.8 cm/yr. Northern Hemisphere: 66.4 % (91 of 137),
    # raised by the mountain sites (European mountains 94.7 %, 18 of 19; Third Pole 90.9 %, 20 of 22).
    # The paper does not give the number of Arctic sites (53 / 0.552 = 96).
    pub = [("Streletskiy et al. 2026, Arctic, 2000-2024, sites with a significant increase", 0.8, "cm/yr", 53),
           ("Streletskiy et al. 2026, Arctic, 2000-2024, share with a significant increase "
            + "(53 sites; number of Arctic sites not given)", 55.2, "%", np.nan),
           ("Streletskiy et al. 2026, Northern Hemisphere incl. mountains, share with a significant increase",
            66.4, "%", 137),
           ("Lin et al. EGUsphere-2026-841, circum-Arctic mean at increasing sites", 1.48, "cm/yr", np.nan),
           ("Smith et al. 2024 (ICOP), Mackenzie Valley thaw tubes, ALT", 1.0, "cm/decade", np.nan),
           ("Smith et al. 2024 (ICOP), Mackenzie Valley thaw tubes, thaw penetration (fixed datum)", 8.0, "cm/decade", np.nan)]
    for w_, v_, u_, n_ in pub:
        cm.append({"what": "PUBLISHED: " + w_, "value": v_, "unit": u_, "sites": n_})
    cmt = pd.DataFrame(cm)
    cmt["sites"] = cmt.sites.astype("Int64")
    cmt.to_csv(out / "aldi_v6_checks_cm.csv", index=False)
    L.append("\nRATES IN CM (site rates times the site's fitted ALT; published values for comparison):")
    L.append(cmt.round(2).to_string(index=False))

    # --- MCMC diagnostics and the survey-date drift flag ---
    mt = mcmc_table(summary)
    mt.to_csv(out / "aldi_v6_checks_diagnostics.csv", index=False)
    if len(mt):
        L.append(f"\nMCMC: {len(mt)} fits; max R-hat {mt.max_rhat.max():.3f}; min ESS {mt.min_ess.min():.0f}; "
                 f"divergences {int(mt.divergences.sum())}")
    if "flag_doy_drift" in res:
        dr = res.survey_doy_drift_days
        L.append(f"survey date drift > {cfg['diagnostics']['survey_doy_drift_days']} days over the record: "
                 f"{int(res.flag_doy_drift.astype(str).str.lower().eq('true').sum())} of {int(dr.notna().sum())} "
                 f"sites with dates (median drift {dr.median():+.1f} days)")
    txt = "\n".join(L)
    (out / "aldi_v6_checks.txt").write_text(txt)
    print(txt)


# =========================================================================================
# PART 5. Simulation check (--validate)
# =========================================================================================
# Simulation check of ALDI v6 on the real sampling design.
#
# Every synthetic network keeps the real sites, years, gaps, method families, regions and
# censored positions, so the check answers "does the method work on THIS network".
# Truths are drawn from the fitted model (hyperparameters at their posterior means),
# plus stress cases. v5.2 (site and pooled classes) is run on the same data for comparison.
#
# Scenarios
#   A_fit     site slopes from the fitted hierarchy, shared region-year shocks, AR(1)
#   A_heavy   as A_fit but site deviations from t(3) (heavy tails, misspecified)
#   A_null    no trend anywhere (mu = tau = 0): every thickening/thinning call is false
#   B_fit     k_i from the fitted family means and spread, residual trends from the fitted
#             hierarchy
#   B_climate residual trend 0 everywhere: every "deeper/shallower than TDD" call is false
#   B_tdderr  as B_climate, but the model sees TDD with error (20 % of its interannual
#             variance): measures the attenuation that moves climate trend into residual
#   B_nok     k_i = 0 (ALT does not follow TDD), residual trends from the fitted hierarchy:
#             checks the model does not credit a trend to TDD

def free_jit():
    """Each fit compiles a new program; drop the compiled code so memory stays flat
    over many fits (without this a long run ran out of memory)."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()


def span_bins_sim(arr):
    d = arr["d"]
    sp = (d.groupby("site_id")["year"].max() - d.groupby("site_id")["year"].min() + 1)
    sp = sp.reindex(arr["sites"]["site_id"]).to_numpy()
    return pd.cut(sp, [0, 15, 20, 25, 60], labels=["12-15", "16-20", "21-25", "26+"])


def synthetic_x(arr, rng, sd_region=0.15, sd_site=0.05, trend=0.03):
    """log-TDD anomalies with a shared region-year part and a small trend (per decade)."""
    d = arr["d"]
    r_obs = arr["r_idx_site"][arr["s_idx"]]
    yrs = d["year"].to_numpy(int)
    cells = {}
    shared = np.array([cells.setdefault((r, y), rng.normal(0, sd_region)) for r, y in zip(r_obs, yrs)])
    x = shared + rng.normal(0, sd_site, len(d)) + trend * (yrs - 2005) / 10
    return x - pd.Series(x).groupby(arr["s_idx"]).transform("mean").to_numpy()


def with_x(arr, x):
    a = copy.copy(arr)
    a["x"] = x
    df = pd.DataFrame({"s": arr["s_idx"], "x": x, "t": arr["t"]})
    sx = df.groupby("s").apply(lambda g: np.polyfit(g.t, g.x, 1)[0], include_groups=False)
    a["sx"] = sx.reindex(range(arr["n_sites"])).to_numpy()
    a["tdd_trend_pct_dec"] = pct(a["sx"])
    a["x_sd"] = df.groupby("s")["x"].std().reindex(range(arr["n_sites"])).to_numpy()
    return a


def simulate(arr, truth, rng):
    """New y on the real design. truth: slope (site, log/decade), a, rho_f, sigma, sd_ry,
    optional k (site) and x_true (obs)."""
    d = arr["d"]
    s_idx, f_site = arr["s_idx"], arr["f_idx_site"]
    years = d["year"].to_numpy(int)
    y = np.empty(len(d))
    for i in range(arr["n_sites"]):
        idx = np.nonzero(s_idx == i)[0]
        yr = years[idx]
        full = np.arange(yr.min(), yr.max() + 1)
        rho = truth["rho_f"][f_site[i]]
        e = np.empty(len(full))
        e[0] = rng.normal()
        for k in range(1, len(full)):
            e[k] = rho * e[k - 1] + np.sqrt(1 - rho ** 2) * rng.normal()
        y[idx] = truth["sigma"][i] * e[np.searchsorted(full, yr)]
    r_obs = arr["r_idx_site"][s_idx]
    cells = {}
    g = np.array([cells.setdefault((r, yy), rng.normal(0, truth["sd_ry"])) for r, yy in zip(r_obs, years)])
    mean = truth["a"][s_idx] + truth["slope"][s_idx] * arr["t"] + g
    if "k" in truth:
        mean = mean + truth["k"][s_idx] * truth["x_true"]
    y = y + mean
    cens = arr["cens"]
    # a censored value is a bound below the true value
    y[cens] = y[cens] - np.abs(rng.normal(0, 0.05, cens.sum()))
    out = copy.copy(arr)
    out["y"] = y
    out["d"] = d.assign(alt_cm=np.exp(y))
    return out


def v52(arr, cfg):
    cfg = copy.deepcopy(cfg)
    cfg["nonlinearity"]["n_boot"] = 19     # breakpoints are not scored here
    res, net = run(arr["d"], cfg)
    res = res.set_index("site_id").reindex(arr["sites"]["site_id"])
    return res, net


def ols_site(arr, cfg, level):
    """Ordinary least squares of log ALT on year, site by site, with a t interval: the
    method of recent syntheses (per-site linear trend tested at p < 0.05, e.g.
    Streletskiy et al. 2026). No autocorrelation, no censoring, no pooling. Returns
    rate, interval and class in %/decade, classed with the same rules as v5.2 and v6."""
    m, rp = cfg["classes"]["stable_margin_pct_dec"], cfg["classes"]["rapid_pct_dec"]
    y, t, s_idx = np.asarray(arr["y"]), np.asarray(arr["t"]), np.asarray(arr["s_idx"])
    est, lo, hi = (np.full(arr["n_sites"], np.nan) for _ in range(3))
    cls = []
    for i in range(arr["n_sites"]):
        idx = s_idx == i
        ti, yi = t[idx], y[idx]
        n = len(ti)
        b, a = np.polyfit(ti, yi, 1)
        se = np.sqrt(np.sum((yi - a - b * ti) ** 2) / (n - 2) / np.sum((ti - ti.mean()) ** 2))
        q = stats.t.ppf(0.5 + level / 2, n - 2)
        est[i], lo[i], hi[i] = pct(b), pct(b - q * se), pct(b + q * se)
        cls.append(classify(est[i], lo[i], hi[i], m, rp))
    return est, lo, hi, np.asarray(cls)


def score(name, method, est, lo, hi, cls, true_rate, bins, mu_hat=None, mu_lo=None, mu_hi=None, mu_true=None):
    cover = (lo <= true_rate) & (true_rate <= hi)
    thick = np.isin(cls, ["rapid_thickening", "gradual_thickening"])
    thin = np.isin(cls, ["rapid_thinning", "gradual_thinning"])
    wrong = (thick & (true_rate <= 0)) | (thin & (true_rate >= 0))
    resolved = ~np.isin(cls, ["indeterminate"])
    row = {"scenario": name, "method": method, "coverage90": cover.mean(),
           "rmse_pct_dec": float(np.sqrt(np.mean((est - true_rate) ** 2))),
           "wrong_direction": wrong.mean(), "resolved": resolved.mean(),
           "called_trend": (thick | thin).mean(),
           "stable_when_|true|<2.5": float(np.mean(np.asarray(cls)[np.abs(true_rate) < 2.5] == "stable"))
           if (np.abs(true_rate) < 2.5).any() else np.nan}
    for b in ["12-15", "16-20", "21-25", "26+"]:
        m = np.asarray(bins == b)
        if m.any():
            row[f"rapid_{b}"] = np.mean(np.asarray(cls)[m] == "rapid_thickening")
            row[f"resolved_{b}"] = resolved[m].mean()
            row[f"coverage_{b}"] = cover[m].mean()
    if mu_hat is not None:
        row["network_err"] = mu_hat - mu_true
        row["network_cover95"] = float(mu_lo <= mu_true <= mu_hi)
    return row


def run_A(arr, cfg, fitted, scen, rng, reps, fit_kw):
    m, rp = cfg["classes"]["stable_margin_pct_dec"], cfg["classes"]["rapid_pct_dec"]
    bins = span_bins_sim(arr)
    rows = []
    for rep in range(reps):
        tr = {"a": arr["ybar"], "rho_f": fitted["rho_f"], "sigma": fitted["sigma"],
              "sd_ry": fitted["sd_ry"]}
        nr, ns = arr["n_regions"], arr["n_sites"]
        mu, tr_, ts = fitted["mu"], fitted["tau_region"], fitted["tau_site"]
        if scen == "A_null":
            slope = np.zeros(ns)
        elif scen == "A_heavy":
            beta = mu + tr_ * rng.normal(size=nr)
            slope = beta[arr["r_idx_site"]] + ts * rng.standard_t(3, ns) / np.sqrt(3)
        else:
            beta = mu + tr_ * rng.normal(size=nr)
            slope = beta[arr["r_idx_site"]] + ts * rng.normal(size=ns)
        tr["slope"] = slope
        sim = simulate(arr, tr, rng)
        true_rate = pct(slope)
        mu_true = float(np.mean(true_rate))
        for ry in [True, False]:
            mc = fit(sim, "A", region_year=ry, seed=rep, **fit_kw)
            s, net, rate = summarise_A(mc, sim, cfg)
            mu_draw = rate.mean(1)                 # network mean of the site rates
            row = score(scen, f"v6{'_ry' if ry else '_nory'}", s.rate_v6.to_numpy(), s.rate_v6_lo.to_numpy(),
                        s.rate_v6_hi.to_numpy(), s.class_v6.to_numpy(), true_rate, bins,
                        np.median(mu_draw), *np.quantile(mu_draw, [0.025, 0.975]), mu_true)
            row.update(rep=rep, **diagnostics(mc))
            rows.append(row)
            del mc
            free_jit()
        res, net = v52(sim, cfg)
        z = 1.959964
        rows.append({**score(scen, "v5.2_site", res.rate_pct_dec.to_numpy(), res.rate_lo.to_numpy(),
                             res.rate_hi.to_numpy(), res.class_site.to_numpy(), true_rate, bins), "rep": rep})
        zz = 1.644854
        rows.append({**score(scen, "v5.2_pooled", res.rate_pooled.to_numpy(),
                             (res.rate_pooled - zz * res.rate_pooled_sd).to_numpy(),
                             (res.rate_pooled + zz * res.rate_pooled_sd).to_numpy(),
                             res.class_pooled.to_numpy(), true_rate, bins,
                             net["mu"], net["mu"] - z * net["se_mu"], net["mu"] + z * net["se_mu"], mu_true),
                     "rep": rep})
        # per-site OLS, the method of recent syntheses: scored with its 90 % interval like
        # the others, plus the share of sites with a trend at p < 0.05 (95 % interval
        # excludes 0), the test those papers report
        est, lo, hi, cls = ols_site(sim, cfg, 0.90)
        _, lo95, hi95, _ = ols_site(sim, cfg, 0.95)
        rows.append({**score(scen, "ols_site", est, lo, hi, cls, true_rate, bins),
                     "trend_p05": float(np.mean((lo95 > 0) | (hi95 < 0))),
                     "wrong_p05": float(np.mean(((lo95 > 0) & (true_rate <= 0)) | ((hi95 < 0) & (true_rate >= 0)))),
                     "rep": rep})
        print(f"{scen} rep {rep}: " + "; ".join(f"{r['method']} cov {r['coverage90']:.2f} wrong {r['wrong_direction']:.3f} "
                                                f"resolved {r['resolved']:.2f}" for r in rows[-5:]), flush=True)
    return rows


def run_B(arr, cfg, fitted, scen, rng, reps, fit_kw, real_x):
    m, rp = cfg["classes"]["stable_margin_pct_dec"], cfg["classes"]["rapid_pct_dec"]
    bins = span_bins_sim(arr)
    rows = []
    ns, nr = arr["n_sites"], arr["n_regions"]
    for rep in range(reps):
        x_true = real_x if real_x is not None else synthetic_x(arr, rng)
        base = with_x(arr, x_true)
        if scen in ("B_climate", "B_tdderr"):
            d_site = np.zeros(ns)
        else:
            beta = fitted["mu_d"] + fitted["tau_region_d"] * rng.normal(size=nr)
            d_site = beta[arr["r_idx_site"]] + fitted["tau_site_d"] * rng.normal(size=ns)
        # k from the fitted model (family means and site spread; about 0.2 on run 7). Up
        # to 2026-11-04 this was N(0.4, 0.1), twice the fitted value.
        k = (np.zeros(ns) if scen == "B_nok" else
             fitted["k_family"][arr["f_idx_site"]] + fitted["tau_k"] * rng.normal(size=ns))
        tr = {"a": arr["ybar"], "rho_f": fitted["rho_f_B"], "sigma": fitted["sigma_B"],
              "sd_ry": 0.0, "slope": d_site, "k": k, "x_true": x_true}
        sim = simulate(base, tr, rng)
        if scen == "B_tdderr":
            sdx = np.std(x_true)
            x_obs = x_true + rng.normal(0, np.sqrt(0.25) * sdx, len(x_true))   # 20 % of var(x_obs)
            x_obs = x_obs - pd.Series(x_obs).groupby(arr["s_idx"]).transform("mean").to_numpy()
            sim = with_x(sim, x_obs)
        mc = fit(sim, "B", region_year=False, seed=rep, target_accept=0.95, **fit_kw)
        s, net = summarise_B(mc, sim, cfg, "hierarchical")
        true_resid = pct(d_site)
        true_exp = pct(k * base["sx"])
        row = score(scen, "v6_B_resid", s.resid_rate.to_numpy(), s.resid_lo.to_numpy(), s.resid_hi.to_numpy(),
                    s.class_resid_raw.to_numpy(), true_resid, bins)
        row["expected_coverage90"] = float(np.mean((s.expected_lo <= true_exp) & (true_exp <= s.expected_hi)))
        row["expected_bias"] = float(np.mean(s.expected_rate - true_exp))
        row["resid_bias"] = float(np.mean(s.resid_rate - true_resid))
        row["k_bias"] = float(np.mean(s.k_site - k))
        row["false_deeper"] = float(np.mean(s.class_resid == "deeper_than_TDD")) if scen in ("B_climate", "B_tdderr") else np.nan
        row.update(rep=rep, **diagnostics(mc))
        rows.append(row)
        del mc
        free_jit()
        print(f"{scen} rep {rep}: resid cov {row['coverage90']:.2f} wrong {row['wrong_direction']:.3f} "
              f"k bias {row['k_bias']:+.3f} resid bias {row['resid_bias']:+.2f} exp cov {row['expected_coverage90']:.2f}",
              flush=True)
    return rows


def fitted_values(arr, arrB, cfg, fit_kw):
    mc = fit(arr, "A", region_year=True, **fit_kw)
    smp = mc.get_samples()
    f = {"mu": float(np.mean(smp["mu"])), "tau_region": float(np.mean(smp["tau_region"])),
         "tau_site": float(np.mean(smp["tau_site"])),
         "rho_f": np.asarray(smp["rho_family"]).mean(0) * 0.95,
         "sigma": np.asarray(smp["sigma_site"]).mean(0),
         "sd_ry": float(np.mean(smp["sd_region_year"])) if "sd_region_year" in smp else 0.0}
    mcB = fit(arrB, "B", region_year=False, target_accept=0.95, **fit_kw)
    sb = mcB.get_samples()
    f.update(mu_d=float(np.mean(sb["mu"])), tau_region_d=float(np.mean(sb["tau_region"])),
             tau_site_d=float(np.mean(sb["tau_site"])), rho_f_B=np.asarray(sb["rho_family"]).mean(0) * 0.95,
             sigma_B=np.asarray(sb["sigma_site"]).mean(0),
             k_family=np.asarray(sb["k_family"]).mean(0), tau_k=float(np.mean(sb["tau_k"])))
    return f


def simulation_check(cfg, argv=None):
    """Simulated networks on the real design (see the section note above)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("annual_file")
    ap.add_argument("out_dir")
    ap.add_argument("tdd_file", nargs="?")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--scenarios", default="A_fit,A_heavy,A_null,B_fit,B_climate,B_tdderr,B_nok")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tag", default="", help="suffix for the output files (run scenarios in parts)")
    args = ap.parse_args(argv)
    import numpyro
    numpyro.set_host_device_count(args.chains)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(args.annual_file, low_memory=False)
    data, _ = load_series(annual, cfg)
    d = eligible_sites(data, cfg)
    fit_kw = dict(chains=args.chains, warmup=args.warmup, draws=args.draws)
    if args.tdd_file:
        tdd = pd.read_csv(args.tdd_file)
        arrB = build_arrays(d, cfg, tdd)
        arr = build_arrays(eligible_sites(arrB["d"].drop(columns=["tdd"]), cfg), cfg)
        real_x = arrB["x"]
    else:
        arr = build_arrays(d, cfg)
        arrB = with_x(arr, synthetic_x(arr, np.random.default_rng(1)))
        real_x = None
    assert arr["n_sites"] == arrB["n_sites"]
    fitted = fitted_values(arr, arrB, cfg, fit_kw)
    print("fitted truth:", {k: (np.round(v, 3) if np.ndim(v) == 0 else np.round(np.mean(v), 3)) for k, v in fitted.items()})

    rng = np.random.default_rng(cfg.get("random_seed", 42))
    rows = []
    for sc in args.scenarios.split(","):
        if sc.startswith("A_"):
            rows += run_A(arr, cfg, fitted, sc, rng, args.reps, fit_kw)
        else:
            rows += run_B(arr, cfg, fitted, sc, rng, args.reps, fit_kw, real_x)
        pd.DataFrame(rows).to_csv(out / f"aldi_v6_simulation_reps{args.tag}.csv", index=False)
    df = pd.DataFrame(rows)
    summ = df.drop(columns=["rep"]).groupby(["scenario", "method"], sort=False).mean(numeric_only=True)
    summ.to_csv(out / f"aldi_v6_simulation{args.tag}.csv")
    txt = (f"ALDI v6 simulation check, {args.reps} networks per scenario, real design "
           f"({arr['n_sites']} sites, {len(arr['y'])} site-years); TDD: "
           f"{'real' if real_x is not None else 'synthetic'}\n\n"
           + summ.round(3).T.to_string())
    (out / f"aldi_v6_simulation{args.tag}.txt").write_text(txt)
    print(txt)



# =========================================================================================
# RUNNING THE STEP
# =========================================================================================
NEEDED = ["numpy", "scipy", "pandas", "yaml", "numpyro", "jax", "matplotlib"]


def require(modules):
    """Stop before anything is touched if this Python lacks a package."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        pip = " ".join({"yaml": "pyyaml"}.get(m, m) for m in missing)
        sys.exit(f"This Python ({sys.version.split()[0]}) is missing: {pip}. Nothing was changed.\n"
                 f"Run it with  py -3.12 {Path(sys.argv[0]).name}  or install them:\n"
                 f"    \"{sys.executable}\" -m pip install {pip}")


def fresh(final):
    """An empty <final>_building folder; the output goes there until the run succeeds."""
    work = final.with_name(final.name + "_building")
    if work.exists():
        shutil.rmtree(work)                  # leftover of an interrupted run
    work.mkdir(parents=True)
    return work


def publish(work, final):
    """The run succeeded: the current output becomes data/previous/<name> (one copy is
    kept), and the new one takes its place."""
    if final.exists():
        prev = DATA / "previous" / final.name
        if prev.exists():
            shutil.rmtree(prev)
        prev.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(final), str(prev))
    shutil.move(str(work), str(final))


def stage(final, job):
    """Run job(work_folder); publish the output only if the job finished without error."""
    work = fresh(final)
    try:
        job(work)
    except SystemExit as err:
        if err.code:
            sys.exit(f"\nStopped: {err.code}\n{final} was NOT changed; the partial output is in {work}")
    except Exception as err:
        print(f"\nStopped: {err!r}\n{final} was NOT changed; the partial output is in {work}")
        raise
    publish(work, final)
    print(f"\nWritten: {final}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--power", action="store_true", help="also run B with NASA POWER temperature")
    ap.add_argument("--validate", action="store_true", help="also run the simulation check")
    ap.add_argument("--sensitivity", action="store_true",
                    help="also run the sensitivity runs (one choice changed at a time; about 30-40 min)")
    ap.add_argument("--checks", action="store_true",
                    help="also the paper checks (detection time, standard per-site test, residual by ice "
                         "and method, rates in cm; about a minute, no refit)")
    ap.add_argument("--no-main", action="store_true",
                    help="do not rerun the main fit: --power, --sensitivity, --checks and --validate use the "
                         "existing data/3_aldi (and its temperature folder)")
    ap.add_argument("--reps", type=int, default=8, help="simulated networks per scenario (--validate)")
    ap.add_argument("--scenarios", default=None, help="simulation scenarios, comma-separated (--validate)")
    ap.add_argument("--annual", type=Path, default=ANNUAL, help="ALT input (default: step 1 output)")
    ap.add_argument("--ee-dir", type=Path, default=EE_DIR, help="Earth Engine files (default: step 2 output)")
    ap.add_argument("--out", type=Path, default=OUT, help="output folder (default data/3_aldi)")
    ap.add_argument("--spatial", action="store_true",
                    help="only the spatial checks, on the existing results (about a minute; writes "
                         "aldi_v6_spatial.txt and aldi_v6_spatial_sites.csv next to them)")
    a = ap.parse_args()
    if a.spatial:
        require(["numpy", "scipy", "pandas"])
        f = a.out / "aldi_v6_results.csv"
        if not f.exists():
            sys.exit(f"{f} not found: run 3_aldi.py first")
        res, lines = spatial_checks(pd.read_csv(f))
        (a.out / "aldi_v6_spatial.txt").write_text("\n".join(lines))
        res[["site_id", "location_id", "pf_extent", "pf_ice"]].to_csv(a.out / "aldi_v6_spatial_sites.csv", index=False)
        print("\n".join(lines))
        return
    require(NEEDED)
    cfg = load_settings()

    if not a.annual.exists():
        sys.exit(f"{a.annual} not found: run 1_alt_target.py first")
    if not list(a.ee_dir.glob("t2m_daily_*.csv")):
        sys.exit(f"No t2m_daily files in {a.ee_dir}: run 2_covariates.py first")

    def main_run(work):
        temp = work / "temperature"
        era5_daily_series([str(a.ee_dir), str(temp)])
        tdd_tables([str(a.annual), str(temp), "--source", "cache"])
        run_aldi(cfg, a.annual, work, temp / "tdd_site_years.csv", temp / "tdd_all_years.csv",
                 temp / "weather_site_years.csv")
    if a.no_main:
        if not (a.out / "temperature" / "tdd_site_years.csv").exists():
            sys.exit(f"--no-main: {a.out / 'temperature'} has no TDD tables; run without --no-main first")
    else:
        stage(a.out, main_run)

    if a.power:
        def power_run(work):
            tdd_tables([str(a.annual), str(POWER_CACHE)])          # downloads only new sites
            run_aldi(cfg, a.annual, work, POWER_CACHE / "tdd_site_years.csv",
                     POWER_CACHE / "tdd_all_years.csv")
        stage(a.out.with_name(a.out.name + "_power"), power_run)

    if a.sensitivity:
        mc_cfg = cfg["model"]
        fit_kw = dict(chains=mc_cfg["chains"], warmup=mc_cfg["warmup"], draws=mc_cfg["draws"],
                      target_accept=mc_cfg["target_accept"], seed=mc_cfg["seed"])
        import numpyro
        numpyro.set_host_device_count(mc_cfg["chains"])

        def sensitivity_run(work):
            sensitivity_runs(cfg, a.annual, a.out / "temperature", work, fit_kw)
        stage(a.out.with_name(a.out.name + "_sensitivity"), sensitivity_run)

    if a.checks:
        def checks_run(work):
            paper_checks(cfg, a.annual, a.out, work, a.out.with_name(a.out.name + "_sensitivity"))
        stage(a.out.with_name(a.out.name + "_checks"), checks_run)

    if a.validate:
        temp = a.out / "temperature"
        argv = [str(a.annual), "", str(temp / "tdd_site_years.csv"), "--reps", str(a.reps)]
        if a.scenarios:
            argv += ["--scenarios", a.scenarios]

        def validate_run(work):
            argv[1] = str(work)
            simulation_check(cfg, argv)
        stage(a.out.with_name(a.out.name + "_validation"), validate_run)

    print(f"\nDone. Start with {a.out / 'aldi_v6_summary.txt'}")


if __name__ == "__main__":
    main()
