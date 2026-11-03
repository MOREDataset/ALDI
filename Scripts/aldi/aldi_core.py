"""ALDI v5.2: trend classes for active-layer thickness series.

What changes from the v4 notebook (ALDI_Calculation_final.ipynb), and why:

1. No gap filling. Theil-Sen and Kendall's S work on irregular years, so PCHIP values
   no longer enter the test as if they were measurements.
2. Lower-bound years (thaw reached the probe or the deepest sensor) stay in as
   right-censored values (Akritas-Theil-Sen, Akritas et al. 1995). Dropping them
   removes the deepest years of deepening sites and pulls their slopes toward zero.
3. Rates are in % of the site's ALT per decade with fixed margins (stable +/-5,
   rapid 10). v4 used the 50th / 80th percentiles of the sample's slopes in cm/yr,
   so a site's class depended on which other sites were in the sample.
4. "No trend" is split into "stable" (the 90 % interval lies inside +/-5 %/dec, an
   equivalence test) and "indeterminate" (the record cannot tell). v4 put both in
   one class, which made short records look stable or rapid.
5. The interval comes from inverting Kendall's S with an AR(1) variance correction
   (Hamed and Rao 1998 with a fitted AR(1) instead of the significant sample lags,
   which are unstable for 10-30 values). For uncensored data without autocorrelation
   it is exactly Gilbert's (1987) rank interval for the Sen slope, and it extends to
   censored years, which the v4 block bootstrap cannot. In simulations
   (validate_aldi.py) both intervals hold their 90 % level when rho = 0 and both fall
   to 80-87 % when rho = 0.5 if rho is estimated per site.
   Since v5.2 the AR(1) coefficient is one value for the whole network, fitted by
   REML on all series together (pooled_ar1). The per-site lag-1 estimate is biased
   low (a fitted trend and gaps both pull it down; at n = 12 and true rho = 0.5 it
   averages about 0.3) and very noisy (SD about 0.3 at n = 12), which is why the
   interval was too narrow. With the pooled value the simulated coverage is 0.90-0.93
   for rho 0 to 0.5. The per-site value is still reported (ar1_rho_site).
6. Reversal is tested with a broken-stick fit and an AR(1) bootstrap of the sup-F
   statistic, so the breakpoint year is estimated, not fixed at the first and last
   seven years, and the test has a known false-positive rate. It is a flag.
7. Partial pooling: site rates are shrunk toward their region with a two-level
   random-effects model. Short noisy records shrink most, which weakens the link
   between record length and "rapid" found in the 2026-10-25 review. The same model
   gives the network trend with regions as clusters.

Units: years in calendar years, ALT in cm, slopes in cm/yr, rates in %/decade.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

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
