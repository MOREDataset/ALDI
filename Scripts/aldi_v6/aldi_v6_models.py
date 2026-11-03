"""ALDI v6: one hierarchical Bayesian model for all ALT series (A), and a climate-
adjusted version that splits each site's trend into a part expected from thawing
degree-days and a residual part (B).

Why (review of 2026-11-03, doc "ALDI paper: publishability and stronger analyses"):
v5.2 classifies each site from its own 10-35 values, so most short records are
"indeterminate" and class shares depend on record length. v5.2 already pools in a
second step (site rate and SE -> REML random effects), but the site SE there treats
rho as known and the censored years only through ranks.

A. Hierarchical trend model (one step, all uncertainty propagated)
   log ALT_it = a_i + b_i * t_it + e_it          t in decades from the site's mean year
   b_i  = beta_region(i) + tau_site * z_i         sites shrink toward their region
   beta_r = mu + tau_region * z_r                 regions shrink toward the network
   e_it  AR(1) in calendar years, rho per method family, exact for gaps (rho^gap)
   sd of e_it per site, log-normal around a family mean (probe, tube and temperature
   ALT have different noise)
   Lower-bound years (thaw reached the probe or deepest sensor) are latent values
   above the bound, so they inform the trend with the right direction and no made-up
   value.
   Rates in % per decade = 100 (exp(b) - 1). Classes use the v5.2 rules applied to
   the 90 % credible interval, and every band probability is reported.

B. Climate-adjusted model (Stefan: ALT = E sqrt(TDD), so log ALT = log E + k log TDD)
   log ALT_it = a_i + k_i * x_it + d_i * t_it + e_it
   x_it = log TDD_it minus the site mean of log TDD
   k_i  = k_family + tau_k * z_i,  k_family ~ N(k0, .), k0 ~ N(0.5, 0.25)
   d_i  hierarchical by region as b_i in A.
   Because d_i takes the linear trend, k_i is identified by the year-to-year swings of
   TDD around their trend (the within-site sensitivity), not by two series that both
   rise. For each site:
       climate-expected trend = k_i * (OLS slope of x_it on t over the site's years)
       residual trend         = d_i
       total                  = expected + residual
   A residual > 0 means the ground thaws deeper over time than the TDD record explains
   (ground ice loss and subsidence, talik, vegetation or snow change, or TDD error);
   < 0 means it lags the TDD record.
   Option k_mode: fixed sets k_i = 0.5 for every site (pure Stefan), as a sensitivity.

Caveat built into the design: error in reanalysis TDD attenuates k, which moves part
of the climate trend into the residual. validate_aldi_v6.py measures this with
simulated TDD error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aldi"))
import aldi_core as A  # noqa: E402  (v5.2 data rules, regions, classes)

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


def build_arrays(d: pd.DataFrame, cfg: dict, tdd: pd.DataFrame | None = None) -> dict:
    """Model inputs. d: output of eligible_sites (one row per site-year).

    tdd: table site_id, year, method, tdd (deg C day). Site-years without TDD are
    dropped for model B only (reported in the summary).
    """
    d = d.copy()
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
    sites["region_id"] = A.assign_regions(sites["lat"], sites["lon"], cfg["pooling"]["region_km"])
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
               t=t, y=y, cens=cens, prev=prev, gap=gap, ybar=ybar, tdd_dropped=dropped)
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
        z_s = numpyro.sample("z_site", dist.StudentT(4.0, 0, 1).expand([ns]))
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
            k0 = numpyro.sample("k0", dist.Normal(0.5, 0.25))
            sd_kf = numpyro.sample("sd_k_family", dist.HalfNormal(0.15))
            zkf = numpyro.sample("z_k_family", dist.Normal(0, 1).expand([nf]))
            k_f = numpyro.deterministic("k_family", k0 + sd_kf * zkf)
            tau_k = numpyro.sample("tau_k", dist.HalfNormal(0.2))
            zk = numpyro.sample("z_k", dist.Normal(0, 1).expand([ns]))
            k = k_f[f_site] + tau_k * zk
        k = numpyro.deterministic("k_site", k)
        mean = mean + k[s_idx] * x

    if region_year and arr["n_ry"] > 0:
        sd_ry = numpyro.sample("sd_region_year", dist.HalfNormal(0.1))
        z_ry = numpyro.sample("z_region_year", dist.Normal(0, 1).expand([arr["n_ry"]]))
        g = jnp.concatenate([sd_ry * z_ry, jnp.zeros(1)])   # index -1 -> 0
        mean = mean + g[jnp.asarray(arr["ry_idx"])]

    # censored years: latent value above the bound (log units)
    y = y_obs
    ci = np.nonzero(cens)[0]
    if len(ci):
        off = numpyro.sample("cens_offset", dist.HalfNormal(p.get("cens_sd", 0.5)).expand([len(ci)]))
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
            rhat.append(np.nanmax(np.atleast_1d(s[k]["r_hat"])))
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
    return [A.classify(m, min(l, m), max(h, m), margin, rapid) for m, l, h in zip(med, lo, hi)], med, lo, hi


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
    for k, v in band_probs(rate, m, rp).items():
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
    if k_mode != "fixed":
        net["k0"] = np.asarray(smp["k0"])
        net["k_family"] = np.asarray(smp["k_family"])
        net["tau_k"] = np.asarray(smp["tau_k"])
    return s, net


def ci_str(v, level=0.95, fmt="{:.2f}"):
    lo, hi = np.quantile(v, [(1 - level) / 2, 1 - (1 - level) / 2])
    return f"{fmt.format(np.median(v))} ({fmt.format(lo)} to {fmt.format(hi)})"
