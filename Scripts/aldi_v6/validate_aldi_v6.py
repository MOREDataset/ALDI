"""Simulation check of ALDI v6 on the real sampling design.

    python validate_aldi_v6.py  ../aldi/aldi_config.yaml  annual.csv  out_dir  [tdd_site_years.csv]  [--reps 10]

Every synthetic network keeps the real sites, years, gaps, method families, regions and
censored positions, so the check answers "does the method work on THIS network".
Truths are drawn from the fitted model (hyperparameters at their posterior means),
plus stress cases. v5.2 (site and pooled classes) is run on the same data for comparison.

Scenarios
  A_fit     site slopes from the fitted hierarchy, shared region-year shocks, AR(1)
  A_heavy   as A_fit but site deviations from t(3) (heavy tails, misspecified)
  A_null    no trend anywhere (mu = tau = 0): every thickening/thinning call is false
  B_fit     k_i ~ N(0.4, 0.1), residual trends from the fitted hierarchy
  B_climate residual trend 0 everywhere: every "deeper/shallower than TDD" call is false
  B_tdderr  as B_climate, but the model sees TDD with error (20 % of its interannual
            variance): measures the attenuation that moves climate trend into residual
  B_nok     k_i = 0 (ALT does not follow TDD), residual trends from the fitted hierarchy:
            checks the model does not credit a trend to TDD
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "aldi"))
import aldi_core as A  # noqa: E402
import aldi_v6_models as M  # noqa: E402


def free_jit():
    """Each fit compiles a new program; drop the compiled code so memory stays flat
    over many fits (without this a long run ran out of memory)."""
    import gc
    import jax
    jax.clear_caches()
    gc.collect()


def span_bins(arr):
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
    a["tdd_trend_pct_dec"] = M.pct(a["sx"])
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
    res, net = A.run(arr["d"], cfg)
    res = res.set_index("site_id").reindex(arr["sites"]["site_id"])
    return res, net


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
    bins = span_bins(arr)
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
        true_rate = M.pct(slope)
        mu_true = float(np.mean(true_rate))
        for ry in [True, False]:
            mc = M.fit(sim, "A", region_year=ry, seed=rep, **fit_kw)
            s, net, rate = M.summarise_A(mc, sim, cfg)
            mu_draw = rate.mean(1)                 # network mean of the site rates
            row = score(scen, f"v6{'_ry' if ry else '_nory'}", s.rate_v6.to_numpy(), s.rate_v6_lo.to_numpy(),
                        s.rate_v6_hi.to_numpy(), s.class_v6.to_numpy(), true_rate, bins,
                        np.median(mu_draw), *np.quantile(mu_draw, [0.025, 0.975]), mu_true)
            row.update(rep=rep, **M.diagnostics(mc))
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
        print(f"{scen} rep {rep}: " + "; ".join(f"{r['method']} cov {r['coverage90']:.2f} wrong {r['wrong_direction']:.3f} "
                                                f"resolved {r['resolved']:.2f}" for r in rows[-4:]), flush=True)
    return rows


def run_B(arr, cfg, fitted, scen, rng, reps, fit_kw, real_x):
    m, rp = cfg["classes"]["stable_margin_pct_dec"], cfg["classes"]["rapid_pct_dec"]
    bins = span_bins(arr)
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
        k = np.zeros(ns) if scen == "B_nok" else rng.normal(0.4, 0.1, ns)
        tr = {"a": arr["ybar"], "rho_f": fitted["rho_f_B"], "sigma": fitted["sigma_B"],
              "sd_ry": 0.0, "slope": d_site, "k": k, "x_true": x_true}
        sim = simulate(base, tr, rng)
        if scen == "B_tdderr":
            sdx = np.std(x_true)
            x_obs = x_true + rng.normal(0, np.sqrt(0.25) * sdx, len(x_true))   # 20 % of var(x_obs)
            x_obs = x_obs - pd.Series(x_obs).groupby(arr["s_idx"]).transform("mean").to_numpy()
            sim = with_x(sim, x_obs)
        mc = M.fit(sim, "B", region_year=False, seed=rep, target_accept=0.95, **fit_kw)
        s, net = M.summarise_B(mc, sim, cfg, "hierarchical")
        true_resid = M.pct(d_site)
        true_exp = M.pct(k * base["sx"])
        row = score(scen, "v6_B_resid", s.resid_rate.to_numpy(), s.resid_lo.to_numpy(), s.resid_hi.to_numpy(),
                    s.class_resid_raw.to_numpy(), true_resid, bins)
        row["expected_coverage90"] = float(np.mean((s.expected_lo <= true_exp) & (true_exp <= s.expected_hi)))
        row["expected_bias"] = float(np.mean(s.expected_rate - true_exp))
        row["resid_bias"] = float(np.mean(s.resid_rate - true_resid))
        row["k_bias"] = float(np.mean(s.k_site - k))
        row["false_deeper"] = float(np.mean(s.class_resid == "deeper_than_TDD")) if scen in ("B_climate", "B_tdderr") else np.nan
        row.update(rep=rep, **M.diagnostics(mc))
        rows.append(row)
        del mc
        free_jit()
        print(f"{scen} rep {rep}: resid cov {row['coverage90']:.2f} wrong {row['wrong_direction']:.3f} "
              f"k bias {row['k_bias']:+.3f} resid bias {row['resid_bias']:+.2f} exp cov {row['expected_coverage90']:.2f}",
              flush=True)
    return rows


def fitted_values(arr, arrB, cfg, fit_kw):
    mc = M.fit(arr, "A", region_year=True, **fit_kw)
    smp = mc.get_samples()
    f = {"mu": float(np.mean(smp["mu"])), "tau_region": float(np.mean(smp["tau_region"])),
         "tau_site": float(np.mean(smp["tau_site"])),
         "rho_f": np.asarray(smp["rho_family"]).mean(0) * 0.95,
         "sigma": np.asarray(smp["sigma_site"]).mean(0),
         "sd_ry": float(np.mean(smp["sd_region_year"])) if "sd_region_year" in smp else 0.0}
    mcB = M.fit(arrB, "B", region_year=False, target_accept=0.95, **fit_kw)
    sb = mcB.get_samples()
    f.update(mu_d=float(np.mean(sb["mu"])), tau_region_d=float(np.mean(sb["tau_region"])),
             tau_site_d=float(np.mean(sb["tau_site"])), rho_f_B=np.asarray(sb["rho_family"]).mean(0) * 0.95,
             sigma_B=np.asarray(sb["sigma_site"]).mean(0))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("annual_file")
    ap.add_argument("out_dir")
    ap.add_argument("tdd_file", nargs="?")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--scenarios", default="A_fit,A_heavy,A_null,B_fit,B_climate,B_tdderr,B_nok")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tag", default="", help="suffix for the output files (run scenarios in parts)")
    args = ap.parse_args()
    import numpyro
    numpyro.set_host_device_count(args.chains)

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(args.annual_file, low_memory=False)
    data, _ = A.load_series(annual, cfg)
    d = M.eligible_sites(data, cfg)
    fit_kw = dict(chains=args.chains, warmup=args.warmup, draws=args.draws)
    if args.tdd_file:
        tdd = pd.read_csv(args.tdd_file)
        arrB = M.build_arrays(d, cfg, tdd)
        arr = M.build_arrays(M.eligible_sites(arrB["d"].drop(columns=["tdd"]), cfg), cfg)
        real_x = arrB["x"]
    else:
        arr = M.build_arrays(d, cfg)
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


if __name__ == "__main__":
    main()
