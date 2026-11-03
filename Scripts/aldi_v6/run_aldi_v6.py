"""Run ALDI v6 (hierarchical trends, A) and the climate-adjusted index (B).

    python run_aldi_v6.py aldi_v6_config.yaml  annual.csv  out_dir  [tdd_site_years.csv [tdd_all_years.csv]]

annual.csv is alt_multisource_annual.csv (same input as v5.2); tdd_site_years.csv comes
from fetch_tdd.py. Without the TDD file only A runs. tdd_all_years.csv (every calendar
year per site) adds the B variants with the previous year's TDD and the winter FDD.

Outputs (out_dir)
  aldi_v6_results.csv     one row per site: v5.2 classes, v6 rates and classes (A),
                          climate-expected and residual trends (B)
  aldi_v6_network.csv     network parameters (median and 95 % interval)
  aldi_v6_summary.txt     counts, crosstabs, sensitivity runs, diagnostics
  aldi_v6_fig_*.png       figures
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "aldi"))
import aldi_core as A  # noqa: E402
import aldi_v6_models as M  # noqa: E402

ORDER = A.CLASSES
RESID_ORDER = ["deeper_than_TDD", "consistent_with_TDD", "shallower_than_TDD", "not_resolved"]


def load_cfg(path):
    path = Path(path)
    v6 = yaml.safe_load(path.read_text())
    base = yaml.safe_load((path.parent / v6["base_config"]).read_text())
    base.update({k: v for k, v in v6.items() if k != "base_config"})
    return base


def span_bins(span):
    return pd.cut(span, [0, 15, 20, 25, 60], labels=["12-15", "16-20", "21-25", "26+"])


def counts(series, order):
    vc = series.value_counts()
    return {k: int(vc.get(k, 0)) for k in order}


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
        full = A.ats_trend(g.year, g.alt_cm, g.censored, 0.9, 0.85, rho)
        part = A.ats_trend(rec.year, rec.alt_cm, rec.censored, 0.9, 0.85, rho)
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


def main():
    cfg_path = Path(sys.argv[1])
    cfg = load_cfg(cfg_path)
    annual_file, out = sys.argv[2], Path(sys.argv[3])
    tdd_file = sys.argv[4] if len(sys.argv) > 4 else None
    tdd_all_file = sys.argv[5] if len(sys.argv) > 5 else None
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
    data, removed = A.load_series(annual, cfg)
    d = M.eligible_sites(data, cfg)

    # --- v5.2 on the same data (reference) ---
    t0 = time.time()
    res52, net52 = A.run(data, cfg)
    log.append(f"v5.2 rerun: {len(res52)} sites, network {net52['mu']:.2f} %/dec ({time.time() - t0:.0f} s)")

    # --- A: main and sensitivity fits ---
    arr = M.build_arrays(d, cfg)
    runs = {}
    for name, ry, sd in [("main", mc_cfg["region_year"], mc_cfg["slope_dist"]),
                         ("no_region_year" if mc_cfg["region_year"] else "region_year", not mc_cfg["region_year"], mc_cfg["slope_dist"]),
                         ("student_slopes", mc_cfg["region_year"], "student")]:
        t0 = time.time()
        mc = M.fit(arr, "A", slope_dist=sd, region_year=ry, priors=pri, **fit_kw)
        s, net, rate = M.summarise_A(mc, arr, cfg)
        runs[name] = dict(mc=mc, s=s, net=net, rate=rate, diag=M.diagnostics(mc), sec=time.time() - t0,
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
        arrB = M.build_arrays(d, cfg, tdd)
        # A on exactly B's site-years, so A and B totals compare like with like
        arrAB = M.build_arrays(M.eligible_sites(arrB["d"].drop(columns=["tdd"]), cfg), cfg)
        mcAB = M.fit(arrAB, "A", region_year=cfg["climate"]["region_year"], priors=pri, **fit_kw)
        sAB, _, _ = M.summarise_A(mcAB, arrAB, cfg)
        bruns = {}
        tdd_all = pd.read_csv(tdd_all_file) if tdd_all_file else None
        for v in cfg["climate"]["variants"]:
            if v.get("extra") and tdd_all is None:
                log.append(f"B {v['name']}: skipped (needs tdd_all_years.csv)")
                continue
            t0 = time.time()
            p2 = dict(pri, k_fixed=cfg["climate"]["k_fixed"])
            if v.get("extra"):
                ex = M.extra_climate(tdd_all, v["extra"])
                arrV = M.build_arrays(d, cfg, tdd, ex, v["extra"])
            else:
                arrV = arrB
            mcB = M.fit(arrV, "B", k_mode=v["k_mode"], region_year=cfg["climate"]["region_year"], priors=p2, **fit_kw)
            sB, netB = M.summarise_B(mcB, arrV, cfg, v["k_mode"])
            bruns[v["name"]] = dict(s=sB, net=netB, diag=M.diagnostics(mcB), sec=time.time() - t0,
                                    n_sites=arrV["n_sites"], n_obs=len(arrV["y"]))
            log.append(f"B {v['name']}: {bruns[v['name']]['diag']}  ({bruns[v['name']]['sec']:.0f} s; "
                       f"{arrV['n_sites']} sites, {len(arrV['y'])} site-years)")
        sB = bruns["main"]["s"].copy()
        sB["tdd_var_explained"] = 1 - (sB.sigma_pct_B / sAB.set_index("site_id").reindex(sB.site_id).sigma_pct.to_numpy()) ** 2
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
        res["two_axis"] = np.where(res.class_resid.isna(), np.nan,
                                   res.class_v6.astype(str) + " | " + res.class_resid.astype(str))

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
        for i, f in enumerate(M.FAMILIES):
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
            for i, f in enumerate(M.FAMILIES):
                netrows[f"B_{km}_rho_{f}"] = q(n["rho_family"][:, i])
            for key in n:
                if key.startswith("k_") and key.endswith("_family") and key != "k_family":
                    for i, f in enumerate(M.FAMILIES):
                        netrows[f"B_{km}_{key[:-7]}_{f}"] = q(n[key][:, i])
            if "k0" in n:
                netrows[f"B_{km}_k0"] = q(n["k0"])
                netrows[f"B_{km}_tau_k"] = q(n["tau_k"])
                for i, f in enumerate(M.FAMILIES):
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
            L.append(f"  {vn:12s} expected {np.median(r['net']['expected_mean']):5.2f}  residual "
                     f"{np.median(r['net']['resid_mean']):5.2f}  total {np.median(r['net']['total_mean']):5.2f}  "
                     f"({r['n_sites']} sites)")
        L.append("\nv6 class (rows) x residual class (columns):")
        L.append(pd.crosstab(b.class_v6, b.class_resid).reindex(index=ORDER, columns=RESID_ORDER, fill_value=0).to_string())
        L.append(f"\nmedian TDD trend {b.tdd_trend_pct_dec.median():.1f} %/dec; median k {b.k_site.median():.2f}; "
                 f"median share of ALT variance explained by TDD {100 * b.tdd_var_explained.median():.0f} %")
        L.append(f"median 90 % interval width: total (A, same years) vs residual: "
                 f"{np.median(res.rate_v6_hi - res.rate_v6_lo):.1f} vs {np.median(b.resid_hi - b.resid_lo):.1f} %/dec")
        L.append("\nby method family (median):")
        L.append(b.groupby("family")[["k_site", "tdd_trend_pct_dec", "expected_rate", "resid_rate", "rate_v6",
                                      "tdd_var_explained"]].median().round(2).to_string())
        L.append("\nresidual class by family:")
        L.append(pd.crosstab(b.family, b.class_resid).reindex(columns=RESID_ORDER, fill_value=0).to_string())
    txt = "\n".join(L)
    (out / f"{pre}_summary.txt").write_text(txt)
    (out / f"{pre}_config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    figures(res, out, pre, have_b)
    print(txt)


if __name__ == "__main__":
    main()
