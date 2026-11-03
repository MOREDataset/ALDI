"""Run ALDI v5.2 on the multi-source ALT file.

    python run_aldi.py aldi_config.yaml [annual_file] [output_dir]

The two optional arguments override the paths in the config (Colab vs local runs).
Writes aldi_v52_results.csv, aldi_v52_removed_site_years.csv, aldi_v52_summary.txt and
a copy of the config used.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aldi_core as A  # noqa: E402


def summary_text(res, net, cfg, removed):
    out = []
    out.append(f"ALDI v5.2  numpy {np.__version__}  scipy {scipy.__version__}  pandas {pd.__version__}")
    out.append(f"sites classified: {len(res)}; site-years: {int(res.n_obs.sum())} uncensored, "
               f"{int(res.n_censored.sum())} censored (lower bounds)")
    out.append(f"site-years removed: {len(removed)}")
    for r, n in removed["reason"].value_counts().items():
        out.append(f"  {r}: {n}")
    for col in ["class_site", "class_pooled"]:
        out.append(f"\n{col}:")
        vc = res[col].value_counts()
        for k in A.CLASSES:
            n = int(vc.get(k, 0))
            out.append(f"  {k:20s} {n:4d} ({100 * n / len(res):5.1f} %)")
    z = 1.959964
    out.append(f"\nnetwork rate (REML, regions as clusters): {net['mu']:.2f} %/dec "
               f"(95 % CI {net['mu'] - z * net['se_mu']:.2f} to {net['mu'] + z * net['se_mu']:.2f}); "
               f"{net['n_sites']} sites in {net['n_regions']} regions of <= {cfg['pooling']['region_km']} km")
    out.append(f"AR(1) for the trend intervals: {net['ar1_method']}; pooled REML rho = "
               f"{net['ar1_rho_pooled']:.2f}; per-site estimates median {res.ar1_rho_site.median():.2f}")
    out.append(f"between-region SD {net['tau_region']:.2f}, between-site SD {net['tau_site']:.2f} %/dec")
    out.append(f"nonlinear (broken stick p < {cfg['nonlinearity']['alpha']}): {int(res.flag_nonlinear.sum())}, "
               f"of which reversals: {int(res.flag_reversal.sum())}  (expected false positives "
               f"~{cfg['nonlinearity']['alpha'] * res.nl_p.notna().sum():.0f} of {int(res.nl_p.notna().sum())} tested)")
    out.append(f"trend q < 0.10 (Benjamini-Hochberg): {int((res.q_trend < 0.10).sum())}")
    out.append(f"median minimum detectable trend: {res.mdt_pct_dec.median():.1f} %/dec")
    out.append(f"survey-date drift flags: {int(res.flag_doy_drift.sum())}")
    bins = pd.cut(res.span_years, [0, 15, 20, 25, 40], labels=["12-15", "16-20", "21-25", "26-36"])
    for col in ["class_site", "class_pooled"]:
        out.append(f"\n{col} by record span (row %):")
        out.append((pd.crosstab(bins, res[col], normalize="index") * 100).round(0)
                   .reindex(columns=A.CLASSES, fill_value=0).to_string())
    return "\n".join(out)


def main():
    cfg_path = Path(sys.argv[1])
    cfg = yaml.safe_load(cfg_path.read_text())
    annual_file = sys.argv[2] if len(sys.argv) > 2 else cfg["input"]["annual_file"]
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else cfg["input"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    annual = pd.read_csv(annual_file, low_memory=False)
    data, removed = A.load_series(annual, cfg)
    res, net = A.run(data, cfg)

    res.to_csv(out_dir / "aldi_v52_results.csv", index=False)
    removed.to_csv(out_dir / "aldi_v52_removed_site_years.csv", index=False)
    txt = summary_text(res, net, cfg, removed)
    (out_dir / "aldi_v52_summary.txt").write_text(txt)
    (out_dir / "aldi_v52_config_used.yaml").write_text(cfg_path.read_text())
    print(txt)


if __name__ == "__main__":
    main()
