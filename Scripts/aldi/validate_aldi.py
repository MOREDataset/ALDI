"""Simulation check of the ALDI v5.1 statistics against known truth.

    python validate_aldi.py aldi_config.yaml out_dir

Synthetic series: ALT = 60 cm * (1 + rate/1000 * (year - mid)) + AR(1) noise with
SD 7.5 % of ALT (the median residual SD of the 221 real series), 10 % of years missing.
For each record length, AR(1) coefficient and true rate it reports:
  - coverage of the nominal 90 % interval for the slope: v5.1 (Kendall inversion with
    AR(1) correction) and v4 (moving-block bootstrap, blocks of max(2, min(5, n//4)))
  - share of each v5.1 class
  - false-positive rate of the broken-stick test when the truth is a straight line
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aldi_core as A  # noqa: E402

ALT0_CM = 60.0
NOISE_FRAC = 0.075
MISSING_FRAC = 0.10


def simulate(n_years, rate, rho, rng):
    years = np.arange(2000, 2000 + n_years, dtype=float)
    keep = rng.random(n_years) >= MISSING_FRAC
    keep[[0, -1]] = True
    e = np.empty(n_years)
    e[0] = rng.standard_normal()
    for k in range(1, n_years):
        e[k] = rho * e[k - 1] + np.sqrt(1 - rho ** 2) * rng.standard_normal()
    mid = years.mean()
    y = ALT0_CM * (1 + rate / 1000 * (years - mid)) + NOISE_FRAC * ALT0_CM * e
    return years[keep], y[keep]


def v4_block_bootstrap_ci(x, y, ci_level, n_boot, rng):
    """The v4 notebook's interval (bootstrap_site + percentile CI), without gap filling."""
    n = len(x)
    b = max(2, min(5, n // 4))
    nb = int(np.ceil(n / b))
    draws = np.empty(n_boot)
    for k in range(n_boot):
        starts = rng.integers(0, n - b + 1, size=nb)
        idx = np.concatenate([np.arange(s, s + b) for s in starts])[:n]
        xb, yb = x[idx], y[idx]
        i, j = np.triu_indices(n, 1)
        ok = xb[j] != xb[i]
        draws[k] = np.median((yb[j][ok] - yb[i][ok]) / (xb[j][ok] - xb[i][ok]))
    a = (1 - ci_level) / 2
    return np.quantile(draws, a), np.quantile(draws, 1 - a)


def main():
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["random_seed"])
    cc, tc, nl = cfg["classes"], cfg["trend"], cfg["nonlinearity"]
    n_rep = 300
    rows = []
    for n_years in [12, 18, 25, 35]:
        for rho in [0.0, 0.3, 0.5]:
            for rate in [0.0, 5.0, 10.0, 20.0]:
                cov51 = cov4 = 0
                classes = []
                nl_pos = nl_tested = 0
                for _ in range(n_rep):
                    x, y = simulate(n_years, rate, rho, rng)
                    c = np.zeros(len(x), bool)
                    true_slope = ALT0_CM * rate / 1000
                    tr = A.ats_trend(x, y, c, tc["ci_level"], tc["ar1_max"])
                    cov51 += tr.lo <= true_slope <= tr.hi
                    lo4, hi4 = v4_block_bootstrap_ci(x, y, tc["ci_level"], 200, rng)
                    cov4 += lo4 <= true_slope <= hi4
                    k = 1000 / tr.intercept_ref
                    classes.append(A.classify(tr.slope * k, tr.lo * k, tr.hi * k,
                                              cc["stable_margin_pct_dec"], cc["rapid_pct_dec"]))
                    if rate in (0.0, 10.0) and n_years >= 2 * nl["min_segment_obs"]:
                        bp = A.breakpoint_test(x, y, nl["min_segment_obs"], 199, tr.rho, rng)
                        if np.isfinite(bp["nl_p"]):
                            nl_tested += 1
                            nl_pos += bp["nl_p"] < nl["alpha"]
                vc = pd.Series(classes).value_counts(normalize=True)
                rows.append({"n_years": n_years, "rho": rho, "true_rate_pct_dec": rate,
                             "coverage_v51": cov51 / n_rep, "coverage_v4_bootstrap": cov4 / n_rep,
                             **{f"share_{k}": round(float(vc.get(k, 0.0)), 3) for k in A.CLASSES},
                             "nl_false_positive": nl_pos / nl_tested if nl_tested else np.nan})
                print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "aldi_v51_simulation.csv", index=False)
    print(df.groupby("rho")[["coverage_v51", "coverage_v4_bootstrap", "nl_false_positive"]].mean())


if __name__ == "__main__":
    main()
