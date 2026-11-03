# ALDI v6: hierarchical trends (A) and climate-adjusted index (B)

v5.2 (`Scripts/aldi`) stays as it is. v6 uses the same sample rules, class margins and
regions, so both classify exactly the same site-years, and the v6 run reruns v5.2 for the
comparison table.

## Why

v5.2 judges each site from its own 10-35 values. Most short records are then
"indeterminate", and class shares depend on record length. v5.2 pools in a second step
(site rate and SE into a REML random-effects model), but that step treats the AR(1)
coefficient as known and sees censored years only through ranks. In simulation its
site intervals cover 81-86 % instead of 90 %.

v6 fits one Bayesian model to all series, so the uncertainty of every part (site
noise, autocorrelation, regional rate, censored values) carries into each site's
rate. B then asks how much of each site's trend the air-temperature record explains.

## A. Hierarchical trend model

```
log ALT_it = a_i + b_i t_it + g_r(i),year + e_it        t in decades from the site's mean year
b_i  ~ N(beta_region, tau_site)                           sites shrink toward their region
beta_r ~ N(mu, tau_region)                                regions shrink toward the network
g    ~ N(0, sd_ry)                                        shared region-year anomaly (cells with >= 2 sites)
e_it AR(1) in calendar years, rho per method family, exact for gaps (rho^gap)
sd(e) per site, log-normal around a family mean
```

- Lower-bound years (thaw reached the probe or the deepest sensor) are latent values
  above the bound, so they count as "at least this deep", with no made-up value.
- Rates in % per decade = 100 (exp(b) - 1). Classes use the v5.2 rules on the 90 %
  credible interval (stable: inside ±5 %/dec; thickening/thinning: excludes 0; rapid:
  median beyond 10 %/dec). Every band probability is reported (`p6_*`).
- Sensitivity runs: without region-year effects; Student-t(4) site deviations.

## B. Climate-adjusted ALDI

Stefan: ALT = E sqrt(TDD), so log ALT = log E + k log TDD with k = 0.5 in theory.

```
log ALT_it = a_i + k_i x_it + d_i t_it + e_it
x_it = log TDD_it minus the site mean;  k_i ~ N(k_family, tau_k), k_family ~ N(0.5, 0.25)
d_i hierarchical by region as b_i in A
```

Because d_i takes the linear trend, k_i is estimated from the year-to-year swings of
TDD around their trend (the within-site sensitivity), not from two series that both rise.
Per site:

- climate-expected trend = k_i x (trend of log TDD over the site's own years)
- residual trend = d_i: thaw that deepens faster (or slower) than the TDD record explains
- residual classes: `deeper_than_TDD` (interval > 0), `consistent_with_TDD` (interval
  inside ±5 %/dec), `shallower_than_TDD` (< 0), `not_resolved`

Variants (config `climate.variants`): `main` (above); `kfixed` (k = 0.5, pure Stefan);
`lag` (adds last year's TDD: a warm summer melts ice and wets the soil, which deepens
the next thaw); `air` (adds last year's TDD and the preceding winter's freezing index:
all that air temperature can explain).

TDD: NASA POWER daily mean air temperature (MERRA-2, 0.5 x 0.625 deg), see
`fetch_tdd.py` for the window rules (probe: 1 Jan to the survey date or the site / network
median survey day; thaw tube and temperature ALT: whole year). ERA5-Land from the GEE
script can replace it (one CSV per site in the cache folder).

## Run

```
pip install numpy scipy pandas pyyaml numpyro jax matplotlib
python fetch_tdd.py   annual.csv  tdd_dir                       # ~10 min, downloads; writes both TDD tables
python run_aldi_v6.py aldi_v6_config.yaml annual.csv out_dir  tdd_dir/tdd_site_years.csv  tdd_dir/tdd_all_years.csv
python validate_aldi_v6.py ../aldi/aldi_config.yaml annual.csv sim_dir tdd_dir/tdd_site_years.csv --reps 8
```

`tdd_all_years.csv` (site_id, year, tdd_full_year, fdd for every calendar year) is only
needed for the `lag` and `air` variants. The main run takes about 10 minutes on 4 cores
(9 model fits); the simulation check 1-2 hours.

Outputs: `aldi_v6_results.csv` (one row per site: v5.2 columns, v6 rate, interval,
class and band probabilities, B expected / residual / total trends and classes per
variant), `aldi_v6_network.csv`, `aldi_v6_summary.txt`, three figures,
`aldi_v6_period_check.csv`.

## Results on run 7 (229 sites, 4,550 site-years, 41 regions)

| class | v5.2 site | v5.2 pooled | v6 | v6 no region-year | v6 Student-t |
|---|---|---|---|---|---|
| rapid thickening | 39 | 35 | 55 | 53 | 55 |
| gradual thickening | 19 | 66 | 60 | 50 | 61 |
| stable | 11 | 23 | 14 | 16 | 13 |
| gradual thinning | 2 | 1 | 1 | 1 | 1 |
| rapid thinning | 4 | 0 | 0 | 0 | 0 |
| indeterminate | 154 | 104 | 99 | 109 | 99 |

- **Network:** +9.0 %/dec (95 % 6.5-11.7) as the mean over regions, +7.1 (6.1-8.1)
  as the mean over sites (v5.2: +7.9, 5.6-10.3). 144 sites thicken with probability
  >= 0.9; one site has probability < 0.1.
- **Persistence and noise** (probe / thaw tube / temperature): rho 0.41 / 0.54 / 0.80,
  noise 8 / 6 / 13 % of ALT; shared region-year anomaly 5.7 %.
- **Record length:** rapid 30 % of 12-20-yr records vs 13 % of 26+ yr. In simulation
  (truth independent of length) v6 gives 43 % vs 47 %, so this is the sites, not the
  method: short records are mostly recent temperature cables and probes, long ones CALM
  probes and slower thaw tubes; within shared regions short records are 1.2 %/dec faster.
- **B:** k = 0.22 (0.19-0.24); median TDD trend +2.4 %/dec. Mean site trends
  (expected / residual / total, %/dec): main 0.7 / 5.6 / 6.3; k = 0.5 1.2 / 5.0 / 6.2;
  + last-year TDD 0.9 / 5.3 / 6.3; + winter FDD 1.1 / 5.1 / 6.2. Residual classes (main):
  87 deeper than TDD, 15 consistent, 1 shallower, 126 not resolved; 67 sites are deeper
  than TDD in the main, k = 0.5 and all-air-temperature variants alike.

Simulation (8 networks per scenario, run-7 design, real TDD):

| | v6 | v5.2 site | v5.2 pooled |
|---|---|---|---|
| 90 % interval coverage (fitted truth) | 0.89 | 0.84 | 0.81 |
| rate RMSE, %/dec | 4.7 | 10.8 | 5.3 |
| wrong-direction calls | 0.7 % | 2.0 % | 1.0 % |
| false trend calls when no trend anywhere | 0 % | 12 % | 1.5 % |
| network-rate bias, %/dec | 0.0 | | -0.9 |

B: residual coverage 0.90; no false "deeper than TDD" calls when every trend is
climatic, also with TDD error of 20 % of its variance (which lowers k by 0.10 and moves
~0.2-0.3 %/dec into the residual).

## Caveats

- **k below Stefan.** k is the year-to-year sensitivity. Error in the coarse TDD
  attenuates it (simulation: about -0.1 with 20 % error variance), and multi-year
  ground-ice and moisture memory is not in a one-year swing. The `kfixed` (k = 0.5) and
  `air` variants bound this; the residual trend stays large in all of them.
- **TDD is not the only climate.** Snow, summer rain and soil moisture are not in
  "expected"; a positive residual means "not explained by air temperature", not
  "not climatic". The driver panel is where those enter.
- **Shrinkage borrows from neighbours.** A short record in a fast region is pulled up.
  `rate_pct_dec` (v5.2 site) stays in the table for statements about one record.
- **Same sample as v5.2**, including U19 1990-2014 and the 60 N cut (open decisions).
