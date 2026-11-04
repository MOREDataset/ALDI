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
the next thaw); `memory` (adds the TDD of each of the two previous years); `air` (adds
last year's TDD and the preceding winter's freezing index); `weather` (`air` plus Nov-Apr
snow depth and JJA rain); `weather_memory` (`memory` plus freezing index, snow and rain:
all that the weather record can explain). Degree-day terms enter as logs; snow and rain
as site anomalies in units of one network SD (they can be near 0 in a dry year).

ALT keeps a memory of warm summers. In `memory` (ERA5-Land, run 7) this year's k is 0.29
for probes and 0.20-0.21 for thaw tubes and temperature ALT; the previous summer adds
0.05-0.08 and the one before 0.02-0.05. The long-run response to a lasting TDD change
(k plus the lag coefficients, `k_long_*` in the results) is 0.40 for probes and
0.27-0.29 for the others, closer to the Stefan 0.5. In a within-site regression with
lags up to four years, lags of three and four years (0.03-0.06) were close to the
coefficient of next year's TDD (0.03), a placebo, since next summer cannot cause this
year's thaw, and did not fall off steadily; they cannot be told from slow variation that
ALT and TDD share, so they are left out. The same slow variation may add a few
hundredths to the two lags kept, so `main` and `memory` bracket what air temperature
explains. A lag of L years needs temperature from L years before the first ALT year;
without it those first years drop out of the variant (the summary says how many).

Not in any variant, on purpose: soil moisture, NDVI, NDWI and other surface states. They
respond to the thaw (mediators), so adjusting for them would remove part of the change B
measures. A positive `weather` residual therefore means change that weather does not
explain: ground ice loss and subsidence, taliks, vegetation, disturbance. That is where
the causal driver analysis starts.

Temperature source. Main: ERA5-Land (0.1 deg) from the Earth Engine exports of
`../covariates/gee_causal_features.py`, corrected to the site elevation in summer
(-6.5 K/km May-Sep, none in winter because of inversions). Robustness: NASA POWER
(MERRA-2, 0.5 x 0.625 deg), which needs no account. B uses each site's year-to-year TDD
swings, which are coherent over hundreds of km, so the grid matters less than for
absolute values; but a coarse cell mixes in sea or mountain air at coastal and mountain
sites, and TDD error lowers k and moves a little trend into the residual (simulation).
`fetch_tdd.py` holds the window rules (probe: 1 Jan to the survey date or the site /
network median survey day; thaw tube and temperature ALT: whole year). The ERA5-Land
exports run by ALT year (October to September), so the last exported year stops on
30 September. The days after it are filled with the site's mean of the same calendar days
in the other years (`tdd_rule` ending in `+tail_filled`; a median 1-2 % of the
whole-year TDD), so the last year of thaw-tube and temperature records and of late
surveys is kept. Before 2026-11-04 those years were dropped, and surveys up to mid-October
were scaled up from the January-September mean (0.2-2 % too high).

Outputs that changed on 2026-11-04 (review): the share of year-to-year ALT variance
explained by TDD (`tdd_var_explained`) is now 1 minus the ratio of the residual variance
with and without TDD on the same site-years (before, a ratio of fitted noise SDs that
also moved with the AR(1) coefficient, which gave negative shares for temperature
records). The summary adds network means with a region bootstrap interval: the model
intervals of a mean over sites treat the monitored sites as the population, so they are
too narrow for a statement about the permafrost region.

## Run

```
pip install numpy scipy pandas pyyaml numpyro jax matplotlib

# ERA5-Land (main). First, in Colab, cell 2b of GEE_causal_features_colab.ipynb exports
# the static, era5 and t2m_daily groups to its own Drive folder (B_FOLDER, now
# ALDI_GEE_exports_v2), ALT years 1988-2025: the two-year TDD lags of 1990 need 1988. Then:
python era5land_for_b.py  path/to/ALDI_GEE_exports_v2  tdd_era5               # daily series + weather table
python fetch_tdd.py       annual.csv  tdd_era5  --source cache                 # TDD tables from those series
python run_aldi_v6.py     aldi_v6_config.yaml annual.csv out_era5 \
       tdd_era5/tdd_site_years.csv tdd_era5/tdd_all_years.csv tdd_era5/weather_site_years.csv

# NASA POWER (robustness; no account, ~10 min of downloads; no weather variant)
python fetch_tdd.py   annual.csv  tdd_power
python run_aldi_v6.py aldi_v6_config.yaml annual.csv out_power  tdd_power/tdd_site_years.csv  tdd_power/tdd_all_years.csv

python validate_aldi_v6.py ../aldi/aldi_config.yaml annual.csv sim_dir tdd_era5/tdd_site_years.csv --reps 8
```

Use separate folders for the two sources: `fetch_tdd.py` never re-downloads a site that
already has a file in `daily_t2m/`.

`tdd_all_years.csv` (site_id, year, tdd_full_year, fdd for every calendar year) is only
needed for the variants with lags or FDD. `fetch_tdd.py` prints which site-years have no
TDD and from which calendar year the lags exist. The main run takes about 25 minutes on
4 cores (11 model fits); the simulation check 1-2 hours.

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
  as the mean over sites (v5.2: +7.9, 5.6-10.3). Resampling whole regions widens the
  site-mean interval to 5.2-9.3: the model interval treats these 229 sites as the
  population. 144 sites thicken with probability >= 0.9; one site has probability < 0.1.
  The rate depends on the window, and so does the site set: site mean +8.8 over
  2000-2024 (205 sites), +6.6 over 2005-2024 (177 sites). Every region counts once in the
  region mean, so large slow regions (Mackenzie Delta, 35 sites, +2.2; North Slope, 21
  sites, +1.9) weigh as much as West Siberian regions of 2-3 sites at +16 to +18.
- **Persistence and noise** (probe / thaw tube / temperature): rho 0.41 / 0.54 / 0.80,
  noise 8 / 6 / 13 % of ALT; shared region-year anomaly 5.7 %.
- **Record length:** rapid 30 % of 12-20-yr records vs 13 % of 26+ yr. In simulation
  (truth independent of length) v6 gives 43 % vs 47 %, so this is the sites, not the
  method: short records are mostly recent temperature cables and probes, long ones CALM
  probes and slower thaw tubes; within shared regions short records are 1.2 %/dec faster.
- **B (ERA5-Land TDD, 221 sites; 8 sites are missing from the Earth Engine export).**
  This year's k = 0.20 (0.18-0.23; probe 0.26, thaw tube 0.17, temperature 0.18); median
  TDD trend +3.9 %/dec; TDD explains a median 15 % of the year-to-year ALT variance
  (below 0 at 46 sites). Mean site trends in %/dec with region-bootstrap 95 % intervals
  (total 6.3-6.4 in every variant):

  | variant | expected | residual | sites deeper than predicted |
  |---|---|---|---|
  | main (this summer's TDD) | 1.3 (0.8-2.1) | 5.1 (3.1-7.4) | 77 |
  | kfixed (k = 0.5) | 2.4 (1.3-4.0) | 4.0 (2.0-6.1) | 58 |
  | lag (+ last summer) | 1.6 (1.0-2.7) | 4.7 (2.7-7.0) | 73 |
  | memory (+ the two previous summers) | 1.9 (1.1-3.2) | 4.3 (2.4-6.6) | 64 |
  | air (+ last summer, winter FDD) | 2.1 (1.5-3.1) | 4.2 (2.2-6.4) | 64 |
  | weather (air + snow depth, summer rain) | 2.6 (1.8-3.5) | 3.7 (1.9-5.8) | 55 |
  | weather_memory | 3.0 (2.0-4.3) | 3.2 (1.5-5.2) | 50 |

  Long-run k with memory: probe 0.40, thaw tube 0.29, temperature 0.27. With NASA POWER
  TDD (229 sites) the expected part is smaller, because POWER's TDD rose a median
  1.6 %/dec more slowly at the same sites: main 0.7 (0.3-1.3) expected and 5.5 (3.4-8.1)
  residual, memory 1.1 (0.5-2.0) and 5.0 (2.9-7.6); TDD explains a median 13 % there.

Simulation (8 networks per scenario, run-7 design, real TDD):

| | v6 | v5.2 site | v5.2 pooled |
|---|---|---|---|
| 90 % interval coverage (fitted truth) | 0.89 | 0.84 | 0.81 |
| rate RMSE, %/dec | 4.7 | 10.8 | 5.3 |
| wrong-direction calls | 0.7 % | 2.0 % | 1.0 % |
| false trend calls when no trend anywhere | 0 % | 12 % | 1.5 % |
| network-rate bias, %/dec | 0.0 | | -0.9 |

B (rerun 2026-11-04 with the fitted k, which the first run had doubled; 4 networks per
scenario, ERA5-Land TDD): residual coverage 0.90; no false "deeper than TDD" calls when
every trend is climatic, also with TDD error of 20 % of its variance. That error, about
four times the real one (ERA5-Land vs POWER), lowers k by 0.06 (a quarter) and moves
0.2-0.3 %/dec from expected into residual. With k = 0 (ALT ignores TDD) the residual
coverage stays 0.90 and the expected trend is unbiased.

## Caveats

- **k is a short-run sensitivity.** Within a year k is 0.2-0.3. The two previous
  summers add 0.06-0.11 (`memory`), so the long-run response is 0.27-0.40 (probes
  0.40), closer to Stefan's 0.5. TDD error lowers k by only about a tenth: ERA5-Land and
  POWER anomalies correlate at 0.92 (reliability of ERA5-Land about 0.95), and using
  POWER as an instrument moves the within-site k from 0.27 to 0.30. `main` and
  `memory` bracket what air temperature explains.
- **The TDD trend depends on the reanalysis.** At the same sites ERA5-Land TDD rose a
  median 1.6 %/dec faster than NASA POWER (site trends correlate at 0.84), which about
  doubles the expected part. This is the largest uncertainty in "expected"; the
  residual stays above 3 %/dec with either source.
- **TDD is not the only climate.** In `main`, snow and summer rain are not in
  "expected", so a positive residual means "not explained by air temperature". The
  `weather` variant adds them; soil moisture stays out (mediator).
- **Shrinkage borrows from neighbours.** A short record in a fast region is pulled up.
  `rate_pct_dec` (v5.2 site) stays in the table for statements about one record.
- **Same sample as v5.2**, including U19 1990-2014 and the 60 N cut (open decisions).
