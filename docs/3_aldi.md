# Step 3: ALDI v6, hierarchical trends (A) and climate-adjusted index (B) (`3_aldi.py`)

v5.2 (described at the end of this note) is part of `3_aldi.py`. v6 uses the same sample rules, class margins and
regions, so both classify exactly the same site-years, and the v6 run reruns v5.2 for the
comparison table.

## Why

v5.2 judges each site from its own 10-35 values. Most short records are then
"indeterminate", and class shares depend on record length. v5.2 pools in a second step
(site rate and SE into a REML random-effects model), but that step treats the AR(1)
coefficient as known and sees censored years only through ranks. In simulation its
site intervals cover 82-86 % instead of 90 %.

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
  above the bound, so they count as "at least this deep", with no made-up value. Since
  7 November 2026 the latent value has no prior beyond "above the bound" (the standard
  censored likelihood). Before, a HalfNormal(0.5) prior on log(latent / bound) held it
  near the bound; where the sensors reach only 10 cm (SALTSIL 2017-2022, Parks Canada
  C28 2015, ALT 90-190 cm) that pulled the latent ALT down to 10-20 cm and turned
  SALTSIL's rate from about +10 into +3 %/dec. The HalfNormal priors are sensitivity runs.
- Rates in % per decade = 100 (exp(b) - 1). Classes use the v5.2 rules on the 90 %
  credible interval (stable: inside ±5 %/dec; thickening/thinning: excludes 0; rapid:
  median beyond 10 %/dec). Every band probability is reported (`p6_*`).
- Sensitivity runs: without region-year effects; Student-t(4) site deviations; and, since
  14 November 2026, a trend-unpooled fit (run `site_only`; called the site-only fit until
  the third review of the paper): each series' trend has its own N(0, 1)
  prior (log units per decade, flat over any plausible rate) and there is no network or
  regional trend; noise model, censoring and region-year term are those of the main fit.
  Against the main fit it isolates what pooling the trends adds to a site's interval and
  P(thickening); the per-site Sen + AR(1) and OLS tests also change the estimator, and their
  P(thickening), Phi(rate / SE), is a normal approximation, not a posterior.
- Network summaries (`aldi_v6_network_summaries.csv`, Table 2 of the paper): mu, the
  equal-weight mean of the regional trends, and means over equal-area cells of 100, 250 and
  500 km, over places and over series, each computed in every posterior draw (median and
  95 % interval). Cells: Lambert azimuthal equal-area projection centred on the North Pole
  (sphere, R = 6371.0088 km), cell = floor(x / size), floor(y / size) from the pole, each
  cell weighted once (1 / series in it). Places: series within 1 km (single linkage).

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

Variants (`SETTINGS`, `climate: variants`): `main` (above); `kfixed` (k = 0.5, pure Stefan);
`lag` (adds last year's TDD: a warm summer melts ice and wets the soil, which deepens
the next thaw); `memory` (adds the TDD of each of the two previous years); `air` (adds
last year's TDD and the preceding winter's freezing index); `weather` (`air` plus Nov-Apr
snow depth and JJA rain); `weather_memory` (`memory` plus freezing index, snow and rain:
all that the weather record can explain). Degree-day terms enter as logs; snow and rain
as site anomalies in units of one network SD (they can be near 0 in a dry year).

ALT keeps a memory of warm summers. In `memory` (ERA5-Land, run of 12 November 2026)
this year's k is 0.28 for probes, 0.20 for thaw tubes and 0.26 for temperature ALT; the
previous summer adds 0.07-0.08 and the one before 0.02-0.05. The long-run response to a
lasting TDD change (k plus the lag coefficients, `k_long_*` in the results) is 0.40 for
probes, 0.29 for thaw tubes and 0.36 for temperature ALT, closer to the Stefan 0.5. In a
within-site regression with lags up to four years (a check on run 7), lags of three and
four years (0.03-0.06) were close to the coefficient of next year's TDD (0.03), a
placebo, since next summer cannot cause this year's thaw, and did not fall off steadily;
they cannot be told from slow variation that ALT and TDD share, so they are left out. The
same slow variation may add a few hundredths to the two lags kept, so `main` and `memory`
bracket what air temperature explains. A lag of L years needs temperature from L years
before the first ALT year; without it those first years drop out of the variant (the
summary says how many).

Not in any variant, on purpose: soil moisture, NDVI, NDWI and other surface states. They
respond to the thaw (mediators), so adjusting for them would remove part of the change B
measures. A positive `weather` residual therefore means change that weather does not
explain: ground ice loss and subsidence, taliks, vegetation, disturbance. That is where
the causal driver analysis starts.

Temperature source. Main: ERA5-Land (0.1 deg) from the Earth Engine exports of
`2_covariates.py`, corrected to the site elevation in summer
(-6.5 K/km May-Sep, none in winter because of inversions). Robustness: NASA POWER
(MERRA-2, 0.5 x 0.625 deg), which needs no account. B uses each site's year-to-year TDD
swings, which are coherent over hundreds of km, so the grid matters less than for
absolute values; but a coarse cell mixes in sea or mountain air at coastal and mountain
sites, and TDD error lowers k and moves a little trend into the residual (simulation).
`3_aldi.py` (`tdd_tables`) holds the window rules (probe: 1 Jan to the survey date or the site /
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

See the top of `3_aldi.py` (and `README.md` in the main folder). Every setting is at the top of the script, with the reason for it.

`py -3.12 3_aldi.py` builds the daily temperature series and TDD tables from the Earth Engine files of step 2 (`data/3_aldi/temperature/`) and runs v5.2, A and every B variant (about 10-30 minutes). `--power` adds the NASA POWER run (no account; the daily files are kept in `data/raw/nasa_power/`), `--validate` the simulation check (1-2 hours), `--sensitivity` the sensitivity runs (about 1 hour, `data/3_aldi_sensitivity/`), `--checks` the paper checks (about a minute, no refit, `data/3_aldi_checks/`; see below), `--mean-check` 50 more simulated networks each of A_fit and A_ppc for the interval of the series mean (model A only, seed 2026, about 25 minutes; `aldi_v6_simulation_mean_reps.csv` in `data/3_aldi_validation/`, apart from the scenario files). `--no-main` reuses the existing `data/3_aldi` instead of refitting it. Since 14 November 2026 (sixth review) the main fit also writes the Monte Carlo standard errors (Vehtari et al. 2021, the quantile method of ArviZ; checked to agree with ArviZ 1.3 to the last digit) of each series' median and interval ends (`rate_v6_mcse`, `rate_v6_lo_mcse`, `rate_v6_hi_mcse`), the margin of its class (`class_margin`: distance of the deciding interval end from its boundary, %/dec), `class_borderline` (margin below 1 %/dec), `class_mc_sensitive` (deciding end within 2 Monte Carlo SE of the boundary) and the same for the rapid threshold (`rapid_margin`, `rapid_mc_sensitive`), and rows `A_main_network_rate_pct_dec_mcse` and `A_main_mean_site_rate_pct_dec_mcse` in `aldi_v6_network.csv` (Monte Carlo SE of the median and of the 2.5 and 97.5 % quantiles). Outputs in `data/3_aldi/`: `aldi_v6_results.csv` (one row per site: v5.2 columns, v6 rate, interval, class and band probabilities, B expected / residual / total trends and classes per variant), `aldi_v6_network.csv`, `aldi_v6_summary.txt`, three figures, `aldi_v6_period_check.csv`.

## Results of 12 November 2026 (230 sites, 4,614 site-years, 44 regions)

Run of 12 November 2026 on the PC (`py -3.12 3_aldi.py --power --sensitivity --checks
--validate`, about 1.5 h) on the step-1 build of the same day (358 sites, 6,605 valid
site-years, with the Isit' and NCL02 fixes of `docs/1_alt_target.md`), with censored years
flat above the bound (see A above). 75 of the 4,614 site-years are lower bounds, and 13 of
the 44 regions hold a single site. Every fit of the main run: R-hat <= 1.007, effective
sample size >= 831, no divergent transitions. The same sample fitted in the cloud on
11 November gave the same network rate (8.82 %/dec) and site mean (7.15) but class counts
that differ by up to three (58 / 60 / 13 / 1 / 98): sites near a class threshold can
change class with floating-point differences in the sampler, so a class count can differ
by a few between computers.

| class | v5.2 site | v5.2 pooled | v6 | v6 no region-year | v6 Student-t |
|---|---|---|---|---|---|
| rapid thickening | 39 | 33 | 57 | 53 | 55 |
| gradual thickening | 19 | 73 | 63 | 54 | 63 |
| stable | 10 | 22 | 13 | 14 | 13 |
| gradual thinning | 1 | 1 | 1 | 1 | 1 |
| rapid thinning | 4 | 0 | 0 | 0 | 0 |
| indeterminate | 157 | 101 | 96 | 108 | 98 |

- **Network:** +8.8 %/dec (95 % 6.6-11.6) as the mean over regions (mu), +7.1 (6.1-8.2)
  as the mean over sites (v5.2: +7.7, 5.5-10.0). Resampling whole regions widens the
  site-mean interval to 5.3-9.2: the model interval treats these 230 sites as the
  population. 150 sites thicken with probability >= 0.9; one site has probability < 0.1.
  The rate depends on the window, and so does the site set: site mean +8.8 over
  2000-2024 (204 sites; sensitivity runs below). Every region counts once in the region
  mean, so large slow regions (Mackenzie Delta and coast, 34 sites, +2.5; North Slope
  foothills, 21 sites, +2.1) weigh as much as the regions of 2-3 sites at Nadym, Urengoy,
  Norilsk and Vorkuta at +16 to +17.
- **Persistence and noise** (probe / thaw tube / temperature): rho 0.41 / 0.54 / 0.72,
  noise 8 / 6 / 13 % of ALT; shared region-year anomaly 5.7 %. Median 90 % interval width
  13.1 %/dec (v5.2 site 19.6, v5.2 pooled 10.5).
- **Record length:** rapid 32 % of 12-20-yr records vs 14 % of 26+ yr. In simulation
  (truth independent of length) v6 gives 39-47 % at every length, fewer in short records,
  so this is the sites, not the method: short records are mostly recent temperature cables
  and probes, long ones CALM probes and slower thaw tubes; within the 19 regions that hold
  both, short records are 1.1 %/dec faster.
- **B (ERA5-Land TDD, all 230 sites, 4,614 site-years).** This year's k = 0.22
  (0.20-0.24; probe 0.26, thaw tube 0.17, temperature 0.23); median TDD trend +4.0 %/dec;
  TDD explains a median 15 % of the year-to-year ALT variance (below 0 at 45 sites). Mean
  site trends in %/dec with region-bootstrap 95 % intervals (total 6.5-6.6 in every
  variant):

  | variant | expected | residual | sites deeper than predicted |
  |---|---|---|---|
  | main (this summer's TDD) | 1.3 (0.8-2.1) | 5.2 (3.2-7.4) | 78 |
  | kfixed (k = 0.5) | 2.6 (1.6-4.1) | 4.0 (2.1-6.1) | 59 |
  | lag (+ last summer) | 1.7 (1.1-2.7) | 4.8 (2.9-7.1) | 74 |
  | memory (+ the two previous summers) | 2.1 (1.3-3.3) | 4.4 (2.5-6.7) | 69 |
  | air (+ last summer, winter FDD) | 2.2 (1.6-3.2) | 4.2 (2.4-6.4) | 66 |
  | weather (air + snow depth, summer rain) | 2.7 (2.0-3.6) | 3.8 (2.0-5.8) | 60 |
  | weather_memory | 3.2 (2.3-4.3) | 3.3 (1.6-5.4) | 54 |

  Long-run k with memory: probe 0.40, thaw tube 0.29, temperature 0.36 (weather_memory:
  0.45 / 0.29 / 0.37). **NASA POWER** (`data/3_aldi_power`, 230 sites; no weather
  variants): POWER's TDD rose more slowly at the same sites (median +2.7 vs +4.0 %/dec), so
  the expected part is smaller: main 0.7 (0.3-1.3) expected and 5.7 (3.6-8.1) residual
  (93 sites deeper than predicted), memory 1.2 (0.6-2.1) and 5.2 (3.1-7.7); TDD explains a
  median 14 % there (15 % with ERA5-Land).

### Sensitivity runs (`--sensitivity`, `data/3_aldi_sensitivity/`)

Each run changes one choice and refits A (main), B (main) and B (weather_memory); every
fit R-hat <= 1.013, no divergences (one divergence in `temp_interval_all`). Means over sites in %/dec; mu = network rate (mean over
regions); declustered = 250 km equal-area cells count once. `aldi_v6_fig_sensitivity.png`
shows the same.

| run | sites | rapid / gradual thick. / stable / indet. | mu (95 %) | site mean | declustered | B expected | B residual | B residual (weather_memory) | deeper than TDD |
|---|---|---|---|---|---|---|---|---|---|
| main | 230 | 57 / 63 / 13 / 96 | 8.8 (6.6-11.6) | 7.05 | 9.8 | 1.33 | 5.24 | 3.40 | 78 |
| censored HalfNormal(0.5) above bound | 230 | 55 / 61 / 13 / 100 | 8.7 | 6.97 | 9.7 | 1.33 | 5.13 | 3.29 | 78 |
| censored HalfNormal(1.0) | 230 | 56 / 61 / 13 / 99 | 8.6 | 6.99 | 9.7 | 1.33 | 5.17 | 3.33 | 77 |
| no censored years | 228 | 58 / 55 / 14 / 100 | 8.4 | 6.87 | 9.4 | 1.29 | 5.07 | 3.25 | 73 |
| undated surveys: TDD to 31 Aug | 230 | (A unchanged) | 8.8 | 7.05 | 9.8 | 1.28 | 5.28 | 3.53 | 81 |
| without coarse / high-dz / sea-cell sites | 170 | 41 / 31 / 10 / 88 | 8.4 (5.6-11.6) | 6.63 | 9.3 | 0.91 | 5.63 | 3.72 | 57 |
| without fire-disturbed BNZ plots | 222 | 50 / 60 / 13 / 98 | 8.8 | 6.87 | 9.8 | 1.36 | 4.96 | 3.18 | 70 |
| one series per place | 174 | 43 / 48 / 11 / 72 | 8.5 (6.2-11.1) | 7.28 | 9.4 | 1.37 | 5.59 | 3.91 | 60 |
| window 2000-2024 | 204 | 63 / 64 / 1 / 76 | 9.6 (7.2-12.3) | 8.76 | 10.4 | 1.91 | 6.28 | 4.11 | 96 |
| 8 instead of 10 years | 240 | 61 / 60 / 15 / 103 | 8.3 | 6.99 | 9.4 | 1.34 | 5.16 | 3.35 | 78 |
| regions 150 km | 230 | 57 / 59 / 14 / 99 | 8.8 | 7.06 | 9.8 | 1.33 | 5.22 | 3.45 | 78 |
| regions 500 km | 230 | 61 / 53 / 14 / 101 | 8.7 | 6.99 | 9.5 | 1.33 | 5.25 | 3.42 | 78 |
| probe survey day as covariate (A only) | 230 | 56 / 56 / 13 / 104 | 8.8 (6.4-11.4) | 6.79 | 9.8 | 1.33 | 5.24 | 3.40 | 78 |
| front_indeterminate years as intervals | 230 | 59 / 59 / 13 / 98 | 8.9 | 7.09 | 9.9 | 1.33 | 5.27 | 3.43 | 81 |
| every temperature year as an interval | 230 | 58 / 60 / 13 / 98 | 8.7 | 7.05 | 9.8 | 1.33 | 5.26 | 3.45 | 81 |

The last three rows (added 13 November 2026) come from
`py -3.12 3_aldi.py --no-main --sensitivity --runs survey_doy,temp_interval_flagged,temp_interval_all`
(written to `data/3_aldi_sensitivity_runs/`); its main row is identical to the one above,
site by site.

Run `mcmc_seed` (added 14 November 2026, fifth review) refits with the main settings and
another seed of the sampler (seed + 1), so that the classes that change by Monte Carlo error
alone can be told from those a choice changes (paper checks, block "CLASS CHANGES BETWEEN
RUNS"). It is run on its own: `py -3.12 3_aldi.py --no-main --sensitivity --runs mcmc_seed`
(written to `data/3_aldi_sensitivity_runs/`, which the paper checks read beside the full
table). Result of the PC run of 14 November 2026: R-hat 1.005, no divergences; 7 of the 230
classes change, all at a class boundary. Five move between groups, all borderline (CALM_U10,
CALM_U11C, CALM_U13, BNZ_WCM1 and HPW_Shakwak_orig_natural from gradual thickening to
indeterminate, their lower ends moving from 0.02-0.23 to -0.23 to -0.05 %/dec), and two cross
the rapid threshold (CALM_U27 9.99 -> 10.01, GSC_90TT08 9.98 -> 10.06). The median |rate
change| of the group changes is 0.05 %/dec and the RMS rate change over all series 0.10.
Three of the four class changes of the run with the low-variability series kept (CALM_U27,
BNZ_WCM1, HPW_Shakwak_orig_natural) also change here, so those are of Monte Carlo size.

Runs `mcmc_long` and `mcmc_long_seed` (added 14 November 2026, sixth review) refit model A only
with 5000 instead of 1500 draws per chain, with the main seed and seed + 1: how many classes
longer sampling settles, and how many still change between two seeds at that length (paper
checks, "class changes, mcmc_long_seed against mcmc_long"; Table S4). Run with the seed run:
`py -3.12 3_aldi.py --no-main --sensitivity --runs mcmc_seed,mcmc_long,mcmc_long_seed` (the
folder `data/3_aldi_sensitivity_runs/` is replaced by each `--runs` call, so all three go in one).

Class margins without refit (main draws): stable +/-2.5 %/dec leaves no stable site (the
13 become indeterminate: no record is precise enough to show a change below 2.5 %/dec);
rapid at 15 %/dec gives 19 rapid instead of 57.

- **The data choices hardly matter.** The censored-value treatment, the survey day of
  undated probes (1,764 site-years), record length and region size move the site mean and
  the B residual by at most 0.2 %/dec, and the network rate mu by at most 0.5.
- **Sample composition matters more.** Dropping the 60 sites with coarse coordinates (26),
  a site more than 200 m from its ERA5-Land cell elevation (13) or a filled sea cell (25)
  lowers the expected part (0.91 vs 1.33) and raises the residual by 0.4. Dropping the 8
  fire-disturbed Bonanza Creek plots lowers the residual by 0.3 and the "deeper than TDD"
  count from 78 to 70. One series per place (174) raises the residual by 0.35 and the
  weather_memory residual by 0.5.
- **The window matters most:** 2000-2024 is faster by 1.7 %/dec in A and 1.0 in B (expected
  part also larger: 1.9), in line with the period check (long records: 4.6 %/dec over the
  full record, 2.3 over the last 15 years).
- Across all runs the B residual stays 5.0-6.3 %/dec (main) and 3.2-4.1 (weather_memory):
  positive in every run.
- **Survey day (run `survey_doy`).** A probe surveyed later in the season meets a deeper
  thaw: the fitted effect is +2.7 % of ALT per 10 days (95 %: 1.7-3.7; 726 dated probe
  site-years at 39 sites, undated years count as the site's mean day). As a covariate of A
  it lowers the probe mean rate from 7.86 to 7.43 %/dec, the mean site rate by 0.26 and the
  network rate by 0.05. The class changes are at Bonanza Creek plots whose survey date
  drifts later by up to 39 days over the record (rapid to gradual: EML_EXTENSIVE,
  EML_MODERATE, UP4C; gradual to indeterminate: BFCONTROL, GSI2, RCBSC, TRM3, WCM1, WCM2),
  plus 4 flips at class boundaries (rates moving by 0.2 or less). Other BNZ plots move
  with them through the region mean. B is unchanged because its TDD runs to the survey day.
- **Temperature ALT as an interval (runs `temp_interval_flagged`, `temp_interval_all`).**
  Entering a year as only known to lie between the ALT read at +0.1 and -0.1 C (94 of the
  735 temperature site-years are front_indeterminate: 90 closed intervals, 4 open above;
  in the second run 506 years with both readings) moves every mean by 0.05 %/dec or less
  and changes 4 classes, 2 of them real (HPW Shakwak, Banks Island, whose intervals are
  wide). The point values are adequate for A and B.

Simulation (`--validate`, 12 November 2026: 8 networks per scenario, the design of this
run (230 sites, 4,614 site-years), real ERA5-Land TDD; 1,840 site intervals per scenario,
so a coverage has a Monte Carlo SE of about 0.007):

| | v6 | v5.2 site | v5.2 pooled | per-site OLS |
|---|---|---|---|---|
| 90 % interval coverage (fitted truth) | 0.90 | 0.85 | 0.82 | 0.73 |
| same, heavy-tailed truth (t, 3 df) | 0.91 | 0.86 | 0.85 | 0.73 |
| rate RMSE, %/dec | 4.4 | 9.4 | 4.8 | 10.1 |
| wrong-direction calls | 0.8 % | 1.5 % | 1.1 % | 3.6 % |
| false trend calls when no trend anywhere | 0 % | 11.5 % | 0.2 % | 24.2 % (16.6 % at p < 0.05) |
| mean site rate: bias, %/dec (95 % interval covers the truth) | +0.2 (7 of 8) | | -0.7 (7 of 8) | |

The last row scores the mean of the site rates of each network, not mu: for v6 the
posterior of the mean of the site rates (`rate.mean(1)` in each draw) against the mean of
the true site rates, for v5.2 pooled its pooled mean against the same truth (the columns
`network_err` and `network_cover95` of the simulation files keep their names). With 8
networks its coverage is known only coarsely (7 of 8: Clopper-Pearson 47-100 %). By
scenario (paper checks, block "SERIES MEAN", `aldi_v6_checks_series_mean.csv`; 14 November
2026): `A_fit` 7 of 8, `A_null` 8 of 8 and `A_ppc` 7 of 10 (one-sided p 0.012), so 22 of 26
networks drawn from Model A; `A_heavy` 7 of 8, `A_misspec` 7 of 8 and `A_thin` 9 of 10, so 23
of 26 misspecified networks; 45 of 52 in all (87 %). Since the fifth review the pooled counts
are descriptive, because the scenarios need not share one coverage (a Clopper-Pearson
interval would assume they do); p = P(Binomial(n, 0.95) <= k), 0.039, 0.139 and 0.014, tests
without that assumption that the coverage is at least 95 % in every scenario pooled. The
shortfall is not caused by the misspecified scenarios, and the interval is probably somewhat
too narrow: the RMSE of the mean site rate over the posterior SD implied by the width is
1.06 (`A_fit`), 1.02 (`A_null`), 1.16 (`A_heavy`) and 1.42 (`A_misspec`; RMSE 0.74 %/dec,
mean width 2.03), the four scenarios that store the width since the rerun after PR #42
(below). Bias -0.51 (`A_ppc`) to +0.29 %/dec. It does not come from the pooling: the same
interval of the trend-unpooled fit covers 22 of 32 networks (bias +0.96, because 100 (e^b - 1)
of imprecise trends has a mean above the truth), while the interval of mu covers 31 of 32.

Per-site OLS (added 9 November 2026, same simulated networks) is the method of recent
syntheses: a linear trend per site tested at p < 0.05 (e.g. Streletskiy et al. 2026), with
no autocorrelation, censoring or pooling. Its intervals are too narrow at every record
length (coverage 0.70-0.77), and with no trend anywhere it finds a significant trend at
17 % of sites against the nominal 5 %.

B (8 networks per scenario): residual coverage 0.90 with the fitted truth (residual bias
+0.04 %/dec). When every trend is climatic, no site is called "deeper than TDD"; with TDD
error of 20 % of its variance, 0.2 % of sites are. That error, about four times the real
one (ERA5-Land vs POWER), lowers k by 0.05 and the expected trend by 0.25 %/dec and raises
the residual by 0.55 (0.17 without the error). With k = 0 (ALT ignores TDD) the residual
coverage stays 0.90 and the expected trend is unbiased.

Two A scenarios added on 14 November 2026 (third review of the paper). They are run on
their own, `py -3.12 3_aldi.py --no-main --validate --scenarios A_thin,A_ppc --reps 10`,
which writes `aldi_v6_simulation_reps_A_thin_A_ppc.csv` beside the main file; the paper
checks read both files. Since 14 November 2026 (fifth review) each scenario is taken whole
from one file, so that all its methods are scored on the same networks: from a file run
later rather than from the main file, and from the most recently written of two such files
(column `source_file` of `aldi_v6_checks_simulation_networks.csv` names it; until then the
first file that held a scenario won).

- **Rare local thinning among thickening neighbours (`A_thin`).** Trends drawn as in
  `A_fit`; then 12 series per network (about 5 %) get a true rate of -10 %/dec, at most one
  per region and only in regions of at least 3 series whose drawn regional trend is
  positive, taken in turn from the four record-length groups. Each network is refitted with
  Model A (normal series trends, the main model), Model A with Student-t(4) series trends,
  the trend-unpooled fit, the per-site Akritas-Theil-Sen method with AR(1) (v5.2) and the
  per-site OLS. The paper checks give the share of the planted series called thinning,
  unresolved, stable or thickening, by record length, their interval coverage and mean
  estimate, and the false thinning calls among the other series
  (`aldi_v6_checks_local_thinning.csv`). The question: does pooling hide a real local
  exception, as it might for the six series that thin on their own record but are
  unresolved in Model A?
- **Posterior predictive check of the class counts (`A_ppc`).** Each network takes one
  posterior draw of the real fit (series trends, noise SDs, AR(1) coefficients times 0.95 as
  in the other scenarios, region-year SD), simulates the real design from it and is refitted
  with Model A; the observed counts of unresolved and stable series by record length are
  compared with the simulated ones (`ppc_*` columns of `aldi_v6_checks_by_span.csv`, block
  "POSTERIOR PREDICTIVE CHECK" of `aldi_v6_checks.txt`). The expected shares of the
  detectability check take the hierarchical rates as true and leave the pooling out; this
  check runs the whole model. Both are conditional on Model A.

- **Trend-unpooled fit in the main scenarios (added 14 November 2026, fifth review of the
  paper).** Model A differs from the per-site methods in two ways at once: its noise model
  (AR(1) per method, region-year anomalies, lower bounds as censored values) and the pooling
  of trends. `run_A` therefore also refits every network of `A_fit`, `A_heavy`, `A_null` and
  `A_misspec` with the trend-unpooled fit (method `v6_unpooled`: Model A's noise model,
  censoring and region-year term, each series trend with its own N(0, 1) prior; target
  acceptance 0.95 as in `A_thin`), scored like the others, with its interval of the mean of
  the series trends in the `network_*` columns (it has no mu). On the same networks, per-site
  method -> trend-unpooled is what the noise model changes and trend-unpooled -> Model A what
  pooling changes (paper checks, block "POOLING AGAINST THE NOISE MODEL"). The fits draw no
  numbers from the generator, so the networks stay those of earlier runs: `--scenarios
  A_fit,A_heavy,A_null` takes the generator in the order of the main run of 12 November 2026
  and should reproduce its networks and scores, and `A_misspec` is run on its own as before.
  `score` also gives the RMSE by record length (`rmse_12-15` etc.).
- **What Model A does not assume (`A_misspec`, added 14 November 2026, first review of the
  paper).** Run on its own (`--scenarios A_misspec`, which writes
  `aldi_v6_simulation_reps_A_misspec.csv`). Trends drawn as in `A_fit`; then in each record
  the trend changes at a random year between 30 and 70 % of the record by N(0, 0.1) log
  units per decade (about +-10 %/dec), region-year anomalies persist (AR(1), 0.5) and the
  innovations are Student-t(4). The reference trend of a record is the least-squares line
  through its noise-free trajectory over the record's years. That is what every method is
  scored against, but not exactly what each estimates once the trend changes: the
  Akritas-Theil-Sen method estimates the median of the pairwise slopes, and a fit with AR(1)
  noise approaches the generalized-least-squares (GLS) line under its noise covariance (the
  fitted AR(1) coefficient of the record's method and the record's SD, plus the region-year
  variance in region-years shared with another series). Since 14 November (fourth review)
  every method is also scored against those two lines of the same noise-free trajectory:
  columns `coverage90_ts`, `rmse_ts`, `wrong_ts` (Theil-Sen) and `coverage90_gls`,
  `rmse_gls`, `wrong_gls` (GLS), and `target_ts_*`, `target_gls_*` give how far each target
  lies from the least-squares line (RMS, mean and 95th percentile of the absolute difference
  over the series of a network). No random numbers are drawn for these, so a rerun of the
  scenario reproduces the networks of the first run.

Columns added on 14 November 2026 (fourth review) to every A scenario run since then:
`network_width95` (width of the 95 % interval of the mean site rate), `mu_err`, `mu_cover95`
and `mu_width95` (the 95 % interval of mu, the mean of the distribution of the regional
trends, against the mu the network was drawn from, 0 in `A_null`), and `boot_cover95`,
`boot_width95` (the region bootstrap of the mean site rate, as in Table 2, against the true
mean site rate of the network). The bootstrap also resamples which regions are monitored,
so for the mean of this network's own sites it should be conservative.

Results of the PC run of 14 November 2026 (10 networks per scenario, so 2,300 series
intervals per method; `aldi_v6_simulation_A_thin_A_ppc.txt`, `aldi_v6_checks.txt`, Tables S6
and S8 of the paper):

- `A_thin`, the 120 planted series. Model A calls 27.5 % thinning, 70.0 % unresolved, 2.5 %
  thickening (3 series, 2 of them of 12-15 years) and none stable; it resolves the thinning
  at 0, 6.7, 30 and 73.3 % of the planted series of 12-15, 16-20, 21-25 and 26+ years. Its
  estimates of them average -2.1 %/dec (true -10), and its 90 % intervals cover 50.8 % of
  them against 91.2 % of the other series. Student-t trends: 30.8 % thinning, 65.8 %
  unresolved, 3.3 % thickening, coverage 61.7 %, mean -2.4. Trend-unpooled fit: 56.7 %
  thinning (30, 47, 60 and 90 % by record length), 43.3 % unresolved, none thickening,
  coverage 92.5 %, mean -8.4. Akritas-Theil-Sen 50 % thinning (coverage 86.7 %), OLS 66.7 %
  (80 %).
- `A_thin`, the other series (218 per network): false thinning (classed thinning with a true
  rate >= 0) at 0.09 % of them for Model A, 0.83 % for the trend-unpooled fit, 1.15 % for
  Akritas-Theil-Sen and 2.39 % for OLS. The trend-unpooled fit classes 6.7 of them per
  network as thinning, 1.8 of them falsely: the fitted spread of trends gives about one
  series in nine a negative rate (P(b < 0) = 0.11), so the 7 thinning series of the
  trend-unpooled fit of the real data are about what that spread predicts.
- `A_thin`, all 230 series: coverage 0.891 for Model A (SE about 0.006), 0.889 with Student-t
  trends, 0.883 for the trend-unpooled fit, 0.862 for Akritas-Theil-Sen and 0.731 for OLS.
- `A_ppc`: series coverage 0.881. Unresolved series by record length, simulated mean (range)
  against observed: 28.6 (13-33) against 26, 28.4 (18-34) against 32, 14.7 (6-19) against
  17, 22.3 (16-28) against 21, and 94.0 (53-111) against 96 in all; stable series of 26+
  years 12.1 (6-15) against 13. Every observed count lies inside the simulated range.
- Mean site rate: its 95 % interval covers the truth in 9 of 10 `A_thin` networks (bias
  +0.28 %/dec) and 7 of 10 `A_ppc` networks (bias -0.51); every scenario is listed above.
  The paper also gives the region-bootstrap interval of the series mean (Table 2), as an
  alternative assessment rather than a verified correction (see `A_misspec` below).
- Sampling: the largest R-hat over the 10 `A_thin` networks is 1.03 with normal, 1.04 with
  Student-t and 1.01 with unpooled series trends; on average the worst parameter of a
  network has R-hat 1.010-1.018 and an effective sample size of 300-414 (`A_ppc` 1.014 and
  337); the Student-t and trend-unpooled refits had one divergent transition each.

Rerun of `A_misspec` on the PC on 14 November 2026 after PR #40 (`py -3.12 3_aldi.py
--no-main --validate --scenarios A_misspec`, about 5 minutes, then `--checks` and
`paper_figures`). It reproduced the first run exactly (coverage 0.878 / 0.874 / 0.920 /
0.811 / 0.665 for v6, v6 without region-year, v5.2 site, v5.2 pooled and OLS; RMSE 5.444 /
5.741 / 10.220 / 6.273 / 10.919) and adds:

- **Trend targets** (block "TREND TARGETS UNDER A CHANGE OF TREND",
  `aldi_v6_checks_trend_targets.csv`). The noise-free Theil-Sen slope differs from the
  least-squares slope by 1.47 %/dec RMS over series (mean -0.03; 95 % of series within
  3.12), the GLS slope by 0.23 (95 % within 0.49). Coverage of the 90 % intervals against
  the least-squares / Theil-Sen / GLS target: v6 0.878 / 0.856 / 0.877, v5.2 site
  (Akritas-Theil-Sen) 0.920 / 0.916 / 0.918, v5.2 pooled 0.811 / 0.797 / 0.813, OLS 0.665 /
  0.656 / 0.664; RMSE v6 5.44 / 5.78 / 5.43, Akritas-Theil-Sen 10.22 / 10.29 / 10.22. Against
  its own target Akritas-Theil-Sen covers no more than against least squares, so the
  difference between the methods does not come from the choice of target.
- **Series mean, mu and bootstrap.** The 95 % interval of the mean site rate covers 7 of 8
  networks with a mean width of 2.03 (RMSE / posterior SD 1.42); the interval of mu covers
  the mu the networks were drawn from in 7 of 8 (mean width 5.1); the region bootstrap of
  the mean site rate covers the true mean in 8 of 8 but is 2.6 times as wide (mean width
  5.2). Eight networks cannot show that the bootstrap's coverage is right; the paper calls
  it an alternative assessment of the uncertainty.
- Sampling as in the first run: on average the worst parameter of a network has R-hat
  1.010 (v6) and 1.013 (without region-year) and an effective sample size of 476 and 344;
  no divergent transitions.

Rerun of the main scenarios on the PC on 14 November 2026 after PR #42 (fifth review): `py
-3.12 3_aldi.py --no-main --validate --scenarios A_fit,A_heavy,A_null` (writes
`aldi_v6_simulation*_A_fit_A_heavy_A_null*`) and `--scenarios A_misspec`, each now with the
trend-unpooled fit (`v6_unpooled`), then `--checks` and `paper_figures`. Every network and
every earlier score was reproduced exactly (Model A coverage 0.900 / 0.905 / 0.980 / 0.878 in
`A_fit` / `A_heavy` / `A_null` / `A_misspec`). New:

- **Pooling against the noise model** (block "POOLING AGAINST THE NOISE MODEL",
  `aldi_v6_checks_pooling_vs_noise.csv`, Table S8). Coverage of the 90 % intervals / RMSE
  (%/dec):

  | scenario | per-site OLS | Akritas-Theil-Sen | trend-unpooled | Model A |
  |---|---|---|---|---|
  | `A_fit` | 0.728 / 10.15 | 0.855 / 9.43 | 0.882 / 9.57 | 0.900 / 4.40 |
  | `A_heavy` | 0.732 / 10.55 | 0.864 / 9.97 | 0.882 / 9.76 | 0.905 / 4.59 |
  | `A_null` | 0.750 / 8.73 | 0.885 / 9.37 | 0.879 / 8.53 | 0.980 / 0.58 |
  | `A_misspec` | 0.665 / 10.92 | 0.920 / 10.22 | 0.890 / 9.91 | 0.878 / 5.44 |
  | `A_thin` | 0.731 / 10.30 | 0.862 / 9.57 | 0.883 / 9.56 | 0.891 / 4.97 |

  In `A_fit` the trend-unpooled fit reaches 90 % of Model A's coverage gain over OLS (Model A
  minus trend-unpooled +0.018, SE 0.009), but 90 % of the RMSE gain comes after it (RMSE
  trend-unpooled minus Model A +5.17, SE 0.45): the calibration comes mostly from the noise
  model, the precision from pooling. RMSE by record length, trend-unpooled / Model A: 14.8 /
  5.2 (12-15 yr), 11.0 / 5.3, 7.0 / 4.3 and 4.1 / 3.1 (26+ yr). With no trend (`A_null`) the
  trend-unpooled fit calls 12.0 % of the series trending (nominal 10 %), Model A none.
- **Series mean, mu and bootstrap** in the four scenarios (`A_fit`, `A_heavy`, `A_null`,
  `A_misspec`): the interval of mu covers 31 of 32 networks (mean width 5.4, 4.6, 1.7, 5.1);
  the region bootstrap of the series mean covers 26 of 32, all 24 with a trend (mean width
  5.6, 4.6 and 5.2, about that of mu) but only 2 of 8 in `A_null` (width 0.4: it resamples
  estimated trends, so it misses their estimation error when the true trends do not differ
  between regions); the trend-unpooled series mean covers 7, 6, 5 and 4 of 8 (bias +0.89,
  +1.03, +0.68, +1.25).
- Sampling as in the first runs: largest R-hat 1.016 and smallest ESS 263 (Model A in
  `A_null`); on average 0.375 divergent transitions per network for Model A in `A_fit` and
  0.125 in `A_null`, none in any other fit.

### Thaw-subsidence bounds (sensitivity runs `subsidence_0.2`, `_0.5`, `_1.0`; added 2026-11-09)

A probe measures ALT from the ground surface, and so do the GSC Mackenzie thaw tubes read
against the surface. Where ground ice melts the surface sinks, so these series miss part of
the deepening of the permafrost table and their trend is too small (Smith et al. 2024: thaw
penetration about 8 cm/decade against about 1 cm/decade of ALT in the Mackenzie Valley).
The three runs add 0.2, 0.5 or 1.0 cm/yr of subsidence, counted from each record's first
year, to the probe and tube ALT of the series on medium- or high-ice ground of the IPA map
(published rates in undisturbed tundra: 0.4-1.0 cm/yr at Utqiagvik 2003-2015, Streletskiy
et al. 2017; 0.4 cm/yr median in the Mackenzie Valley, Smith et al. 2024; 12-15 cm in 18
years on the North Slope, Nyland et al. 2021), and refit A and B. They bound how much the
network rate and the B residual can grow; they are not a correction (no site has its own
subsidence record here). Fig S2 of `paper_figures.m` shows their network rates; with
MODEL_B = true its Model B figure marks the B means of the three runs.

Results (PC, 12 November 2026; 146 probe and tube series on medium- or high-ice ground;
every fit R-hat <= 1.013, no divergences):

| run | network rate mu (95 %) | site mean | B expected | B residual | B residual (weather_memory) | deeper than TDD |
|---|---|---|---|---|---|---|
| main | 8.8 (6.6-11.6) | 7.05 | 1.33 | 5.24 | 3.40 | 78 |
| +0.2 cm/yr | 10.5 (8.4-13.1) | 9.23 | 1.30 | 7.38 | 5.55 | 124 |
| +0.5 cm/yr | 13.2 (11.2-15.7) | 12.51 | 1.27 | 10.64 | 8.78 | 189 |
| +1.0 cm/yr | 17.7 (15.1-20.5) | 17.75 | 1.20 | 15.88 | 13.98 | 208 |

Even the low end (0.2 cm/yr, 2 cm per decade) raises the residual by 2.1 %/dec. Read with
care: the corrected quantity is the lowering of the permafrost table (thaw penetration from
a fixed datum), not ALT itself, so the runs say that 5.2 %/dec is a lower bound for how fast
the permafrost table drops beyond what air temperature explains, not that ALT grows faster.

Since the third review of the paper (14 November 2026) the three runs are no longer among
the sensitivity runs of the paper (Table S4, Fig S2 now show the 13 runs that change the
analysis of ALT). They change the quantity, so they are reported apart, as hypothetical
scenarios in Text S1 and Table S11 of the supplement (`paper_figures.m` writes
`TableS11_subsidence_scenarios.csv`).

## Paper checks (`paper_checks`, `--checks`, `data/3_aldi_checks/`; added 2026-11-09)

Computed from the outputs of the main run (and of `--sensitivity` for the 2000-2024 cm
row), with no refit; about a minute. They turn the novelty of the paper (its uncertainty
accounting) into numbers a reader can compare with the usual analysis:

- **Years of record a site needs** (`aldi_v6_checks_detection.csv`, Table S5), per method
  family. With the fitted noise (each family's AR(1) coefficient and sigma) the SD of a
  trend from a given set of years is exact, so the chance of each class follows from the
  true rate. Two noise levels: in the network (the shared region-year anomalies removed,
  as the hierarchical model does) and alone (left in, as a per-site test). For consecutive
  years this is the detection time of Weatherhead et al. (1998), written for annual values:
  n* = [C sqrt(12) sigma sqrt((1 + rho)/(1 - rho)) / |w0|]^(2/3), with C = 2.93 for
  thickening at 90 % and 3.29 for showing "stable"; the table gives both.
- **Expected shares by record length** (`aldi_v6_checks_by_span.csv`, Table S6): the share
  of sites expected to stay unresolved from each record's own years and noise, beside the
  observed share (the dashes of Fig 3). Pooling toward the region narrows the hierarchical
  intervals further, so its expected share is an upper bound. This is a model-conditional
  consistency check, not a test of the model: it takes the hierarchical rates and the
  fitted noise as true. Since 14 November 2026 the same table also holds the posterior
  predictive check of scenario `A_ppc` (`ppc_*` columns; see the simulation above), which
  runs the whole model on networks drawn from its posterior.
- **The standard per-site test on the same series**: OLS of log ALT on year with the
  t (F) test at p < 0.05 (the test of recent CALM syntheses, e.g. Streletskiy et al. 2026),
  Benjamini-Hochberg false-discovery-rate q values, the same 90 % class rules on its
  intervals, and from the noise model its power, its false-positive rate with no trend
  (above 5 % because the years are autocorrelated), and the sign error (Type S) and
  exaggeration (Type M) of its significant rates (Gelman and Carlin 2014), with the
  hierarchical rate as the true rate.
- **Shrinkage** (Fig S4): how far the hierarchical rate sits from the site's own OLS rate
  toward its region's rate (0 = own rate, 1 = region rate), by record length; the class a
  site with no data would get from its region alone; the regional share of the between-site
  spread of trends (tau_region^2 / (tau_region^2 + tau_site^2)).
- **Expected class counts** (sums of the band probabilities) and the sites with
  P(rate > 5 %/dec) > 0.9.
- **TDD share of the variance vs share of the trend**: R2 of ALT on sqrt(TDD) with site
  fixed effects (mostly differences between sites, the number reported as about 92 % by
  Streletskiy et al. 2026), within sites, and year to year (each site detrended), beside
  the share of the trend that TDD explains (B main expected / total). The k = 0.5 run (B
  kfixed) is a theory-informed sensitivity scenario, not a bound: errors in TDD pull a
  fitted k toward 0, but 0.5 is the Stefan value for the thawing index of the ground
  surface, which air TDD is not.
- **Residual by ground ice and method** (`aldi_v6_checks_ice_method.csv`; Model B, not in
  the paper since 14 November 2026): the
  low-minus-high ice contrast within regions, per method family, since probes and
  surface-referenced tubes carry the subsidence bias and ground temperatures do not.
- **Agreement of the methods** (`aldi_v6_checks_method.csv`, Table S3; added 2026-11-14):
  are the trends of probes, thaw tubes and ground temperatures consistent enough to be
  analysed together under the model? Raw family means mix
  method with place (most thaw tubes are in the slowly deepening Mackenzie Delta), so each
  family is compared with the probes within regions (region fixed effects, 95 % interval
  from resampling whole regions), on the hierarchical rates, on the per-site OLS rates (no
  pooling, so the model cannot pull the methods together) and on the model B residuals.
  Second test at one place (series within 1 km): if the methods measured different things,
  two series of different methods at one place would differ more than two series of the
  same method there.
  Since 14 November 2026 (second review of the paper) also on the trends of the
  trend-unpooled fit (same noise model, no pooling of trends), and every within-region difference is
  recomputed with each shared region left out in turn (`loro_min`, `loro_max` and
  `aldi_v6_checks_method_loro.csv`): the thaw-tube contrast rests on four regions, and a
  bootstrap of four regions cannot show that it generalizes. The pairs at one place share
  series and places, so their medians describe the data; they are not a test.
  Unpooled trends are noisy and, as percentages, skewed to the right (100 (e^b - 1)
  stretches large b), so a few short records with extreme rates dominate the unweighted
  contrast of the trend-unpooled trends. The row "trend-unpooled Bayesian rate, precision-weighted"
  makes the same contrast on the log scale with each series weighted by
  1 / (its posterior variance + tau_site^2), as in a random-effects meta-analysis, and turns
  the differences back into % per decade (its `mean` is the weighted mean).
- **Classes against the record alone**: how many rapid series have their whole 90 %
  interval above the rapid threshold (the rapid / gradual split uses the median only), and
  how many stable series are stable on their own record (trend-unpooled fit, same noise
  model) and by the per-site Akritas-Theil-Sen method.
- **Records that thin on their own record** (`aldi_v6_checks_thinning_records.csv`, second
  file of Table S10; added 2026-11-14, third review): every series called thinning by the
  trend-unpooled fit or by Model A, with its region (number of series, regional rate,
  centre), record years and number of values, noise SD against its method's median, and its
  rate, 90 % interval, P(thickening) and class in each fit (trend-unpooled, hierarchical,
  Student-t trends, Akritas-Theil-Sen). A summary line says how many of those left
  unresolved by Model A are short, noisier than their method's median, still negative in
  Model A, and thinning by Akritas-Theil-Sen, and in which regions they lie.
- **Rare local thinning** (`aldi_v6_checks_local_thinning.csv`, second file of Table S8;
  added 2026-11-14): the `A_thin` simulation summarized by method and record length.
- **Series mean by scenario** (`aldi_v6_checks_series_mean.csv`, Table S8
  `TableS8_simulation_series_mean.csv`; added 2026-11-14, fourth review): the 95 % interval
  of the mean site rate of Model A, one per simulated network, per scenario and pooled over
  the scenarios drawn from Model A (`A_fit`, `A_null`, `A_ppc`), over the misspecified ones
  (`A_heavy`, `A_misspec`, `A_thin`) and over all; the share it covers with its exact
  (Clopper-Pearson) 95 % interval, bias and RMSE, and, where the runs hold the columns, its
  mean width, the RMSE over the posterior SD that width implies, and the coverage and width
  of the interval of mu and of the region bootstrap. Fifth review: the scenarios need not
  share one coverage, so the pooled rows (`kind` "pooled, descriptive") are descriptive
  counts, and their Clopper-Pearson interval would assume one common coverage. Without that
  assumption a pooled count still tests that the coverage is at least 95 % in every scenario
  pooled, because a sum of independent binomial counts whose probabilities are all at least
  0.95 is stochastically at least Binomial(n, 0.95): `p_nominal_in_every_scenario` =
  P(Binomial(n, 0.95) <= k) (for one scenario the exact one-sided test). Where the runs hold
  the trend-unpooled fit, the same interval of that fit is scored too (`unpooled_*`), and the
  interval of mu over the networks that store it (`mu_networks`, `mu_covered`, `mu_p_nominal`).
- **Pooling against the noise model** (`aldi_v6_checks_pooling_vs_noise.csv`, Table S8
  `TableS8_simulation_pooling_vs_noise.csv`; added 2026-11-14, fifth review): per scenario
  that holds both the trend-unpooled fit and Model A (`A_fit`, `A_heavy`, `A_null`,
  `A_misspec`, `A_thin`), the coverage of the 90 % intervals, RMSE, wrong direction and
  resolved share of per-site OLS, per-site Akritas-Theil-Sen, the trend-unpooled fit and Model
  A, overall and by record length; the paired differences between networks (trend-unpooled
  minus Model A in RMSE, trend-unpooled minus OLS and minus Akritas-Theil-Sen and Model A
  minus trend-unpooled in coverage, each with its SE from the spread between networks); and
  the share of the RMSE gain of Model A over OLS that comes after the trend-unpooled fit
  (pooling) and the share of its coverage gain that the trend-unpooled fit already reaches
  (noise model).
- **Class changes between runs** (`aldi_v6_checks_class_changes.csv` and
  `aldi_v6_checks_class_changes_sites.csv`, Table S4 `TableS4_class_changes.csv` and
  `TableS4_class_changes_by_series.csv`; added 2026-11-14, fifth review): for every run that
  refits Model A with one choice changed (`data/3_aldi_sensitivity`,
  `data/3_aldi_sensitivity_runs` for runs not in the full table, such as `mcmc_seed`, and
  `data/3_aldi_keep_pinned` with the low-variability temperature series kept), the classes of
  the series it shares with its own main fit: changes of class, changes between the groups
  thickening / stable / unresolved / thinning, rapid <-> gradual, and how many group changes
  are borderline, that is, at a series whose deciding interval end lay within 1 %/dec of its
  class boundary in the main fit (`class_margin`); the median rate change of the group
  changes; and a summary over the processing runs with the same period (without the
  2000-2024 window, which changes the period, the subsidence scenarios, which change the
  quantity, `undated_doy_243`, which does not refit A, and the reference runs `mcmc_seed` and
  `keep_low_variability`). The changed series of `survey_doy`, `keep_low_variability` and
  `mcmc_seed` are listed in the text.
- **Trend targets** (`aldi_v6_checks_trend_targets.csv`, Table S8
  `TableS8_simulation_trend_targets.csv`; added 2026-11-14, fourth review): for scenario
  `A_misspec`, per method, the coverage of the 90 % intervals, RMSE and wrong-direction
  share against the least-squares, Theil-Sen and GLS lines of the noise-free trajectory,
  and how far the targets lie apart.
- **Excluded years that mark a lost active layer** (added 2026-11-14): site-years flagged
  no_refreeze (the deepest thawed sensor did not refreeze the next winter) or talik /
  possible_talik (thaw reached unfrozen ground; every later year of the site), in the ALT
  target and in the analysed series, with each analysed series that loses such years. Such a
  year has no valid ALT; that is not the same as no degradation. ALT thickening and talik
  formation are different observational outcomes: the trends describe the years with a
  valid ALT, not the degradation that follows talik formation.
- **Acceleration** (`aldi_v6_checks_acceleration.csv`; added 2026-11-14): a hinge at the
  midpoint of each record of at least 25 years with 8 or more measured values on each side,
  GLS with the record's noise covariance; the change of trend per record with its p value
  and, since the second review, its Benjamini-Hochberg q value over the records, pooled with
  the random-effects model of v5.2 and as a plain mean with a region bootstrap.
- **Rates in cm** (`aldi_v6_checks_cm.csv`, Table S7): site rates times the site's fitted
  ALT, and the per-site OLS on ALT in cm as in Streletskiy et al. (2026) and Lin et al.
  (EGUsphere-2026-841), beside their published values and Smith et al. (2024). The OLS is
  run on the whole record and again on the rule of Streletskiy et al. (2000-2024 only, at
  least ten years in that window). Their published shares (Commun. Earth Environ. 7:671,
  156 CALM sites): 55.2 % of Arctic sites increase significantly (53 sites, 0.8 +- 0.8
  cm/yr; the number of Arctic sites is not given), and 66.4 % in the Northern Hemisphere
  (91 of 137 sites), raised by the mountain sites (European mountains 18 of 19, Third Pole
  20 of 22). The Arctic share is the comparator for ALDI.
- **MCMC diagnostics** of every fit of the main run (Table S2) and the count of sites whose
  survey date drifts by more than 14 days over the record.

Results of the 12 November 2026 run (PC; `data/3_aldi_checks/aldi_v6_checks.txt`):

- **Years needed** (consecutive, to class a site thickening at the network rate of
  8.8 %/dec with 90 % chance, network / alone): probes 29 / 30, thaw tubes 25 / 28, ground
  temperature 50 / 50; to show a trendless site as "stable": 44 / 47, 39 / 43, 78 / 79. A
  coin flip (50 %) needs 19, 17 and 33 years. The shared region-year noise is 17-47 % of
  the year-to-year variance, but removing it saves only 0-4 years: the gain of the
  hierarchical model is mainly its pooling, not the shared anomalies.
- **Unresolved sites match what record length allows.** Hierarchical, observed vs expected
  at most: 12-15 yr 53 vs 74 %, 16-20 yr 62 vs 65 %, 21-25 yr 41 vs 49 %, 26+ yr 24 vs 27 %.
  The per-site Sen + AR(1) method leaves 62 % unresolved at 26+ yr where an OLS test with
  the same noise would leave 36 %: it is more conservative than its noise needs.
- **Standard per-site test**: 104 sites (45 %) significantly increasing at p < 0.05, 14
  decreasing; 94 (41 %) after the FDR correction. With no trend anywhere it would call 12 %
  of sites significant (nominal 5 %, because the years are autocorrelated). Its significant
  rates overstate the true rate by 1.9 times (4.1 times for 12-15 year records); wrong sign
  2.3 %. Of the 112 sites it calls not significant, the hierarchical model classes 24
  thickening and 12 stable.
- **Pooling**: median shrinkage toward the region 0.84 (12-15 yr), 0.70, 0.51, 0.25 (26+);
  47 % of the between-site spread of trends is regional (tau_region 4.8, tau_site 5.1).
- **Bands**: expected class counts 69 rapid, 66 gradual thickening, 89 stable, 5 gradual
  thinning (the interval rules give 57 / 63 / 13 / 1 and 96 not resolved: the 89 "stable"
  is posterior mass inside +-5 %/dec, not a class); P(rate > 5 %/dec) > 0.9 at 53 sites.
- **TDD**: R2 of ALT on sqrt(TDD) is 95 % with site fixed effects but 9 % within sites and
  10 % from year to year (each site detrended); TDD explains 20 % of the trend (1.3 of
  6.6 %/dec). The same fit as the panel of Streletskiy et al. (2026: R2 92 % with site
  fixed effects, 1.05 cm per unit sqrt(DDT) in the Arctic lowlands; here 95 % and 1.56)
  owes its R2 to the differences between sites, not to the change at a site. With k fixed
  at 0.5 the expected part is 2.6 and the residual 4.1 %/dec.
- **Ground ice by method** (mean residual low / medium / high): probes 8.4 / 5.4 / 4.5,
  thaw tubes 7.4 / 3.8 / 1.6, ground temperature 6.6 / 3.2 / 6.3. Raw means are higher on
  low-ice ground in all three families, clearly in probes and tubes but barely in ground
  temperatures, which subsidence cannot bias. Only 2-5 regions hold both low- and high-ice
  sites, so the within-region contrasts are not resolved (all families +1.3, -1.8 to +3.3;
  ground temperatures -1.9 in 2 regions).
- **Methods agree within regions** (run of 14 November 2026 on the outputs of 12 November):
  raw means probes 7.9, thaw tubes 4.6, ground temperatures 7.5 %/dec, but within the same
  regions tubes trend +1.7 (-3.7 to +3.8; 4 shared regions) and ground temperatures +1.3
  (+0.1 to +3.1; 7 regions) %/dec faster than probes, small beside the between-site spread
  (tau_site 5.1). The per-site OLS rates, which no pooling can pull together, give the same
  signs with wide intervals (+2.3, -7.8 to +10.4; +4.8, -6.1 to +13.4); the model B
  residuals +1.4 (-3.0 to +3.4) and +1.4 (+0.2 to +2.8). At the 12 places that hold two
  methods (30 series), two series of different methods differ by a median 3.0 %/dec (19
  pairs) and two series of the same method by 3.3 (68 pairs); per-site OLS 7.5 and 5.9,
  the larger spread of the different-method pairs following the noisier ground-temperature
  records (sigma 12.7 against 8.4 % for probes). 11 of the 44 regions hold two or more methods.
- **cm**: hierarchical site rates average 6.1 cm/decade (median 4.3; 7.2 in the 2000-2024
  run). The per-site OLS on cm gives 1.27 cm/yr at the 106 sites with a significant
  increase (46 % of 230) and 0.93 cm/yr over all 175 increasing sites (Lin et al.: 1.48).
  On the rule of Streletskiy et al. (2026), 2000-2024 and at least ten years, 52 % of 218
  series increase significantly, by 1.35 cm/yr on average (114 sites), against their Arctic
  55 % and 0.8 +- 0.8 cm/yr (their Northern Hemisphere 66 % includes the mountain sites).
  Averages over "significant" sites are 2x the hierarchical mean, as the exaggeration above
  predicts.
- **MCMC**: 10 fits of the main run, max R-hat 1.007, min ESS 831, no divergences. Survey
  date drifts by more than 14 days at 12 of the 39 sites with dates (median +6 days).
- **Class changes between runs** (run of 14 November 2026 after PR #42; block "CLASS CHANGES
  BETWEEN RUNS", Table S4): 54 of the 230 series are borderline in the main fit (21 of the 120
  thickening series with a lower end below 1 %/dec, 23 of the 96 indeterminate ones with a
  lower end above -1), and 12 thickening series have a median within 0.5 of the rapid
  threshold. Over the 12 processing runs with the main period a median of 9 classes change
  (4-22), 7 between groups (2-15); 69 of the 85 group changes are at borderline series, and 37
  series change group in at least one run. Most changes: `drop_site_quality` 22 of 170,
  `region_500km` 20 of 230, `one_per_place` 15 of 174 and `survey_doy` 13 of 230 (the nine BNZ
  plots, rates -0.9 to -3.0 %/dec: six to indeterminate, three rapid to gradual; plus four at
  a boundary); `window_2000_2024` changes 64 of 204 but also changes the period. Reference
  runs: `mcmc_seed` 7, all at a boundary (sensitivity section); `keep_low_variability` 4
  (CALM_U27 gradual to rapid; BNZ_BFCONTROL, BNZ_WCM1 and HPW_Shakwak_orig_natural gradual to
  indeterminate), with rates moving by less than 0.1 and three of the four also changing in
  `mcmc_seed`. A network rate that holds across choices does not make every class robust.

Added 14 November 2026 (sixth review of the paper):

- **Robustness of each class** (`aldi_v6_checks_series_robustness.csv`, released as
  `TableS10_series_results.csv`): one row per series with the rate, interval, P(thickening)
  and class of model A, the trend-unpooled fit, the Akritas-Theil-Sen method and per-site OLS;
  the class margin and deciding end, the Monte Carlo SE of the interval ends and the flags
  above; `class_changes_in` (every run of the same period whose class differs, with the new
  class), `runs_compared`, `runs_changed`; and one label, the first that applies:
  `monte_carlo` (changes between runs of the sampler, or the deciding end within 2 Monte Carlo
  SE of its boundary), `choice` (changes in a processing run or with the low-variability series
  kept), `borderline` (margin below 1 %/dec), `robust`.
- **Per-site GLS in the simulation** (`gls_site`, `gls_true`; block "POOLING AGAINST THE NOISE
  MODEL", Table S8): generalized least squares of each series under the noise of model A
  (AR(1) in calendar years with its sigma and its method's rho, plus the region-year variance
  in region-years shared with another series), with the posterior means of the trend-unpooled
  fit (`gls_site`, the comparator of the review) or with the true noise of the network
  (`gls_true`, an oracle); bounds used as values as in OLS, normal-quantile interval. Per-site
  OLS -> GLS is what modelling the temporal dependence and the shared anomalies gives; GLS ->
  trend-unpooled what the Bayesian fit adds (censoring, region-year anomalies estimated from
  the other series, uncertainty of the noise parameters, priors).
- **Series mean on more networks** (`aldi_v6_checks_series_mean_more_networks.csv`, Table S8):
  the networks of `--mean-check`, with 95 and 90 % coverage, bias and its SE, RMSE over the
  posterior SD, the standardized errors (mean 0 and SD 1 if calibrated), the error split into
  regional trends and the series' departures from them, by record length and method, mu and
  the region bootstrap on the same networks. The main scenarios now also store `network_sd`,
  `network_cover90`, `network_z`, and A_thin and A_ppc the interval of mu and the bootstrap.
- **Randomization test of the method contrasts** (block "AGREEMENT OF THE METHODS",
  `aldi_v6_checks_method.csv` columns `randomization_*` and `regional_differences`): the
  difference of each method from the probes in every region that holds both, their weighted
  mean (fixed-effects weights) and Fisher's exact sign-flip test (no difference; equivalence
  within +-5 %/dec as two one-sided tests). With R regions the smallest attainable one-sided p
  is 1 / 2^R, so with the four regions that thaw tubes share with probes equivalence cannot be
  shown at the 5 % level by any test that treats regions as the independent units.

## Spatial checks (`spatial_checks`, also `--spatial`)

Added 2026-11-07 after a geodata review. They use only the fitted rates, so
`py -3.12 3_aldi.py --spatial` reruns them on the existing results in about a minute
(the permafrost-map part needs geopandas, as step 2). Results of the 12 November 2026 run:

- **Places, not series.** 230 series stand at 174 places: series within 1 km of each other
  (a CALM thaw tube and its probing grid, a grid and the borehole beside it, CALM and LTER
  plots of one catchment) are one place. 36 places hold 92 series. Their rates correlate at
  only 0.49 (median difference 3.3 %/dec), and at 26 of those places the series fall in
  different classes. The build keeps them apart on purpose (the methods read different
  levels), so class counts should be reported per series and per place, and the
  difference between series at one place is a direct measure of method and micro-site
  uncertainty. `location_id` in the results.
- **Declustered network means.** The mean over series lets dense clusters dominate
  (Mackenzie Delta and coast 34 series, mean +2.3 %/dec; Fairbanks 33 series, mean +9.8).
  Giving each cell of an equal-area grid (Lambert azimuthal, North Pole) one vote gives
  +9.0, +9.8 and +9.4 %/dec for 100, 250 and 500 km cells, close to the mean over regions
  (+9.1) and far from the mean over series (+7.0). Residual (main): 6.6-7.2 declustered vs
  5.2 over series; residual (weather_memory): 4.6-5.1 vs 3.3.
- **Do the 250 km regions capture the spatial dependence?** Site rates correlate at 0.36
  within 100 km and 0.19 at 100-250 km, and not beyond 250 km. Deviations of site rates
  from their region rate do not correlate across region borders at any distance (-0.06 to
  0.03), so the region level of the model absorbs the spatial dependence of the trends.
- **Permafrost zone and ground ice (IPA map, Brown et al. 2002).** 150 series are in
  continuous, 75 in discontinuous, 4 in sporadic and 1 in isolated permafrost; 92 in high,
  87 in medium and 51 in low ground-ice terrain (sites in lakes or on the coast take the
  nearest polygon within 20 km). Raw medians differ a lot (residual 7.6 %/dec in
  discontinuous vs 3.5 in continuous; 7.9 in low-ice vs 3.7 in high-ice terrain), but
  almost all of that is between regions, mainly Fairbanks. Within the 5 regions that hold
  both kinds, discontinuous minus continuous is -0.1 %/dec (-2.0 to +3.2) and low minus
  high ice +1.3 (-1.8 to +3.3). The map is independent of the ALT data, so it is a valid
  stratifier for B; with 5 regions the within-region contrasts are weak. `pf_extent` and
  `pf_ice` in the results.

## Caveats

- **k is a short-run sensitivity.** Within a year k is 0.2-0.3. The two previous
  summers add 0.10-0.11 (`memory`), so the long-run response is 0.29-0.40 (probes
  0.40), closer to Stefan's 0.5. TDD error lowers k by only about a tenth: ERA5-Land and
  POWER anomalies correlate at 0.92 (reliability of ERA5-Land about 0.95), and using
  POWER as an instrument moves the within-site k from 0.27 to 0.30 (a check of
  4 November 2026 on run 7). `main` and `memory` bracket what air temperature explains.
- **The TDD trend depends on the reanalysis.** At the same sites ERA5-Land TDD rose a
  median 1.7 %/dec faster than NASA POWER (site trends correlate at 0.84), which nearly
  doubles the expected part (1.3 vs 0.7 %/dec). This is the largest uncertainty in
  "expected"; the residual stays above 3 %/dec with either source.
- **TDD is not the only climate.** In `main`, snow and summer rain are not in
  "expected", so a positive residual means "not explained by air temperature". The
  `weather` variant adds them; soil moisture stays out (mediator).
- **Shrinkage borrows from neighbours.** A short record in a fast region is pulled up.
  `rate_pct_dec` (v5.2 site) stays in the table for statements about one record.
- **Same sample as v5.2**, including U19 1990-2014 (kept, decision of 9 November 2026)
  and the 60 N cut. The cut costs little: only 3 sites south of 60 N have long enough
  records (Kamchatka R30C, Nunavik TSJQ304 and Marmot Mountain C21 in the Canadian
  Rockies, 1990-2002).
- **Most probe surveys have no date.** 1,764 of 2,783 probe site-years (the CALM summary
  table) take the network median survey day (DOY 263, 20 September, in this build) as the
  end of their TDD window, 293 the site's median day, and 726 their own date. Where a site
  is surveyed in mid-August, its TDD includes about a month after the survey. That adds
  noise to the site's TDD (and lowers k a little) and puts September warming into the
  expected trend. The sensitivity run that ends their window on 31 August instead moves
  the B residual by only +0.04 %/dec (weather_memory +0.13).
- **Disturbed plots.** Bonanza Creek LTER (27 of the 33 series of the Fairbanks region)
  includes fire and fire-line plots. The 6 series flagged burned all fall in
  `deeper_than_TDD` (residuals +8.5 to +15.0 %/dec): B responds to a known non-climatic
  disturbance as it should, but those plots also raise the network residual (mean 5.0
  without them, 5.2 with). Report B with and without the disturbed plots.

## Appendix: ALDI v5.2, the site-by-site method (rerun inside `3_aldi.py`)

`extras/ALDI_Calculation_final.ipynb` (v4) is left as it was, so its results can still be reproduced. v5.2 is the revised index; v6 reruns it on the same site-years for the comparison columns of `aldi_v6_results.csv`.


### What changed from v4, and why

| | v4 notebook | v5.2 |
|---|---|---|
| Gaps | PCHIP fills gaps of up to 2 years, and the filled values are tested as data | No filling. Theil-Sen and Kendall's S work with irregular years |
| Probe or sensor limit reached | The year is dropped | The year stays in as a right-censored value (Akritas-Theil-Sen). Dropping it removes the deepest years of deepening sites |
| Rate | cm/yr | % of the site's ALT per decade, so sites with 40 cm and 120 cm ALT are compared on one scale |
| Class thresholds | 50th and 80th percentiles of the sample's slopes, clipped. In the 1990+ data the 50th percentile (0.47 cm/yr) was clipped to 0.30, and the 80th gave 1.27 | Fixed: stable ±5 %/dec, rapid 10 %/dec (v5 decision). A site's class no longer depends on which other sites are in the sample |
| No trend | One class for "flat" and for "cannot tell" | `stable` means the 90 % interval lies inside ±5 %/dec (an equivalence test). `indeterminate` means the record cannot decide |
| Interval | Moving-block bootstrap (2-5-year blocks) | Kendall-S inversion with an AR(1) variance factor. Without autocorrelation this is exactly Gilbert's Sen-slope interval, and it also works with censored years. The AR(1) coefficient is one network value fitted by REML over all series (v5.2); v5.1 used each site's own lag-1 estimate, which is biased low for 10-35 values and made the interval too narrow |
| Reversal | Fixed first and last 7 years, slopes of opposite sign, both intervals excluding 0 | Broken-stick fit with an estimated breakpoint. The p-value comes from an AR(1) bootstrap of the sup-F statistic. A reversal needs both segments beyond ±5 %/dec with opposite signs. It is reported as a flag, not a class (v5 decision) |
| Method changes in a series | Mixed | Only the series' main method family is kept (probe, thaw tube or temperature). This removed 24 site-years at 20 sites |
| Pooling | None | Two-level random-effects model (regions of ≤250 km, then sites), fitted by REML. It gives each site's shrunk rate (`rate_pooled`) and class (`class_pooled`), the regional rate, and the network rate with regions as clusters |
| Extra outputs | | Minimum detectable trend, Benjamini-Hochberg q-values, survey-date drift flag, burned-site flag |

### v5.2 change (2026-11-03): pooled autocorrelation

v5.1 estimated the AR(1) coefficient of each series from its own lag-1 residual
correlation. With 10-35 values that estimate is biased low (a fitted trend and missing
years both pull it down: at 12 years and a true rho of 0.5 it averaged about 0.3) and
noisy (SD about 0.3), so the variance correction was too small and the 90 % interval
too narrow. The lag-1 estimate also divided a sum over consecutive pairs by a sum over
all years, which shrank it further when years were missing (fixed, used by
`ar1_method: site`).

v5.2 fits one AR(1) coefficient for the whole network by REML: each series keeps its own
level, slope and noise variance, missing years are handled exactly (correlation
rho^gap), and the restricted likelihood removes most of the trend bias. On the run-7
data the pooled rho is 0.46 (probe 0.40, temperature 0.54, thaw tube 0.57; 0.36
without the sites with a breakpoint), so year-to-year persistence in ALT is larger
than v5.1 assumed. `ar1_method: site` in the config gives the v5.1 behaviour.

### Results on the run-7 data (2026-11-02 build; 1990-2025, >=60 N, >=10 years over >=12 years: 229 sites)

| class | site v5.1 | site v5.2 | pooled v5.1 | pooled v5.2 |
|---|---|---|---|---|
| rapid thickening | 43 | 39 | 44 | 35 |
| gradual thickening | 26 | 19 | 59 | 66 |
| stable | 18 | 11 | 24 | 23 |
| gradual thinning | 3 | 2 | 1 | 1 |
| rapid thinning | 5 | 4 | 1 | 0 |
| indeterminate | 134 | 154 | 100 | 104 |

- **Network rate:** +7.9 %/dec (95 % CI 5.6 to 10.3), 41 regions (v5.1: +8.4). Between-region SD 6.0, between-site SD 3.7 %/dec.
- **Detectable trend:** median minimum detectable trend 14.0 %/dec (v5.1: 12.4).
- **Trend q < 0.10 (Benjamini-Hochberg):** 22 sites (v5.1: 39).
- **Breakpoints:** 33 of 197 tested (about 10 expected by chance), 20 reversals.
- **Left out:** 1,307 site-years before 1990 or south of 60 N; 749 site-years at 123 sites that are too short inside the window (46 of them have long records before 1990, mostly Roshydromet 1930-1990 and KWBS); 24 minority-method years.

### Simulation check (v5.2, 2026-11-03)

Synthetic series with 7.5 % noise (the median of the real series) and 10 % missing years, over 12-35 years, AR(1) rho 0, 0.3 and 0.5, 300 series per cell. The pooled rho is fitted on the 300 series of a cell, as it is on the network.

- **Interval coverage (nominal 90 %), mean over lengths and rates:**

  | rho | v5.2 pooled | v5.1 site | v4 bootstrap |
  |---|---|---|---|
  | 0 | 0.905 | 0.922 | 0.934 |
  | 0.3 | 0.905 | 0.879 | 0.906 |
  | 0.5 | 0.911 | 0.853 | 0.870 |

  v5.2 stays within 0.89-0.92 in every cell (12 to 35 years).
- **No true trend:** 5-11 % of records are called thickening or thinning (a 90 % interval allows 10 %); v5.1 reached 20 % for 12-year records at rho 0.5.
- **Stable needs long records:** with a true rate of 0 the share called stable is 0 at 12 and 18 years, 2-37 % at 25 years and 43-93 % at 35 years. The equivalence test needs the 90 % interval inside +/-5 %/dec, and with 7.5 % noise a 12-year record has a slope SE of about 6 %/dec. This is a property of the margin and the data, not a fault: use `p_stable` or `class_pooled` for short records.
- **Breakpoint test:** false-positive rate 4 % against a nominal 5 %.

### Caveats

- **Sharp cut-offs, uncertain inputs.** Classes have sharp edges, but the inputs are uncertain. Use `p_*` (posterior band probabilities from the pooled model) when one site's class matters.
- **Censored years.** The variance of S ignores censoring. With 35 censored values in 4,515 site-years the effect is small.
- **What the pooled model assumes.** It assumes normal site and region effects, and a few BNZ and Mackenzie sites have heavy-tailed rates. Pooling also borrows from neighbours, so `class_pooled` is a regional-context class. Report `class_site` for statements about a single record.
- **U19 Pearl Creek.** Its 1990-2014 years remain, because the build flags talik only from 2015. The v5 decision was to exclude the site.
