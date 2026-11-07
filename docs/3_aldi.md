# Step 3: ALDI v6, hierarchical trends (A) and climate-adjusted index (B) (`3_aldi.py`)

v5.2 (described at the end of this note) is part of `3_aldi.py`. v6 uses the same sample rules, class margins and
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

Variants (`SETTINGS`, `climate: variants`): `main` (above); `kfixed` (k = 0.5, pure Stefan);
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

`py -3.12 3_aldi.py` builds the daily temperature series and TDD tables from the Earth Engine files of step 2 (`data/3_aldi/temperature/`) and runs v5.2, A and every B variant (about 10-30 minutes). `--power` adds the NASA POWER run (no account; the daily files are kept in `data/raw/nasa_power/`), `--validate` the simulation check (1-2 hours), `--sensitivity` the sensitivity runs (30-40 minutes, `data/3_aldi_sensitivity/`). `--no-main` reuses the existing `data/3_aldi` instead of refitting it. Outputs in `data/3_aldi/`: `aldi_v6_results.csv` (one row per site: v5.2 columns, v6 rate, interval, class and band probabilities, B expected / residual / total trends and classes per variant), `aldi_v6_network.csv`, `aldi_v6_summary.txt`, three figures, `aldi_v6_period_check.csv`.

## Results on run 7 (229 sites, 4,550 site-years, 41 regions)

| class | v5.2 site | v5.2 pooled | v6 | v6 no region-year | v6 Student-t |
|---|---|---|---|---|---|
| rapid thickening | 39 | 35 | 56 | 55 | 54 |
| gradual thickening | 19 | 66 | 60 | 47 | 62 |
| stable | 11 | 23 | 13 | 16 | 14 |
| gradual thinning | 2 | 1 | 1 | 1 | 1 |
| rapid thinning | 4 | 0 | 0 | 0 | 0 |
| indeterminate | 154 | 104 | 99 | 110 | 98 |

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
- **B (ERA5-Land TDD, all 229 sites, 4,550 site-years; Earth Engine export of 7 November
  2026).** This year's k = 0.21 (0.19-0.23; probe 0.26, thaw tube 0.17, temperature
  0.20); median TDD trend +3.9 %/dec; TDD explains a median 15 % of the year-to-year ALT
  variance (below 0 at 49 sites). Mean site trends in %/dec with region-bootstrap 95 %
  intervals (total 6.3 in every variant):

  | variant | expected | residual | sites deeper than predicted |
  |---|---|---|---|
  | main (this summer's TDD) | 1.3 (0.8-2.0) | 5.0 (3.0-7.4) | 77 |
  | kfixed (k = 0.5) | 2.4 (1.3-3.9) | 3.9 (2.0-6.1) | 61 |
  | lag (+ last summer) | 1.6 (1.0-2.6) | 4.6 (2.7-7.0) | 73 |
  | memory (+ the two previous summers) | 1.9 (1.2-3.1) | 4.3 (2.4-6.7) | 68 |
  | air (+ last summer, winter FDD) | 2.1 (1.4-3.0) | 4.1 (2.2-6.4) | 65 |
  | weather (air + snow depth, summer rain) | 2.5 (1.8-3.4) | 3.7 (1.9-5.8) | 59 |
  | weather_memory | 3.0 (2.1-4.0) | 3.2 (1.5-5.4) | 54 |

  Long-run k with memory: probe 0.40, thaw tube 0.29, temperature 0.29. With NASA POWER
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

## Spatial checks (`spatial_checks`, also `--spatial`)

Added 2026-11-07 after a geodata review. They use only the fitted rates, so
`py -3.12 3_aldi.py --spatial` reruns them on the existing results in about a minute
(the permafrost-map part needs geopandas, as step 2). Results on run 7:

- **Places, not series.** 229 series stand at 173 places: series within 1 km of each other
  (a CALM thaw tube and its probing grid, a grid and the borehole beside it, CALM and LTER
  plots of one catchment) are one place. 36 places hold 92 series. Their rates correlate at
  only 0.50 (median difference 2.7 %/dec), and at 25 of those places the series fall in
  different classes. The build keeps them apart on purpose (the methods read different
  levels), so class counts should be reported per series and per place, and the
  difference between series at one place is a direct measure of method and micro-site
  uncertainty. `location_id` in the results.
- **Declustered network means.** The mean over series lets dense clusters dominate
  (Mackenzie Delta 35 series at +1.3 %/dec, Fairbanks 33 at +10.3). Giving each cell of an
  equal-area grid (Lambert azimuthal, North Pole) one vote gives +9.2, +10.0 and +9.6 %/dec
  for 100, 250 and 500 km cells, close to the mean over regions (+9.3) and far from the mean
  over series (+7.0). Residual (main): 6.6-7.2 declustered vs 5.0 over series; residual
  (weather_memory): 4.6-5.2 vs 3.2.
- **Do the 250 km regions capture the spatial dependence?** Site rates correlate at 0.38
  within 100 km and 0.19 at 100-250 km, and not beyond 250 km. Deviations of site rates
  from their region rate do not correlate across region borders at any distance (-0.06 to
  0.03), so the region level of the model absorbs the spatial dependence of the trends.
- **Permafrost zone and ground ice (IPA map, Brown et al. 2002).** 150 series are in
  continuous, 76 in discontinuous and 3 in sporadic permafrost; 92 in high, 87 in medium
  and 50 in low ground-ice terrain (sites in lakes or on the coast take the nearest polygon
  within 20 km). Raw medians differ a lot (residual 7.5 %/dec in discontinuous vs 3.3 in
  continuous; 7.7 in low-ice vs 3.3 in high-ice terrain), but almost all of that is between
  regions, mainly Fairbanks. Within the 5 regions that hold both kinds, discontinuous minus
  continuous is +0.5 %/dec (-0.3 to +3.6) and low minus high ice +2.0 (-0.7 to +4.3). The
  map is independent of the ALT data, so it is a valid stratifier for B; with 5 regions the
  within-region contrasts are weak. `pf_extent` and `pf_ice` in the results.

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
- **Same sample as v5.2**, including U19 1990-2014 and the 60 N cut (open decisions). The
  cut costs little: only 2 permafrost sites south of 60 N have long enough records
  (Kamchatka R30C and Nunavik TSJQ304).
- **Most probe surveys have no date.** 1,822 of 2,916 probe site-years (the CALM summary
  table) take the network median survey day (DOY 262, 19 September) as the end of their
  TDD window, 293 the site's median day, and 794 their own date. Where a site is surveyed
  in mid-August, its TDD includes about a month after the survey. That adds noise to the
  site's TDD (and lowers k a little) and puts September warming into the expected trend.
  A sensitivity run with an earlier default day for undated surveys would bound it.
- **Disturbed plots.** Bonanza Creek LTER (27 of the 33 series of the Fairbanks region)
  includes fire and fire-line plots. The 6 series flagged burned all fall in
  `deeper_than_TDD` (residuals +8.5 to +14.6 %/dec): B responds to a known non-climatic
  disturbance as it should, but those plots also raise the network residual. Report B with
  and without the disturbed plots.

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
