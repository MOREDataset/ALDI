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

`py -3.12 3_aldi.py` builds the daily temperature series and TDD tables from the Earth Engine files of step 2 (`data/3_aldi/temperature/`) and runs v5.2, A and every B variant (about 10-30 minutes). `--power` adds the NASA POWER run (no account; the daily files are kept in `data/raw/nasa_power/`), `--validate` the simulation check (1-2 hours), `--sensitivity` the sensitivity runs (about 1 hour, `data/3_aldi_sensitivity/`), `--checks` the paper checks (about a minute, no refit, `data/3_aldi_checks/`; see below). `--no-main` reuses the existing `data/3_aldi` instead of refitting it. Outputs in `data/3_aldi/`: `aldi_v6_results.csv` (one row per site: v5.2 columns, v6 rate, interval, class and band probabilities, B expected / residual / total trends and classes per variant), `aldi_v6_network.csv`, `aldi_v6_summary.txt`, three figures, `aldi_v6_period_check.csv`.

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
| network-rate bias, %/dec (95 % interval covers the truth) | +0.2 (7 of 8) | | -0.7 (7 of 8) | |

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
subsidence record here). Fig 5 of `paper_figures.m` marks the B means of the three runs.

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

## Paper checks (`paper_checks`, `--checks`, `data/3_aldi_checks/`; added 2026-11-09)

Computed from the outputs of the main run (and of `--sensitivity` for the 2000-2024 cm
row), with no refit; about a minute. They turn the novelty of the paper (its uncertainty
accounting) into numbers a reader can compare with the usual analysis:

- **Years of record a site needs** (`aldi_v6_checks_detection.csv`, Table S4), per method
  family. With the fitted noise (each family's AR(1) coefficient and sigma) the SD of a
  trend from a given set of years is exact, so the chance of each class follows from the
  true rate. Two noise levels: in the network (the shared region-year anomalies removed,
  as the hierarchical model does) and alone (left in, as a per-site test). For consecutive
  years this is the detection time of Weatherhead et al. (1998), written for annual values:
  n* = [C sqrt(12) sigma sqrt((1 + rho)/(1 - rho)) / |w0|]^(2/3), with C = 2.93 for
  thickening at 90 % and 3.29 for showing "stable"; the table gives both.
- **Expected shares by record length** (`aldi_v6_checks_by_span.csv`, Table S5): the share
  of sites expected to stay unresolved from each record's own years and noise, beside the
  observed share (the dashes of Fig 3). Pooling toward the region narrows the hierarchical
  intervals further, so its expected share is an upper bound.
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
  kfixed) is the bound for an attenuated k: errors in TDD pull a fitted k toward 0, and
  Stefan's exponent is about the most the expected part can take.
- **Residual by ground ice and method** (`aldi_v6_checks_ice_method.csv`, Table S6): the
  low-minus-high ice contrast within regions, per method family, since probes and
  surface-referenced tubes carry the subsidence bias and ground temperatures do not.
- **Agreement of the methods** (`aldi_v6_checks_method.csv`, Table S9; added 2026-11-14):
  can probes, thaw tubes and ground temperatures share one index? Raw family means mix
  method with place (most thaw tubes are in the slowly deepening Mackenzie Delta), so each
  family is compared with the probes within regions (region fixed effects, 95 % interval
  from resampling whole regions), on the hierarchical rates, on the per-site OLS rates (no
  pooling, so the model cannot pull the methods together) and on the model B residuals.
  Second test at one place (series within 1 km): if the methods measured different things,
  two series of different methods at one place would differ more than two series of the
  same method there.
- **Rates in cm** (`aldi_v6_checks_cm.csv`, Table S7): site rates times the site's fitted
  ALT, and the per-site OLS on ALT in cm as in Streletskiy et al. (2026) and Lin et al.
  (EGUsphere-2026-841), beside their published values and Smith et al. (2024). The OLS is
  run on the whole record and again on the rule of Streletskiy et al. (2000-2024 only, at
  least ten years in that window). Their published shares (Commun. Earth Environ. 7:671,
  156 CALM sites): 55.2 % of Arctic sites increase significantly (53 sites, 0.8 +- 0.8
  cm/yr; the number of Arctic sites is not given), and 66.4 % in the Northern Hemisphere
  (91 of 137 sites), raised by the mountain sites (European mountains 18 of 19, Third Pole
  20 of 22). The Arctic share is the comparator for ALDI.
- **MCMC diagnostics** of every fit of the main run (Table S8) and the count of sites whose
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
