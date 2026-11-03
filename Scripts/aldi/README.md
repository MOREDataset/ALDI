# ALDI v5.2 (trend classes for the multi-source ALT data)

`ALDI_Calculation_final.ipynb` (v4) is left as it was, so its results can still be reproduced. This folder is the revised index.

## Run

```
pip install numpy scipy pandas pyyaml
python run_aldi.py aldi_config.yaml  path/to/alt_multisource_annual.csv  out_dir
python validate_aldi.py aldi_config.yaml out_dir      # simulation check, about 5 minutes
```

Input is `alt_multisource_annual.csv`, not `_for_ALDI.csv`, because only the annual file keeps the lower-bound years. Every setting is in `aldi_config.yaml`, with the reason next to it.

Outputs:
- `aldi_v52_results.csv`: one row per site.
- `aldi_v52_removed_site_years.csv`: every used site-year of the input that was left out, with the reason (latitude / year window, minority method family, or a site too short inside the window).
- `aldi_v52_summary.txt`: class counts, network rate, flags, and classes by record length.
- `aldi_v52_config_used.yaml`: a copy of the settings.

## What changed from v4, and why

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

## v5.2 change (2026-11-03): pooled autocorrelation

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

## Results on the run-7 data (2026-11-02 build; 1990-2025, >=60 N, >=10 years over >=12 years: 229 sites)

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

## Simulation check (`validate_aldi.py`)

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

## Caveats

- **Sharp cut-offs, uncertain inputs.** Classes have sharp edges, but the inputs are uncertain. Use `p_*` (posterior band probabilities from the pooled model) when one site's class matters.
- **Censored years.** The variance of S ignores censoring. With 35 censored values in 4,515 site-years the effect is small.
- **What the pooled model assumes.** It assumes normal site and region effects, and a few BNZ and Mackenzie sites have heavy-tailed rates. Pooling also borrows from neighbours, so `class_pooled` is a regional-context class. Report `class_site` for statements about a single record.
- **U19 Pearl Creek.** Its 1990-2014 years remain, because the build flags talik only from 2015. The v5 decision was to exclude the site.
