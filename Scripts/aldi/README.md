# ALDI v5.1 (trend classes for the multi-source ALT data)

`ALDI_Calculation_final.ipynb` (v4) is left as it was, so its results can still be reproduced. This folder is the revised index.

## Run

```
pip install numpy scipy pandas pyyaml
python run_aldi.py aldi_config.yaml  path/to/alt_multisource_annual.csv  out_dir
python validate_aldi.py aldi_config.yaml out_dir      # simulation check, about 5 minutes
```

Input is `alt_multisource_annual.csv`, not `_for_ALDI.csv`, because only the annual file keeps the lower-bound years. Every setting is in `aldi_config.yaml`, with the reason next to it.

Outputs:
- `aldi_v51_results.csv`: one row per site.
- `aldi_v51_removed_site_years.csv`: every site-year that was left out, with the reason.
- `aldi_v51_summary.txt`: class counts, network rate, flags, and classes by record length.
- `aldi_v51_config_used.yaml`: a copy of the settings.

## What changed from v4, and why

| | v4 notebook | v5.1 |
|---|---|---|
| Gaps | PCHIP fills gaps of up to 2 years, and the filled values are tested as data | No filling. Theil-Sen and Kendall's S work with irregular years |
| Probe or sensor limit reached | The year is dropped | The year stays in as a right-censored value (Akritas-Theil-Sen). Dropping it removes the deepest years of deepening sites |
| Rate | cm/yr | % of the site's ALT per decade, so sites with 40 cm and 120 cm ALT are compared on one scale |
| Class thresholds | 50th and 80th percentiles of the sample's slopes, clipped. In the 1990+ data the 50th percentile (0.47 cm/yr) was clipped to 0.30, and the 80th gave 1.27 | Fixed: stable ±5 %/dec, rapid 10 %/dec (v5 decision). A site's class no longer depends on which other sites are in the sample |
| No trend | One class for "flat" and for "cannot tell" | `stable` means the 90 % interval lies inside ±5 %/dec (an equivalence test). `indeterminate` means the record cannot decide |
| Interval | Moving-block bootstrap (2-5-year blocks) | Kendall-S inversion with an AR(1) variance factor. Without autocorrelation this is exactly Gilbert's Sen-slope interval, and it also works with censored years |
| Reversal | Fixed first and last 7 years, slopes of opposite sign, both intervals excluding 0 | Broken-stick fit with an estimated breakpoint. The p-value comes from an AR(1) bootstrap of the sup-F statistic. A reversal needs both segments beyond ±5 %/dec with opposite signs. It is reported as a flag, not a class (v5 decision) |
| Method changes in a series | Mixed | Only the series' main method family is kept (probe, thaw tube or temperature). This removed 24 site-years at 20 sites |
| Pooling | None | Two-level random-effects model (regions of ≤250 km, then sites), fitted by REML. It gives each site's shrunk rate (`rate_pooled`) and class (`class_pooled`), the regional rate, and the network rate with regions as clusters |
| Extra outputs | | Minimum detectable trend, Benjamini-Hochberg q-values, survey-date drift flag, burned-site flag |

## Results on the 1990+ data (≥60°N, ≥10 years over ≥12 years: 221 sites)

| class | site only | pooled |
|---|---|---|
| rapid thickening | 39 | 40 |
| gradual thickening | 26 | 60 |
| stable | 18 | 23 |
| gradual thinning | 3 | 1 |
| rapid thinning | 4 | 1 |
| indeterminate | 131 | 96 |

- **Network rate:** +8.3 %/dec (95 % CI 5.8 to 10.8), from 41 regions. The between-region SD is 6.4 %/dec and the between-site SD 4.9 %/dec.
- **Detectable trend:** the median minimum detectable trend is 12.3 %/dec. That is why most single records are indeterminate.
- **Breakpoints:** 29 sites have a significant breakpoint, against about 10 expected by chance among the 191 tested. 17 of them are reversals.
- **Comparison with v4 on the same 221 sites:**
  - Of the 96 sites v4 called `no_trend`, 17 are stable, 74 indeterminate and 5 trending.
  - Of the 16 v4 `gradual_thinning` sites, 11 are indeterminate and 3 are rapid thinning.
  - All 5 v4 `transitional` sites are indeterminate.
- **Sensitivity** (site-only counts; the network rate stays 8.2-8.9 %/dec in every run):
  - ≥8 years: 272 sites, same pattern.
  - Margins 2.5/10: stable drops from 18 to 4.
  - Margins 5/15: rapid thickening drops from 39 to 25.
  - Without the censored years, or with mixed methods: within 1-2 sites of the main run.
  - Regions of 150 or 500 km: the pooled counts move by 1-3.
  - From 2000 instead of 1990: 200 sites, stable 8.

## Simulation check (`validate_aldi.py`)

Synthetic series were generated with 7.5 % noise (the median of the real series) and 10 % missing years, over 12-35 years, with AR(1) ρ of 0, 0.3 and 0.5.

- **Interval coverage (nominal 90 %):**
  - v5.1: 92 % at ρ = 0, 87 % at ρ = 0.3 and 84 % at ρ = 0.5.
  - v4 bootstrap: 93 %, 91 % and 86 %.
  - Both run a little narrow when autocorrelation is strong. The real series have a median raw ρ of 0.13.
- **No true trend:**
  - With ρ = 0, 5-9 % of records are called thickening or thinning. A 90 % interval allows 10 %.
  - With ρ = 0.5, up to 20 % of 12-year records are.
  - Only records of 35 years are mostly called stable (89 % at ρ = 0). Shorter ones are mostly indeterminate.
- **Breakpoint test:** the false-positive rate is 2-6 % against a nominal 5 %.

## Caveats

- **Sharp cut-offs, uncertain inputs.** Classes have sharp edges, but the inputs are uncertain. Use `p_*` (posterior band probabilities from the pooled model) when one site's class matters.
- **Censored years.** The variance of S ignores censoring. With 34 censored values in 4,424 site-years the effect is small.
- **What the pooled model assumes.** It assumes normal site and region effects, and a few BNZ and Mackenzie sites have heavy-tailed rates. Pooling also borrows from neighbours, so `class_pooled` is a regional-context class. Report `class_site` for statements about a single record.
- **U19 Pearl Creek.** Its 1990-2014 years remain, because the build flags talik only from 2015. The v5 decision was to exclude the site.
