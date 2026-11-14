# Active Layer Dynamics Index (ALDI)

Active layer thickness (ALT), the depth of summer thaw above permafrost, is a key indicator
of how permafrost responds to a warming climate. This repository builds a circumarctic
multi-source ALT dataset and the Active Layer Dynamics Index (ALDI) computed from it:

- **ALT target (step 1).** End-of-season ALT from probing grids, thaw tubes and
  ground-temperature profiles of about 35 sources, merged into one series per site, with
  quality flags. Build of 12 November 2026: 358 sites and 6,605 valid site-years.
- **ALDI A (step 3).** One hierarchical Bayesian model of all series (log ALT, AR(1) noise
  per method family, shared region-year anomalies, partial pooling of site trends within
  250 km regions, lower-bound years as censored values). Each site gets a rate of change in
  % per decade, its 90 % interval and a class: rapid or gradual thickening, stable, gradual
  or rapid thinning, or indeterminate (the record cannot decide). Applied to 230 sites
  north of 60 N with at least 10 years over 12 or more years since 1990 (4,614
  site-years, 44 regions). Run of 12 November 2026: network rate +8.8 %/dec (95 %
  6.6-11.6); 57 sites thicken rapidly and 63 gradually, 13 are stable, 1 thins and 96
  are not yet resolved.
- **ALDI B (step 3).** Each site's trend split into the part expected from the air
  temperature record (thawing degree-days, Stefan's law, ERA5-Land corrected to the site
  elevation) and a residual that the modelled weather does not explain, in seven variants
  from this summer's thawing degree-days alone to degree-day memory, winter, snow and rain.
  The residual is a statistical decomposition, not an attribution of cause. Run of
  12 November 2026: thawing degree-days explain 1.3 of the mean site trend of
  6.6 %/dec, and 78 sites deepen faster than they predict.

The figure below shows the framework of the earlier version (v4, six classes and
class-stratified causal models on 129 CALM sites, EGUsphere-2026-2639). It is kept for
reference; the current method is described in `docs/3_aldi.md`.

![Framework of the earlier ALDI version (v4).](docs/Index_ProcessV3.jpg)

## How to run it

Three scripts, run in this order from this folder. Each one is the complete code of its
step, with its settings at the top.

| Step | Command | Time | Reads | Writes |
|---|---|---|---|---|
| 1. ALT target | `py -3.12 1_alt_target.py` | ~3 min | `data/raw/alt_sources/` | `data/1_alt_target/` |
| 2. Covariates (Earth Engine) | `py -3.12 2_covariates.py` | ~2 h the first time | step 1 | `data/2_covariates/` |
| 3. ALDI A + B | `py -3.12 3_aldi.py` | ~10-30 min | steps 1 and 2 | `data/3_aldi/` |
| 4. Paper figures and tables | in MATLAB, from this folder: `paper_figures` | ~1 min | step 3 | `data/4_paper/` |

- **Step 1** rebuilds the ALT target from the raw files on the computer. `--download` first
  fetches new or missing source files (NEON then needs your token for that terminal:
  `$env:NEON_TOKEN="..."`, and MET Norway's Frost API your client ID:
  `$env:FROST_CLIENT_ID="..."`; neither is ever saved).
- **Step 2**, one-time setup: `py -3.12 -m pip install earthengine-api geopandas scikit-learn`,
  then `earthengine authenticate` (browser sign-in), then the first run with
  `--project YOUR-EE-PROJECT-ID` (remembered in `data/ee_project.txt`). `--test` checks 3
  sites first. The default gets what ALDI B needs; `--all` gets every variable and merges
  them for the causal analysis. A stopped run continues where it left off.
- **Step 3** options: `--power` (NASA POWER temperature, robustness run), `--validate`
  (simulation check, 1-2 h), `--sensitivity` (fifteen runs that each change one choice:
  censored prior, undated surveys, site quality, disturbed plots, one series per place,
  2000-2024 window, record length, region size, and three thaw-subsidence bounds; about
  1 h), `--checks` (the paper checks: years of record a site needs, the standard per-site
  test on the same series, shrinkage, residual by ground ice and method, rates in cm; about
  a minute, no refit), `--spatial` (only the spatial checks on the existing results, about
  a minute; needs geopandas for the permafrost-map part). `--no-main` skips the main fit
  and adds the other runs to the existing `data/3_aldi`. Everything at once:
  `py -3.12 3_aldi.py --power --sensitivity --checks --validate`.
- **Step 4** (MATLAB R2021a or newer, no toolboxes) draws Figs 1-5 and S1-S4 of the
  Results (PNG and PDF) and writes Tables 1-3 and S1-S9 (CSV) and `key_numbers.txt` (the
  numbers quoted in the text). It uses the outputs of `3_aldi.py --power --sensitivity
  --checks --validate`; a missing run only drops its figure, table or markers.
- A new output replaces the current one only when its run finished without error. The
  output before it is kept in `data/previous/` (one copy).
- Every script prints its options with `-h`. Python 3.12 is needed (on this laptop `py -3.12`;
  plain `python` is a 3.15 without some of the packages). A script started with a Python that
  lacks a package stops at once and says what to install.

## What is where

| | |
|---|---|
| `1_alt_target.py` | ALT from every source and method (probing grids, thaw tubes, ground temperatures): download, read, merge, flags. The list of source files is at its end |
| `2_covariates.py` | climate, snow, soil, vegetation, fire, terrain and permafrost-map variables per site and year, from Google Earth Engine |
| `3_aldi.py` | ALDI v6: A (hierarchical trends and classes) and B (trend expected from air temperature and residual), the v5.2 comparison, the simulation check |
| `paper_figures.m` | the figures and tables of the paper's Results section (MATLAB), from the outputs of step 3 |
| `data/` | everything that is data. Only on the computer, not on GitHub: `raw/` (downloads: ALT sources, permafrost map, NASA POWER), `1_alt_target/`, `2_covariates/`, `3_aldi/`, `4_paper/`, `previous/` |
| `docs/` | the method of each step in detail (`1_alt_target.md`, `2_covariates.md`, `3_aldi.md`) and the list of covariate columns |
| `extras/` | earlier and exploratory work, not part of the current analysis: `ALDI_causality.ipynb` (the class-stratified causal analysis of v4, kept for reference), `IndexVX_FiguresScriptV2.m` (v4 figures, MATLAB), and the v4 paper version: `ALDI_Calculation_final.ipynb` and its data in `v4_data/` |
