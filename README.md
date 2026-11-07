# Active Layer Dynamics Index (ALDI)

![Conceptual diagram of the proposed framework for ALDI construction and causal inference.](docs/Index_ProcessV3.jpg)

Active layer thickness (ALT) is a key indicator of permafrost response to environmental change. However, its long-term variation across regions can be driven by multiple geospatial factors whose relative contributions remain difficult to disentangle. In this work, we propose a framework combining two processes: long-term trajectory classification and causal inference. Accordingly, we introduce the Active Layer Dynamics Index (ALDI), categorizing ALT time series into six classes of direction and rate of change: Rapid and Gradual Thickening, Rapid and Gradual Thinning, Transitional and No Trend. ALDI relies on nonparametric trend detection, reversal identification, and bootstrap uncertainty quantification. We then conduct causal inference using site fixed-effects and first-difference estimators to examine how environmental drivers vary across ALDI classes. Applied to 129 CALM network sites (1990–2024), ALDI identifies a 3.5:1 thickening-to-thinning ratio across the pan-Arctic and substantial regional heterogeneity. In terms of causality, thermal forcing dominates at thickening sites but is absent in thinning and transitional regimes, where hydrological and disturbance processes prevail. These class-dependent relationships were not detectable in pooled analyses, underscoring the need for trajectory-aware analysis in permafrost research.

## How to run it

Three scripts, run in this order from this folder. Each one is the complete code of its
step, with its settings at the top.

| Step | Command | Time | Reads | Writes |
|---|---|---|---|---|
| 1. ALT target | `py -3.12 1_alt_target.py` | ~3 min | `data/raw/alt_sources/` | `data/1_alt_target/` |
| 2. Covariates (Earth Engine) | `py -3.12 2_covariates.py` | ~2 h the first time | step 1 | `data/2_covariates/` |
| 3. ALDI A + B | `py -3.12 3_aldi.py` | ~10-30 min | steps 1 and 2 | `data/3_aldi/` |

- **Step 1** rebuilds the ALT target from the raw files on the computer. `--download` first
  fetches new or missing source files (NEON then needs your token for that terminal:
  `$env:NEON_TOKEN="..."`; it is never saved).
- **Step 2**, one-time setup: `py -3.12 -m pip install earthengine-api geopandas scikit-learn`,
  then `earthengine authenticate` (browser sign-in), then the first run with
  `--project YOUR-EE-PROJECT-ID` (remembered in `data/ee_project.txt`). `--test` checks 3
  sites first. The default gets what ALDI B needs; `--all` gets every variable and merges
  them for the causal analysis. A stopped run continues where it left off.
- **Step 3** options: `--power` (NASA POWER temperature, robustness run), `--validate`
  (simulation check, 1-2 h).
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
| `data/` | everything that is data. Only on the computer, not on GitHub: `raw/` (downloads: ALT sources, permafrost map, NASA POWER), `1_alt_target/`, `2_covariates/`, `3_aldi/`, `previous/` |
| `docs/` | the method of each step in detail (`1_alt_target.md`, `2_covariates.md`, `3_aldi.md`) and the list of covariate columns |
| `extras/` | `ALDI_causality.ipynb` (causal analysis, after steps 2 and 3), `IndexVX_FiguresScriptV2.m` (figures, MATLAB), and the v4 paper version: `ALDI_Calculation_final.ipynb` and its data in `v4_data/` |
