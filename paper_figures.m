%% paper_figures.m - figures and tables of the ALDI paper (Results section)
%
% Step 4, after 1_alt_target.py, 2_covariates.py and 3_aldi.py. Reads what 3_aldi.py
% wrote in data\ and writes every figure (PNG at 300 dpi and vector PDF) and table (CSV)
% of the Results section to data\4_paper\, plus key_numbers.txt with the numbers quoted
% in the text.
%
% Run from the Index folder:   >> paper_figures
% Needs MATLAB R2021a or newer, no toolboxes.
%
% Reads (all written by 3_aldi.py; the runs of 7 November 2026 are the canonical ones):
%   data\3_aldi\aldi_v6_results.csv, aldi_v6_network.csv, aldi_v6_summary.txt
%   data\3_aldi_power\                (py -3.12 3_aldi.py --power)
%   data\3_aldi_sensitivity\          (--sensitivity)
%   data\3_aldi_validation\           (--validate; the per-site OLS rows need the
%                                      3_aldi.py of 9 November 2026 or later)
%   data\3_aldi_checks\               (--checks: detection time, the standard per-site
%                                      test, shrinkage, rates in cm; 3_aldi.py of
%                                      9 November 2026 or later)
%   data\1_alt_target\alt_multisource_annual.csv, source_inventory.csv
%   data\raw\permafrost_map\nhipa.byte  (IPA permafrost map, NSIDC GGD318; map background)
%
% Every figure has one panel and makes one point of the Results. Numbering follows the
% draft of 14 November 2026 (three research questions; supplement numbered by first
% citation). Model B (the part of the trend associated with air temperature) was set
% aside for a follow-up paper on that day: MODEL_B below brings its figure and tables
% back, under names that start with B_.
% Main text
%   Fig 1  Sites on the IPA permafrost map, coloured by ALDI class          (Results 3.3)
%   Fig 2  Regional and network rates of ALT change                         (Results 3.2)
%   Fig 3  Site classes by record length, per-site method vs hierarchical   (Results 3.3)
%   Fig 4  Simulation: coverage of 90 % intervals by record length          (Results 3.3)
%   Table 2  Summaries of the network trend: mu, mean of the regions, means over equal-area
%            cells, places and series, from the posterior draws (Table 1, published
%            studies, is not made here)
% Supplement
%   Fig S1  Site-years per year by measurement method
%   Fig S2  Sensitivity runs: network rate of each run
%   Fig S3  Regional trends of all regions with their 90 and 95 % intervals
%   Fig S4  Site rate on its own (per-site OLS) vs in the hierarchical model
%   Table S1  Data sources of the analysed series
%   Table S2  MCMC diagnostics of the fits of Model A
%   Table S3  Agreement of the three methods: within regions (with each shared region left
%             out in turn) and at one place (TableS3_method_agreement, TableS3_method_loro),
%             and the series of each method (TableS3_by_method)
%   Table S4  Sensitivity runs and class margins
%   Table S5  Years of record a site needs, per method
%   Table S6  By record length: observed and expected shares, the standard per-site test
%   Table S7  Rates in cm per decade beside published values
%   Table S8  Simulation check
%   Table S9  Regional trends: median, 90 and 95 % intervals, P(trend > 0)
%   Table S10 Rates and classes from the record alone (per-site Sen + AR(1), site-only
%             Bayesian fit) and from the hierarchical model
% With MODEL_B = true also: B_Fig_expected_vs_residual (Fig 5 until 14 November 2026),
% B_Fig_ground_ice_within_regions, B_Table_decomposition, B_Table_residual_by_ice_and_method,
% and the Model B columns and markers of the tables above and of Fig S2.

clear; close all;

%% Settings
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
DATA = fullfile(here, 'data');
OUT  = fullfile(DATA, '4_paper');
if ~exist(OUT, 'dir'), mkdir(OUT); end

MIN_SITES_REGION = 4;          % regions with fewer sites are pooled in Fig 2
MODEL_B = false;               % Model B outputs (follow-up paper): false since 14 Nov 2026
W1 = 8.3; W2 = 17.5;           % Copernicus one- and two-column widths, cm
FONT = 8;
set(groot, 'defaultAxesFontSize', FONT, 'defaultTextFontSize', FONT, ...
    'defaultAxesTickDir', 'out', 'defaultAxesBox', 'off', ...
    'defaultAxesTickLength', [0.015 0.015]);

% classes in display order (as written by 3_aldi.py), labels and colours
CLS   = ["rapid_thickening" "gradual_thickening" "stable" "gradual_thinning" "rapid_thinning" "indeterminate"];
CLS_L = ["Rapid thickening" "Gradual thickening" "Stable" "Gradual thinning" "Rapid thinning" "Not yet resolved"];
CLS_C = [0.70 0.09 0.17; 0.96 0.55 0.33; 0.36 0.62 0.36; 0.47 0.67 0.86; 0.13 0.30 0.62; 0.82 0.82 0.82];
RES   = ["deeper_than_TDD" "consistent_with_TDD" "shallower_than_TDD" "not_resolved"];
RES_L = ["Positive residual after TDD" "Residual within ±5 % per decade" ...
         "Negative residual after TDD" "Residual not resolved"];
RES_C = [0.70 0.09 0.17; 0.36 0.62 0.36; 0.13 0.30 0.62; 0.75 0.75 0.75];
FAM   = ["probe" "thaw_tube" "temperature"];
FAM_L = ["Probing" "Thaw tube" "Ground temperature"];
FAM_M = ['o' 's' '^'];
FAM_C = [0.27 0.45 0.70; 0.90 0.60 0.20; 0.40 0.66 0.40];
SPAN_L = ["12-15" "16-20" "21-25" "26+"];
METH   = ["ols_site" "v5.2_site" "v6_ry"];                  % simulation methods
METH_L = ["Per-site OLS" "Per-site Sen + AR(1)" "Hierarchical (this study)"];
METH_C = [0.15 0.15 0.15; 0.62 0.62 0.62; 0.70 0.09 0.17];

%% Read the results
rd = @(f) readtable(f, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
res  = rd(fullfile(DATA, '3_aldi', 'aldi_v6_results.csv'));
net  = rd(fullfile(DATA, '3_aldi', 'aldi_v6_network.csv'));
summ = fileread(fullfile(DATA, '3_aldi', 'aldi_v6_summary.txt'));
pw_dir = fullfile(DATA, '3_aldi_power');
have_power = isfile(fullfile(pw_dir, 'aldi_v6_results.csv'));
if have_power
    pres = rd(fullfile(pw_dir, 'aldi_v6_results.csv'));
    pnet = rd(fullfile(pw_dir, 'aldi_v6_network.csv'));
end
sens_f = fullfile(DATA, '3_aldi_sensitivity', 'aldi_v6_sensitivity.csv');
have_sens = isfile(sens_f);
if have_sens, sens = rd(sens_f); end
sim_f = fullfile(DATA, '3_aldi_validation', 'aldi_v6_simulation.csv');
have_sim = isfile(sim_f);
if have_sim, sim = rd(sim_f); end
CK = fullfile(DATA, '3_aldi_checks');
have_ck = isfile(fullfile(CK, 'aldi_v6_checks_by_span.csv'));
if have_ck
    cksp = read_text_cols(fullfile(CK, 'aldi_v6_checks_by_span.csv'), {'span'});
    cks  = read_text_cols(fullfile(CK, 'aldi_v6_checks_sites.csv'), {'site_id', 'span_bin', 'family', ...
        'class_v6', 'class_site', 'class_ols', 'pf_ice'});
    ckd  = rd(fullfile(CK, 'aldi_v6_checks_detection.csv'));
    ck_txt = fileread(fullfile(CK, 'aldi_v6_checks.txt'));
else
    warning('No %s: run py -3.12 3_aldi.py --no-main --checks for the dashes of Fig 3, Fig S4 and Tables S2, S3 and S5-S7', CK);
end

n_sites = height(res);
span_bin = strings(n_sites, 1);
span_bin(res.span_years <= 15) = SPAN_L(1);
span_bin(res.span_years > 15 & res.span_years <= 20) = SPAN_L(2);
span_bin(res.span_years > 20 & res.span_years <= 25) = SPAN_L(3);
span_bin(res.span_years > 25) = SPAN_L(4);
burned = as_logical(res.flag_burned);
if MODEL_B
    ex = res.expected_rate; rs = res.resid_rate;         % Model B, air temperature, ERA5-Land
end
fprintf('%d sites read from data\\3_aldi\n', n_sites);

% all figures and tables are remade below; this also clears the names of earlier versions
% (the numbering changed on 14 November 2026)
old = [dir(fullfile(OUT, 'Fig*.png')); dir(fullfile(OUT, 'Fig*.pdf')); dir(fullfile(OUT, 'Table*.csv')); ...
       dir(fullfile(OUT, 'B_*.*'))];
for i = 1:numel(old), delete(fullfile(OUT, old(i).name)); end

% analysed site-years (one per site and year, inside the analysed window): Fig S1, Table S1
ann = rd(fullfile(DATA, '1_alt_target', 'alt_multisource_annual.csv'));
ann = ann(as_logical(ann.use), :);
[tf, loc] = ismember(ann.site_id, res.site_id);
ann = ann(tf, :); loc = loc(tf);
inwin = ann.year >= res.year_min(loc) & ann.year <= res.year_max(loc);
ann = ann(inwin, :); loc = loc(inwin);
ann.family = res.family(loc);
% only the series' own method family: 3_aldi.py drops the years of a second method in a
% series (single_method_family), so Fig S1 counts exactly the measured values in the model
mfam = strings(height(ann), 1);
mfam(ann.method == "probe") = "probe";
mfam(ann.method == "thaw_tube") = "thaw_tube";
mfam(ismember(ann.method, ["borehole_temp" "soil_temp" "soil_temp_monthly"])) = "temperature";
ann = ann(mfam == ann.family, :);
[~, iu] = unique(ann.site_id + "_" + string(ann.year));
ann = ann(iu, :);
fprintf('%d analysed site-years (sum of n_obs in the results: %d)\n', height(ann), sum(res.n_obs));

%% Figure 1: sites on the permafrost map, coloured by class
fig = new_figure(W2, 11.5);
ax = axes(fig); hold(ax, 'on');
XL = [-3.9e6 3.3e6]; YL = [-3.4e6 3.2e6];             % map window, m (EASE-Grid North)
ipa = fullfile(DATA, 'raw', 'permafrost_map', 'nhipa.byte');
zone_l = strings(0); zone_c = zeros(0, 3);
if isfile(ipa)
    [rgb, xg, yg, zone_l, zone_c] = ipa_image(ipa, XL, YL);
    image(ax, xg, yg, rgb);
else
    warning('No %s: map without permafrost background', ipa);
end
set(ax, 'YDir', 'normal');
for la = [60 70 80]                                     % parallels
    [x, y] = ease_xy(la * ones(1, 181), 0:2:360);
    plot(ax, x, y, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.4);
    [x, y] = ease_xy(la, -10);                          % labels along 10°W
    text(ax, x, y, sprintf('%d°N', la), 'FontSize', FONT - 1, 'Color', [0.35 0.35 0.35]);
end
for lo = 0:45:315                                       % meridians
    [x, y] = ease_xy(55:90, lo * ones(1, 36));
    plot(ax, x, y, ':', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.3);
end
[sx, sy] = ease_xy(res.lat, res.lon);
for c = numel(CLS):-1:1                                 % not resolved first, underneath
    for f = 1:numel(FAM)
        k = res.class_v6 == CLS(c) & res.family == FAM(f);
        if ~any(k), continue; end
        scatter(ax, sx(k), sy(k), 16, CLS_C(c, :), FAM_M(f), 'filled', ...
            'MarkerEdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.3);
    end
end
axis(ax, 'equal'); xlim(ax, XL); ylim(ax, YL); axis(ax, 'off');
h = gobjects(0); lab = strings(0);
for c = 1:numel(CLS)
    if any(res.class_v6 == CLS(c))
        h(end+1) = scatter(ax, nan, nan, 25, CLS_C(c, :), 's', 'filled', 'MarkerEdgeColor', 'k'); %#ok<SAGROW>
        lab(end+1) = sprintf('%s (%d)', CLS_L(c), sum(res.class_v6 == CLS(c))); %#ok<SAGROW>
    end
end
for f = 1:numel(FAM)
    h(end+1) = scatter(ax, nan, nan, 16, [1 1 1], FAM_M(f), 'MarkerEdgeColor', 'k'); %#ok<SAGROW>
    lab(end+1) = sprintf('%s (%d)', FAM_L(f), sum(res.family == FAM(f))); %#ok<SAGROW>
end
for z = 1:numel(zone_l)                                 % map background
    h(end+1) = patch(ax, NaN, NaN, zone_c(z, :), 'EdgeColor', [0.6 0.6 0.6], 'LineWidth', 0.3); %#ok<SAGROW>
    lab(end+1) = zone_l(z); %#ok<SAGROW>
end
legend(ax, h, lab, 'Location', 'eastoutside', 'Box', 'off', 'FontSize', FONT - 1);
save_figure(fig, OUT, 'Fig1_site_map');

%% Figure 2: regional and network rates
nw  = net_value(net, 'A_main_network_rate_pct_dec');
nsm = net_value(net, 'A_main_mean_site_rate_pct_dec');
reg = region_table(res, MIN_SITES_REGION);
fig = new_figure(11, 9);
ax = axes(fig); hold(ax, 'on');
nr = height(reg);
patch(ax, [nw(2) nw(3) nw(3) nw(2)], [0.4 0.4 nr + 0.6 nr + 0.6], CLS_C(1, :), ...
    'FaceAlpha', 0.12, 'EdgeColor', 'none');
xline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
h1 = xline(ax, nw(1), '-', 'Color', CLS_C(1, :), 'LineWidth', 1);
h2 = xline(ax, nsm(1), '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);
rng(1);                                                  % jitter, reproducible
for i = 1:nr
    r = reg.rates{i};
    scatter(ax, r, i + 0.25 * (rand(size(r)) - 0.5), 9, [0.55 0.55 0.55], 'filled', ...
        'MarkerFaceAlpha', 0.6);
    plot(ax, reg.region_rate(i), i, 'd', 'MarkerSize', 6, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
end
yticks(ax, 1:nr); yticklabels(ax, reg.label); ylim(ax, [0.4 nr + 0.6]);
xlabel(ax, 'ALT change (% per decade)');
legend(ax, [h1 h2], {sprintf('Network rate, %.1f (%.1f to %.1f)', nw), ...
    sprintf('Mean of sites, %.1f', nsm(1))}, 'Location', 'southoutside', 'Box', 'off');
save_figure(fig, OUT, 'Fig2_regional_rates');

%% Figure 3: site classes by record length, per-site method vs hierarchical
% Black dashes (3_aldi.py --checks): the share of sites expected NOT to be resolved yet,
% from each record's years and noise and the hierarchical rate as the true rate; for a
% site on its own on the S bars, with the shared year-to-year anomalies of its region
% removed on the H bars. The hierarchical model also pools toward the region, so it can
% resolve more sites than its dash. Where a bar's grey top reaches its dash, the
% unresolved sites are what record length allows, not a weakness of the method.
fig = new_figure(12, 7.5);
ax = axes(fig); hold(ax, 'on');
Y = zeros(8, numel(CLS)); xpos = [1 2 4 5 7 8 10 11]; nbin = zeros(1, 4);
for b = 1:4
    k = span_bin == SPAN_L(b); nbin(b) = sum(k);
    Y(2*b - 1, :) = 100 * mean(res.class_site(k) == CLS, 1);
    Y(2*b, :)     = 100 * mean(res.class_v6(k) == CLS, 1);
end
hb = bar(ax, xpos, Y, 0.85, 'stacked', 'EdgeColor', 'none');
for c = 1:numel(CLS), hb(c).FaceColor = CLS_C(c, :); end
ylim(ax, [0 100]); xlim(ax, [0 12]);
xticks(ax, [1.5 4.5 7.5 10.5]); xticklabels(ax, compose('%s\\newlinen=%d', SPAN_L', nbin'));
text(ax, xpos, 101 * ones(size(xpos)), repmat(["S" "H"], 1, 4), 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontSize', FONT - 1);
xlabel(ax, {'Record length (years)', 'S: per-site Sen + AR(1), H: hierarchical (this study)'});
ylabel(ax, 'Share of sites (%)');
hL = fliplr(hb); lL = fliplr(CLS_L); hE = [];
if have_ck
    for b = 1:4
        k = cksp.span == SPAN_L(b);
        if ~any(k), continue; end
        yS = 100 - cksp.expected_unresolved_alone_pct(k);
        yH = 100 - cksp.expected_unresolved_network_pct(k);
        hE = plot(ax, xpos(2*b - 1) + [-0.42 0.42], [yS yS], 'k-', 'LineWidth', 1.4);
        plot(ax, xpos(2*b) + [-0.42 0.42], [yH yH], 'k-', 'LineWidth', 1.4);
    end
end
if ~isempty(hE)                         % bars and a line: concatenate, do not assign
    hL = [hL hE]; lL = [lL sprintf('Expected not resolved\n(record length, noise)')];
end
legend(ax, hL, lL, 'Location', 'eastoutside', 'Box', 'off', 'FontSize', FONT - 1);
save_figure(fig, OUT, 'Fig3_classes_by_record_length');

%% Figure 4: simulation check, coverage of 90 % intervals by record length
% The rates of false and wrong-sign trends of the same simulation are quoted in the text
% (key_numbers.txt) and listed in Table S8.
if have_sim
    fig = new_figure(W1, 8);
    ax = axes(fig); hold(ax, 'on');
    yline(ax, 90, '--', 'Nominal 90 %', 'Color', [0.4 0.4 0.4], 'LabelHorizontalAlignment', 'left', ...
        'FontSize', FONT - 1);
    hm = gobjects(0); ml = strings(0);
    for m = 1:numel(METH)
        k = sim.scenario == "A_fit" & sim.method == METH(m);
        if ~any(k), continue; end
        cov = arrayfun(@(b) sim.(char("coverage_" + SPAN_L(b)))(k), 1:4);
        hm(end+1) = plot(ax, 1:4, 100 * cov, '-o', 'Color', METH_C(m, :), 'MarkerFaceColor', METH_C(m, :), ...
            'MarkerSize', 4, 'LineWidth', 1); %#ok<SAGROW>
        ml(end+1) = sprintf('%s: %.0f %% overall', METH_L(m), 100 * sim.coverage90(k)); %#ok<SAGROW>
    end
    xticks(ax, 1:4); xticklabels(ax, SPAN_L); xlim(ax, [0.6 4.4]); ylim(ax, [50 100]);
    xlabel(ax, 'Record length (years)'); ylabel(ax, 'Coverage of 90 % intervals (%)');
    legend(ax, hm, ml, 'Location', 'southoutside', 'Box', 'off', 'FontSize', FONT - 1);
    save_figure(fig, OUT, 'Fig4_simulation_coverage');
end

%% Model B (MODEL_B): expected from air temperature vs residual trend (Fig 5 until 14 Nov 2026)
% Large markers: the mean over sites of the main fit; with k fixed at 0.5 (Stefan's
% exponent: errors in TDD pull a fitted k toward 0, so this is about the most the
% expected part can be); and with thaw subsidence added at the probe and thaw-tube
% series on medium- or high-ice ground (sensitivity runs: 0.2, 0.5 and 1.0 cm/yr; the
% surface these methods measure from sinks, so their ALT trend is too small).
if MODEL_B
    fig = new_figure(14, 12.5);
    ax = axes(fig); hold(ax, 'on');
    plot(ax, [res.expected_lo res.expected_hi]', [rs rs]', '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.4);
    plot(ax, [ex ex]', [res.resid_lo res.resid_hi]', '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.4);
    xline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
    yline(ax, 0, '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 0.8);          % zero residual
    h = gobjects(0); lab = strings(0);
    for c = numel(RES):-1:1
        k = res.class_resid == RES(c);
        if ~any(k), continue; end
        h(end+1) = scatter(ax, ex(k), rs(k), 14, RES_C(c, :), 'filled', 'MarkerEdgeColor', [0.2 0.2 0.2], ...
            'LineWidth', 0.3); %#ok<SAGROW>
        lab(end+1) = sprintf('%s (%d)', RES_L(c), sum(k)); %#ok<SAGROW>
    end
    if any(burned)
        h(end+1) = scatter(ax, ex(burned), rs(burned), 40, 'k', 'LineWidth', 0.8);
        lab(end+1) = sprintf('Burned plots (%d)', sum(burned));
    end
    h = fliplr(h); lab = fliplr(lab);
    bm = [net_value(net, 'B_main_mean_expected_pct_dec'); net_value(net, 'B_main_mean_residual_pct_dec')];
    bk = [net_value(net, 'B_kfixed_mean_expected_pct_dec'); net_value(net, 'B_kfixed_mean_residual_pct_dec')];
    sub_runs = ["main" "subsidence_0.2" "subsidence_0.5" "subsidence_1.0"];
    if have_sens
        [tf, loc] = ismember(sub_runs, sens.run);
        if sum(tf) >= 2
            h(end+1) = plot(ax, sens.B_mean_expected(loc(tf)), sens.B_mean_residual(loc(tf)), '-^', 'Color', 'k', ...
                'MarkerFaceColor', [0.6 0.6 0.6], 'MarkerSize', 5, 'LineWidth', 0.8);
            lab(end+1) = "Mean with hypothetical subsidence (0.2, 0.5, 1.0 cm/yr)";
            dx = 0.02 * (max(res.expected_hi) - min(res.expected_lo));   % label beside its marker
            for r = find(tf & sub_runs ~= "main")
                text(ax, sens.B_mean_expected(loc(r)) + dx, sens.B_mean_residual(loc(r)), ...
                    "+" + erase(sub_runs(r), "subsidence_") + " cm/yr", 'FontSize', FONT - 1, ...
                    'VerticalAlignment', 'middle', 'BackgroundColor', 'w', 'Margin', 0.5);
            end
        end
    end
    if ~isnan(bk(1, 1))
        h(end+1) = plot(ax, bk(1, 1), bk(2, 1), 'd', 'Color', 'k', 'MarkerFaceColor', 'w', 'MarkerSize', 7, ...
            'LineWidth', 1);
        lab(end+1) = "Mean, k fixed at Stefan's 0.5";
    end
    bw = [net_value(net, 'B_weather_memory_mean_expected_pct_dec'); net_value(net, 'B_weather_memory_mean_residual_pct_dec')];
    if ~isnan(bw(1, 1))
        h(end+1) = plot(ax, bw(1, 1), bw(2, 1), 's', 'Color', 'k', 'MarkerFaceColor', 'w', 'MarkerSize', 7, ...
            'LineWidth', 1);
        lab(end+1) = "Mean, all weather terms + 2-year memory";
    end
    h(end+1) = plot(ax, bm(1, 1), bm(2, 1), 'd', 'Color', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
    lab(end+1) = "Mean, summer TDD (main fit)";
    xlim(ax, [floor(min(res.expected_lo)) - 0.5, ceil(max(res.expected_hi)) + 0.5]);
    ylim(ax, [floor(min(res.resid_lo)) - 1, ceil(max(res.resid_hi)) + 1]);
    xl = xlim(ax);
    text(ax, xl(2), 0, 'zero residual ', 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom', ...
        'FontSize', FONT - 1, 'Color', [0.3 0.3 0.3]);
    xlabel(ax, 'Trend associated with summer air temperature, TDD (% per decade)');
    ylabel(ax, 'Residual trend after TDD (% per decade)');
    legend(ax, h, lab, 'Location', 'southoutside', 'NumColumns', 2, 'Box', 'off', 'FontSize', FONT - 1);
    save_figure(fig, OUT, 'B_Fig_expected_vs_residual');
end

%% Table S1: data sources of the analysed series
inv_f = fullfile(DATA, '1_alt_target', 'source_inventory.csv');
[src, ~, gi] = unique(ann.source);
T1 = table(src, accumarray(gi, 1), 'VariableNames', {'source', 'site_years'});
T1.series = accumarray(gi, (1:height(ann))', [], @(i) numel(unique(ann.site_id(i))));
T1.methods = accumarray(gi, (1:height(ann))', [], @(i) {strjoin(unique(ann.method(i)), ', ')});
T1.methods = string(T1.methods);
if isfile(inv_f)
    inv = rd(inv_f);
    [tf, loc] = ismember(T1.source, inv.source);
    T1.title = strings(height(T1), 1); T1.reference = T1.title; T1.licence = T1.title;
    T1.title(tf) = inv.title(loc(tf)); T1.reference(tf) = inv.reference(loc(tf));
    T1.licence(tf) = inv.licence(loc(tf));
end
T1 = sortrows(T1, 'site_years', 'descend');
writetable(T1, fullfile(OUT, 'TableS1_data_sources.csv'));

%% Table 2: summaries of the network trend (from the posterior draws; 3_aldi.py of 14 Nov 2026)
sum_f = fullfile(DATA, '3_aldi', 'aldi_v6_network_summaries.csv');
if isfile(sum_f)
    copyfile(sum_f, fullfile(OUT, 'Table2_network_summaries.csv'));
else
    warning('No %s: rerun py -3.12 3_aldi.py (main run of 14 November 2026 or later)', sum_f);
end

%% Table S10: rates and classes from the record alone and from the hierarchical model
% per site: the Akritas-Theil-Sen trend with AR(1) correction (v5.2) and the site-only
% Bayesian fit (Model A without pooling of trends, same noise model); hierarchical: Model A.
% P(thickening) of Sen + AR(1) is the normal approximation Phi(rate / SE), not a posterior.
row = @(name, a, b, c) {string(name), string(a), string(b), string(c)};
iw = @(lo, hi) sprintf('%.1f', median(hi - lo, 'omitnan'));
have_so = ismember('class_v6_site_only', res.Properties.VariableNames);
nso = net_value(net, 'A_site_only_mean_site_rate_pct_dec');
p_ats = 0.5 * erfc(-(res.rate_pct_dec ./ res.rate_se) / sqrt(2));
if have_so
    so_w = iw(res.rate_v6_lo_site_only, res.rate_v6_hi_site_only);
    so_p = sum(res.p6_thickening_site_only > 0.9);
else
    so_w = '-'; so_p = '-';
end
rows = [row('Series', n_sites, n_sites, n_sites)
        row('Network rate (mu, mean of regions), % per decade', '-', '-', ci(nw))
        row('Mean of series rates, % per decade', sprintf('%.1f', mean(res.rate_pct_dec, 'omitnan')), ...
            ci(nso), ci(nsm))
        row('Median width of 90 % interval, % per decade', iw(res.rate_lo, res.rate_hi), so_w, ...
            iw(res.rate_v6_lo, res.rate_v6_hi))
        row('Series with P(thickening) > 0.9', sum(p_ats > 0.9), so_p, sum(res.p6_thickening > 0.9))];
for c = 1:numel(CLS)
    if have_so, nc = sum(res.class_v6_site_only == CLS(c)); else, nc = '-'; end
    rows = [rows; row(CLS_L(c), sum(res.class_site == CLS(c)), nc, sum(res.class_v6 == CLS(c)))]; %#ok<AGROW>
end
T2 = cell2table(rows, 'VariableNames', {'quantity', 'per_site_Sen_AR1', 'site_only_bayesian', 'hierarchical'});
writetable(T2, fullfile(OUT, 'TableS10_rates_and_classes_by_method.csv'));

%% Model B (MODEL_B): decomposition for every climate input and check (Table 3 until 14 Nov 2026)
if MODEL_B
    if ~have_power, pnet = []; end
    B = b_inputs(net, have_power, pnet);                      % used in key_numbers below
    VAR_ID = ["main" "kfixed" "lag" "memory" "air" "weather" "weather_memory" ...
              "kprior_zero" "kprior_wide" "dprior_wide" "region_year"];
    VAR_L  = ["Summer TDD (main)" "k fixed at 0.5 (Stefan)" "+ previous summer's TDD" ...
              "+ TDD of the two previous summers" "+ previous summer's TDD and winter freezing index" ...
              "+ winter freezing index, winter snow and summer rain" "All weather terms + 2-year memory" ...
              "Check: k prior centred on 0" "Check: k prior SD 1.0 (four times wider)" ...
              "Check: wider priors of the residual-trend hierarchy" "Check: with a region-year term"];
    T3 = table();
    for i = 1:numel(VAR_ID) + have_power
        if i <= numel(VAR_ID)
            tb = net; pre = "B_" + VAR_ID(i); lab3 = VAR_L(i); src3 = "ERA5-Land"; tres = res;
            col = "class_resid"; if VAR_ID(i) ~= "main", col = "class_resid_" + VAR_ID(i); end
        else
            tb = pnet; pre = "B_main"; lab3 = "Summer TDD (main)"; src3 = "NASA POWER"; tres = pres; col = "class_resid";
        end
        e3 = net_value(tb, char(pre + "_mean_expected_pct_dec"));
        if isnan(e3(1)), continue; end
        r3 = net_value(tb, char(pre + "_mean_residual_pct_dec"));
        t3 = net_value(tb, char(pre + "_mean_total_pct_dec"));
        k3 = net_value(tb, char(pre + "_k0"));
        if ismember(col, tres.Properties.VariableNames), cc = tres.(col); else, cc = strings(0, 1); end
        T3 = [T3; table(lab3, src3, string(ci(e3)), string(ci(r3)), string(ci(t3)), round(100 * e3(1) / t3(1)), ...
            k3(1), sum(cc == RES(1)), sum(cc == RES(2)), sum(cc == RES(3)), sum(cc == RES(4)), ...
            'VariableNames', {'climate_input', 'temperature', 'expected_pct_dec', 'residual_pct_dec', ...
            'total_pct_dec', 'expected_share_pct', 'k_this_summer', 'positive_residual', 'residual_within_5', ...
            'negative_residual', 'not_resolved'})]; %#ok<AGROW>
    end
    writetable(T3, fullfile(OUT, 'B_Table_decomposition.csv'));
end

%% Supplement: Fig S1, analysed site-years per year by measurement method
fig = new_figure(12, 6.5);
ax = axes(fig); hold(ax, 'on');
years = (min(res.year_min):max(res.year_max))';
cnt = zeros(numel(years), numel(FAM));
for f = 1:numel(FAM)
    cnt(:, f) = histcounts(ann.year(ann.family == FAM(f)), [years; years(end) + 1] - 0.5);
end
hb = bar(ax, years, cnt, 1, 'stacked', 'EdgeColor', 'none');
for f = 1:numel(FAM), hb(f).FaceColor = FAM_C(f, :); end
xlim(ax, [years(1) - 1, years(end) + 1]);
xlabel(ax, 'Year'); ylabel(ax, 'Site-years in the analysis');
legend(ax, hb, FAM_L, 'Location', 'northwest', 'Box', 'off');
save_figure(fig, OUT, 'FigS1_site_years');

%% Supplement: Fig S2 and Table S4, sensitivity runs
if have_sens
    cols_s = {'run', 'n_sites', 'n_site_years', 'A_network_rate', 'A_network_lo95', 'A_network_hi95', ...
        'A_mean_site_rate', 'A_declustered_250km', 'A_rapid_thickening', 'A_gradual_thickening', 'A_stable', ...
        'A_gradual_thinning', 'A_rapid_thinning', 'A_indeterminate', 'n_subsidence_series'};
    if MODEL_B
        cols_s = [cols_s, {'B_mean_expected', 'B_mean_residual', 'B_deeper_than_TDD', 'Bwm_mean_expected', ...
            'Bwm_mean_residual', 'Bwm_deeper_than_TDD'}];
    else
        sens = sens(sens.run ~= "undated_doy_243", :);       % changes Model B only
    end
    keep = intersect(cols_s, sens.Properties.VariableNames, 'stable');
    RUN_ID = ["main" "cens_sd_0.5" "cens_sd_1.0" "no_censored" "undated_doy_243" "drop_site_quality" ...
              "drop_disturbed" "one_per_place" "window_2000_2024" "min_obs_8" "region_150km" "region_500km" ...
              "subsidence_0.2" "subsidence_0.5" "subsidence_1.0" ...
              "survey_doy" "temp_interval_flagged" "temp_interval_all" "margins_2.5_10" "margins_5_15"];
    RUN_L  = ["Main run" "Censored years: HalfNormal(0.5)" "Censored years: HalfNormal(1.0)" ...
              "Censored years left out" "Undated surveys on 31 August" "Without imprecise locations" ...
              "Without fire-disturbed plots" "One series per place" "2000-2024 only" ...
              "At least 8 years (not 10)" "Regions of 150 km" "Regions of 500 km" ...
              "Subsidence 0.2 cm/yr, ice-rich" "Subsidence 0.5 cm/yr, ice-rich" "Subsidence 1.0 cm/yr, ice-rich" ...
              "Probe survey day as covariate" "Flagged temperature years as intervals" ...
              "All temperature years as intervals" "Class margins 2.5 / 10 (no refit)" ...
              "Class margins 5 / 15 (no refit)"];
    % Table S4 with a readable label beside each run id
    T_s1 = sens(:, keep);
    lab_s1 = strrep(T_s1.run, '_', ' ');
    [tf, loc] = ismember(T_s1.run, RUN_ID); lab_s1(tf) = RUN_L(loc(tf));
    T_s1 = addvars(T_s1, lab_s1, 'After', 'run', 'NewVariableNames', 'label');
    writetable(T_s1, fullfile(OUT, 'TableS4_sensitivity.csv'));

    % one row per run, both models on one rate axis; dotted lines mark the main run
    sf = sens(~isnan(sens.A_network_rate), :);           % margins_* runs only reclassify: no rates
    nrun = height(sf); yy = (nrun:-1:1)';
    m0 = find(sf.run == "main", 1);
    C_EXP = [0.27 0.45 0.70];
    fig = new_figure(12, 4 + 0.4 * nrun);
    ax = axes(fig); hold(ax, 'on');
    hS = gobjects(1, 1);
    hS(1) = errorbar(ax, sf.A_network_rate, yy, sf.A_network_rate - sf.A_network_lo95, ...
        sf.A_network_hi95 - sf.A_network_rate, 'horizontal', 'o', 'Color', 'k', 'MarkerFaceColor', 'k', ...
        'MarkerSize', 3, 'CapSize', 2);
    sl = {'Network rate (95 % interval)'};
    vals = {sf.A_network_rate}; cols = {'k'};
    if MODEL_B
        sl{1} = 'Model A: network rate (95 % interval)';
        hS(2) = plot(ax, sf.B_mean_expected, yy, 'o', 'Color', C_EXP, 'MarkerFaceColor', C_EXP, 'MarkerSize', 4);
        hS(3) = plot(ax, sf.B_mean_residual, yy, 'o', 'Color', CLS_C(1, :), 'MarkerFaceColor', CLS_C(1, :), 'MarkerSize', 4);
        sl(2:3) = {'Model B: expected from air temperature', 'Model B: residual, air temperature'};
        vals(2:3) = {sf.B_mean_expected, sf.B_mean_residual}; cols(2:3) = {C_EXP, CLS_C(1, :)};
        if ismember('Bwm_mean_residual', sf.Properties.VariableNames)
            hS(4) = plot(ax, sf.Bwm_mean_residual, yy, 'o', 'Color', CLS_C(1, :), 'MarkerSize', 4);
            sl{4} = 'Model B: residual, all weather + memory';
            vals{4} = sf.Bwm_mean_residual; cols{4} = CLS_C(1, :);
        end
    end
    if ~isempty(m0)
        for j = 1:numel(vals), xline(ax, vals{j}(m0), ':', 'Color', cols{j}, 'LineWidth', 0.6); end
    end
    xline(ax, 0, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.4);
    run_l = strrep(sf.run, '_', ' ');
    [tf, loc] = ismember(sf.run, RUN_ID); run_l(tf) = RUN_L(loc(tf));
    yticks(ax, 1:nrun); yticklabels(ax, flipud(run_l)); ylim(ax, [0.5 nrun + 0.5]);
    if MODEL_B
        xlim(ax, [min([0; sf.B_mean_expected]) - 0.5, max(sf.A_network_hi95) + 0.5]);
    else
        xlim(ax, [min([0; sf.A_network_lo95]) - 0.5, max(sf.A_network_hi95) + 0.5]);
    end
    xlabel(ax, 'Rate of ALT change (% per decade)');
    legend(ax, hS, sl, 'Location', 'southoutside', 'Box', 'off', 'FontSize', FONT - 1);
    save_figure(fig, OUT, 'FigS2_sensitivity');
end

%% Model B (MODEL_B): residual trend by ground-ice content (IPA map), within regions
% (Fig S3 / S5 until 14 Nov 2026)
if MODEL_B && ismember('pf_ice', res.Properties.VariableNames)
    reg_ids = unique(res.region_id); d = []; lbl = strings(0); nn = [];
    for r = reg_ids'
        lo_k = res.region_id == r & res.pf_ice == "low"; hi_k = res.region_id == r & res.pf_ice == "high";
        if any(lo_k) && any(hi_k)
            d(end+1) = mean(rs(lo_k), 'omitnan') - mean(rs(hi_k), 'omitnan'); %#ok<SAGROW>
            lbl(end+1) = region_name(mean(res.lat(res.region_id == r)), circmean(res.lon(res.region_id == r))); %#ok<SAGROW>
            nn(end+1, :) = [sum(lo_k) sum(hi_k)]; %#ok<SAGROW>
        end
    end
    % the summary is the region fixed-effects contrast of 3_aldi.py (spatial checks): it
    % weights each region by how much it holds of both groups, so one small region with an
    % extreme difference counts little; the bars are the plain per-region differences.
    % Regions are compared with themselves only, because ground ice and climate both
    % change from region to region.
    ice_fe = fe_contrast(rs, res.pf_ice, res.region_id, "low", "high", 2000);
    % quote the same interval as B_Table_residual_by_ice_and_method and the summary (the bootstrap of 3_aldi.py; this
    % one draws other regions, so its interval differs in the first decimal)
    if have_ck && isfile(fullfile(CK, 'aldi_v6_checks_ice_method.csv'))
        im6 = rd(fullfile(CK, 'aldi_v6_checks_ice_method.csv'));
        k6 = im6.family == "all" & im6.pf_ice == "low_minus_high_within_regions";
        if any(k6), ice_fe(1:3) = [im6.mean_resid(k6) im6.lo95(k6) im6.hi95(k6)]; end
    end
    if ~isempty(d)
        fig = new_figure(11, 3.5 + 0.6 * numel(d));
        ax = axes(fig); hold(ax, 'on');
        patch(ax, [ice_fe(2) ice_fe(3) ice_fe(3) ice_fe(2)], [0.4 0.4 numel(d) + 0.6 numel(d) + 0.6], ...
            CLS_C(1, :), 'FaceAlpha', 0.12, 'EdgeColor', 'none');
        hB = barh(ax, 1:numel(d), d, 0.6, 'FaceColor', [0.55 0.55 0.55], 'EdgeColor', 'none');
        hF = xline(ax, ice_fe(1), '--', 'Color', CLS_C(1, :), 'LineWidth', 1);
        xline(ax, 0, ':');
        yticks(ax, 1:numel(d)); yticklabels(ax, lbl(:) + " (" + nn(:, 1) + "/" + nn(:, 2) + ")");
        ylim(ax, [0.4 numel(d) + 0.6]);
        xlabel(ax, 'Residual trend, low minus high ground ice (% per decade)');
        legend(ax, [hB hF], {'Difference within each region (sites low/high)', ...
            sprintf('All regions, fixed effects: %+.1f (95 %%: %+.1f to %+.1f)', ice_fe(1:3))}, ...
            'Location', 'southoutside', 'Box', 'off', 'FontSize', FONT - 1);
        save_figure(fig, OUT, 'B_Fig_ground_ice_within_regions');
    end
end

%% Supplement: Fig S4, site rate on its own vs in the hierarchical model (shrinkage)
% Each site's own per-site OLS rate against its hierarchical rate. On the 1:1 line the
% model keeps the site's own rate; flat means it takes the region's rate. Short records
% move most, because their own rate is the least certain. Own rates beyond the axis are
% drawn at its edge (open triangles).
if have_ck
    SPAN_C = [0.78 0.86 0.94; 0.52 0.68 0.85; 0.26 0.47 0.72; 0.08 0.25 0.50];
    fig = new_figure(W1 + 3, 9);
    ax = axes(fig); hold(ax, 'on');
    XLIM = [-40 60];
    plot(ax, XLIM, XLIM, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    xline(ax, 0, ':', 'Color', [0.4 0.4 0.4]); yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
    hs = gobjects(1, 4); ls = strings(1, 4);
    for b = 1:4
        k = cks.span_bin == SPAN_L(b);
        x = cks.ols_rate(k); y = cks.rate_v6(k);
        in = x >= XLIM(1) & x <= XLIM(2);
        hs(b) = scatter(ax, x(in), y(in), 12, SPAN_C(b, :), 'filled', 'MarkerEdgeColor', [0.3 0.3 0.3], ...
            'LineWidth', 0.2);
        if any(x < XLIM(1)), scatter(ax, XLIM(1) * ones(sum(x < XLIM(1)), 1), y(x < XLIM(1)), 14, SPAN_C(b, :), '<'); end
        if any(x > XLIM(2)), scatter(ax, XLIM(2) * ones(sum(x > XLIM(2)), 1), y(x > XLIM(2)), 14, SPAN_C(b, :), '>'); end
        ls(b) = sprintf('%s yr (%d): moved %.0f %% of the way', SPAN_L(b), sum(k), ...
            100 * cksp.median_shrinkage(cksp.span == SPAN_L(b)));
    end
    xlim(ax, XLIM + [-2 2]); ylim(ax, [min(cks.rate_v6) - 3, max(cks.rate_v6) + 3]);
    xlabel(ax, 'Site on its own: per-site OLS rate (% per decade)');
    ylabel(ax, 'Hierarchical rate (% per decade)');
    legend(ax, hs, ls, 'Location', 'southoutside', 'Box', 'off', 'FontSize', FONT - 1);
    save_figure(fig, OUT, 'FigS4_shrinkage');
end

%% Supplement: Table S3 (series of each method), Table S8 simulation
T = table();
for f = 1:numel(FAM)
    k = res.family == FAM(f);
    Tf = table(FAM_L(f), sum(k), sum(res.n_obs(k)), median(res.span_years(k)), mean(res.rate_v6(k), 'omitnan'), ...
        median(res.rate_v6(k), 'omitnan'), 'VariableNames', {'method', 'series', 'site_years', 'median_span_yr', ...
        'mean_rate', 'median_rate'});
    if MODEL_B
        Tf = [Tf table(mean(ex(k), 'omitnan'), mean(rs(k), 'omitnan'), sum(res.class_resid(k) == RES(1)), ...
            'VariableNames', {'mean_expected_B', 'mean_residual_B', 'deeper_than_TDD'})]; %#ok<AGROW>
    end
    T = [T; Tf]; %#ok<AGROW>
end
writetable(T, fullfile(OUT, 'TableS3_by_method.csv'));
if have_sim
    cols_m = {'scenario', 'method', 'coverage90', 'rmse_pct_dec', 'wrong_direction', 'resolved', ...
        'called_trend', 'trend_p05', 'network_err', 'network_cover95', 'expected_coverage90'};
    if MODEL_B, cols_m = [cols_m, {'resid_bias', 'false_deeper'}]; end
    keep = intersect(cols_m, sim.Properties.VariableNames, 'stable');
    TS3 = sim(:, keep);
    if ~MODEL_B, TS3 = TS3(~startsWith(TS3.scenario, "B_"), :); end
    % standard error of the mean coverage from the spread between the simulated networks, and
    % the scenarios run later (A_misspec), from the paper checks
    nf = fullfile(CK, 'aldi_v6_checks_simulation_networks.csv');
    if have_ck && isfile(nf)
        sn = rd(nf);
        if ~MODEL_B, sn = sn(~startsWith(sn.scenario, "B_"), :); end
        TS3.networks = nan(height(TS3), 1); TS3.coverage90_se = nan(height(TS3), 1);
        for i = 1:height(TS3)
            k = sn.scenario == TS3.scenario(i) & sn.method == TS3.method(i);
            if any(k), TS3.networks(i) = sn.networks(k); TS3.coverage90_se(i) = sn.coverage90_se(k); end
        end
        later = sn(~ismember(sn.scenario, TS3.scenario), :);
        for i = 1:height(later)
            r = TS3(1, :);
            for v = string(r.Properties.VariableNames)
                if isnumeric(r.(v)), r.(v) = NaN; end
            end
            r.scenario = later.scenario(i); r.method = later.method(i);
            % intersect of two rows is a row; a for loop over a column would run once with the
            % whole column, so no transpose here
            for v = intersect(string(r.Properties.VariableNames), string(later.Properties.VariableNames))
                if v ~= "scenario" && v ~= "method", r.(v) = later.(v)(i); end
            end
            TS3 = [TS3; r]; %#ok<AGROW>
        end
    end
    writetable(TS3, fullfile(OUT, 'TableS8_simulation.csv'));
end

%% Supplement: Fig S3 and Table S9, trends of all regions with their intervals
% (review of 14 November 2026: Fig 2 pools the 32 regions with fewer than four series)
reg_f = fullfile(DATA, '3_aldi', 'aldi_v6_regions.csv');
if isfile(reg_f)
    rg = rd(reg_f);
    rg.name = strings(height(rg), 1);
    for i = 1:height(rg), rg.name(i) = region_name(rg.lat(i), rg.lon(i)); end
    rg = sortrows(rg, 'rate_median');
    writetable(rg(:, [{'region_id', 'name'}, setdiff(rg.Properties.VariableNames, {'region_id', 'name'}, 'stable')]), ...
        fullfile(OUT, 'TableS9_regions.csv'));
    nrg = height(rg); yy = (1:nrg)';
    fig = new_figure(12, 4 + 0.33 * nrg);
    ax = axes(fig); hold(ax, 'on');
    patch(ax, [nw(2) nw(3) nw(3) nw(2)], [0.4 0.4 nrg + 0.6 nrg + 0.6], CLS_C(1, :), 'FaceAlpha', 0.12, ...
        'EdgeColor', 'none');
    xline(ax, 0, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.4);
    hN = xline(ax, nw(1), '-', 'Color', CLS_C(1, :), 'LineWidth', 1);
    plot(ax, [rg.lo95 rg.hi95]', [yy yy]', '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6);
    plot(ax, [rg.lo90 rg.hi90]', [yy yy]', '-', 'Color', [0.15 0.15 0.15], 'LineWidth', 1.6);
    one = rg.n_series == 1;
    h1 = plot(ax, rg.rate_median(~one), yy(~one), 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'k', 'MarkerEdgeColor', 'k');
    h2 = plot(ax, rg.rate_median(one), yy(one), 'o', 'MarkerSize', 4, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
    yticks(ax, yy); yticklabels(ax, rg.name + " (" + rg.n_series + ")"); ylim(ax, [0.4 nrg + 0.6]);
    xlabel(ax, 'Regional trend of ALT (% per decade)');
    legend(ax, [h1 h2 hN], {'Two or more series: median, 90 % (thick) and 95 % (thin) interval', ...
        'One series', sprintf('Network rate, %.1f (%.1f to %.1f)', nw)}, 'Location', 'southoutside', ...
        'Box', 'off', 'FontSize', FONT - 1);
    save_figure(fig, OUT, 'FigS3_regional_trends');
end

%% Supplement: Tables S2, S3, S5, S6 and S7 from the paper checks (data\3_aldi_checks)
if have_ck
    copies = {'aldi_v6_checks_detection.csv', 'TableS5_detection_time.csv'
              'aldi_v6_checks_by_span.csv', 'TableS6_by_record_length.csv'
              'aldi_v6_checks_cm.csv', 'TableS7_rates_in_cm.csv'
              'aldi_v6_checks_method_loro.csv', 'TableS3_method_loro.csv'};
    if MODEL_B
        copies = [copies; {'aldi_v6_checks_ice_method.csv', 'B_Table_residual_by_ice_and_method.csv'}];
    end
    for i = 1:size(copies, 1)
        if isfile(fullfile(CK, copies{i, 1}))
            copyfile(fullfile(CK, copies{i, 1}), fullfile(OUT, copies{i, 2}));
        else
            warning('No %s: rerun py -3.12 3_aldi.py --no-main --checks', copies{i, 1});
        end
    end
    % without Model B: the MCMC diagnostics of the fits of Model A only, and the method
    % agreement without the rows of the Model B residual
    f2 = fullfile(CK, 'aldi_v6_checks_diagnostics.csv');
    if isfile(f2)
        T = rd(f2);
        if ~MODEL_B, T = T(T.model == "A", :); end
        writetable(T, fullfile(OUT, 'TableS2_mcmc_diagnostics.csv'));
    end
    f3 = fullfile(CK, 'aldi_v6_checks_method.csv');
    if isfile(f3)
        T = rd(f3);
        if ~MODEL_B, T = T(T.measure ~= "model B residual trend", :); end
        writetable(T, fullfile(OUT, 'TableS3_method_agreement.csv'));
    end
    f4 = fullfile(OUT, 'TableS3_method_loro.csv');
    if ~MODEL_B && isfile(f4)
        T = rd(f4);
        writetable(T(T.measure ~= "model B residual trend", :), f4);
    end
end

%% Key numbers for the text
fid = fopen(fullfile(OUT, 'key_numbers.txt'), 'w');
p = @(varargin) fprintf(fid, [varargin{1} '\n'], varargin{2:end});
p('ALDI paper: numbers quoted in the Results (from data\\3_aldi*, %s)', char(datetime('now', 'Format', 'yyyy-MM-dd')));
p('');
p('DATA');
n_lb = 0;
if ismember('n_censored', res.Properties.VariableNames), n_lb = sum(res.n_censored, 'omitnan'); end
p('  series %d, site-years %d (%d measured values + %d lower bounds; Fig S1 counts %d measured), regions %d', ...
    n_sites, sum(res.n_obs) + n_lb, sum(res.n_obs), n_lb, height(ann), numel(unique(res.region_id)));
p('  series not from CALM: %d of %d', sum(~startsWith(res.site_id, "CALM_")), n_sites);
for f = 1:numel(FAM), p('  %s: %d series', FAM_L(f), sum(res.family == FAM(f))); end
p('  record span: median %d years (%d to %d)', median(res.span_years), min(res.span_years), max(res.span_years));
p('');
p('MODEL A (hierarchical)');
p('  network rate (mean of regions): %s %%/dec', ci(nw));
p('  mean of site rates: %s %%/dec', ci(nsm));
if have_sens
    m0 = sens.run == "main";
    if any(m0)
        p('  declustered (250 km cells): %.1f; mean over regions %.1f', sens.A_declustered_250km(m0), ...
            sens.A_mean_over_regions(m0));
    end
    ks = ~startsWith(sens.run, "subsidence");               % scenarios, listed below
    if MODEL_B
        p('  sensitivity runs: network %.1f to %.1f, residual %.1f to %.1f, residual all weather %.1f to %.1f', ...
            min(sens.A_network_rate(ks)), max(sens.A_network_rate(ks)), min(sens.B_mean_residual(ks)), ...
            max(sens.B_mean_residual(ks)), min(sens.Bwm_mean_residual(ks)), max(sens.Bwm_mean_residual(ks)));
    else
        kr = ks & sens.run ~= "main" & ~startsWith(sens.run, "margins");
        p('  sensitivity runs (%d, without the 3 subsidence scenarios): network %.1f to %.1f %%/dec', sum(kr), ...
            min(sens.A_network_rate(kr)), max(sens.A_network_rate(kr)));
        for r = ["subsidence_0.2" "subsidence_0.5" "subsidence_1.0"]
            k = sens.run == r;
            if any(k)
                p('  hypothetical subsidence of %s cm/yr at %d probe and tube series on medium- or high-ice ground: network %s %%/dec', ...
                    erase(r, "subsidence_"), sens.n_subsidence_series(k), ...
                    ci([sens.A_network_rate(k) sens.A_network_lo95(k) sens.A_network_hi95(k)]));
            end
        end
    end
end
for c = 1:numel(CLS)
    p('  %s: %d hierarchical, %d per site (v5.2)', CLS_L(c), sum(res.class_v6 == CLS(c)), sum(res.class_site == CLS(c)));
end
p('  P(thickening) > 0.90: %d sites; > 0.95: %d; < 0.10: %d', sum(res.p6_thickening > 0.9), ...
    sum(res.p6_thickening > 0.95), sum(res.p6_thickening < 0.1));
for b = 1:4
    k = span_bin == SPAN_L(b);
    p('  %s yr (n=%d): not resolved %.0f %% hierarchical vs %.0f %% per site; stable %.0f %%', SPAN_L(b), sum(k), ...
        100 * mean(res.class_v6(k) == "indeterminate"), 100 * mean(res.class_site(k) == "indeterminate"), ...
        100 * mean(res.class_v6(k) == "stable"));
end
p('  regional rates (Fig 2):');
for i = height(reg):-1:1, p('    %s: %.1f', reg.label(i), reg.region_rate(i)); end
if MODEL_B
    p('');
    p('MODEL B');
    for i = 1:numel(B)
        p('  %s: expected %s, residual %s, total %s %%/dec, k %.2f', B(i).name, ci(B(i).exp), ci(B(i).res), ...
            ci(B(i).tot), B(i).k(1));
    end
    for c = 1:numel(RES), p('  %s: %d sites', RES_L(c), sum(res.class_resid == RES(c))); end
    p('  burned plots: %d, all deeper than temperature explains: %d', sum(burned), ...
        sum(burned & res.class_resid == RES(1)));
    p('  residual without burned plots: %.1f (all %.1f)', mean(rs(~burned), 'omitnan'), mean(rs, 'omitnan'));
    bk = [net_value(net, 'B_kfixed_mean_expected_pct_dec'); net_value(net, 'B_kfixed_mean_residual_pct_dec')];
    if ~isnan(bk(1, 1))
        p('  k fixed at 0.5 (most the expected part can take): expected %s, residual %s %%/dec', ci(bk(1, :)), ci(bk(2, :)));
    end
    if have_sens
        for r = ["subsidence_0.2" "subsidence_0.5" "subsidence_1.0"]
            k = sens.run == r;
            if any(k)
                p('  %s cm/yr at ice-rich probe and tube series (%d series): network %.1f, expected %.2f, residual %.2f %%/dec', ...
                    erase(r, "subsidence_"), sens.n_subsidence_series(k), sens.A_network_rate(k), ...
                    sens.B_mean_expected(k), sens.B_mean_residual(k));
            end
        end
    end
    if exist('d', 'var') && ~isempty(d)
        p('  low minus high ground ice within regions (region fixed effects, %d regions): %+.1f (%+.1f to %+.1f)', ...
            ice_fe(4), ice_fe(1:3));
        p('    plain per-region differences: %.1f to %.1f', min(d), max(d));
    end
end
if have_sim
    p('');
    p('SIMULATION (real design, scenario A_fit unless stated)');
    for m = 1:numel(METH)
        k = sim.scenario == "A_fit" & sim.method == METH(m);
        if ~any(k), continue; end
        p('  %s: coverage %.2f, wrong sign %.3f, resolved %.2f, RMSE %.1f %%/dec', METH_L(m), sim.coverage90(k), ...
            sim.wrong_direction(k), sim.resolved(k), sim.rmse_pct_dec(k));
        k0 = sim.scenario == "A_null" & sim.method == METH(m);
        if any(k0), p('     no true trend: trend called at %.3f of sites', sim.called_trend(k0)); end
        if any(k0) && ismember('trend_p05', sim.Properties.VariableNames) && ~isnan(sim.trend_p05(k0))
            p('     no true trend: significant at p < 0.05 at %.3f of sites', sim.trend_p05(k0));
        end
    end
    k = sim.method == "v6_B_resid" & sim.scenario == "B_fit";
    if MODEL_B && any(k), p('  Model B residual: coverage %.2f, bias %+.2f %%/dec', sim.coverage90(k), sim.resid_bias(k)); end
    k = sim.method == "v6_B_resid" & ismember(sim.scenario, ["B_climate" "B_tdderr"]);
    if MODEL_B && any(k), p('  Model B, no residual trend: sites called deeper %.3f', max(sim.false_deeper(k))); end
end
if have_ck
    p('');
    p('PAPER CHECKS (data\\3_aldi_checks; all lines in aldi_v6_checks.txt)');
    p('  years of record a site needs, consecutive (network / alone):');
    for f = 1:numel(FAM)
        kn = ckd.family == FAM(f) & ckd.noise == "network"; ka = ckd.family == FAM(f) & ckd.noise == "alone";
        if ~any(kn), continue; end
        p('    %s: thickening at the network rate, 90 %% chance %d / %d yr, 50 %% %d / %d yr; stable %d / %d yr', ...
            FAM_L(f), ckd.years_thickening_90pct(kn), ckd.years_thickening_90pct(ka), ...
            ckd.years_thickening_50pct(kn), ckd.years_thickening_50pct(ka), ...
            ckd.years_stable_90pct(kn), ckd.years_stable_90pct(ka));
    end
    for b = 1:4
        k = cksp.span == SPAN_L(b);
        p('  %s yr: not resolved %.0f %% hierarchical (expected at most %.0f %%), %.0f %% per site (expected %.0f %% for OLS)', ...
            SPAN_L(b), cksp.unresolved_hierarchical_pct(k), cksp.expected_unresolved_network_pct(k), ...
            cksp.unresolved_per_site_pct(k), cksp.expected_unresolved_alone_pct(k));
    end
    sig = cks.ols_p < 0.05;
    a = cksp(cksp.span == "all", :); a1 = cksp(cksp.span == SPAN_L(1), :);
    p('  standard per-site test (OLS on log ALT): p < 0.05 increasing %d (%.0f %%), decreasing %d; after FDR %d', ...
        sum(sig & cks.ols_rate > 0), 100 * mean(sig & cks.ols_rate > 0), sum(sig & cks.ols_rate < 0), sum(cks.ols_q < 0.05));
    p('    with no trend at all it would call %.0f %% of sites significant (nominal 5 %%: autocorrelation)', ...
        a.ols_false_pos_if_no_trend_pct);
    p('    significant rates overstate the true rate %.1f times (12-15 yr: %.1f times); wrong sign %.1f %%', ...
        a.ols_exaggeration_among_significant, a1.ols_exaggeration_among_significant, ...
        a.ols_wrong_sign_among_significant_pct);
    p('    of the sites it calls not significant, the hierarchical model classes %d thickening and %d stable', ...
        sum(~sig & contains(cks.class_v6, "thickening")), sum(~sig & cks.class_v6 == "stable"));
    p('  shrinkage toward the region (median): %s', strjoin(compose('%s yr %.2f', cksp.span, cksp.median_shrinkage)', ', '));
    keys = ["share that is regional" "class of a site with NO data" "POSTERIOR EXPECTED CLASS COUNTS" ...
            "P(rate >" "MCMC:" "survey date drift" "site-years by method, all" "UNRESOLVED RECORDS" ...
            "site-only Bayesian fit (same noise" "site-only fit, classes of all" "ACCELERATION" ...
            "median trend before" "p < 0.05: faster" "Benjamini-Hochberg over the records" ...
            "records, p < 0.05 faster" "EXCLUDED YEARS THAT MARK" "  no_refreeze: " "talik or possible_talik: " ...
            "wide_sensor_spacing: " "profile_inversion: " "analysed series with" ...
            "coverage of the 95 % interval of the network rate"];
    if MODEL_B, keys = [keys "with site fixed effects R2" "share of the TREND"]; end
    lines = splitlines(string(ck_txt));
    for kk = keys
        for ln = lines(contains(lines, kk))'
            p('  %s', strtrim(ln));
        end
    end
    if MODEL_B
        im = rd(fullfile(CK, 'aldi_v6_checks_ice_method.csv'));
        p('  residual trend by ground ice, mean %%/dec (low / medium / high; sites):');
        for f = [FAM "all"]
            v = nan(1, 3); n = zeros(1, 3); ICE = ["low" "medium" "high"];
            for j = 1:3
                k = im.family == f & im.pf_ice == ICE(j);
                if any(k), v(j) = im.mean_resid(k); n(j) = im.sites(k); end
            end
            k = im.family == f & im.pf_ice == "low_minus_high_within_regions" & f ~= "all";  % "all": under MODEL B
            s = sprintf('    %s: %.1f / %.1f / %.1f (%d / %d / %d)', f, v, n);
            if any(k)
                s = [s sprintf('; low minus high within %d shared regions %+.1f (%+.1f to %+.1f)', im.sites(k), ...
                    im.mean_resid(k), im.lo95(k), im.hi95(k))]; %#ok<AGROW>
            end
            p('%s', s);
        end
    end
    if isfile(fullfile(CK, 'aldi_v6_checks_method.csv'))
        ma = rd(fullfile(CK, 'aldi_v6_checks_method.csv'));
        p('  agreement of the methods (Table S3), mean %%/dec and difference from probes within regions');
        p('  [difference, 95 %% interval, shared regions; smallest and largest with one shared region left out]:');
        mlist = ["hierarchical rate" "site-only Bayesian rate" "per-site OLS rate"];
        if MODEL_B, mlist = [mlist "model B residual trend"]; end
        has_loro = ismember('loro_min', ma.Properties.VariableNames);
        for m = mlist
            s = sprintf('    %s:', m);
            for f = FAM
                k = ma.measure == m & ma.group == f;
                if ~any(k), continue; end
                s = [s sprintf(' %s %.1f (%d)', f, ma.mean(k), ma.n(k))]; %#ok<AGROW>
                if isfinite(ma.minus_probe_within_regions(k))
                    s = [s sprintf(' [%+.1f, %+.1f to %+.1f, %d regions', ma.minus_probe_within_regions(k), ...
                        ma.lo95(k), ma.hi95(k), ma.shared_regions(k))]; %#ok<AGROW>
                    if has_loro && isfinite(ma.loro_min(k))
                        s = [s sprintf('; %+.1f to %+.1f', ma.loro_min(k), ma.loro_max(k))]; %#ok<AGROW>
                    end
                    s = [s ']']; %#ok<AGROW>
                end
            end
            p('%s', s);
        end
        for m = ["hierarchical rate" "site-only Bayesian rate" "per-site OLS rate"]
            k1 = ma.measure == m + ", |difference| of two series at one place" & ma.group == "same method";
            k2 = ma.measure == m + ", |difference| of two series at one place" & ma.group == "different methods";
            if any(k1) && any(k2)
                p('    two series at one place, %s: median |difference| %.1f same method (%d pairs), %.1f different methods (%d pairs)', ...
                    m, ma.median(k1), ma.n(k1), ma.median(k2), ma.n(k2));
            end
        end
    end
    cm = rd(fullfile(CK, 'aldi_v6_checks_cm.csv'));
    p('  rates in cm (Table S7):');
    for i = 1:height(cm)
        if ~startsWith(cm.what(i), "ALDI region")
            p('    %s: %.2f %s', cm.what(i), cm.value(i), cm.unit(i));
        end
    end
end
% --- checks added after the review of 14 November 2026 ---
p('');
p('REVIEW CHECKS (14 November 2026)');
erm = net_value(net, 'A_main_equal_region_mean_pct_dec');
p('  network mean mu %s; equal-weight mean of the fitted regional trends %s %%/dec', ci(nw), ci(erm));
if isfile(reg_f)
    one = rg.n_series == 1;
    p('  regions: median above 0 at %d of %d; 95 %% interval above 0 at %d (%d of the %d single-series regions); 90 %% interval above 0 at %d (%d); P(trend > 0) min %.3f, median %.3f', ...
        sum(rg.rate_median > 0), height(rg), sum(rg.lo95 > 0), sum(rg.lo95 > 0 & one), sum(one), ...
        sum(rg.lo90 > 0), sum(rg.lo90 > 0 & one), min(rg.p_positive), median(rg.p_positive));
    k0 = rg.lo95 <= 0;
    p('  regions whose 95 %% interval reaches 0: %s', strjoin(compose('%s (%d series, %.1f, %.1f to %.1f)', ...
        rg.name(k0), rg.n_series(k0), rg.rate_median(k0), rg.lo95(k0), rg.hi95(k0)), '; '));
end
lines = splitlines(string(summ));
kl = contains(lines, "site-years by method family") | contains(lines, "site-only fit (no pooling");
if MODEL_B, kl = kl | contains(lines, "corr(expected, residual mean)"); end
for ln = lines(kl)'
    p('  %s', strtrim(ln));
end
if isfile(sum_f)
    st = rd(sum_f);
    p('  Table 2, summaries of the network trend (per posterior draw; region bootstrap for the series mean):');
    for i = 1:height(st)
        s = sprintf('    %s (%d %s): %.1f (%.1f to %.1f)', st.summary(i), st.n_units(i), st.unit(i), st.median(i), ...
            st.lo95(i), st.hi95(i));
        if isfinite(st.region_bootstrap_lo95(i))
            s = [s sprintf('; region bootstrap %.1f to %.1f', st.region_bootstrap_lo95(i), st.region_bootstrap_hi95(i))]; %#ok<AGROW>
        end
        p('%s', s);
    end
end
dep_f = fullfile(DATA, '3_aldi', 'aldi_v6_dependence.csv');
if isfile(dep_f)
    dp = rd(dep_f);
    if ~MODEL_B, dp = dp(startsWith(dp.fit, "A "), :); end
    p('  dependence left in the residuals: lag-1 autocorrelation of innovations | same-year correlation of two series of a region');
    for i = 1:height(dp)
        p('    %s: %+.2f (%d pairs) | %+.2f (%d pairs)', dp.fit(i), dp.lag1(i), dp.n_lag1(i), dp.same_region_year(i), dp.n_pairs(i));
    end
end
nf = fullfile(CK, 'aldi_v6_checks_simulation_networks.csv');
if have_ck && isfile(nf)
    sn = rd(nf);
    p('  simulation, coverage of 90 %% intervals: mean over networks (SE from the spread between networks; range)');
    for i = 1:height(sn)
        p('    %s %s: %.3f (SE %.3f; %.2f to %.2f; %d networks), RMSE %.1f, trend called %.3f', sn.scenario(i), ...
            sn.method(i), sn.coverage90(i), sn.coverage90_se(i), sn.coverage90_min(i), sn.coverage90_max(i), ...
            sn.networks(i), sn.rmse_pct_dec(i), sn.called_trend(i));
    end
end
kp = fullfile(DATA, '3_aldi_keep_pinned');
if isfile(fullfile(kp, 'aldi_v6_network.csv'))
    kn = rd(fullfile(kp, 'aldi_v6_network.csv')); kr = rd(fullfile(kp, 'aldi_v6_results.csv'));
    p('  with the series pinned to a sensor kept (data\\3_aldi_keep_pinned): %d series; network %s; equal-weight regions %s; mean of sites %s %%/dec', ...
        height(kr), ci(net_value(kn, 'A_main_network_rate_pct_dec')), ci(net_value(kn, 'A_main_equal_region_mean_pct_dec')), ...
        ci(net_value(kn, 'A_main_mean_site_rate_pct_dec')));
    p('    classes: %s', strjoin(compose('%s %d', CLS_L', arrayfun(@(c) sum(kr.class_v6 == c), CLS)'), ', '));
    new_s = setdiff(kr.site_id, res.site_id);
    for sid = new_s'
        k = kr.site_id == sid;
        p('    added: %s (%s, %d values, %d-%d): %.1f %%/dec (%.1f to %.1f), %s', sid, kr.family(k), kr.n_obs(k), ...
            kr.year_min(k), kr.year_max(k), kr.rate_v6(k), kr.rate_v6_lo(k), kr.rate_v6_hi(k), kr.class_v6(k));
    end
    if MODEL_B
        p('    model B main: expected %s, residual %s %%/dec', ci(net_value(kn, 'B_main_mean_expected_pct_dec')), ...
            ci(net_value(kn, 'B_main_mean_residual_pct_dec')));
    end
end
fclose(fid);
fprintf('Done: figures and tables in %s\n', OUT);
type(fullfile(OUT, 'key_numbers.txt'));


%% ======================================================================================
%  Helper functions
%  ======================================================================================
function fig = new_figure(w_cm, h_cm)
fig = figure('Units', 'centimeters', 'Position', [2 2 w_cm h_cm], 'Color', 'w', ...
    'PaperPositionMode', 'auto');
end

function save_figure(fig, out, name)
exportgraphics(fig, fullfile(out, [name '.png']), 'Resolution', 300);
exportgraphics(fig, fullfile(out, [name '.pdf']), 'ContentType', 'vector');
fprintf('  %s\n', name);
end

function tf = as_logical(x)
% True/False columns may come in as logical, numbers or text
if islogical(x), tf = x;
elseif isnumeric(x), tf = x ~= 0 & ~isnan(x);
else, tf = lower(strtrim(string(x))) == "true";
end
end

function s = ci(v)
s = sprintf('%.1f (%.1f to %.1f)', v(1), v(2), v(3));
end

function T = read_text_cols(f, cols)
% readtable with the named columns kept as text (a label such as 12-15 could be read as
% a date otherwise)
opts = detectImportOptions(f, 'TextType', 'string', 'VariableNamingRule', 'preserve', 'Delimiter', ',');
opts = setvartype(opts, intersect(cols, opts.VariableNames), 'string');
T = readtable(f, opts);
end

function v = net_value(T, name)
% median and 95 % interval of one network parameter (aldi_v6_network.csv)
k = find(string(T{:, 1}) == name, 1);
if isempty(k), v = [NaN NaN NaN]; else, v = [T.median(k) T.lo95(k) T.hi95(k)]; end
end

function B = b_inputs(net, have_power, pnet)
% expected, residual and total mean site trend for the three climate inputs of Table 3
B = struct('name', {}, 'exp', {}, 'res', {}, 'tot', {}, 'k', {});
B(1) = b_one(net, 'B_main', 'ERA5-Land, air temperature');
B(2) = b_one(net, 'B_weather_memory', 'ERA5-Land, all weather + 2-year memory');
if have_power
    B(3) = b_one(pnet, 'B_main', 'NASA POWER, air temperature');
end
end

function b = b_one(T, pre, name)
b.name = name;
b.exp = net_value(T, [pre '_mean_expected_pct_dec']);
b.res = net_value(T, [pre '_mean_residual_pct_dec']);
b.tot = net_value(T, [pre '_mean_total_pct_dec']);
b.k   = net_value(T, [pre '_k0']);
end

function [x, y] = ease_xy(lat, lon)
% EASE-Grid North (Lambert azimuthal equal-area, north pole, sphere R = 6371228 m), the
% grid of the IPA map file; Greenwich points down
R = 6371228;
rho = 2 * R * sind(45 - lat / 2);
x = rho .* sind(lon);
y = -rho .* cosd(lon);
end

function [rgb, xg, yg, zone_l, zone_c] = ipa_image(file, XL, YL)
% IPA permafrost map (Brown et al. 2002, NSIDC GGD318), 12.5 km EASE-Grid North,
% 1441 x 1441 bytes. Codes 1-20: extent x ground ice x landform (continuous 1 5 9 13 17,
% discontinuous 2 6 10 14 18, sporadic 3 7 11 15 19, isolated 4 8 12 16 20); 21 glacier,
% 22 relict, 23 lake, 24 ocean, 25 land without permafrost.
fid = fopen(file, 'r'); g = fread(fid, [1441 1441], 'uint8=>uint8')'; fclose(fid);
d = 12533.7625; ul = 9024309;                           % centre of the upper-left cell
xg = -ul + (0:1440) * d; yg = ul - (0:1440) * d;
ix = xg >= XL(1) - d & xg <= XL(2) + d; iy = yg >= YL(1) - d & yg <= YL(2) + d;
g = g(iy, ix); xg = xg(ix); yg = yg(iy);
% zone_l and zone_c: legend labels and colours of the background
zone_l = ["Continuous" "Discontinuous" "Sporadic" "Isolated"] + " permafrost";
zone_l(5) = "Land without permafrost";
zone_c = [0.62 0.71 0.81; 0.74 0.80 0.87; 0.84 0.88 0.92; 0.89 0.92 0.95; 0.92 0.92 0.92];
groups = {[22 25], zone_c(5, :); 21, [0.97 0.97 0.97]; [1 5 9 13 17], zone_c(1, :); ...
          [2 6 10 14 18], zone_c(2, :); [3 7 11 15 19], zone_c(3, :); [4 8 12 16 20], zone_c(4, :)};
rgb = ones([size(g) 3]);
for i = 1:size(groups, 1)
    m = ismember(g, groups{i, 1});
    for c = 1:3
        ch = rgb(:, :, c); ch(m) = groups{i, 2}(c); rgb(:, :, c) = ch;
    end
end
end

function out = fe_contrast(y, g, r, level, base, nboot)
% Difference of group `level` from `base` within regions (least squares with region fixed
% effects, other groups as extra levels), as in the spatial checks of 3_aldi.py; 95 %
% interval from resampling whole regions. Returns [estimate lo hi n_shared_regions].
ok = ~isnan(y) & ~ismissing(g);
y = y(ok); g = string(g(ok)); r = r(ok);
b = fe_est(y, g, r, level, base);
ur = unique(r); bb = nan(nboot, 1);
rs = RandStream('mt19937ar', 'Seed', 0);
for i = 1:nboot
    pick = ur(randi(rs, numel(ur), numel(ur), 1));
    idx = arrayfun(@(k) find(r == pick(k)), (1:numel(pick))', 'UniformOutput', false);
    lab = arrayfun(@(k) k * ones(numel(idx{k}), 1), (1:numel(pick))', 'UniformOutput', false);
    idx = vertcat(idx{:}); lab = vertcat(lab{:});
    bb(i) = fe_est(y(idx), g(idx), lab, level, base);
end
bb = sort(bb(isfinite(bb)));
q = @(p) interp1(((1:numel(bb))' - 0.5) / numel(bb), bb, p, 'linear', 'extrap');
out = [b q(0.025) q(0.975) numel(intersect(r(g == level), r(g == base)))];
end

function b = fe_est(y, g, r, level, base)
levels = setdiff(unique(g), base);
if ~any(g == level) || isempty(intersect(r(g == level), r(g == base))), b = NaN; return; end
X = double(g == levels(:)');
[~, ~, ri] = unique(r);
for j = 1:size(X, 2)
    mj = accumarray(ri, X(:, j), [], @mean); X(:, j) = X(:, j) - mj(ri);
end
my = accumarray(ri, y, [], @mean); y = y - my(ri);
coef = lsqminnorm(X, y);
b = coef(levels == level);
end

function m = circmean(lon)
m = atan2d(mean(sind(lon)), mean(cosd(lon)));
end

function name = region_name(lat, lon)
% readable names of the regions of the 7 November 2026 run, matched by centroid (150 km)
P = {69.3 -134.3 'Mackenzie Delta and coast'; 64.9 -148.2 'Fairbanks'; 69.7 -149.0 'North Slope foothills';
     68.4 -149.6 'Toolik and Imnavait'; 70.7 -155.8 'Utqiagvik and coastal plain'; 69.1 160.4 'Lower Kolyma';
     68.0 -133.5 'Inuvik uplands'; 62.8 -122.9 'Southern Mackenzie Valley'; 65.5 -127.5 'Central Mackenzie Valley';
     70.2 68.5 'Yamal'; 78.0 15.1 'Svalbard'; 73.0 -80.0 'Bylot Island and Borden';
     67.3 63.6 'Vorkuta'; 65.3 72.7 'Nadym'; 66.9 76.8 'Urengoy'; 68.5 87.5 'Norilsk-Igarka';
     68.3 18.9 'Abisko'; 74.5 -20.6 'Zackenberg'; 62.1 129.6 'Yakutsk'};
dkm = 6371 * acos(min(1, sind(lat) * sind([P{:, 1}]) + cosd(lat) * cosd([P{:, 1}]) .* cosd(lon - [P{:, 2}])));
[dmin, i] = min(dkm);
if dmin < 150
    name = string(P{i, 3});
else
    ew = 'E'; if lon < 0, ew = 'W'; end
    name = sprintf('%.0f°N %.0f°%s', lat, abs(lon), ew);
end
end

function reg = region_table(res, min_sites)
% one row per region with at least min_sites series, sorted by rate, and one row for the
% other regions pooled (first, so it ends up at the bottom of Fig 2)
[G, rid] = findgroups(res.region_id);
n = splitapply(@numel, res.region_id, G);
big = find(n >= min_sites);
label = strings(numel(big), 1); nn = zeros(numel(big), 1); rr = nn; rates = cell(numel(big), 1);
for j = 1:numel(big)
    k = res.region_id == rid(big(j));
    label(j) = sprintf('%s (%d)', region_name(mean(res.lat(k)), circmean(res.lon(k))), sum(k));
    nn(j) = sum(k); rr(j) = median(res.region_rate_v6(k)); rates{j} = res.rate_v6(k);
end
reg = sortrows(table(label, nn, rr, rates, 'VariableNames', {'label', 'n', 'region_rate', 'rates'}), 'region_rate');
small = ~ismember(res.region_id, rid(big));
if any(small)
    [Gs, ~] = findgroups(res.region_id(small));
    rs = splitapply(@median, res.region_rate_v6(small), Gs);   % one value per small region
    other = table(string(sprintf('Other %d regions (%d)', numel(rs), sum(small))), sum(small), mean(rs), ...
        {res.rate_v6(small)}, 'VariableNames', {'label', 'n', 'region_rate', 'rates'});
    reg = [other; reg];
end
end
