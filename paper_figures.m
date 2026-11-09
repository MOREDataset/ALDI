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
% Every figure has one panel and makes one point of the Results.
% Main text
%   Fig 1  Sites on the IPA permafrost map, coloured by ALDI class         (Results 1)
%   Fig 2  Regional and network rates of ALT change                        (Results 1)
%   Fig 3  Site classes by record length, per-site method vs hierarchical  (Results 2)
%   Fig 4  Simulation: coverage of 90 % intervals by record length         (Results 2)
%   Fig 5  Expected from air temperature vs residual trend, per site       (Results 3)
%   Table 1  Data sources of the analysed series
%   Table 2  Model A: network rates and class counts, three methods
%   Table 3  Model B: expected and residual trend for three climate inputs
% Supplement
%   Fig S1  Site-years per year by measurement method
%   Fig S2  Sensitivity runs, Models A and B
%   Fig S3  Residual trend, low minus high ground ice within regions
%   Fig S4  Site rate on its own (per-site OLS) vs in the hierarchical model
%   Table S1  Sensitivity runs   Table S2  By measurement method   Table S3  Simulation check
%   Table S4  Years of record a site needs, per method    Table S5  By record length:
%             observed and expected shares, the standard per-site test
%   Table S6  Residual trend by ground ice and method     Table S7  Rates in cm per decade
%             beside published values                     Table S8  MCMC diagnostics

clear; close all;

%% Settings
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
DATA = fullfile(here, 'data');
OUT  = fullfile(DATA, '4_paper');
if ~exist(OUT, 'dir'), mkdir(OUT); end

MIN_SITES_REGION = 4;          % regions with fewer sites are pooled in Fig 2
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
RES_L = ["Deeper than temperature explains" "Consistent with temperature" ...
         "Shallower than temperature explains" "Not resolved"];
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
    warning('No %s: run py -3.12 3_aldi.py --no-main --checks for the dashes of Fig 3, Fig S4 and Tables S4-S8', CK);
end

n_sites = height(res);
span_bin = strings(n_sites, 1);
span_bin(res.span_years <= 15) = SPAN_L(1);
span_bin(res.span_years > 15 & res.span_years <= 20) = SPAN_L(2);
span_bin(res.span_years > 20 & res.span_years <= 25) = SPAN_L(3);
span_bin(res.span_years > 25) = SPAN_L(4);
burned = as_logical(res.flag_burned);
ex = res.expected_rate; rs = res.resid_rate;             % Model B, air temperature, ERA5-Land
fprintf('%d sites read from data\\3_aldi\n', n_sites);

% all figures are remade below; this also clears file names of earlier versions
old = [dir(fullfile(OUT, 'Fig*.png')); dir(fullfile(OUT, 'Fig*.pdf'))];
for i = 1:numel(old), delete(fullfile(OUT, old(i).name)); end

% analysed site-years (one per site and year, inside the analysed window): Fig S1, Table 1
ann = rd(fullfile(DATA, '1_alt_target', 'alt_multisource_annual.csv'));
ann = ann(as_logical(ann.use), :);
[tf, loc] = ismember(ann.site_id, res.site_id);
ann = ann(tf, :); loc = loc(tf);
inwin = ann.year >= res.year_min(loc) & ann.year <= res.year_max(loc);
ann = ann(inwin, :); loc = loc(inwin);
ann.family = res.family(loc);
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
% (key_numbers.txt) and listed in Table S3.
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

%% Figure 5: Model B, expected from air temperature vs residual trend
% Large markers: the mean over sites of the main fit; with k fixed at 0.5 (Stefan's
% exponent: errors in TDD pull a fitted k toward 0, so this is about the most the
% expected part can be); and with thaw subsidence added at the probe and thaw-tube
% series on medium- or high-ice ground (sensitivity runs: 0.2, 0.5 and 1.0 cm/yr; the
% surface these methods measure from sinks, so their ALT trend is too small).
fig = new_figure(16, 9);
ax = axes(fig); hold(ax, 'on');
plot(ax, [res.expected_lo res.expected_hi]', [rs rs]', '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.4);
plot(ax, [ex ex]', [res.resid_lo res.resid_hi]', '-', 'Color', [0.85 0.85 0.85], 'LineWidth', 0.4);
xline(ax, 0, ':', 'Color', [0.4 0.4 0.4]); yline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
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
        lab(end+1) = sprintf('Mean, subsidence %s cm/yr: residual %s', ...
            strjoin(erase(sub_runs(tf & sub_runs ~= "main"), "subsidence_"), ' / '), ...
            strjoin(compose('%.1f', sens.B_mean_residual(loc(tf & sub_runs ~= "main"))'), ' / '));
    end
end
if ~isnan(bk(1, 1))
    h(end+1) = plot(ax, bk(1, 1), bk(2, 1), 'd', 'Color', 'k', 'MarkerFaceColor', 'w', 'MarkerSize', 7, ...
        'LineWidth', 1);
    lab(end+1) = sprintf('Mean, k fixed at 0.5: %.1f expected, %.1f residual', bk(1, 1), bk(2, 1));
end
h(end+1) = plot(ax, bm(1, 1), bm(2, 1), 'd', 'Color', 'k', 'MarkerFaceColor', 'k', 'MarkerSize', 7);
lab(end+1) = sprintf('Mean of sites: %.1f expected, %.1f residual', bm(1, 1), bm(2, 1));
xlabel(ax, 'Expected from air temperature (% per decade)');
ylabel(ax, 'Residual trend (% per decade)');
legend(ax, h, lab, 'Location', 'eastoutside', 'Box', 'off', 'FontSize', FONT - 1);
save_figure(fig, OUT, 'Fig5_expected_vs_residual');

%% Table 1: data sources of the analysed series
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
writetable(T1, fullfile(OUT, 'Table1_data_sources.csv'));

%% Table 2: Model A network rates and classes, three methods
v52 = regexp(summ, 'v5\.2 rerun: \d+ sites, network ([\-\d.]+)', 'tokens', 'once');
v52_net = NaN; if ~isempty(v52), v52_net = str2double(v52{1}); end
row = @(name, a, b, c) {string(name), string(a), string(b), string(c)};
iw = @(lo, hi) sprintf('%.1f', median(hi - lo, 'omitnan'));
pw = 2 * 1.644854 * res.rate_pooled_sd;
rows = [row('Sites', n_sites, n_sites, n_sites)
        row('Network rate (mean of regions), % per decade', '-', sprintf('%.1f', v52_net), ci(nw))
        row('Mean of site rates, % per decade', sprintf('%.1f', mean(res.rate_pct_dec, 'omitnan')), ...
            sprintf('%.1f', mean(res.rate_pooled, 'omitnan')), ci(nsm))
        row('Median width of 90 % interval, % per decade', iw(res.rate_lo, res.rate_hi), ...
            sprintf('%.1f', median(pw, 'omitnan')), iw(res.rate_v6_lo, res.rate_v6_hi))];
for c = 1:numel(CLS)
    rows = [rows; row(CLS_L(c), sum(res.class_site == CLS(c)), sum(res.class_pooled == CLS(c)), ...
        sum(res.class_v6 == CLS(c)))]; %#ok<AGROW>
end
T2 = cell2table(rows, 'VariableNames', {'quantity', 'per_site_Sen_AR1', 'pooled_REML', 'hierarchical_v6'});
writetable(T2, fullfile(OUT, 'Table2_modelA_network_and_classes.csv'));

%% Table 3: Model B decomposition for three climate inputs
if ~have_power, pnet = []; end
B = b_inputs(net, have_power, pnet);
cls_cols = {'class_resid', 'class_resid_weather_memory', 'class_resid'};
tabs = {res, res, []}; if have_power, tabs{3} = pres; end
T3 = table();
for i = 1:numel(B)
    t = tabs{i}; cc = t.(cls_cols{i});
    T3 = [T3; table(string(B(i).name), string(ci(B(i).exp)), string(ci(B(i).res)), string(ci(B(i).tot)), ...
        B(i).k(1), sum(cc == RES(1)), sum(cc == RES(2)), sum(cc == RES(3)), sum(cc == RES(4)), height(t), ...
        'VariableNames', {'climate_input', 'expected_pct_dec', 'residual_pct_dec', 'total_pct_dec', ...
        'k_this_summer', 'deeper', 'consistent', 'shallower', 'not_resolved', 'sites'})]; %#ok<AGROW>
end
writetable(T3, fullfile(OUT, 'Table3_modelB_decomposition.csv'));

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

%% Supplement: Fig S2 and Table S1, sensitivity runs
if have_sens
    keep = intersect({'run', 'n_sites', 'n_site_years', 'A_network_rate', 'A_network_lo95', 'A_network_hi95', ...
        'A_mean_site_rate', 'A_declustered_250km', 'A_rapid_thickening', 'A_gradual_thickening', 'A_stable', ...
        'A_gradual_thinning', 'A_rapid_thinning', 'A_indeterminate', 'B_mean_expected', 'B_mean_residual', ...
        'B_deeper_than_TDD', 'Bwm_mean_expected', 'Bwm_mean_residual', 'Bwm_deeper_than_TDD'}, ...
        sens.Properties.VariableNames, 'stable');
    writetable(sens(:, keep), fullfile(OUT, 'TableS1_sensitivity.csv'));

    % one row per run, both models on one rate axis; dotted lines mark the main run
    sf = sens(~isnan(sens.A_network_rate), :);           % margins_* runs only reclassify: no rates
    nrun = height(sf); yy = (nrun:-1:1)';
    m0 = find(sf.run == "main", 1);
    C_EXP = [0.27 0.45 0.70];
    fig = new_figure(12, 10);
    ax = axes(fig); hold(ax, 'on');
    hS = gobjects(1, 3);
    hS(1) = errorbar(ax, sf.A_network_rate, yy, sf.A_network_rate - sf.A_network_lo95, ...
        sf.A_network_hi95 - sf.A_network_rate, 'horizontal', 'o', 'Color', 'k', 'MarkerFaceColor', 'k', ...
        'MarkerSize', 3, 'CapSize', 2);
    sl = {'Model A: network rate (95 % interval)'};
    hS(2) = plot(ax, sf.B_mean_expected, yy, 'o', 'Color', C_EXP, 'MarkerFaceColor', C_EXP, 'MarkerSize', 4);
    hS(3) = plot(ax, sf.B_mean_residual, yy, 'o', 'Color', CLS_C(1, :), 'MarkerFaceColor', CLS_C(1, :), 'MarkerSize', 4);
    sl(2:3) = {'Model B: expected from air temperature', 'Model B: residual, air temperature'};
    vals = {sf.A_network_rate, sf.B_mean_expected, sf.B_mean_residual}; cols = {'k', C_EXP, CLS_C(1, :)};
    if ismember('Bwm_mean_residual', sf.Properties.VariableNames)
        hS(4) = plot(ax, sf.Bwm_mean_residual, yy, 'o', 'Color', CLS_C(1, :), 'MarkerSize', 4);
        sl{4} = 'Model B: residual, all weather + memory';
        vals{4} = sf.Bwm_mean_residual; cols{4} = CLS_C(1, :);
    end
    if ~isempty(m0)
        for j = 1:numel(vals), xline(ax, vals{j}(m0), ':', 'Color', cols{j}, 'LineWidth', 0.6); end
    end
    xline(ax, 0, '-', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.4);
    RUN_ID = ["main" "cens_sd_0.5" "cens_sd_1.0" "no_censored" "undated_doy_243" "drop_site_quality" ...
              "drop_disturbed" "one_per_place" "window_2000_2024" "min_obs_8" "region_150km" "region_500km" ...
              "subsidence_0.2" "subsidence_0.5" "subsidence_1.0"];
    RUN_L  = ["Main run" "Censored years: HalfNormal(0.5)" "Censored years: HalfNormal(1.0)" ...
              "Censored years left out" "Undated surveys on 31 August" "Without imprecise locations" ...
              "Without fire-disturbed plots" "One series per place" "2000-2024 only" ...
              "At least 8 years (not 10)" "Regions of 150 km" "Regions of 500 km" ...
              "Subsidence 0.2 cm/yr, ice-rich" "Subsidence 0.5 cm/yr, ice-rich" "Subsidence 1.0 cm/yr, ice-rich"];
    run_l = strrep(sf.run, '_', ' ');
    [tf, loc] = ismember(sf.run, RUN_ID); run_l(tf) = RUN_L(loc(tf));
    yticks(ax, 1:nrun); yticklabels(ax, flipud(run_l)); ylim(ax, [0.5 nrun + 0.5]);
    xlim(ax, [min([0; sf.B_mean_expected]) - 0.5, max(sf.A_network_hi95) + 0.5]);
    xlabel(ax, 'Rate of ALT change (% per decade)');
    legend(ax, hS, sl, 'Location', 'southoutside', 'Box', 'off', 'FontSize', FONT - 1);
    save_figure(fig, OUT, 'FigS2_sensitivity');
end

%% Supplement: Fig S3, residual trend by ground-ice content (IPA map), within regions
if ismember('pf_ice', res.Properties.VariableNames)
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
        save_figure(fig, OUT, 'FigS3_ground_ice_within_regions');
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

%% Supplement: Table S2 by measurement method, Table S3 simulation
T = table();
for f = 1:numel(FAM)
    k = res.family == FAM(f);
    T = [T; table(FAM_L(f), sum(k), sum(res.n_obs(k)), median(res.span_years(k)), mean(res.rate_v6(k), 'omitnan'), ...
        mean(ex(k), 'omitnan'), mean(rs(k), 'omitnan'), sum(res.class_resid(k) == RES(1)), ...
        'VariableNames', {'method', 'sites', 'site_years', 'median_span_yr', 'mean_rate_A', ...
        'mean_expected_B', 'mean_residual_B', 'deeper_than_TDD'})]; %#ok<AGROW>
end
writetable(T, fullfile(OUT, 'TableS2_by_method.csv'));
if have_sim
    keep = intersect({'scenario', 'method', 'coverage90', 'rmse_pct_dec', 'wrong_direction', 'resolved', ...
        'called_trend', 'trend_p05', 'network_err', 'network_cover95', 'expected_coverage90', 'resid_bias', ...
        'false_deeper'}, sim.Properties.VariableNames, 'stable');
    writetable(sim(:, keep), fullfile(OUT, 'TableS3_simulation.csv'));
end

%% Supplement: Tables S4-S8 from the paper checks (copied from data\3_aldi_checks)
if have_ck
    copies = {'aldi_v6_checks_detection.csv', 'TableS4_detection_time.csv'
              'aldi_v6_checks_by_span.csv', 'TableS5_by_record_length.csv'
              'aldi_v6_checks_ice_method.csv', 'TableS6_residual_by_ice_and_method.csv'
              'aldi_v6_checks_cm.csv', 'TableS7_rates_in_cm.csv'
              'aldi_v6_checks_diagnostics.csv', 'TableS8_mcmc_diagnostics.csv'};
    for i = 1:size(copies, 1)
        copyfile(fullfile(CK, copies{i, 1}), fullfile(OUT, copies{i, 2}));
    end
end

%% Key numbers for the text
fid = fopen(fullfile(OUT, 'key_numbers.txt'), 'w');
p = @(varargin) fprintf(fid, [varargin{1} '\n'], varargin{2:end});
p('ALDI paper: numbers quoted in the Results (from data\\3_aldi*, %s)', char(datetime('now', 'Format', 'yyyy-MM-dd')));
p('');
p('DATA');
p('  series %d, site-years %d (Fig S1 count %d), regions %d', n_sites, sum(res.n_obs), height(ann), ...
    numel(unique(res.region_id)));
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
    ks = ~startsWith(sens.run, "subsidence");               % bounds, reported under MODEL B
    p('  sensitivity runs: network %.1f to %.1f, residual %.1f to %.1f, residual all weather %.1f to %.1f', ...
        min(sens.A_network_rate(ks)), max(sens.A_network_rate(ks)), min(sens.B_mean_residual(ks)), ...
        max(sens.B_mean_residual(ks)), min(sens.Bwm_mean_residual(ks)), max(sens.Bwm_mean_residual(ks)));
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
    if any(k), p('  Model B residual: coverage %.2f, bias %+.2f %%/dec', sim.coverage90(k), sim.resid_bias(k)); end
    k = sim.method == "v6_B_resid" & ismember(sim.scenario, ["B_climate" "B_tdderr"]);
    if any(k), p('  Model B, no residual trend: sites called deeper %.3f', max(sim.false_deeper(k))); end
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
            "P(rate >" "with site fixed effects R2" "share of the TREND" "MCMC:" "survey date drift" ...
            "low_minus_high_within_regions"];
    lines = splitlines(string(ck_txt));
    for kk = keys
        for ln = lines(contains(lines, kk))'
            p('  %s', strtrim(ln));
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
     70.2 68.5 'Yamal'; 78.0 15.1 'Svalbard'; 73.0 -80.0 'Bylot Island and Borden'};
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
