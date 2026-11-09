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
%   data\1_alt_target\alt_multisource_annual.csv, source_inventory.csv
%   data\raw\permafrost_map\nhipa.byte  (IPA permafrost map, NSIDC GGD318; map background)
%
% Main text
%   Fig 1  Sites on the IPA permafrost map, coloured by ALDI class; site-years per year
%   Fig 2  Regional and network rates of ALT change
%   Fig 3  Classes by record length; simulation check of three trend methods
%   Fig 4  Climate-expected vs residual trend per site; network decomposition
%   Table 1  Data sources of the analysed series
%   Table 2  Model A: network rates and class counts, three methods
%   Table 3  Model B: expected and residual trend for three climate inputs
% Supplement
%   Fig S1  Sensitivity runs        Table S1  Sensitivity runs
%   Fig S2  Residual by ground ice   Table S2  By measurement method
%                                    Table S3  Simulation check

clear; close all;

%% Settings
here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
DATA = fullfile(here, 'data');
OUT  = fullfile(DATA, '4_paper');
if ~exist(OUT, 'dir'), mkdir(OUT); end

MIN_SITES_REGION = 4;          % regions with fewer sites are pooled in Fig 2
FIG_WIDTH_CM = 17.5;           % Copernicus double-column width
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

n_sites = height(res);
span_bin = strings(n_sites, 1);
span_bin(res.span_years <= 15) = SPAN_L(1);
span_bin(res.span_years > 15 & res.span_years <= 20) = SPAN_L(2);
span_bin(res.span_years > 20 & res.span_years <= 25) = SPAN_L(3);
span_bin(res.span_years > 25) = SPAN_L(4);
burned = as_logical(res.flag_burned);
fprintf('%d sites read from data\\3_aldi\n', n_sites);

%% Figure 1: map and data coverage
fig = new_figure(FIG_WIDTH_CM, 9.5);
tl = tiledlayout(fig, 1, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl, 1, [1 3]); hold(ax, 'on');
XL = [-3.9e6 3.3e6]; YL = [-3.4e6 3.2e6];             % map window, m (EASE-Grid North)
ipa = fullfile(DATA, 'raw', 'permafrost_map', 'nhipa.byte');
if isfile(ipa)
    [rgb, xg, yg] = ipa_image(ipa, XL, YL);
    image(ax, xg, yg, rgb);
else
    warning('No %s: map without permafrost background', ipa);
end
set(ax, 'YDir', 'normal');
for la = [60 70 80]                                     % parallels
    [x, y] = ease_xy(la * ones(1, 181), 0:2:360);
    plot(ax, x, y, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 0.4);
    [x, y] = ease_xy(la, -35);
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
legend(ax, h, lab, 'Location', 'southwest', 'Box', 'off', 'FontSize', FONT - 1);
panel_label(ax, 'a');

% (b) analysed site-years per year, by measurement method
ax = nexttile(tl, 4, [1 2]); hold(ax, 'on');
ann = rd(fullfile(DATA, '1_alt_target', 'alt_multisource_annual.csv'));
ann = ann(as_logical(ann.use), :);
[tf, loc] = ismember(ann.site_id, res.site_id);
ann = ann(tf, :); loc = loc(tf);
inwin = ann.year >= res.year_min(loc) & ann.year <= res.year_max(loc);
ann = ann(inwin, :); loc = loc(inwin);
ann.family = res.family(loc);
[~, iu] = unique(ann.site_id + "_" + string(ann.year));
ann = ann(iu, :);
years = (min(res.year_min):max(res.year_max))';
cnt = zeros(numel(years), numel(FAM));
for f = 1:numel(FAM)
    cnt(:, f) = histcounts(ann.year(ann.family == FAM(f)), [years; years(end) + 1] - 0.5);
end
hb = bar(ax, years, cnt, 1, 'stacked', 'EdgeColor', 'none');
for f = 1:numel(FAM), hb(f).FaceColor = FAM_C(f, :); end
xlim(ax, [years(1) - 1, years(end) + 1]);
ylabel(ax, 'Site-years in the analysis');
legend(ax, FAM_L, 'Location', 'northwest', 'Box', 'off');
panel_label(ax, 'b');
fprintf('Fig 1b: %d site-years (sum of n_obs in the results: %d)\n', height(ann), sum(res.n_obs));
save_figure(fig, OUT, 'Fig1_sites_map_coverage');

%% Figure 2: regional and network rates
nw  = net_value(net, 'A_main_network_rate_pct_dec');
nsm = net_value(net, 'A_main_mean_site_rate_pct_dec');
reg = region_table(res, MIN_SITES_REGION);
fig = new_figure(FIG_WIDTH_CM * 0.62, 9);
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

%% Figure 3: classes by record length and simulation check
fig = new_figure(FIG_WIDTH_CM, 7);
tl = tiledlayout(fig, 1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl); hold(ax, 'on');
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
xlabel(ax, 'Record length (years)'); ylabel(ax, 'Share of sites (%)');
title(ax, 'Classes (S: per site, H: hierarchical)', 'FontWeight', 'normal');
legend(ax, fliplr(hb), fliplr(CLS_L), 'Location', 'southoutside', 'NumColumns', 2, 'Box', 'off', ...
    'FontSize', FONT - 1);
panel_label(ax, 'a');
if have_sim
    ax = nexttile(tl); hold(ax, 'on');
    yline(ax, 90, '--', 'Color', [0.4 0.4 0.4]);
    hm = gobjects(0); ml = strings(0);
    for m = 1:numel(METH)
        k = sim.scenario == "A_fit" & sim.method == METH(m);
        if ~any(k), continue; end
        cov = arrayfun(@(b) sim.(char("coverage_" + SPAN_L(b)))(k), 1:4);
        hm(end+1) = plot(ax, 1:4, 100 * cov, '-o', 'Color', METH_C(m, :), 'MarkerFaceColor', METH_C(m, :), ...
            'MarkerSize', 4, 'LineWidth', 1); %#ok<SAGROW>
        ml(end+1) = METH_L(m); %#ok<SAGROW>
    end
    xticks(ax, 1:4); xticklabels(ax, SPAN_L); xlim(ax, [0.6 4.4]); ylim(ax, [50 100]);
    xlabel(ax, 'Record length (years)'); ylabel(ax, '90 % intervals containing the true trend (%)');
    title(ax, 'Simulation: interval coverage', 'FontWeight', 'normal');
    legend(ax, hm, ml, 'Location', 'southoutside', 'Box', 'off', 'FontSize', FONT - 1);
    panel_label(ax, 'b');

    ax = nexttile(tl); hold(ax, 'on');
    G = ["A_null" "called_trend"; "A_fit" "wrong_direction"; "A_fit" "resolved"];
    V = nan(3, numel(METH));
    for g = 1:3
        for m = 1:numel(METH)
            k = sim.scenario == G(g, 1) & sim.method == METH(m);
            if any(k), V(g, m) = 100 * sim.(char(G(g, 2)))(k); end
        end
    end
    hb = bar(ax, V, 0.85, 'EdgeColor', 'none');
    for m = 1:numel(METH), hb(m).FaceColor = METH_C(m, :); end
    xticks(ax, 1:3);
    xticklabels(ax, {sprintf('Trend called,\\newlineno true trend'), sprintf('Wrong sign'), ...
        sprintf('Trend resolved')});
    ylabel(ax, 'Share of sites (%)');
    title(ax, 'Simulation: decisions', 'FontWeight', 'normal');
    for g = 1:3
        for m = 1:numel(METH)
            if ~isnan(V(g, m))
                text(ax, hb(m).XEndPoints(g), V(g, m) + 1.5, sprintf('%.0f', V(g, m)), ...
                    'HorizontalAlignment', 'center', 'FontSize', FONT - 2);
            end
        end
    end
    panel_label(ax, 'c');
end
save_figure(fig, OUT, 'Fig3_classes_and_simulation');

%% Figure 4: Model B, expected vs residual trend
fig = new_figure(FIG_WIDTH_CM, 8);
tl = tiledlayout(fig, 1, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl, 1, [1 3]); hold(ax, 'on');
ex = res.expected_rate; rs = res.resid_rate;
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
xlabel(ax, 'Expected from air temperature (% per decade)');
ylabel(ax, 'Residual trend (% per decade)');
legend(ax, fliplr(h), fliplr(lab), 'Location', 'northwest', 'Box', 'off', 'FontSize', FONT - 1);
panel_label(ax, 'a');

ax = nexttile(tl, 4, [1 2]); hold(ax, 'on');
if ~have_power, pnet = []; end
B = b_inputs(net, have_power, pnet);
E = vertcat(B.exp); R = vertcat(B.res);
hb = bar(ax, 1:numel(B), [E(:, 1) R(:, 1)], 0.6, 'stacked', 'EdgeColor', 'none');
hb(1).FaceColor = [0.55 0.66 0.80]; hb(2).FaceColor = CLS_C(1, :);
errorbar(ax, 1:numel(B), E(:, 1), E(:, 1) - E(:, 2), E(:, 3) - E(:, 1), 'k', 'LineStyle', 'none', 'CapSize', 3);
errorbar(ax, 1:numel(B), E(:, 1) + R(:, 1), R(:, 1) - R(:, 2), R(:, 3) - R(:, 1), 'k', 'LineStyle', 'none', 'CapSize', 3);
xticks(ax, 1:numel(B)); xticklabels(ax, {B.label}); xtickangle(ax, 0);
ylabel(ax, 'Mean site trend (% per decade)');
legend(ax, hb, {'Expected from weather', 'Not explained by weather'}, 'Location', 'northoutside', 'Box', 'off');
panel_label(ax, 'b');
save_figure(fig, OUT, 'Fig4_expected_vs_residual');

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

%% Supplement: Fig S1 and Table S1, sensitivity runs
if have_sens
    keep = intersect({'run', 'n_sites', 'n_site_years', 'A_network_rate', 'A_network_lo95', 'A_network_hi95', ...
        'A_mean_site_rate', 'A_declustered_250km', 'A_rapid_thickening', 'A_gradual_thickening', 'A_stable', ...
        'A_gradual_thinning', 'A_rapid_thinning', 'A_indeterminate', 'B_mean_expected', 'B_mean_residual', ...
        'B_deeper_than_TDD', 'Bwm_mean_expected', 'Bwm_mean_residual', 'Bwm_deeper_than_TDD'}, ...
        sens.Properties.VariableNames, 'stable');
    writetable(sens(:, keep), fullfile(OUT, 'TableS1_sensitivity.csv'));

    fig = new_figure(FIG_WIDTH_CM, 8);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nrun = height(sens); yy = (nrun:-1:1)';
    ax = nexttile(tl); hold(ax, 'on');
    patch(ax, [nw(2) nw(3) nw(3) nw(2)], [0.5 0.5 nrun + 0.5 nrun + 0.5], CLS_C(1, :), 'FaceAlpha', 0.1, ...
        'EdgeColor', 'none');
    hE = errorbar(ax, sens.A_network_rate, yy, sens.A_network_rate - sens.A_network_lo95, ...
        sens.A_network_hi95 - sens.A_network_rate, 'horizontal', 'o', 'Color', 'k', 'MarkerFaceColor', 'k', ...
        'MarkerSize', 3, 'CapSize', 2);
    hM = plot(ax, sens.A_mean_site_rate, yy, 's', 'Color', [0.45 0.45 0.45], 'MarkerSize', 4);
    yticks(ax, 1:nrun); yticklabels(ax, flipud(strrep(sens.run, '_', ' '))); ylim(ax, [0.5 nrun + 0.5]);
    xlabel(ax, 'ALT change (% per decade)');
    legend(ax, [hE hM], {'Network rate (95 %)', 'Mean of sites'}, 'Location', 'southoutside', 'Box', 'off');
    panel_label(ax, 'a');
    ax = nexttile(tl); hold(ax, 'on');
    hS = plot(ax, sens.B_mean_expected, yy, 'o', 'Color', [0.55 0.66 0.80], 'MarkerFaceColor', [0.55 0.66 0.80], 'MarkerSize', 4);
    hS(2) = plot(ax, sens.B_mean_residual, yy, 'o', 'Color', CLS_C(1, :), 'MarkerFaceColor', CLS_C(1, :), 'MarkerSize', 4);
    sl = {'Expected, air temperature', 'Residual, air temperature'};
    if ismember('Bwm_mean_residual', sens.Properties.VariableNames)
        hS(3) = plot(ax, sens.Bwm_mean_residual, yy, 'o', 'Color', CLS_C(1, :), 'MarkerSize', 4);
        sl{3} = 'Residual, all weather + memory';
    end
    xline(ax, 0, ':', 'Color', [0.4 0.4 0.4]);
    yticks(ax, 1:nrun); yticklabels(ax, {}); ylim(ax, [0.5 nrun + 0.5]);
    xlabel(ax, 'Mean site trend (% per decade)');
    legend(ax, hS, sl, 'Location', 'southoutside', 'Box', 'off');
    panel_label(ax, 'b');
    save_figure(fig, OUT, 'FigS1_sensitivity');
end

%% Supplement: Fig S2, residual by ground-ice content (IPA map), within regions
if ismember('pf_ice', res.Properties.VariableNames)
    ICE = ["low" "medium" "high"];
    fig = new_figure(FIG_WIDTH_CM * 0.8, 7);
    tl = tiledlayout(fig, 1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax = nexttile(tl); hold(ax, 'on'); rng(2);
    for i = 1:3
        k = res.pf_ice == ICE(i);
        scatter(ax, i + 0.3 * (rand(sum(k), 1) - 0.5), rs(k), 9, [0.5 0.5 0.5], 'filled', 'MarkerFaceAlpha', 0.5);
        plot(ax, i + [-0.25 0.25], median(rs(k), 'omitnan') * [1 1], 'k-', 'LineWidth', 1.5);
        text(ax, i, max(rs) + 2, sprintf('n=%d', sum(k)), 'HorizontalAlignment', 'center', 'FontSize', FONT - 1);
    end
    yline(ax, 0, ':');
    xticks(ax, 1:3); xticklabels(ax, ICE + " ice"); xlim(ax, [0.5 3.5]);
    ylabel(ax, 'Residual trend (% per decade)'); title(ax, 'All sites', 'FontWeight', 'normal');
    panel_label(ax, 'a');
    ax = nexttile(tl); hold(ax, 'on');
    reg_ids = unique(res.region_id); d = []; lbl = strings(0); nn = [];
    for r = reg_ids'
        lo_k = res.region_id == r & res.pf_ice == "low"; hi_k = res.region_id == r & res.pf_ice == "high";
        if any(lo_k) && any(hi_k)
            d(end+1) = mean(rs(lo_k), 'omitnan') - mean(rs(hi_k), 'omitnan'); %#ok<SAGROW>
            lbl(end+1) = region_name(mean(res.lat(res.region_id == r)), circmean(res.lon(res.region_id == r))); %#ok<SAGROW>
            nn(end+1, :) = [sum(lo_k) sum(hi_k)]; %#ok<SAGROW>
        end
    end
    if ~isempty(d)
        barh(ax, 1:numel(d), d, 0.6, 'FaceColor', [0.55 0.55 0.55], 'EdgeColor', 'none');
        xline(ax, mean(d), '--', 'Color', CLS_C(1, :), 'LineWidth', 1);
        xline(ax, 0, ':');
        yticks(ax, 1:numel(d)); yticklabels(ax, lbl(:) + " (" + nn(:, 1) + "/" + nn(:, 2) + ")");
        xlabel(ax, 'Low minus high ice, residual (% per decade)');
        title(ax, sprintf('Within regions, mean %+.1f', mean(d)), 'FontWeight', 'normal');
    end
    panel_label(ax, 'b');
    save_figure(fig, OUT, 'FigS2_residual_by_ground_ice');
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

%% Key numbers for the text
fid = fopen(fullfile(OUT, 'key_numbers.txt'), 'w');
p = @(varargin) fprintf(fid, [varargin{1} '\n'], varargin{2:end});
p('ALDI paper: numbers quoted in the Results (from data\\3_aldi*, %s)', char(datetime('now', 'Format', 'yyyy-MM-dd')));
p('');
p('DATA');
p('  series %d, site-years %d (Fig 1b count %d), regions %d', n_sites, sum(res.n_obs), height(ann), ...
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
    p('  sensitivity runs: network %.1f to %.1f, residual %.1f to %.1f, residual all weather %.1f to %.1f', ...
        min(sens.A_network_rate), max(sens.A_network_rate), min(sens.B_mean_residual), max(sens.B_mean_residual), ...
        min(sens.Bwm_mean_residual), max(sens.Bwm_mean_residual));
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
if exist('d', 'var') && ~isempty(d)
    p('  low minus high ground ice within %d regions: %+.1f (%.1f to %.1f)', numel(d), mean(d), min(d), max(d));
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

function panel_label(ax, s)
text(ax, -0.02, 1.02, ['(' s ')'], 'Units', 'normalized', 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
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

function v = net_value(T, name)
% median and 95 % interval of one network parameter (aldi_v6_network.csv)
k = find(string(T{:, 1}) == name, 1);
if isempty(k), v = [NaN NaN NaN]; else, v = [T.median(k) T.lo95(k) T.hi95(k)]; end
end

function B = b_inputs(net, have_power, pnet)
% expected, residual and total mean site trend for the three climate inputs of Table 3
B = struct('name', {}, 'label', {}, 'exp', {}, 'res', {}, 'tot', {}, 'k', {});
B(1) = b_one(net, 'B_main', 'ERA5-Land, air temperature', sprintf('ERA5-Land\\newlineair temp.'));
B(2) = b_one(net, 'B_weather_memory', 'ERA5-Land, all weather + 2-year memory', ...
    sprintf('ERA5-Land\\newlineall weather'));
if have_power
    B(3) = b_one(pnet, 'B_main', 'NASA POWER, air temperature', sprintf('NASA POWER\\newlineair temp.'));
end
end

function b = b_one(T, pre, name, label)
b.name = name; b.label = label;
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

function [rgb, xg, yg] = ipa_image(file, XL, YL)
% IPA permafrost map (Brown et al. 2002, NSIDC GGD318), 12.5 km EASE-Grid North,
% 1441 x 1441 bytes. Codes 1-20: extent x ground ice x landform (continuous 1 5 9 13 17,
% discontinuous 2 6 10 14 18, sporadic 3 7 11 15 19, isolated 4 8 12 16 20); 21 glacier,
% 22 relict, 23 lake, 24 ocean, 25 land without permafrost.
fid = fopen(file, 'r'); g = fread(fid, [1441 1441], 'uint8=>uint8')'; fclose(fid);
d = 12533.7625; ul = 9024309;                           % centre of the upper-left cell
xg = -ul + (0:1440) * d; yg = ul - (0:1440) * d;
ix = xg >= XL(1) - d & xg <= XL(2) + d; iy = yg >= YL(1) - d & yg <= YL(2) + d;
g = g(iy, ix); xg = xg(ix); yg = yg(iy);
groups = {[22 25], [0.92 0.92 0.92]; 21, [0.97 0.97 0.97]; [1 5 9 13 17], [0.62 0.71 0.81]; ...
          [2 6 10 14 18], [0.74 0.80 0.87]; [3 7 11 15 19], [0.84 0.88 0.92]; [4 8 12 16 20], [0.89 0.92 0.95]};
rgb = ones([size(g) 3]);
for i = 1:size(groups, 1)
    m = ismember(g, groups{i, 1});
    for c = 1:3
        ch = rgb(:, :, c); ch(m) = groups{i, 2}(c); rgb(:, :, c) = ch;
    end
end
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
