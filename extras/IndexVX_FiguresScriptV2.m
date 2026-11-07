%% ** First, CircumArctic map with all original circumarctic sites

clear
clc
% Load the data file
filePath = 'G:\My Drive\UND\Index\CALM_ALT_all_Max.csv';
[data,~,~,~] = preprocess_CALM(filePath, 0);
region_names = unique(data.MainRegion);

mk = {'o','s','^','d','>','p'};
faceclr = turbo(length(region_names)+4);

figure('units','normalized','OuterPosition',[0 0 1 1]);
ax = axesm('stereo', 'MapLatLimit', [60 90], 'Frame', 'on', 'Grid', 'on');
setm(ax, 'MeridianLabel', 'off', 'ParallelLabel', 'on');

% Load land shapefile (Natural Earth land data)
land = shaperead('landareas.shp', 'UseGeoCoords', true);

% Plot land areas in light gray
geoshow(ax, land, 'FaceColor', [0.85 0.85 0.85], 'EdgeColor', 'k');

%read permafrost map dataset
shapefile = 'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\CALM\ggd318_map_circumarctic\permaice.shp';
permafrost = shaperead(shapefile, 'UseGeoCoords', false); % Faster reading

% Extract projection information
shape_info = shapeinfo(shapefile);
source_proj = shape_info.CoordinateReferenceSystem;

% Convert all projected coordinates to lat/lon **in one step**
[X, Y] = deal({permafrost.X}, {permafrost.Y});
[lat, lon] = cellfun(@(x, y) projinv(source_proj, x, y), X, Y, 'UniformOutput', false);

%%**Categorize Data: Permafrost, Ground Ice, Landform**
categories = struct( ...
    'permafrost', struct( ...
    'continuous', {'c'}, ...
    'discontinuous', {'d'}, ...
    'sporadic', {'s'}, ...
    'isolated', {'i'}), ...
    'ground_ice', struct( ...
    'high', {'h'}, ...
    'medium', {'m'}, ...
    'low', {'l'}), ...
    'landform', struct( ...
    'lowlands', {'f'}, ...
    'mountains', {'r'}));

% Define Colors for Each Layer
colors.permafrost = struct( ...
    'continuous', [0.0,0.2,0.8], ...
    'discontinuous', [0.2,0.4,0.9], ...
    'sporadic', [0.4,0.6,1.0], ...
    'isolated', [0.7,0.85,1.0]);

% Extract all `COMBO` values at once
combo_values = {permafrost.COMBO};

% Initialize category assignments
num_polygons = length(permafrost);
permafrost_types = repmat({'other'}, num_polygons, 1);

% Assign categories
for i = 1:num_polygons
    combo_code = lower(combo_values{i}); % Convert to lowercase

    % Extract permafrost type
    if numel(combo_code) > 2
        permafrost_types{i} = combo_code(1);
    end
end

field = fieldnames(categories.permafrost)';
for i = 1:numel(field)
    category =  categories.permafrost.(field{i});
    idx = strcmp(permafrost_types, category);
    if any(idx)
        h1(i) = geoshow([lat{idx}], [lon{idx}], 'DisplayType', 'polygon', ...
            'FaceColor', colors.permafrost.(field{i}), 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    end
end
lgd1 = legend([h1(1) h1(2) h1(3) h1(4)],{'Continuous (90-100\%)', 'Discontinuous (50-90\%)', 'Sporadic (10-50\%)', 'Isolated (0-10\%)'});
set(lgd1,'Location', 'northeastoutside', 'Orientation', 'vertical', 'Interpreter','latex','FontSize',18);

% Plot CALM sites
for i = 2:length(region_names)
    tmp = data(data.MainRegion==region_names(i),:);
    [regionlocations,~,~] = unique([tmp.Lat tmp.Long],"rows");
    g = geoshow(regionlocations(:,1), regionlocations(:,2), 'DisplayType', 'point', ...
        'Marker', mk{i}, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', faceclr(i+3,:),'MarkerSize',10, 'DisplayName',string(region_names(i)));
end
set(lgd1, 'Location', 'eastoutside', 'Orientation', 'vertical', 'Interpreter','latex','FontSize',18);
lgd1.Title.String = 'Permafrost extent';

exportgraphics(gcf,'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\Index\CALM167locationsPerma_map.pdf', 'Resolution',300);
lgd1.Title.String = 'Main regions';
exportgraphics(gcf,'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\Index\CALM167locationsPerma_map_2.pdf', 'Resolution',300);


%% ** then Regions' ALT figures
clear
clc

% Load the data
filePath = 'G:\My Drive\UND\Index\CALM_ALT_all_Max.csv';
[data,~,~,~] = preprocess_CALM(filePath, 0);
region_names = unique(data.MainRegion);

% Region names and colors
faceclr = turbo(length(region_names)+4);
mk = {'o','s','^','d','>','p'};

% Create figure with subplots
figure('units','normalized','OuterPosition',[0 0 1 1]);
t = tiledlayout('flow','TileSpacing','compact','Padding','compact');
for r = 1:length(region_names)
    region_data = data(data.MainRegion==region_names(r),:);
    
    if isempty(region_data)
        continue;
    end
    
    % Get unique sites in this region
    sites = unique([region_data.Lat, region_data.Long], 'rows');
    
    % Create subplot
    nexttile;
    hold on;
    
    % Plot individual site time series
    for s = 1:size(sites, 1)
        site_data = region_data(region_data.Lat == sites(s,1) & region_data.Long == sites(s,2), :);
        if height(site_data)<8% not now this is an early on figure about the dataset %remove less than 8 years worth of ALT time series 
            continue,
        else
            [~, idx] = sort(site_data .Year, 'ascend');
            if s == 1
                plot(site_data.Year(idx), site_data.Max(idx), '-', 'Color', [faceclr(r+3,:) 0.2], 'LineWidth', 1, 'DisplayName','Sites');
            else
                plot(site_data.Year(idx), site_data.Max(idx), '-', 'Color', [faceclr(r+3,:) 0.2], 'LineWidth', 1, 'HandleVisibility','off');
            end
        end
    end
    
    % Calculate and plot regional average
    years = unique(region_data.Year);
    regional_avg = zeros(length(years), 1);
    
    for y = 1:length(years)
        year_data = region_data(region_data.Year == years(y), :);
        regional_avg(y) = mean(year_data.Max, 'omitnan');
    end
    
    plot(years, regional_avg, '-', 'Color', faceclr(r+3,:), 'Marker',mk{r},'MarkerEdgeColor','auto','MarkerFaceColor','w','LineWidth', 2.3, 'DisplayName', 'Regional Average\,\,');
    
    % Add trend line
    valid_idx = ~isnan(regional_avg);
    if sum(valid_idx) > 2
        p = polyfit(years(valid_idx), regional_avg(valid_idx), 1);
        trend_line = polyval(p, years);
        plot(years, trend_line, '--k', 'LineWidth', 2, 'DisplayName', 'Linear Trend\,\,\,\,\,');
        switch (p(1)*10<0) %define the sign to add a plus sign or not
            case 0
                sign_trnd = '+';
            otherwise
                sign_trnd = '';
        end

        % Add trend value to title
        title(sprintf('%s (n=%d sites)', ...%\nTrend: %s%.2f cm/decade
            region_names(r), size(sites,1)), 'FontSize', 18, 'Interpreter', 'latex');%, sign_trnd, p(1)*10
    else
        title(sprintf('%s (n=%d sites)', region_names{r}, size(sites,1)), ...
            'FontSize', 18, 'Interpreter', 'latex');
    end
    
    % xlabel('Year', 'FontSize', 18, 'Interpreter', 'latex');
    set(gca, "FontSize",18, "TickLabelInterpreter", "latex")
    xlabel('')
    ylabel('Active Layer Thickness (cm)', 'FontSize', 18, 'Interpreter', 'latex');
    grid on;
    xlim([min(years) max(years)]);
    box on
    hold off;
end
lgd = legend(gca, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 12);
lgd.Layout.Tile = 'East'; % <-- place legend east of tiles
% Add overall title
% sgtitle('Arctic Active Layer Thickness Time Series by Region', ...
    % 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'latex');
exportgraphics(gcf,'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\Index\CALM167locations_timeseries.pdf', 'Resolution',300);


%% ** CALM ALT preprocessing stats
% This script:
% 1) Loads and cleans CALM ALT Max table
% 2) Filters to Arctic (Lat>=60) + sites with >=8 unique years
% 3) Assigns MainRegion (requires assign_CALM_regions.m on path)
% 4) Reports EDA: missingness, duplicates, censoring, coverage, distributions

clear; clc; close all;

filePath = 'G:\My Drive\UND\Index\CALM_ALT_all_Max.csv';

[result_table,region_counts,alt_by_region,cov_by_region] = preprocess_CALM(filePath, 1);


%% ** Simple grouped horizontal bar chart by region
%  Robust version: positions read from rendered bar objects after drawnow.
%  Includes diagnostic table so you can verify counts before inspecting fig.
%
%  MATLAB R2019b+ required (XEndPoints / YEndPoints).

clear; clc; close all;

%---- user paths --------------------------------------------------------
filePath = 'G:\My Drive\UND\Index\aldi_v4_results_wMainRegion.csv';

%---- load & clean ------------------------------------------------------
T = readtable(filePath, 'TextType','string');
T = T(:, {'MainRegion','class_classification'});
T = rmmissing(T);

T.MainRegion = strtrim(T.MainRegion);

% Normalise class names: snake_case → Title Case
raw = lower(replace(strtrim(T.class_classification), "_", " "));
raw = regexprep(raw, '(\<[a-z])', '${upper($1)}');
T.class_classification = raw;

%---- Verify class names actually match expected set --------------------
classOrder = ["Rapid Thickening"
              "Gradual Thickening"
              "No Trend"
              "Gradual Thinning"
              "Rapid Thinning"
              "Transitional"];

foundClasses = unique(T.class_classification);
extra   = setdiff(foundClasses, classOrder);
if ~isempty(extra)
    warning('Unexpected class names in data: %s', strjoin(extra, ', '));
end

nClass = numel(classOrder);

%---- Colours -----------------------------------------------------------
classColors = [0.84  0.19  0.15        % Rapid Thickening  — strong red
               0.96  0.51  0.35        % Gradual Thickening — salmon
               0.70  0.70  0.70        % No Trend           — grey
               0.40  0.68  0.84        % Gradual Thinning   — light blue
               0.12  0.38  0.65        % Rapid Thinning     — dark blue
               0.58  0.40  0.74];      % Transitional       — purple

%---- build count matrix ------------------------------------------------
% Rows = regions (alphabetical), Columns = classes (classOrder)
regions = ["Alaska"
    "Russia"
    "Canada"
    "Svalbard"
    "Greenland"    
    "Scandinavia"];
nReg    = numel(regions);

countMat = zeros(nReg, nClass);
for r = 1:nReg
    for c = 1:nClass
        countMat(r, c) = sum(T.MainRegion == regions(r) & ...
                             T.class_classification == classOrder(c));
    end
end

% ---- Print verification table -------------------------------------------
fprintf('\n===== COUNT TABLE (verify before looking at figure) =====\n');
verifyT = array2table(countMat, ...
    'VariableNames', cellstr(replace(classOrder," ","_")), ...
    'RowNames',      cellstr(regions));
disp(verifyT);
fprintf('Row sums:  ');  fprintf('%d  ', sum(countMat,2));  fprintf('\n');
fprintf('Col sums:  ');  fprintf('%d  ', sum(countMat,1));  fprintf('\n');
fprintf('Total: %d\n\n', sum(countMat(:)));

%---- Append "All Regions" summary row ----------------------------------
countMat = [countMat; sum(countMat, 1)];
regions  = [regions; "All Regions"];
nReg     = numel(regions);

%---- Flip region order so first alphabetical is at TOP -----------------
%  barh() plots row 1 at the BOTTOM by default. We flip so that
%  alphabetical regions are top-down, with "All Regions" at the bottom.
regions  = flipud(regions);
countMat = flipud(countMat);

%---- Plot --------------------------------------------------------------
figure('units','normalized','OuterPosition',[0           0        1         1]);
t = tiledlayout('flow','TileSpacing','compact','Padding','compact');
b = barh(1:nReg, countMat, 'grouped', 'EdgeColor','none', 'BarWidth', .9, 'GroupWidth',.95);

% Assign colours
for k = 1:nClass
    b(k).FaceColor = 'flat';
    b(k).CData     = classColors(k,:);
end

% Y-axis labels
set(gca, 'YTick', 1:nReg, 'YTickLabel', regions);

% Styling
xlabel('Number of Sites', 'FontSize',18, 'Interpreter','latex');
ylabel('');
set(gca, 'FontSize',18, 'TickLabelInterpreter','latex', ...
         'Box','on', 'TickDir','in', 'XMinorGrid','off');
ax = gca;
ax.YAxis.TickLength = [0 0];

xlim([0, max(countMat(:)) * 1.13]);

%---- Force MATLAB to render bar positions before reading them ----------
drawnow;

%---- Value labels (non-zero only) --------------------------------------
%  CRITICAL: for barh, MATLAB swaps the meaning of EndPoints:
%    XEndPoints → VERTICAL category-axis positions  (y on screen)
%    YEndPoints → HORIZONTAL value-axis positions    (x on screen)
%  This is because barh transposes the data axes.

hold on;
xNudge = max(countMat(:)) * 0.005;   % small gap past bar tip

for k = 1:nClass
    yCenters = b(k).XEndPoints;   % vertical bar-centre positions
    xTips    = b(k).YEndPoints;   % horizontal bar-tip positions
    vals     = b(k).YData;        % the actual plotted values

    % --- sanity check (uncomment to debug) ---
    % fprintf('b(%d) [%s]  YData: %s\n', k, classOrder(k), mat2str(vals));
    % fprintf('   XEndPoints: %s\n', mat2str(round(xTips,2)));
    % fprintf('   YEndPoints: %s\n', mat2str(round(yCenters,2)));

    for r = 1:nReg
        v = vals(r);
        if v > 0 && r > 1 
            text(xTips(r) + xNudge, yCenters(r), num2str(v), ...
                 'FontSize', 15, 'Interpreter', 'latex', ...
                 'VerticalAlignment', 'middle', ...
                 'HorizontalAlignment', 'left');
        end
        if v > 0 && r == 1
            text(xTips(r) + xNudge, yCenters(r), sprintf('%d (%.2f\\%%)',v,100*v/sum([b(1).YData(r) b(2).YData(r) b(3).YData(r) b(4).YData(r) b(5).YData(r) b(6).YData(r)])), ...
                 'FontSize', 15, 'Interpreter', 'latex', ...
                 'VerticalAlignment', 'middle', ...
                 'HorizontalAlignment', 'left');
        end
    end
end
hold off;

%---- Separator lines between region groups -----------------------------
%  After flipud, row 1 = "All Regions" (bottom of plot).
%  Give it a thicker separator to distinguish summary from individual regions.
for i = 1:(nReg-1)
    if i == 1   % line between "All Regions" and the first individual region
        lw = 1.8;
    else
        lw = 0.6;
    end
    yline(i + 0.5, 'Color','k', 'LineWidth', lw, 'HandleVisibility','off');
end

%---- Legend ------------------------------------------------------------
lgd = legend(b, strrep(classOrder, "Gradual Thickening", "Gradual Thickening\,\,\,\,\,"), ...
    'Location','northeast', 'Orientation','vertical', ...
    'Interpreter','latex', 'FontSize',18, 'Box','on');
lgd.Title.String = 'Active Layer Dynamics';
lgd.Title.Interpreter = 'latex';

exportgraphics(gcf, 'G:\My Drive\UND\Index\Index_BarPlotByRegion.pdf', 'Resolution', 300);

%% ** Maps indx state and confidence - polar
close('all')
clear
clc

set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

%Load data
data = readtable('G:\My Drive\UND\Index\aldi_v4_results_wMainRegion.csv');

data.MainRegion = strtrim(data.MainRegion);

% Normalise class names: snake_case → Title Case
raw = lower(replace(strtrim(data.class_classification), "_", " "));
raw = regexprep(raw, '(\<[a-z])', '${upper($1)}');
data.class_classification = raw;

%Parse classification into class and rate
n_sites = height(data);
data.class = strings(n_sites, 1);
data.rate = strings(n_sites, 1);
data.full_class = strings(n_sites, 1);

for i = 1:n_sites
    clf = string(data.class_classification{i});
    
    switch clf
        case 'Rapid Thickening'
            data.class(i) = "Thickening";
            data.rate(i) = "Rapid";
            data.full_class(i) = "Rapid Thickening";
        case 'Gradual Thickening'
            data.class(i) = "Thickening";
            data.rate(i) = "Gradual";
            data.full_class(i) = "Gradual Thickening";
        case 'Rapid Thinning'
            data.class(i) = "Thinning";
            data.rate(i) = "Rapid";
            data.full_class(i) = "Rapid Thinning";
        case 'Gradual Thinning'
            data.class(i) = "Thinning";
            data.rate(i) = "Gradual";
            data.full_class(i) = "Gradual Thinning";
        case 'No Trend'
            data.class(i) = "No Trend";
            data.rate(i) = "";
            data.full_class(i) = "No Trend";
        case 'Transitional'
            data.class(i) = "Transitional";
            data.rate(i) = "";
            data.full_class(i) = "Transitional";
    end
end

% Convert to categorical with specified order
fullClassOrder = ["No Trend", "Rapid Thickening", "Gradual Thickening", ...
                  "Gradual Thinning", "Rapid Thinning", "Transitional"];

data.full_class = categorical(data.full_class, fullClassOrder);

% IMPORTANT: Convert region to categorical/string for comparison
data.MainRegion = categorical(strtrim(string(data.MainRegion)));

%---- Colours -----------------------------------------------------------
rgbList = [0.70  0.70  0.70            % No Trend           — grey
               0.84  0.19  0.15        % Rapid Thickening  — strong red
               0.96  0.51  0.35        % Gradual Thickening — salmon
               0.40  0.68  0.84        % Gradual Thinning   — light blue
               0.12  0.38  0.65        % Rapid Thinning     — dark blue
               0.58  0.40  0.74];      % Transitional       — purple
                    
for i = 1:height(data.class_confidence)
    tmp = data.class_confidence{i};
    switch tmp
        case 'high'
            data.class_confidence_int(i) = 10;
        case 'moderate'
            data.class_confidence_int(i) = 5;
        case 'low'
            data.class_confidence_int(i) = 1;
    end
end

% find only unique location data
[data_unq,~,ic] = unique([data.lat data.lon],"rows");
data = data(ic,:);
mk = {'*','^','^','v','v','s','o'};


figure('units','normalized','OuterPosition',[0 0 1 1]);
% Create a polar map projection
ax_actual = axesm('stereo', 'MapLatLimit', [60 90], 'Frame', 'on', 'Grid', 'on',...
    'MeridianLabel', 'off', 'ParallelLabel', 'on','GLineStyle',':',...
    'GColor',[0.8 0.8 0.8],'FLineWidth',.2,'FFaceColor',"#92c9ff");
% Plot the land boundaries
land = shaperead('landareas.shp', 'UseGeoCoords', true);
geoshow(land, 'FaceColor', [1 1 1], 'EdgeColor', [0 0 0]);
hold on,
%read permafrost map dataset
shapefile = 'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\CALM\ggd318_map_circumarctic\permaice.shp';
permafrost = shaperead(shapefile, 'UseGeoCoords', false); % Faster reading

% Extract projection information
shape_info = shapeinfo(shapefile);
source_proj = shape_info.CoordinateReferenceSystem;

% Convert all projected coordinates to lat/lon **in one step**
[X, Y] = deal({permafrost.X}, {permafrost.Y});
[lat, lon] = cellfun(@(x, y) projinv(source_proj, x, y), X, Y, 'UniformOutput', false);

%%**Categorize Data: Permafrost, Ground Ice, Landform**
categories = struct( ...
    'permafrost', struct( ...
    'continuous', {'c'}, ...
    'discontinuous', {'d'}, ...
    'sporadic', {'s'}, ...
    'isolated', {'i'}), ...
    'ground_ice', struct( ...
    'high', {'h'}, ...
    'medium', {'m'}, ...
    'low', {'l'}), ...
    'landform', struct( ...
    'lowlands', {'f'}, ...
    'mountains', {'r'}));

% Define Colors for Each Layer
colors.permafrost = struct( ...
    'continuous', [0.0,0.2,0.8], ...
    'discontinuous', [0.2,0.4,0.9], ...
    'sporadic', [0.4,0.6,1.0], ...
    'isolated', [0.7,0.85,1.0]);

% Extract all `COMBO` values at once
combo_values = {permafrost.COMBO};

% Initialize category assignments
num_polygons = length(permafrost);
permafrost_types = repmat({'other'}, num_polygons, 1);

% Assign categories
for i = 1:num_polygons
    combo_code = lower(combo_values{i}); % Convert to lowercase

    % Extract permafrost type
    if numel(combo_code) > 2
        permafrost_types{i} = combo_code(1);
    end
end

field = fieldnames(categories.permafrost)';
for i = 1:numel(field)
    category =  categories.permafrost.(field{i});
    if i == 1
    idx = strcmp(permafrost_types, category);
    if any(idx)
        h1(i) = geoshow([lat{idx}], [lon{idx}], 'DisplayType', 'polygon', ...
            'FaceColor', '#bcf6e8', 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    end        
    end
    if i == 2
    idx = strcmp(permafrost_types, category);
    if any(idx)
        h1(i) = geoshow([lat{idx}], [lon{idx}], 'DisplayType', 'polygon', ...
            'FaceColor', '#bcf6e8', 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    end        
    end    
    if i == 3
    idx = strcmp(permafrost_types, category);
    if any(idx)
        h1(i) = geoshow([lat{idx}], [lon{idx}], 'DisplayType', 'polygon', ...
            'FaceColor', '#bcf6e8', 'EdgeColor', 'none', 'FaceAlpha', 0.2);
    end        
    end      
    if i == 4
    idx = strcmp(permafrost_types, category);
    if any(idx)
        h1(i) = geoshow([lat{idx}], [lon{idx}], 'DisplayType', 'polygon', ...
            'FaceColor', '#bcf6e8', 'EdgeColor', 'none', 'FaceAlpha', 0.1);
    end        
    end       
end
% geoshow(lat, lon, 'FaceColor', '#bcf6e8', 'EdgeColor', 'none', 'DisplayType', 'polygon');

for ii = 1:length(fullClassOrder)
    sprintf(fullClassOrder(ii))
    tmp = data(data.full_class==fullClassOrder(ii),:);
    if ~strcmp(fullClassOrder(ii),'No Trend')
        g(ii) = geoshow(tmp.lat, tmp.lon, 'DisplayType', 'point', ...
    'Marker', mk(ii), 'MarkerEdgeColor', 'w', 'MarkerFaceColor', rgbList(ii,:),'MarkerSize',10, 'DisplayName',string(unique(tmp.full_class)));
    else
        g(ii) = geoshow(tmp.lat, tmp.lon, 'DisplayType', 'point', ...
    'Marker', mk(ii), 'MarkerEdgeColor', 'k', 'MarkerFaceColor', rgbList(ii,:),'MarkerSize',10, 'DisplayName',string(unique(tmp.full_class)));
    end
end
lgd2 = legend(g,fullClassOrder, "Location","northeastoutside","Interpreter","latex",FontSize=20);
lgd2.AutoUpdate = 'off';
lgd2.PlotChildren = lgd2.PlotChildren([2:6, 1]); % Reorder handles internally
lgd2.Title.String = '\,Active Layer Dynamics\,\,';
% exportgraphics(gcf,'G:\My Drive\UND\Index\Index_ResultsMap_All.pdf','Resolution',300);

%% ** Maps index state and confidence - polar - CALM locations by region
%Regional zoom maps using AXESM + GEOSHOW
%Reuses the exact permafrost zonation logic that already works
%Only the axes/zoom are changed for regional panels

close all
clear
clc

set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

%Load data
data = readtable('G:\My Drive\UND\Index\aldi_v4_results_wMainRegion.csv');
data.MainRegion = strtrim(data.MainRegion);

% Normalize class names: snake_case -> Title Case
raw = lower(replace(strtrim(data.class_classification), "_", " "));
raw = regexprep(raw, '(\<[a-z])', '${upper($1)}');
data.class_classification = raw;

%Parse classification into class and rate
n_sites = height(data);
data.class = strings(n_sites,1);
data.rate = strings(n_sites,1);
data.full_class = strings(n_sites,1);

for i = 1:n_sites
    clf = string(data.class_classification{i});

    switch clf
        case 'Rapid Thickening'
            data.class(i) = "Thickening";
            data.rate(i) = "Rapid";
            data.full_class(i) = "Rapid Thickening";

        case 'Gradual Thickening'
            data.class(i) = "Thickening";
            data.rate(i) = "Gradual";
            data.full_class(i) = "Gradual Thickening";

        case 'Rapid Thinning'
            data.class(i) = "Thinning";
            data.rate(i) = "Rapid";
            data.full_class(i) = "Rapid Thinning";

        case 'Gradual Thinning'
            data.class(i) = "Thinning";
            data.rate(i) = "Gradual";
            data.full_class(i) = "Gradual Thinning";

        case 'No Trend'
            data.class(i) = "No Trend";
            data.rate(i) = "";
            data.full_class(i) = "No Trend";

        case 'Transitional'
            data.class(i) = "Transitional";
            data.rate(i) = "";
            data.full_class(i) = "Transitional";
    end
end

%Convert to categorical with specified order
fullClassOrder = ["No Trend", "Rapid Thickening", "Gradual Thickening", ...
                  "Gradual Thinning", "Rapid Thinning", "Transitional"];
data.full_class = categorical(data.full_class, fullClassOrder);

% IMPORTANT: Convert region to categorical/string for comparison
data.MainRegion = categorical(strtrim(string(data.MainRegion)));

%Colors
rgbList = [0.70  0.70  0.70            % No Trend
           0.84  0.19  0.15            % Rapid Thickening
           0.96  0.51  0.35            % Gradual Thickening
           0.40  0.68  0.84            % Gradual Thinning
           0.12  0.38  0.65            % Rapid Thinning
           0.58  0.40  0.74];          % Transitional

%Confidence to numeric
data.class_confidence_int = nan(height(data),1);
for i = 1:height(data)
    tmp = data.class_confidence{i};
    switch tmp
        case 'high'
            data.class_confidence_int(i) = 10;
        case 'moderate'
            data.class_confidence_int(i) = 5;
        case 'low'
            data.class_confidence_int(i) = 1;
    end
end

%Keep only unique locations
[~,~,ic] = unique([data.lat data.lon],"rows");
data = data(ic,:);

%Marker set
mk = {'*','^','^','v','v','s','o'};

%Load land boundaries
land = shaperead('landareas.shp', 'UseGeoCoords', true);

%Read permafrost map dataset
shapefile = 'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\CALM\ggd318_map_circumarctic\permaice.shp';
permafrost = shaperead(shapefile, 'UseGeoCoords', false); % Faster reading

% Extract projection information
shape_info = shapeinfo(shapefile);
source_proj = shape_info.CoordinateReferenceSystem;

% Convert all projected coordinates to lat/lon in one step
[X, Y] = deal({permafrost.X}, {permafrost.Y});
[lat, lon] = cellfun(@(x, y) projinv(source_proj, x, y), X, Y, 'UniformOutput', false);

%Categorize permafrost polygons
categories = struct( ...
    'permafrost', struct( ...
        'continuous', {'c'}, ...
        'discontinuous', {'d'}, ...
        'sporadic', {'s'}, ...
        'isolated', {'i'}), ...
    'ground_ice', struct( ...
        'high', {'h'}, ...
        'medium', {'m'}, ...
        'low', {'l'}), ...
    'landform', struct( ...
        'lowlands', {'f'}, ...
        'mountains', {'r'}));

% Extract all COMBO values at once
combo_values = {permafrost.COMBO};

% Initialize category assignments
num_polygons = length(permafrost);
permafrost_types = repmat({'other'}, num_polygons, 1);

% Assign categories
for i = 1:num_polygons
    combo_code = lower(combo_values{i});
    if numel(combo_code) > 2
        permafrost_types{i} = combo_code(1);
    end
end

%Region definitions
regions = struct();

regions(1).name   = 'Alaska';
regions(1).latlim = [60 71.5];
regions(1).lonlim = [-170 -140];

regions(2).name   = 'Russia';
regions(2).latlim = [60 80];
regions(2).lonlim = [50 -170];

% regions(3).name   = 'Canada';
% regions(3).latlim = [50 84];
% regions(3).lonlim = [-142 -50];
% 
% regions(4).name   = 'Scandinavia-Greenland-Svalbard';
% regions(4).latlim = [54 85];
% regions(4).lonlim = [-78 40];

%Figure

for r = 1:numel(regions)
    if ~strcmp(regions(r).name, 'Russia')
        figure('Units','normalized','OuterPosition',[0 0 .5 .7], 'Color','w');
    else
        figure('Units','normalized','OuterPosition',[0 0 1 .7], 'Color','w');
    end
    % Use a standard projection that respects MapLatLimit and MapLonLimit
    axesm('MapProjection', 'eqdconicstd',...
        'MapLatLimit', regions(r).latlim, ...
        'MapLonLimit', regions(r).lonlim, ...        
        'Frame', 'on', ...
        'Grid', 'on', ...
        'MeridianLabel', 'of', ...
        'ParallelLabel', 'on', ...
        'GLineStyle', ':', ...
        'GColor', [0.8 0.8 0.8], ...
        'FLineWidth', 0.5, ...
        'FFaceColor', "#92c9ff");

    tightmap
    setm(gca, 'MLabelParallel', 'south');

    % Land
    geoshow(land, 'FaceColor', [1 1 1], 'EdgeColor', [0 0 0]);
    hold on

    % Permafrost zonation: EXACT SAME WORKING LOGIC
    field = fieldnames(categories.permafrost)';
    for i = 1:numel(field)
        category = categories.permafrost.(field{i});
        idx = strcmp(permafrost_types, category);

        if any(idx)
            switch i
                case 1
                    thisAlpha = 0.5;
                case 2
                    thisAlpha = 0.3;
                case 3
                    thisAlpha = 0.2;
                case 4
                    thisAlpha = 0.1;
            end

            geoshow([lat{idx}], [lon{idx}], 'DisplayType', 'polygon', ...
                'FaceColor', '#bcf6e8', ...
                'EdgeColor', 'none', ...
                'FaceAlpha', thisAlpha);
        end
    end

    % Subset points to the region
    inLat = data.lat >= regions(r).latlim(1) & data.lat <= regions(r).latlim(2);

    lon1 = regions(r).lonlim(1);
    lon2 = regions(r).lonlim(2);

    if lon1 <= lon2
        inLon = data.lon >= lon1 & data.lon <= lon2;
    else
        % for dateline-crossing cases if needed later
        inLon = data.lon >= lon1 | data.lon <= lon2;
    end

    dataReg = data(inLat & inLon, :);

    % Plot class points
    for ii = 1:numel(fullClassOrder)
        tmp = dataReg(dataReg.full_class == fullClassOrder(ii), :);

        if isempty(tmp)
            continue
        end

        if ~strcmp(char(fullClassOrder(ii)), 'No Trend')
            geoshow(tmp.lat, tmp.lon, 'DisplayType', 'point', ...
                'Marker', mk{ii}, ...
                'MarkerEdgeColor', 'w', ...
                'MarkerFaceColor', rgbList(ii,:), ...
                'MarkerSize', 10);
        else
            geoshow(tmp.lat, tmp.lon, 'DisplayType', 'point', ...
                'Marker', mk{ii}, ...
                'MarkerEdgeColor', 'k', ...
                'MarkerFaceColor', rgbList(ii,:), ...
                'MarkerSize', 10);
        end
    end

    % exportgraphics(gca,['G:\My Drive\UND\Index\Index_ResultsMap_' regions(r).name '.pdf'],'Resolution',300);

end

% %Shared legend with dummy handles
% legendAx = axes('Position',[0 0 1 1], 'Visible','off');
% hold(legendAx,'on')
% 
% legendHandles = gobjects(numel(fullClassOrder),1);
% for ii = 1:numel(fullClassOrder)
%     if ~strcmp(char(fullClassOrder(ii)), 'No Trend')
%         edgeCol = 'w';
%     else
%         edgeCol = 'k';
%     end
% 
%     legendHandles(ii) = plot(legendAx, nan, nan, ...
%         'LineStyle', 'none', ...
%         'Marker', mk{ii}, ...
%         'MarkerSize', 9, ...
%         'MarkerFaceColor', rgbList(ii,:), ...
%         'MarkerEdgeColor', edgeCol);
% end

% lgd = legend(legendAx, legendHandles, cellstr(fullClassOrder), ...
%     'Location', 'eastoutside', ...
%     'Interpreter', 'latex', ...
%     'FontSize', 14);
% lgd.AutoUpdate = 'off';
% lgd.PlotChildren = lgd.PlotChildren([2:6, 1]); % Reorder handles internally
% lgd.Title.String = '\,Active Layer Dynamics\,\,';



%% ** FIGURE 3: Slopes vs Regional
close('all')
clear
clc
%Load data
data = readtable('G:\My Drive\UND\Index\aldi_v4_results_wMainRegion.csv');
data.MainRegion = categorical(data.MainRegion);

figure('units','normalized','OuterPosition',[0           0        1         1]);
t = tiledlayout('flow','TileSpacing','compact','Padding','compact');

region_names = categories(data.MainRegion);
region_labels = categories(data.MainRegion);
n_regions = length(region_names);

% Calculate statistics per region
region_stats = struct();
for r = 1:n_regions
    idx = find(region_names(r)==data.MainRegion);
    region_stats(r).n = numel(idx);
    region_stats(r).mean_slope = mean(data.trend_sen_slope(idx));
    region_stats(r).se_slope = std(data.trend_sen_slope(idx)) / sqrt(numel(idx));
    region_stats(r).pct_thick = 100 * sum(strcmp(data.class_classification(idx), 'rapid_thickening') | ...
                                          strcmp(data.class_classification(idx), 'gradual_thickening')) / numel(idx);
    region_stats(r).pct_thin = 100 * sum(strcmp(data.class_classification(idx), 'rapid_thinning') | ...
                                         strcmp(data.class_classification(idx), 'gradual_thinning')) / numel(idx);    
end

% Mean Sen slope by region
nexttile,
mean_slopes = [region_stats.mean_slope];
se_slopes = [region_stats.se_slope];

bar_colors = repmat([0.8, 0.8, 0.8], n_regions, 1);

b = bar(mean_slopes);
b.FaceColor = 'flat';
b.CData = bar_colors;
hold on;
errorbar(1:n_regions, mean_slopes, se_slopes, 'k.', 'LineWidth', 1.2, 'CapSize', 6);
yline(0, 'k--', 'LineWidth', 1);

set(gca, 'XTickLabel', region_labels, 'FontSize', 20, 'TickLabelInterpreter','latex', XGrid='off', YGrid='on', YMinorGrid='on', YMinorTick='on');
xtickangle(0);
ylabel('Mean Sen slope (cm yr$^{-1}$)', 'FontSize', 20, interpreter='latex');
% ylim([-0.8, 1.8]);
box on;

% title('(a) Regional trend magnitude', 'FontSize', 11, 'FontWeight', 'normal');

% add scatter plot in the same plot per region and colored per class classification 
% add scatter plot in the same plot per region and colored per class classification 
hold on;
% Define class colors (consistent with later figure)
class_order = {'rapid_thickening', 'gradual_thickening', 'no_trend', ...
               'gradual_thinning', 'rapid_thinning', 'transitional'};
class_colors = containers.Map( ...
    class_order, ...
    {[0.84, 0.15, 0.16], [1.0, 0.50, 0.05], [0.6,0.6,0.6], ...
     [0.47,0.67,0.19], [0.12,0.47,0.71], [0.58,0.40,0.74]});
markerSizes = 50; % base marker size

% jitter x for visibility
rng(0); % reproducible jitter
jitterAmt = 0.12;

for r = 1:n_regions
    fprintf('- %s:\n',region_names{r})
    idx_region = find(region_names(r) == data.MainRegion);
    x = r * ones(numel(idx_region),1) + (rand(numel(idx_region),1)-0.5)*2*jitterAmt;
    y = data.trend_sen_slope(idx_region);
    classes_here = data.class_classification(idx_region);
    for iii = 1:length(y)
        if y(iii)>5.7 || y(iii)<-3 || (y(iii)==1.25) || (y(iii)==-1.25 && r==4) || (round(y(iii),2)==-0.22 && r==6) || (round(y(iii),2)==0.71 && r==2)
            fprintf('  + Found site (lat=%.5f lon=%.5f, %s) with slope = %.3f\n', data.lat(idx_region(iii)), data.lon(idx_region(iii)), region_names{r}, y(iii))
            % permadata = assignEnvironmentalCategories(data.lat(idx_region(iii)), data.lon(idx_region(iii)));    %uncomment this and the next line if you want to recompute permafrost params        
            % fprintf('  + Found site (lat=%.5f lon=%.5f, %s) with slope = %.3f [%s, %s, %s]\n', data.lat(idx_region(iii)), data.lon(idx_region(iii)), region_names{r}, y(iii), permadata.permafrostType{1}, permadata.groundIceType{1}, permadata.landformType{1})
        end
    end    
    for c = 1:length(class_order)
        % fprintf('  + %s:\n',class_order{c})
        match = strcmp(classes_here, class_order{c});
        if any(match)
            bb = scatter(x(match), y(match), markerSizes, 'MarkerFaceColor', class_colors(class_order{c}), ...
                'MarkerEdgeColor','white','MarkerFaceAlpha',0.85);
            if c==6
                bb.Marker = 'diamond';
            end
        end
    end
end

% Create custom legend for classes
h_legend = gobjects(length(class_order),1);
for c = 1:length(class_order)
    h_legend(c) = scatter(NaN, NaN, markerSizes, 'MarkerFaceColor', class_colors(class_order{c}), ...
        'MarkerEdgeColor','k');
end
lgd = legend(h_legend, {'Rapid thickening','Gradual thickening\,\,\,','No trend','Gradual thinning','Rapid thinning','Transitional'}, ...
    'Location','northeast','FontSize',20, interpreter='latex');
lgd.Title.String = '\,Active Layer Dynamics\,';
lgd.Title.Interpreter = 'latex';

set(gca, 'XLim', [0.5285 6.6437], 'YLim', [-3.4    9.0905])

%first
annotation(gcf,'textarrow', [0.140756302521008 0.128151260504202],...
    [0.898321816386969 0.932872655478776],...
'String', {'(64.71444^{\circ}, -148.14528^{\circ})', '[Discontinuous permafrost with low ground ice content]'},...
'Interpreter','tex',...
'FontSize',13, 'Units', 'normalized');

%second
annotation(gcf,'textarrow', [0.224789915966386 0.268382352941176],...
    [0.109575518262586 0.0631786771964462],...
'String', {'(73.0054^{\circ}, -80.68725^{\circ})', '[Continuous permafrost with',' high ground ice content]'},...
'Interpreter','tex',...
'FontSize',13, 'Units', 'normalized');

%five
annotation(gcf,'textarrow',[0.570395658263305 0.554114145658261],...
    [0.161422405768837 0.198934745354227],...
    'String', {'(68.74^{\circ}, 161.3928^{\circ})','[Continuous permafrost with high ground ice content]'},...
'Interpreter','tex',...
'FontSize',13, 'Units', 'normalized');


%four
annotation(gcf,'textarrow', [0.526260504201681 0.545181197478992],...
    [0.418558736426455 0.392870260517452],...
'String', {'(70.28333^{\circ}, 68.9^{\circ})','[Continuous permafrost with','medium ground ice content]'},...
'Interpreter','tex',...
'FontSize',13, 'Units', 'normalized');


%third
annotation(gcf,'textarrow', [0.680142682072827 0.701676295518205],...
    [0.764107627348041 0.724633787269069],...
'String', {'(62.2958^{\circ}, 9.352^{\circ})','[Discontinuous permafrost with low ground ice content]'},...
'Interpreter','tex',...
'FontSize',13, 'Units', 'normalized');

%six
annotation(gcf,'textarrow',[0.802521008403361 0.846638655462185],...
    [0.249753208292201 0.275419545903257],...
    'String', {'(77.548^{\circ}, 14.473^{\circ})','[Continuous permafrost with','medium ground ice content]'},...
    'Interpreter','tex', 'FontSize',13, 'Units', 'normalized');

%seven
annotation(gcf,'textarrow',[0.279936974789916 0.279936974789915],...
    [0.512339585389929 0.356367226061202],...
    'String', {'(62.69667^{\circ}, -123.065^{\circ})','[Discontinuous permafrost with low ground ice content]'},...
    'Interpreter','tex', 'FontSize',13, 'Units', 'normalized');
hold off;


exportgraphics(gca,'G:\My Drive\UND\Index\Index_BarPlot_slopesVsRegions.pdf', 'Resolution', 300) % Use vector for best quality

%% ** Representative ALT time series by index class
% Updates:
%   - Transitional panels now plot EARLY and RECENT Sen trends (trajectory component),
%     using the same fixed 7-year (or short-span) window logic as your Python.
%   - Overall long-term Theil–Sen trend is always plotted.
%   - Legend explains plotted elements; for Transitional it includes early/recent trends.

set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

close('all'); clear; clc;

summary_fp = 'G:\My Drive\UND\Index\aldi_v4_results_wMainRegion.csv';
raw_fp     = 'G:\My Drive\UND\Index\CALM_ALT_all_Max.csv';

%%---- Load summary table ----
data = readtable(summary_fp);
fprintf('loaded summary data\n')
%%---- Parse class labels into class / rate / full_class ----
n_sites = height(data);
data.class      = strings(n_sites, 1);
data.rate       = strings(n_sites, 1);
data.full_class = strings(n_sites, 1);

fprintf('-- Processing individual sites'' data:\n')
for i = 1:n_sites
    clf = string(data.class_classification{i});
    fprintf('-- site %d\n', i)

    switch clf
        case 'rapid_thickening'
            data.class(i) = "Thickening";   data.rate(i) = "Rapid";   data.full_class(i) = "Rapid Thickening";
        case 'gradual_thickening'
            data.class(i) = "Thickening";   data.rate(i) = "Gradual"; data.full_class(i) = "Gradual Thickening";
        case 'no_trend'
            data.class(i) = "No Trend";     data.rate(i) = "";        data.full_class(i) = "No Trend";
        case 'gradual_thinning'
            data.class(i) = "Thinning";     data.rate(i) = "Gradual"; data.full_class(i) = "Gradual Thinning";
        case 'rapid_thinning'
            data.class(i) = "Thinning";     data.rate(i) = "Rapid";   data.full_class(i) = "Rapid Thinning";
        case 'transitional'
            data.class(i) = "Transitional"; data.rate(i) = "";        data.full_class(i) = "Transitional";
        otherwise
            data.class(i) = "No Trend";     data.rate(i) = "";        data.full_class(i) = "No Trend";
    end
end

fullClassOrder = ["Rapid Thickening", "Gradual Thickening", "No Trend", ...
                  "Gradual Thinning", "Rapid Thinning", "Transitional"];
data.full_class = categorical(data.full_class, fullClassOrder);

data.region = categorical(strtrim(string(data.MainRegion)));


%%---- Load raw time series ----
fprintf('-- Loading raw ALT time series:\n')
% Load the data
raw_fp = 'G:\My Drive\UND\Index\CALM_ALT_all_Max.csv';
[df,~,~,~] = preprocess_CALM(raw_fp, 0);

% ---- 9) PCHIP fill small gaps per site (max gap size = 2 years) ----
max_gap_fill = 2;

sites = unique(df.site_id);
out = cell(numel(sites),1);

for k = 1:numel(sites)
    S = df(df.site_id == sites(k), :);
    S = sortrows(S, 'Year');

    yrs = S.Year;
    vals = S.Max;

    % Build complete yearly axis inside [min,max]
    y0 = min(yrs);
    y1 = max(yrs);
    yrs_complete = (y0:y1)';
    if numel(yrs)<numel(yrs_complete)
        fprintf('Site %d has missing %d years\n', k, numel(yrs_complete)-numel(yrs))
    end
    % Identify which missing years are in fillable gaps (gap_size <= max_gap_fill)
    fillable_missing = false(size(yrs_complete));
    for i = 1:(numel(yrs)-1)
        gap_size = yrs(i+1) - yrs(i) - 1;
        if gap_size > 0 && gap_size <= max_gap_fill
            gap_years = (yrs(i)+1):(yrs(i+1)-1);
            fillable_missing(ismember(yrs_complete, gap_years)) = true;
        end
    end

    % Start with observed values placed on complete axis
    vals_complete = nan(size(yrs_complete));
    [tfObs, locObs] = ismember(yrs_complete, yrs);
    vals_complete(tfObs) = vals(locObs(tfObs));

    % Apply PCHIP only on fillable missing years
    idx_fill = fillable_missing & ~tfObs;
    if any(idx_fill) && numel(yrs) >= 2
        vals_complete(idx_fill) = pchip(yrs, vals, yrs_complete(idx_fill));
        was_interp = idx_fill;
    else
        was_interp = false(size(yrs_complete));
    end

    % Keep only observed years + filled years (drop non-fillable missing years)
    keep = tfObs | was_interp;
    yrs_f = yrs_complete(keep);
    vals_f = vals_complete(keep);
    was_interp = was_interp(keep);

    % Censor flag: only meaningful for observed points; interpolated are false
    cens_f = false(size(yrs_f));
    [tf2, loc2] = ismember(yrs_f, yrs);
    cens_f(tf2) = S.is_censored_high(loc2(tf2));

    % Constant lat/lon for site
    Lat0 = S.Lat(1);
    Lon0 = S.Long(1);

    out{k} = table( ...
        repmat(sites(k), numel(yrs_f), 1), ...
        yrs_f, vals_f, ...
        repmat(Lat0, numel(yrs_f), 1), ...
        repmat(Lon0, numel(yrs_f), 1), ...
        cens_f, was_interp, ...
        'VariableNames', {'site_id','Year','Max','Lat','Long','is_censored_high','was_interpolated'});
end

df_clean = vertcat(out{:});
df_clean = sortrows(df_clean, {'site_id','Year'});
% df_clean is now the cleaned + deduped + (small-gap) PCHIP-filled dataset
raw_data = df_clean;
clear df_clean

%%---- Choose 3 representative sites per class (smallest MK p-values within class) ----
MIN_YEARS        = 8;
MIN_COMPLETENESS = 0.0;
MAX_INTERP_FRAC  = 1.0;

rep_sites = strings(numel(fullClassOrder), 3);

fprintf('-- Finding the best examples per category:\n')
for k = 1:numel(fullClassOrder)
    cls = fullClassOrder(k);
    fprintf('-- class %s\n', cls)
    rows = data(data.full_class == cls, :);
    if isempty(rows), continue; end

    rows = rows(rows.trend_n_years >= MIN_YEARS, :);
    rows = rows(rows.completeness >= MIN_COMPLETENESS, :);

    interp_frac = rows.n_interpolated ./ max(rows.n_after_gapfill, 1);
    rows = rows(interp_frac <= MAX_INTERP_FRAC, :);

    if isempty(rows), continue; end

    pvals = rows.trend_mk_pvalue;
    [~, j] = mink(pvals, min(2, numel(pvals)));
    rep_sites(k,1:numel(j)) = string(rows.site_id(j))';
end

%%---- Plot: 6 figures (one per class), each has 3 panels (3 sites) ----
fprintf('-- Plotting:\n')
for k = 1:6
    fig = figure('units','normalized','OuterPosition',[0 0 .5 .5], 'Color','w');
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    for kk = 1:2
        nexttile;

        if rep_sites(k,kk) == ""
            axis off;
            text(0.5,0.5, sprintf('No site\n%s', string(fullClassOrder(k))), ...
                'HorizontalAlignment','center', 'FontSize', 12, 'Interpreter','latex');
        else
            panelLetter = char('a' + (kk-1));
            localPlotSite(rep_sites(k,kk), panelLetter, data, raw_data);
        end

        if kk > 1
            ylabel(gca,'');
        end
    end
    exportgraphics(gcf,strcat('G:\My Drive\UND\Index\Index_temporalPlots_',fullClassOrder(k),'.pdf'), 'Resolution', 300) % Use vector for best quality
end
% close('all')

%% ** Add additional static features (permafrost related)
clear; clc; close all;

%Load Data
dataTargets = readtable('G:\My Drive\UND\Index\aldi_v4_results_wMainRegion.csv');
latitudes = dataTargets.lat;
longitudes = dataTargets.lon;

% Permafrost related features
permadata = assignEnvironmentalCategories(latitudes, longitudes);
[~, ia] = unique([permadata.latitudes, permadata.longitudes], 'rows');
permadata_unique = permadata(ia, :);
augmentedData = outerjoin(dataTargets, permadata_unique, ...
    Type="left", ...
    LeftKeys=["lat","lon"], ...
    RightKeys=["latitudes","longitudes"], ...
    MergeKeys=true); 
augmentedData = renamevars(augmentedData, "lat_latitudes", "lat");
augmentedData = renamevars(augmentedData, "lon_longitudes", "lon");
augmentedData(:,{'site_id','n_observations','n_after_gapfill', 'n_interpolated', 'gap_fill_method', 'span_years', 'completeness', 'max_gap_spacing', 'year_min', 'year_max', 'mean_alt', 'median_alt', 'std_alt'})=[]; %remove non useful features
augmentedData = augmentedData(:,{'lat','lon','permafrostType','groundIceType','landformType','class_classification','class_confidence'});
% writetable(augmentedData, 'G:\My Drive\UND\Index\aldi_v4_mainResults_wMainRegion_wPermafrost.csv');


% 
% Load yearly features
filePath = 'G:\My Drive\UND\Index\ALDI_causal_features_complete.csv';
opts = detectImportOptions(filePath);
yearFeatures = readtable(filePath, opts);
yearFeatures = movevars(yearFeatures, "lat", "Before", "FDD");
yearFeatures = movevars(yearFeatures, "lon", "after", "lat");
yearFeatures = movevars(yearFeatures, "year", "after", "lon");

joinedData = innerjoin(yearFeatures,augmentedData,Keys=["lat","lon"]);

filePath = 'G:\My Drive\UND\Index\CALM_ALT_all_Max.csv';
[result_table,region_counts,alt_by_region,cov_by_region] = preprocess_CALM(filePath, 1);


joinedData2 = innerjoin(joinedData,result_table,LeftKeys=["lat","lon", "year"],RightKeys=["Lat","Long", "Year"]);
joinedData2.site_id_result_table = [];
joinedData2.Properties.VariableNames(contains(joinedData2.Properties.VariableNames,'site_id')) = {'site_id'};


joinedData2 = movevars(joinedData2, "Max", 'Before', 'class_classification');
% writetable(joinedData2, 'G:\My Drive\UND\Index\ALDI_withYearlyData_augmented.csv');


% %add permafrost static features to the yearly data
% joinedData = innerjoin(yearFeatures,augmentedData,Keys=["lat","lon"]);
% joinedData = movevars(joinedData, "Max", 'Before', 'class_classification');

% % add additional (static) features
% filePath = 'G:\My Drive\UND\Index\alti_FINAL_additional_features.csv';
% opts = detectImportOptions(filePath);
% addFeatures = readtable(filePath, opts);
% addFeatures(:,{'site_id'})=[]; %remove non useful features

% joinedData2 = innerjoin(joinedData,addFeatures,Keys=["lat","lon"]);
% joinedData2 = movevars(joinedData2, "class_classification");
% joinedData2 = movevars(joinedData2, "class_confidence");
% joinedData2 = movevars(joinedData2, "Max");
% writetable(joinedData2, 'G:\My Drive\UND\Index\ALDI_withYearlyData_augmented.csv');

unq_class = unique(joinedData2.class_classification);
for i = 1:length(unq_class)
    writetable(joinedData2(strcmp(joinedData2.class_classification,unq_class(i)),:), ['G:\My Drive\UND\Index\ALDI_withYearlyData_augmented_' unq_class{i}, '.csv']);
end


%% ** split into permafrost zonation sets
clear
clc
filePath = 'G:\My Drive\UND\Index\ALDI_withYearlyData_augmented.csv';
opts = detectImportOptions(filePath);
data = readtable(filePath, opts);
unq_permafrost = string(unique(data.permafrostType));
for i = 1:length(unq_permafrost)
    writetable(data(strcmp(data.permafrostType,unq_permafrost{i}),:), ['G:\My Drive\UND\Index\ALDI_withYearlyData_augmented_' unq_permafrost{i}, 'Permafrost.csv']);
end


%% ** Vertical forest-style plot by class
% y-axis  : standardized effect (or raw effect if plotMode='raw')
% x-axis  : treatments
% CI      : vertical confidence intervals
% Method  : FE filled black, FD open black
% Sig     : shape (* circle, ** square, *** diamond)

clear; clc; close all;

%------------------------------------------------------------------------
% Raw TSV data
% -------------------------------------------------------------------------
txt = strjoin([
"Class	Treatment	Method	Effect	SE	Effect_Std	SD_Treatment	CI_Lower	CI_Upper	p_value	N	Sites	Sig"
"All	TDD	fe_cluster	0.022686418	0.004158091	3.083234702	135.9066365	0.014536709	0.030836126	4.87E-08	1704	97	***"
"All	TDD	first_diff	0.013159777	0.003689408	2.381002657	180.9303156	0.00592867	0.020390884	0.000361219	1704	97	***"
"All	FDD	fe_cluster	-0.004005637	0.001106695	-1.539502703	384.3340377	-0.006174719	-0.001836556	2.95E-04	1704	97	***"
"All	FDD	first_diff	-0.001629597	0.00108264	-0.800321189	491.1158745	-0.003751533	0.000492338	1.32E-01	1704	97	"
"All	SWE$_{max}$	fe_cluster	6.919276981	8.773614326	0.307900353	0.04449892	-10.27669111	24.11524507	4.30E-01	1704	97	"
"All	SWE$_{max}$	first_diff	11.96637326	7.997799241	0.730874367	0.06107735	-3.709025207	27.64177173	1.35E-01	1704	97	"
"All	SDO	fe_cluster	-0.093888404	0.084058517	-0.579021891	6.167128889	-0.25864007	0.070863261	2.64E-01	1704	97	"
"All	SDO	first_diff	-0.05810803	0.057333783	-0.478479837	8.234315306	-0.17048018	0.054264121	3.11E-01	1704	97	"
"All	P$_{summer}$	fe_cluster	0.032074018	0.01436959	1.382853534	43.11444637	0.00391014	0.060237896	2.56E-02	1704	97	*"
"All	P$_{summer}$	first_diff	0.027268644	0.008428796	1.722862604	63.18108742	0.010748508	0.04378878	1.22E-03	1704	97	**"
"All	SM	fe_cluster	-54.87701307	24.37260217	-1.146106122	0.020884995	-102.6464355	-7.107590601	2.43E-02	1594	89	*"
"All	SM	first_diff	-40.04719859	23.54350083	-0.891026639	0.022249412	-86.19161228	6.097215106	8.89E-02	1594	89	"
"All	NDWI	fe_cluster	1.01128781	6.678326572	0.08495656	0.084008291	-12.07799175	14.10056737	8.80E-01	2064	119	"
"All	NDWI	first_diff	-6.423467125	2.826942569	-0.741611302	0.115453428	-11.96417275	-0.882761504	2.31E-02	2064	119	*"
"All	FireDays	fe_cluster	-2.501999623	5.024652765	-0.106881363	0.042718377	-12.35013808	7.346138832	6.19E-01	1254	74	"
"All	FireDays	first_diff	1.688654928	1.240925815	0.106029114	0.062789095	-0.743514978	4.120824833	1.74E-01	1254	74	"
"All	HWD	fe_cluster	0.192851528	0.093305685	1.029401877	5.337794762	0.009975746	0.375727311	3.87E-02	1704	97	*"
"All	HWD	first_diff	0.023681066	0.06905384	0.187491265	7.91734915	-0.111661974	0.159024105	7.32E-01	1704	97	"
"All	NDVI$_{summer}$	fe_cluster	58.55990157	17.55787696	2.28454245	0.039012061	24.14709509	92.97270805	8.52E-04	1201	71	***"
"All	NDVI$_{summer}$	first_diff	28.85877262	11.81653268	1.551237842	0.053752731	5.698794136	52.0187511	1.46E-02	1201	71	*"
"All	ROS	fe_cluster	-0.565966011	0.400638579	-0.875708337	1.547280792	-1.351203197	0.219271176	1.58E-01	1704	97	"
"All	ROS	first_diff	-0.683528111	0.324723256	-1.496325402	2.189120504	-1.319973997	-0.047082224	3.53E-02	1704	97	*"
"Rapid Thickening	TDD	fe_cluster	0.046242806	0.013335689	6.795501227	146.9526134	0.020105335	0.072380277	0.000525133	252	18	***"
"Rapid Thickening	TDD	first_diff	0.011637564	0.010861259	2.42563947	208.4318836	-0.009650112	0.032925241	0.283956033	252	18	"
"Rapid Thickening	FDD	fe_cluster	-0.008010009	0.003209805	-3.252484663	406.0525636	-0.014301111	-0.001718906	0.012578636	252	18	*"
"Rapid Thickening	FDD	first_diff	0.000943074	0.002271617	0.522831818	554.3912599	-0.003509214	0.005395362	0.678028296	252	18	"
"Rapid Thickening	SWE$_{max}$	fe_cluster	12.54440574	12.55488063	0.899050251	0.071669417	-12.06270812	37.15151961	0.317714442	252	18	"
"Rapid Thickening	SWE$_{max}$	first_diff	5.636801428	12.25744886	0.563479383	0.099964384	-18.38735688	29.66095973	0.645611384	252	18	"
"Rapid Thickening	SDO	fe_cluster	-0.286938088	0.36918317	-1.924411137	6.706712061	-1.010523804	0.436647628	0.437026516	252	18	"
"Rapid Thickening	SDO	first_diff	-0.1033839	0.200159516	-0.952348979	9.211772635	-0.495689342	0.288921542	0.605499971	252	18	"
"Rapid Thickening	P$_{summer}$	fe_cluster	0.092347836	0.077245245	4.241113273	45.92542117	-0.059050062	0.243745734	0.231885944	252	18	"
"Rapid Thickening	P$_{summer}$	first_diff	0.034673666	0.014590613	2.390168273	68.93324454	0.006076589	0.063270742	0.017480772	252	18	*"
"Rapid Thickening	SM	fe_cluster	-216.9817064	198.5238393	-2.975273794	0.013712095	-606.0812815	172.1178687	0.274404526	219	15	"
"Rapid Thickening	SM	first_diff	-144.8665339	100.1056656	-2.557577745	0.017654718	-341.0700331	51.33696537	0.147858777	219	15	"
"Rapid Thickening	NDWI	fe_cluster	38.16517312	23.95467105	3.463981332	0.090762888	-8.785119398	85.11546563	0.111109797	297	20	"
"Rapid Thickening	NDWI	first_diff	-17.21200323	9.252121056	-2.236474162	0.129936889	-35.34582728	0.92182082	0.062838777	297	20	"
"Rapid Thickening	FireDays	fe_cluster	-27.35624225	2.798138467	-1.050256477	0.03839184	-32.84049287	-21.87199163	1.42E-22	174	11	***"
"Rapid Thickening	FireDays	first_diff	-7.668385703	5.4338235	-0.310116378	0.040440895	-18.31848406	2.981712654	0.158176212	174	11	"
"Rapid Thickening	HWD	fe_cluster	0.610790162	0.192247404	3.982660556	6.520505415	0.233992173	0.98758815	0.001487532	252	18	**"
"Rapid Thickening	HWD	first_diff	0.156319488	0.182246662	1.546385839	9.892469968	-0.200877405	0.513516381	0.391038362	252	18	"
"Rapid Thickening	NDVI$_{summer}$	fe_cluster	244.7593211	129.4622927	7.684129589	0.031394635	-8.982109906	498.5007521	0.058679916	161	10	"
"Rapid Thickening	NDVI$_{summer}$	first_diff	56.41286684	63.26017449	2.387977276	0.042330366	-67.57479681	180.4005305	0.372521809	161	10	"
"Rapid Thickening	ROS	fe_cluster	-2.016354282	0.76632003	-4.257683893	2.111575297	-3.518313942	-0.514394622	0.00850797	252	18	**"
"Rapid Thickening	ROS	first_diff	-1.094831308	0.823255673	-3.455907965	3.156566622	-2.708382777	0.51872016	0.18355779	252	18	"
"Gradual Thickening	TDD	fe_cluster	0.037860451	0.009034311	4.849423349	128.0867833	0.020153527	0.055567375	2.78E-05	425	21	***"
"Gradual Thickening	TDD	first_diff	0.028425916	0.007357673	4.773574103	167.930354	0.014005142	0.042846689	0.000111802	425	21	***"
"Gradual Thickening	FDD	fe_cluster	-0.007799896	0.003211236	-2.693062693	345.2690527	-0.014093802	-0.00150599	0.015143073	425	21	*"
"Gradual Thickening	FDD	first_diff	-0.005979002	0.003446817	-2.639700747	441.49521	-0.012734639	0.000776635	0.082803782	425	21	"
"Gradual Thickening	SWE$_{max}$	fe_cluster	3.475282608	19.20807815	0.15010989	0.043193578	-34.17185878	41.122424	0.85642395	425	21	"
"Gradual Thickening	SWE$_{max}$	first_diff	2.877498818	11.61594776	0.175053211	0.060835198	-19.88934043	25.64433806	0.804351307	425	21	"
"Gradual Thickening	SDO	fe_cluster	-0.212204905	0.099142726	-1.140498056	5.374513176	-0.406521076	-0.017888733	0.032322606	425	21	*"
"Gradual Thickening	SDO	first_diff	-0.266291249	0.099596734	-1.867318453	7.012316254	-0.461497261	-0.071085237	0.007502074	425	21	**"
"Gradual Thickening	P$_{summer}$	fe_cluster	0.011891957	0.01553479	0.536421675	45.10793942	-0.018555673	0.042339587	0.443970996	425	21	"
"Gradual Thickening	P$_{summer}$	first_diff	0.012046672	0.025927451	0.746119902	61.93576991	-0.038770198	0.062863542	0.642196433	425	21	"
"Gradual Thickening	SM	fe_cluster	-92.06223066	34.18134075	-1.785060284	0.019389714	-159.0564275	-25.06803385	0.007073849	408	20	**"
"Gradual Thickening	SM	first_diff	-67.45664695	64.65272772	-1.365241223	0.020238795	-194.1736648	59.26037089	0.296777461	408	20	"
"Gradual Thickening	NDWI	fe_cluster	-2.636158791	11.01665191	-0.231818312	0.087937917	-24.22839977	18.95608219	0.810881834	689	34	"
"Gradual Thickening	NDWI	first_diff	-7.505003147	4.098703996	-0.945365866	0.125964753	-15.53831536	0.52830907	0.067090488	689	34	"
"Gradual Thickening	FireDays	fe_cluster	1.13E-15	1.99E-17		0	1.09E-15	1.17E-15	0	329	18	***"
"Gradual Thickening	FireDays	first_diff	-1.67E-16	1.08E-15		0	-2.29E-15	1.96E-15	0.877514679	329	18	"
"Gradual Thickening	HWD	fe_cluster	0.524577314	0.224769696	1.969582422	3.754608464	0.084036804	0.965117823	0.019603914	425	21	*"
"Gradual Thickening	HWD	first_diff	-0.001051074	0.219342626	-0.005619988	5.346899869	-0.430954721	0.428852573	0.996176609	425	21	"
"Gradual Thickening	NDVI$_{summer}$	fe_cluster	46.27820516	24.31088186	2.101974765	0.045420404	-1.37024773	93.92665804	0.056962243	312	17	"
"Gradual Thickening	NDVI$_{summer}$	first_diff	25.7059805	21.04644572	1.71835607	0.066846548	-15.54429511	66.95625612	0.221937274	312	17	"
"Gradual Thickening	ROS	fe_cluster	0.736173733	0.540603173	1.176971604	1.598768811	-0.323389016	1.795736483	0.173272497	425	21	"
"Gradual Thickening	ROS	first_diff	-0.040596248	0.410754934	-0.089893821	2.21433817	-0.845661126	0.76446863	0.921270663	425	21	"
"No Trend	TDD	fe_cluster	0.009729289	0.002836015	1.346668305	138.4138516	0.004170802	0.015287775	0.000602204	791	41	***"
"No Trend	TDD	first_diff	0.007213038	0.004522761	1.308826351	181.4528586	-0.001651411	0.016077487	0.110750111	791	41	"
"No Trend	FDD	fe_cluster	-0.001483161	0.000816004	-0.584960107	394.400877	-0.0030825	0.000116177	0.069126754	791	41	"
"No Trend	FDD	first_diff	-0.000339046	0.000947939	-0.164228255	484.3838007	-0.002196972	0.00151888	0.720593084	791	41	"
"No Trend	SWE$_{max}$	fe_cluster	1.791559276	13.63000438	0.064167759	0.03581671	-24.92275842	28.50587697	0.89542542	791	41	"
"No Trend	SWE$_{max}$	first_diff	25.97210544	12.07270846	1.241172123	0.04778866	2.31003165	49.63417922	0.031451955	791	41	*"
"No Trend	SDO	fe_cluster	-0.004875287	0.058653111	-0.032256227	6.61627295	-0.119833271	0.110082698	0.93375558	791	41	"
"No Trend	SDO	first_diff	-0.046390733	0.066502013	-0.409947443	8.836838991	-0.176732283	0.083950816	0.485437451	791	41	"
"No Trend	P$_{summer}$	fe_cluster	0.025047169	0.008664041	1.055578507	42.14362462	0.008065961	0.042028378	0.003840989	791	41	**"
"No Trend	P$_{summer}$	first_diff	0.032957354	0.008605754	2.082881738	63.19930118	0.016090386	0.049824322	0.000128306	791	41	***"
"No Trend	SM	fe_cluster	-17.5438203	21.4839974	-0.412623615	0.023519599	-59.65168146	24.56404085	0.414157374	741	38	"
"No Trend	SM	first_diff	-22.58909859	25.27851577	-0.545539919	0.024150584	-72.13407909	26.9558819	0.371531354	741	38	"
"No Trend	NDWI	fe_cluster	-2.197347669	4.101112186	-0.160472612	0.073030142	-10.23537985	5.840684513	0.592101542	843	48	"
"No Trend	NDWI	first_diff	1.389140632	6.043836437	0.131991618	0.095016743	-10.45656111	13.23484238	0.818212853	843	48	"
"No Trend	FireDays	fe_cluster	1.209470825	0.508714063	0.069970624	0.057852262	0.212409583	2.206532066	0.017430151	608	35	*"
"No Trend	FireDays	first_diff	2.394190805	0.441365764	0.209550109	0.087524398	1.529129804	3.259251806	5.81E-08	608	35	***"
"No Trend	HWD	fe_cluster	-0.019506911	0.065995837	-0.111031062	5.691883236	-0.148856375	0.109842552	0.767552448	791	41	"
"No Trend	HWD	first_diff	-0.032882755	0.070828542	-0.277750806	8.446701151	-0.171704146	0.105938635	0.642462526	791	41	"
"No Trend	NDVI$_{summer}$	fe_cluster	36.20598833	13.70482887	1.40808852	0.038891039	9.34501733	63.06695933	0.008245659	585	34	**"
"No Trend	NDVI$_{summer}$	first_diff	29.7173982	11.96794707	1.549523107	0.052141951	6.260652972	53.17414343	0.013025097	585	34	*"
"No Trend	ROS	fe_cluster	-0.150523153	0.46179401	-0.186513774	1.239103555	-1.055622782	0.754576475	0.744459898	791	41	"
"No Trend	ROS	first_diff	-0.555044978	0.394964849	-0.95269479	1.716428086	-1.329161858	0.219071902	0.15993143	791	41	"
"Gradual Thinning	TDD	fe_cluster	-0.002985374	0.010376603	-0.362895714	121.5578712	-0.023323141	0.017352393	0.773574508	151	12	"
"Gradual Thinning	TDD	first_diff	0.008885139	0.005465728	1.453707445	163.611114	-0.001827491	0.019597768	0.104032769	151	12	"
"Gradual Thinning	FDD	fe_cluster	0.001254181	0.005195475	0.430525486	343.2723006	-0.008928763	0.011437125	0.809246158	151	12	"
"Gradual Thinning	FDD	first_diff	-0.00226439	0.004259538	-1.078726843	476.3874943	-0.01061293	0.006084151	0.594999915	151	12	"
"Gradual Thinning	SWE$_{max}$	fe_cluster	22.67314491	37.59821756	0.836133633	0.036877709	-51.01800738	96.3642972	0.546483569	151	12	"
"Gradual Thinning	SWE$_{max}$	first_diff	46.10398873	44.89659833	2.323722864	0.050401775	-41.89172703	134.0997045	0.304470996	151	12	"
"Gradual Thinning	SDO	fe_cluster	0.215886964	0.368481453	1.112666313	5.153930063	-0.506323413	0.938097341	0.557954243	151	12	"
"Gradual Thinning	SDO	first_diff	0.110080749	0.273850663	0.774931362	7.039662878	-0.426656688	0.646818186	0.687703423	151	12	"
"Gradual Thinning	P$_{summer}$	fe_cluster	0.021784986	0.031592829	0.723670691	33.2187822	-0.040135821	0.083705792	0.490474224	151	12	"
"Gradual Thinning	P$_{summer}$	first_diff	0.030959379	0.020143454	1.580675502	51.05643454	-0.008521066	0.070439824	0.124306806	151	12	"
"Gradual Thinning	SM	fe_cluster	9.765544539	51.55008918	0.182189053	0.018656313	-91.27077365	110.8018627	0.849749563	141	11	"
"Gradual Thinning	SM	first_diff	-23.31847879	29.641868	-0.53338728	0.022874017	-81.4154725	34.77851492	0.431472888	141	11	"
"Gradual Thinning	NDWI	fe_cluster	-25.71336462	5.157558266	-2.650299288	0.103070887	-35.82199307	-15.60473617	6.18E-07	153	12	***"
"Gradual Thinning	NDWI	first_diff	-6.10451782	3.900920521	-0.871082282	0.142694691	-13.75018155	1.541145908	0.117608325	153	12	"
"Gradual Thinning	FireDays	fe_cluster	-2.64E-16	2.12E-15		0	-4.42E-15	3.89E-15	0.900914918	82	6	"
"Gradual Thinning	FireDays	first_diff	-9.60E-16	5.64E-16		0	-2.06E-15	1.45E-16	0.088659439	82	6	"
"Gradual Thinning	HWD	fe_cluster	-0.193509981	0.315580789	-1.056405456	5.459178143	-0.812036961	0.425017	0.539752818	151	12	"
"Gradual Thinning	HWD	first_diff	-0.032379903	0.232999352	-0.26138683	8.072501812	-0.489050241	0.424290434	0.889473911	151	12	"
"Gradual Thinning	NDVI$_{summer}$	fe_cluster	72.66260257	44.65965271	1.941916946	0.026725122	-14.86870829	160.1939134	0.103730689	82	6	"
"Gradual Thinning	NDVI$_{summer}$	first_diff	73.40030355	21.39787357	2.442971	0.033282846	31.46124201	115.3393651	0.000603	82	6	***"
"Gradual Thinning	ROS	fe_cluster	-1.699268448	0.998228207	-3.382999266	1.990856283	-3.655759782	0.257222887	0.088702041	151	12	"
"Gradual Thinning	ROS	first_diff	-1.788735565	0.852212994	-4.941327978	2.7624698	-3.459042341	-0.118428789	0.035823057	151	12	*"
"Transitional	TDD	fe_cluster	0.03132237	0.021179728	5.377905936	171.695372	-0.010189133	0.072833873	0.1391712	56	2	"
"Transitional	TDD	first_diff	0.010620164	0.017531421	2.308385837	217.3587817	-0.02374079	0.044981117	0.54466163	56	2	"
"Transitional	FDD	fe_cluster	-0.007856299	0.004770878	-4.091502793	520.792671	-0.017207048	0.00149445	0.099615669	56	2	"
"Transitional	FDD	first_diff	-0.001602582	0.001403279	-1.01625441	634.1357017	-0.004352958	0.001147794	0.253442899	56	2	"
"Transitional	SWE$_{max}$	fe_cluster	6.089497436	34.97604025	0.180889284	0.029705125	-62.46228177	74.64127664	0.86178308	56	2	"
"Transitional	SWE$_{max}$	first_diff	-46.34207954	45.89861857	-1.682605318	0.036308369	-136.3017189	43.6175598	0.312657374	56	2	"
"Transitional	SDO	fe_cluster	-0.26330929	0.256712349	-1.817564718	6.902774738	-0.766456249	0.239837668	0.305034054	56	2	"
"Transitional	SDO	first_diff	0.118613365	0.114849455	1.016249788	8.567751116	-0.106487431	0.343714161	0.301710353	56	2	"
"Transitional	P$_{summer}$	fe_cluster	0.025986417	0.005820422	1.33157956	51.24136752	0.014578601	0.037394234	8.02E-06	56	2	***"
"Transitional	P$_{summer}$	first_diff	0.028844687	0.002807343	1.979635231	68.63084398	0.023342397	0.034346978	9.16E-25	56	2	***"
"Transitional	SM	fe_cluster	-182.9198978	316.2593902	-4.462313175	0.024394903	-802.7769125	436.9371168	0.56300377	56	2	"
"Transitional	SM	first_diff	-48.31269173	112.1119626	-1.190639281	0.024644441	-268.0481006	171.4227172	0.666517429	56	2	"
"Transitional	NDWI	fe_cluster	12.50536355	2.587330676	0.983715935	0.078663522	7.434288608	17.57643849	1.34E-06	54	2	***"
"Transitional	NDWI	first_diff	13.43785445	0.916378797	1.297011943	0.096519273	11.64178502	15.23392389	1.09E-48	54	2	***"
"Transitional	HWD	fe_cluster	0.683557059	0.543673784	3.919797451	5.734411491	-0.382023977	1.749138094	0.20864768	56	2	"
"Transitional	HWD	first_diff	0.416032319	0.255761999	3.550320707	8.533761785	-0.085251987	0.917316625	0.103813893	56	2	"
"Transitional	ROS	fe_cluster	-4.081220197	0.491129666	-3.156585734	0.773441662	-5.043816653	-3.11862374	9.58E-17	56	2	***"
"Transitional	ROS	first_diff	-3.188480498	0.7159143	-3.306074976	1.03688104	-4.591646741	-1.785314255	8.44E-06	56	2	***"
], newline);

tmpFile = [tempname, '.tsv'];
fid = fopen(tmpFile,'w');
fprintf(fid,'%s',txt);
fclose(fid);

T = readtable(tmpFile, 'FileType','text', 'Delimiter','\t');
T = T(ismember(T.Sig, ["*","**","***"]), :);

%------------------------------------------------------------------------
% Clean / convert
% -------------------------------------------------------------------------
varsToNum = {'Effect','SE','Effect_Std','SD_Treatment','CI_Lower','CI_Upper',...
             'p_value','N','Sites'};
for k = 1:numel(varsToNum)
    if ismember(varsToNum{k}, T.Properties.VariableNames)
        T.(varsToNum{k}) = str2double(string(T.(varsToNum{k})));
    end
end

T.Class     = string(T.Class);
T.Treatment = string(T.Treatment);
T.Method    = string(T.Method);
T.Sig       = string(T.Sig);

% Standardized CI from raw CI * SD_Treatment
T.CI_Lower_Std = T.CI_Lower .* T.SD_Treatment;
T.CI_Upper_Std = T.CI_Upper .* T.SD_Treatment;

% If SD=0 or Effect_Std missing, standardized values should remain missing
badStd = ~isfinite(T.Effect_Std) | ~isfinite(T.SD_Treatment) | T.SD_Treatment<=0;
T.CI_Lower_Std(badStd) = NaN;
T.CI_Upper_Std(badStd) = NaN;

%------------------------------------------------------------------------
% Settings
% -------------------------------------------------------------------------
plotMode = 'std';   % 'std' or 'raw'

classOrder = ["Rapid Thickening","Gradual Thickening", "Gradual Thinning","Transitional", ...
              "No Trend","All"];

preferredOrder = ["TDD","FDD","SWE$_{max}$","SDO","P$_{summer}$", ...
                  "SM","NDWI","FireDays","HWD","NDVI$_{summer}$","ROS"];

allTreatments = preferredOrder(ismember(preferredOrder, unique(T.Treatment,'stable')));

methodOrder = ["fe_cluster","first_diff"];
methodEdgeColor = [0 0 0];
methodFaceColors = containers.Map({'fe_cluster','first_diff'}, ...
                                  {[0 0 0], [1 1 1]});   % FE filled, FD open

sigLevels  = ["*","**","***"];
sigMarkers = {'o','s','d'};

% x-jitter now, since treatments are on x-axis
xJitterMap = containers.Map({'fe_cluster','first_diff'}, {-0.12, +0.12});

markerSize = 55;
capW = 0.08;   % horizontal half-width of CI caps
lineW = 1.7;

%------------------------------------------------------------------------
% Select plotting columns
% -------------------------------------------------------------------------
switch lower(plotMode)
    case 'raw'
        T.Effect_plot   = T.Effect;
        T.CI_Lower_plot = T.CI_Lower;
        T.CI_Upper_plot = T.CI_Upper;
        yLabelText = 'Raw effect';
    case 'std'
        T.Effect_plot   = T.Effect_Std;
        T.CI_Lower_plot = T.CI_Lower_Std;
        T.CI_Upper_plot = T.CI_Upper_Std;
        yLabelText = 'ALT change (cm)';
    otherwise
        error('plotMode must be ''raw'' or ''std''.');
end

validY = [T.CI_Lower_plot(isfinite(T.CI_Lower_plot)); T.CI_Upper_plot(isfinite(T.CI_Upper_plot))];
yPad = 0.05 * (max(validY) - min(validY) + eps);
yLims = [min(validY)-yPad, max(validY)+yPad];

%------------------------------------------------------------------------
% Figure
% -------------------------------------------------------------------------
f = figure('units','normalized','OuterPosition',[0 0 1 1]);
tl = tiledlayout(3,2,'TileSpacing','compact','Padding','tight');
axs = gobjects(numel(classOrder),1);

%------------------------------------------------------------------------
% Plot
% -------------------------------------------------------------------------
for c = 1:numel(classOrder)
    thisClass = classOrder(c);
    Tc = T(T.Class == thisClass, :);

    ax = nexttile(tl, c);
    axs(c) = ax;
    hold(ax, 'on');

    % horizontal zero line now
    yline(ax, 0, '--k', 'LineWidth', 1, 'HandleVisibility','off');

    % faint vertical guides per treatment
    for r = 1:numel(allTreatments)
        plot(ax, [r r], yLims, '-', ...
            'Color', [0.92 0.92 0.92], ...
            'LineWidth', 0.5, ...
            'HandleVisibility','off');
    end

    for i = 1:height(Tc)
        trt  = Tc.Treatment(i);
        meth = Tc.Method(i);
        sig  = Tc.Sig(i);

        if ~ismember(trt, allTreatments)
            continue;
        end
        if ~isfinite(Tc.Effect_plot(i)) || ~isfinite(Tc.CI_Lower_plot(i)) || ~isfinite(Tc.CI_Upper_plot(i))
            continue;
        end

        baseX = find(allTreatments == trt, 1);
        x = baseX + xJitterMap(char(meth));

        sIdx = find(sigLevels == sig, 1);
        thisMarker = sigMarkers{sIdx};
        thisFaceColor = methodFaceColors(char(meth));

        % vertical CI
        plot(ax, [x x], [Tc.CI_Lower_plot(i), Tc.CI_Upper_plot(i)], '-', ...
            'Color', methodEdgeColor, 'LineWidth', lineW, 'HandleVisibility','off');

        % CI caps
        plot(ax, [x-capW x+capW], [Tc.CI_Lower_plot(i) Tc.CI_Lower_plot(i)], '-', ...
            'Color', methodEdgeColor, 'LineWidth', 1.0, 'HandleVisibility','off');

        plot(ax, [x-capW x+capW], [Tc.CI_Upper_plot(i) Tc.CI_Upper_plot(i)], '-', ...
            'Color', methodEdgeColor, 'LineWidth', 1.0, 'HandleVisibility','off');

        % point
        scatter(ax, x, Tc.Effect_plot(i), markerSize, ...
            'Marker', thisMarker, ...
            'MarkerFaceColor', thisFaceColor, ...
            'MarkerEdgeColor', methodEdgeColor, ...
            'LineWidth', 1.0, ...
            'HandleVisibility','off');
    end

    set(ax, ...
        'XTick', 1:numel(allTreatments), ...
        'XTickLabel', cellstr(allTreatments), ...
        'FontSize', 16, ...
        'Box','on', ...
        'Layer','top', ...
        'TickLabelInterpreter','latex');

    xlim(ax, [0.5, numel(allTreatments)+0.5]);
    ylim(ax, yLims);

    grid(ax, 'on');
    ax.GridAlpha = 0.10;
    ax.XGrid = 'off';

    title(ax, sprintf('%s', thisClass), ...
        'FontSize', 16, 'Interpreter','latex');

    if ~mod(c,2)==1
        set(ax, 'YTickLabel', '');
    end
    if c<5
        set(ax, 'XTickLabel', '');
    else
        set(ax, 'XTickLabelRotation', 90);
    end
end

ylabel(tl, yLabelText, 'FontSize', 16, 'Interpreter','latex');
% xlabel(tl, 'Treatments', 'FontSize', 18, 'Interpreter','latex');

%------------------------------------------------------------------------
% Legend
% -------------------------------------------------------------------------
axLeg = axs(1);
hold(axLeg,'on');

h1 = plot(axLeg, nan, nan, 'o', 'Color', 'k', ...
    'LineStyle', '-', ...
    'MarkerFaceColor', 'k', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerSize', 7, ...
    'DisplayName', 'FE');

h2 = plot(axLeg, nan, nan, 'o', 'Color', 'k', ...
    'LineStyle', '-', ...
    'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerSize', 7, ...
    'DisplayName', 'FD');

h3 = plot(axLeg, nan, nan, 'o',...
    'LineStyle', 'none', ...
    'MarkerFaceColor', 'k', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerSize', 7, ...
    'DisplayName', '$p<.05$');

h4 = plot(axLeg, nan, nan, 's', ...
    'LineStyle', 'none', ...
    'MarkerFaceColor', 'k', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerSize', 7, ...
    'DisplayName', '$p<.01$');

h5 = plot(axLeg, nan, nan, 'd', ...
    'LineStyle', 'none', ...
    'MarkerFaceColor', 'k', ...
    'MarkerEdgeColor', 'k', ...
    'MarkerSize', 7, ...
    'DisplayName', '$p<.001$');

lgd = legend(axLeg, [h1 h2 h3 h4 h5], 'FontSize',15,...
    'Orientation','horizontal', ...
    'Box','on', 'Interpreter','latex');
lgd.Layout.Tile = 'north';
lgd.Title.String = 'Estimation Methods and Significance';
exportgraphics(gcf,'G:\My Drive\UND\Index\Index_CausalityInferenceSigResults.pdf', 'Resolution', 300) 


%%
clear
clc
filePath = 'G:\My Drive\UND\Index\CALM_ALT_all_Max.csv';
[result_table,~,~,~] = preprocess_CALM(filePath, 1);

% Read location names from Excel and join to result_table by lat/lon
locFile = 'G:\My Drive\UND\Index\CALM_ALT_all.xlsx';
if isfile(locFile)
    optsLoc = detectImportOptions(locFile);
    tblLoc = readtable(locFile, optsLoc);

    % Ensure latitude/longitude variable names exist - try common names
    latNames = 'Lat';
    lonNames = 'Long';
    latVar = intersect(latNames, tblLoc.Properties.VariableNames);
    lonVar = intersect(lonNames, tblLoc.Properties.VariableNames);
    if isempty(latVar) || isempty(lonVar)
        error('Could not find latitude/longitude columns in %s', locFile);
    end
    latVar = latVar{1}; lonVar = lonVar{1};

    % Find a location name column (try common names)
    nameCandidates = 'Location';
    nameVar = intersect(nameCandidates, tblLoc.Properties.VariableNames);
    if isempty(nameVar)
        % If none, create a synthetic name from lat/lon
        tblLoc.LocationName = strcat('Loc_', string(tblLoc.(latVar)), '_', string(tblLoc.(lonVar)));
        nameVar = {'LocationName'};
    else
        nameVar = nameVar{1};
    end

    % Prepare keys for join: round to reasonable precision to avoid floating mismatches
    tolDigits = 6; % adjust if needed
    tblLoc.key_lat = round(tblLoc.(latVar), tolDigits);
    tblLoc.key_lon = round(tblLoc.(lonVar), tolDigits);

    % Ensure result_table has lat/lon columns
    if ~any(ismember(result_table.Properties.VariableNames, latNames)) || ~any(ismember(result_table.Properties.VariableNames, lonNames))
        error('result_table does not contain expected latitude/longitude columns.');
    end
    % pick actual names in result_table
    rlatVar = intersect(latNames, result_table.Properties.VariableNames); rlatVar = rlatVar{1};
    rlonVar = intersect(lonNames, result_table.Properties.VariableNames); rlonVar = rlonVar{1};

    result_table.key_lat = round(result_table.(rlatVar), tolDigits);
    result_table.key_lon = round(result_table.(rlonVar), tolDigits);

    % Use outer join to preserve all rows in result_table
    tblLocSmall = tblLoc(:, {'key_lat','key_lon', nameVar});
    tblLocSmall.Properties.VariableNames = {'key_lat','key_lon','LocationName'};

    result_table = outerjoin(result_table, tblLocSmall, ...
        'Keys', {'key_lat','key_lon'}, ...
        'MergeKeys', true, ...
        'Type','left');

    % % If LocationName is missing, fill with synthetic string from coords
    % missingIdx = ismissing(result_table.LocationName);
    % if any(missingIdx)
    %     result_table.LocationName(missingIdx) = strcat('Loc_', string(result_table.key_lat(missingIdx)), '_', string(result_table.key_lon(missingIdx)));
    % end

    % Clean up temporary key variables if not needed
    result_table.key_lat = [];
    result_table.key_lon = [];
else
    warning('Location file not found: %s. Adding empty LocationName column.', locFile);
    result_table.LocationName = strings(height(result_table),1);
end
writetable(result_table(result_table.MainRegion=="Alaska",:), 'G:\My Drive\UND\Index\CALM_ALT_all_Max_Alaska.csv');


%% (old) Classification Distribution with Confidence Donuts
clear
close('all')
clc

%Load data
filePath = 'G:\My Drive\UND\Index\aldi_v4_results.csv';
opts = detectImportOptions(filePath);
data = readtable(filePath, opts);

%Add main regions
data.MainRegion = strings(height(data),1);
[CALMlocations,~,ic] = unique([data.lat data.lon],"rows");

for i = 1:size(CALMlocations,1)
    tmp = data(i==ic,:);
    % assign_CALM_regions must return a region string
    data.MainRegion(i==ic) = strings(height(tmp),1) + assign_CALM_regions(tmp.lat(1), tmp.lon(1));
end
data.MainRegion = categorical(strtrim(string(data.MainRegion)));
writetable(data, 'G:\My Drive\UND\Index\aldi_v4_results_wMainRegion.csv');

%Parse classification into class and rate
n_sites = height(data);
data.class = strings(n_sites, 1);
data.rate = strings(n_sites, 1);
data.full_class = strings(n_sites, 1);

for i = 1:n_sites
    clf = string(data.class_classification{i});
    
    switch clf
        case 'rapid_thickening'
            data.class(i) = "Thickening";
            data.rate(i) = "Rapid";
            data.full_class(i) = "Rapid Thickening";
        case 'gradual_thickening'
            data.class(i) = "Thickening";
            data.rate(i) = "Gradual";
            data.full_class(i) = "Gradual Thickening\,\,\,";
        case 'rapid_thinning'
            data.class(i) = "Thinning";
            data.rate(i) = "Rapid";
            data.full_class(i) = "Rapid Thinning";
        case 'gradual_thinning'
            data.class(i) = "Thinning";
            data.rate(i) = "Gradual";
            data.full_class(i) = "Gradual Thinning";
        case 'no_trend'
            data.class(i) = "No Trend";
            data.rate(i) = "";
            data.full_class(i) = "No Trend";
        case 'transitional'
            data.class(i) = "Transitional";
            data.rate(i) = "";
            data.full_class(i) = "Transitional";
    end
end

% Convert to categorical with specified order
classOrder = ["Thickening", "No Trend", "Thinning", "Transitional"];
fullClassOrder = ["Rapid Thickening", "Gradual Thickening\,\,\,", "No Trend", ...
                  "Gradual Thinning", "Rapid Thinning", "Transitional"];

data.class = categorical(data.class, classOrder);
data.full_class = categorical(data.full_class, fullClassOrder);

%Define color scheme
colors_full = [
    0.84, 0.15, 0.16;   % Rapid Thickening - dark red
    1.00, 0.55, 0.20;   % Gradual Thickening - orange
    0.60, 0.60, 0.60;   % No Trend - gray
    0.55, 0.75, 0.30;   % Gradual Thinning - light green
    0.20, 0.50, 0.70;   % Rapid Thinning - blue
    0.58, 0.40, 0.74;   % Transitional - purple
];

%Count data for main plot
full_counts = zeros(6, 1);
for i = 1:6
    full_counts(i) = sum(data.full_class == fullClassOrder(i));
end

%Main plot
figure('units','normalized','OuterPosition',[0 0 1 1]);
t = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Main circum-Arctic donut
nexttile(t,[3,2])
d = donutchart(full_counts, fullClassOrder, ...
    'ExplodedWedges', 6, ...  
    'LabelStyle', 'percent', ...
    'FaceAlpha', 0.9, ...
    'EdgeColor', 'w', ...
    'Direction', 'counterclockwise');
d.ColorOrder = colors_full;
d.CenterLabel = 'Circum-Arctic';
d.InnerRadius = 0.5;
d.FontSize = 14;
d.LegendVisible = 'on';
d.LegendTitle = 'Active Layer Dynamics';

d.Interpreter = 'latex';

%Regional donuts
regionCategories = ["Alaska"; "Russia"; "Canada"; "Greenland"; "Scandinavia"; "Svalbard"];

for r = 1:numel(regionCategories)
    nexttile;
    region = regionCategories(r);
    
    % Subset data for this region
    idx = data.MainRegion == region;
    n_region = sum(idx);
    
    if n_region == 0
        title(sprintf('%s', region), 'FontSize', 12);
        axis off;
        continue;
    end
    
    % Count by FULL_CLASS (not class!) for this region
    % This is the key fix - use full_class to match fullClassOrder
    region_full_class = data.full_class(idx);
    
    region_counts = zeros(6, 1);
    for c = 1:6
        region_counts(c) = sum(region_full_class == fullClassOrder(c));
    end
    
    % Get year range
    years_str = sprintf('%d–%d', min(data.year_min(idx)), max(data.year_max(idx)));
    
    % Create donut chart
    % Only include non-zero categories for cleaner display
    valid_idx = region_counts > 0;
    
    if sum(valid_idx) == 0
        title(region, 'FontSize', 12, 'Interpreter','latex');
        axis off;
        continue;
    end
    
    d = donutchart(region_counts(valid_idx), fullClassOrder(valid_idx), ...
        'LabelStyle', 'percent', ...
        'FaceAlpha', 0.9, ...
        'EdgeColor', 'w', ...
        'Direction', 'counterclockwise');
    
    % Match colors to the correct categories
    d.ColorOrder = colors_full(valid_idx, :);
    
    d.CenterLabel = {char(region)};
    d.InnerRadius = 0.5;
    d.FontSize = 12;
    d.LegendVisible = 'off';
    d.Interpreter = 'latex';

    % title(years_str, 'FontSize', 10, 'FontWeight', 'normal');
end
% exportgraphics(t, 'G:\My Drive\UND\Index\Index_DonutsRegionsV2.pdf','Resolution',300)

%Print verification table
fprintf('\n=== VERIFICATION: Regional Counts ===\n');
fprintf('%-12s', 'Region');
for c = 1:6
    fprintf('%15s', fullClassOrder(c));
end
fprintf('%10s\n', 'Total');
fprintf('%s\n', repmat('-', 1, 120));

for r = 1:numel(regionCategories)
    region = regionCategories(r);
    idx = data.MainRegion == region;
    
    fprintf('%-12s', region);
    row_total = 0;
    for c = 1:6
        cnt = sum(data.full_class(idx) == fullClassOrder(c));
        fprintf('%15d', cnt);
        row_total = row_total + cnt;
    end
    fprintf('%10d\n', row_total);
end

fprintf('%s\n', repmat('-', 1, 120));
fprintf('%-12s', 'Total');
for c = 1:6
    fprintf('%15d', full_counts(c));
end
fprintf('%10d\n', sum(full_counts));



%% Add missing factors to dataset
clear; clc; close all;

%Load Data
dataTargets = readtable('G:\My Drive\UND\Index\ALTI_withYearlyData_augmented.csv');
latitudes = dataTargets.lat;
longitudes = dataTargets.lon;


% Load yearly features
filePath = 'G:\My Drive\UND\Index\CALM_ERA5_missing_features.csv';
opts = detectImportOptions(filePath);
yearFeatures = readtable(filePath, opts);
yearFeatures = movevars(yearFeatures, "lat", "Before", "LAI_high_summer");
yearFeatures = movevars(yearFeatures, "lon", "after", "lat");
yearFeatures = movevars(yearFeatures, "year", "after", "lon");

%add permafrost static features to the yearly data
joinedData = innerjoin(yearFeatures,dataTargets,Keys=["lat","lon", "year"]);
joinedData = movevars(joinedData, "Max", 'Before', 'class_classification');
% writetable(joinedData, 'G:\My Drive\UND\Index\ALTI_withYearlyData_augmentedV2.csv'); 


% Load yearly features
filePath = 'G:\My Drive\UND\Index\CALM_NDWI_FIRMS_features.csv';
opts = detectImportOptions(filePath);
yearFeatures = readtable(filePath, opts);
yearFeatures.FIRMS_T21_max_C = yearFeatures.FIRMS_T21_max_K-273.15;
yearFeatures.FIRMS_T21_mean_C = yearFeatures.FIRMS_T21_mean_K-273.15;
yearFeatures.FIRMS_T21_max_K = [];
yearFeatures.FIRMS_T21_mean_K = [];
yearFeatures = renamevars(yearFeatures, "mean", "NDWI_annual_mean");
yearFeatures = removevars(yearFeatures, 'batch_id');
yearFeatures = removevars(yearFeatures, 'site_index');
yearFeatures = movevars(yearFeatures, "NDWI_annual_mean", "Before", "FIRMS_fire_days");

%add more features to the yearly data
joinedData2 = innerjoin(yearFeatures,joinedData,Keys=["lat","lon", "year"]);
writetable(joinedData2, 'G:\My Drive\UND\Index\ALTI_withYearlyData_augmentedV3.csv'); 


%split to index classes
unq_class = unique(joinedData2.class_classification);
for i = 1:length(unq_class)
    writetable(joinedData2(strcmp(joinedData2.class_classification,unq_class(i)),:), ['G:\My Drive\UND\Index\ALTI_withYearlyData_augmentedV3_' unq_class{i}, '.csv']);
end





%% ========================================================================
%% Summary Statistics for Paper
%% ========================================================================
fprintf('\n');
fprintf('================================================================\n');
fprintf('SUMMARY STATISTICS FOR MANUSCRIPT\n');
fprintf('================================================================\n');

fprintf('\nClassification counts:\n');
for c = 1:6
    n = sum(strcmp(results.class_classification, class_order{c}));
    fprintf('  %s: %d (%.1f%%)\n', class_labels{c}, n, 100*n/n_sites);
end

fprintf('\nAggregate:\n');
fprintf('  Thickening: %d (%.1f%%)\n', n_thick, 100*n_thick/n_sites);
fprintf('  Thinning: %d (%.1f%%)\n', n_thin, 100*n_thin/n_sites);
fprintf('  Ratio: %.1f:1\n', n_thick/n_thin);

fprintf('\nSen slope statistics:\n');
fprintf('  Mean: %.2f cm/yr\n', mean(results.trend_sen_slope));
fprintf('  Median: %.2f cm/yr\n', median(results.trend_sen_slope));
fprintf('  SD: %.2f cm/yr\n', std(results.trend_sen_slope));
fprintf('  Range: %.2f to %.2f cm/yr\n', min(results.trend_sen_slope), max(results.trend_sen_slope));

fprintf('\nRegional mean slopes:\n');
for r = 1:n_regions
    fprintf('  %s: %.2f ± %.2f cm/yr (n=%d)\n', ...
        region_names{r}, region_stats(r).mean_slope, region_stats(r).se_slope, region_stats(r).n);
end

fprintf('\nConfidence levels:\n');
for cf = 1:3
    n = sum(strcmp(results.class_confidence, conf_levels{cf}));
    fprintf('  %s: %d (%.1f%%)\n', conf_levels{cf}, n, 100*n/n_sites);
end

fprintf('\nRegime shifts:\n');
fprintf('  Sites with shifts: %d (%.1f%%)\n', n_with_shifts, 100*n_with_shifts/n_sites);

fprintf('\nTemporal coverage:\n');
fprintf('  Years: %d - %d\n', min(results.year_min), max(results.year_max));
fprintf('  Mean span: %.1f years\n', mean(results.span_years));
fprintf('  Mean completeness: %.1f%%\n', 100*mean(results.completeness));

fprintf('\n================================================================\n');
fprintf('Figures saved: Fig1-Fig4 (.png and .eps)\n');
fprintf('================================================================\n');






%% Functions
function region = assign_CALM_regions(lat, lon)
% ASSIGN_CALM_REGIONS Assigns region based on CALM site coordinates
% This function uses the official CALM network site categories to assign
% each site to one of 5 main regions: Alaska, Canada, Russia, Greenland, Svalbard
%
% Input:
%   lat - latitude (scalar or array)
%   lon - longitude (scalar or array)
%
% Output:
%   region - cell array of region names
%
% Usage:
%   region = assign_CALM_regions(71.31667, -156.6)
%   region = assign_CALM_regions([71.31667; 78.92], [-156.6; 11.86])

% Define all CALM sites with their official regions based on the CALM metadata
% Format: [lat, lon, region_name]

sites = {
    % ===== ALASKA =====
    % Alaska North Slope
    71.31667, -156.6, 'Alaska'
    71.31667, -156.5833, 'Alaska'
    70.45, -157.4, 'Alaska'
    70.3745, -148.5522, 'Alaska'
    70.36667, -148.5667, 'Alaska'
    70.1613, -148.4653, 'Alaska'
    70.28333, -148.8667, 'Alaska'
    70.2835, -148.8928, 'Alaska'
    70.275, -148.919, 'Alaska'
    69.6739, -148.7219, 'Alaska'
    69.1466, -148.8483, 'Alaska'
    69.1482, -148.8505, 'Alaska'
    69.12883, -148.5928, 'Alaska'
    68.6114, -149.3101, 'Alaska'
    68.611, -149.3145, 'Alaska'
    68.611, -149.30933, 'Alaska'
    68.6215, -149.6063, 'Alaska'
    68.624, -149.61817, 'Alaska'
    68.61667, -149.6, 'Alaska'
    68.4774, -149.5024, 'Alaska'
    68.0691, -149.5804, 'Alaska'
    68.48333, -155.7333, 'Alaska'
    69.5006, -148.5592, 'Alaska'
    69.441, -148.67033, 'Alaska'
    69.401, -148.8056, 'Alaska'
    70.8645, -153.9067, 'Alaska'
    69.98962, -153.0938, 'Alaska'
    70.33523, -152.052, 'Alaska'
    69.1704, -158.0067, 'Alaska'
    69.3957, -152.1428, 'Alaska'
    70.1959, -161.0781, 'Alaska'
    69.8894, -142.9839, 'Alaska'
    69.7516, -154.6176, 'Alaska'
    69.7776, -144.7933, 'Alaska'
    70.6285, -156.8353, 'Alaska'
    69.972, -144.7706, 'Alaska'
    69.1555, -158.0305, 'Alaska'
    68.6816, -144.8421, 'Alaska'
    70.0366, -157.0814, 'Alaska'
    70.5685, -152.965, 'Alaska'
    70.4417, -154.3656, 'Alaska'
    70.723267, -153.83604, 'Alaska'
    
    % Alaska Interior
    66.45, -150.6167, 'Alaska'
    65.1667, -147.9, 'Alaska'
    64.7, -148.1333, 'Alaska'
    64.71444, -148.14528, 'Alaska'  % Bonanza Creek (additional grid)
    64.9, -147.8167, 'Alaska'
    64.88, -147.67, 'Alaska'
    62.55, -143.3667, 'Alaska'
    62.51667, -143.3667, 'Alaska'
    63.02131, -163.5607, 'Alaska'
    62.78478, -164.52623, 'Alaska'
    61.52786, -165.61778, 'Alaska'
    61.87901, -162.05759, 'Alaska'
    62.04755, -163.23412, 'Alaska'
    65.82819, -144.07843, 'Alaska'
    66.62076, -145.1035, 'Alaska'
    65.65724, -149.08443, 'Alaska'
    64.7333, -156.78334, 'Alaska'
    64.7777, -141.10954, 'Alaska'
    66.54491, -152.64539, 'Alaska'
    64.73486, -155.48461, 'Alaska'
    65.36988, -147.0609, 'Alaska'
    68.13013, -145.54007, 'Alaska'
    67.03829, -146.40263, 'Alaska'
    64.55446, -149.08339, 'Alaska'
    63.67475, -144.14597, 'Alaska'
    
    % Alaska Seward Peninsula
    64.8442, -163.7202, 'Alaska'
    65.454, -164.6268, 'Alaska'
    
    % ===== CANADA =====
    78.88333, -75.91667, 'Canada'
    78.54000, -75.55000, 'Canada'  % Alexandria Fiord (additional)
    69.71972, -134.4619, 'Canada'
    69.36917, -134.9486, 'Canada'
    69.21889, -134.2911, 'Canada'
    68.96667, -133.55, 'Canada'
    68.68472, -134.1458, 'Canada'
    67.795, -134.1261, 'Canada'
    65.67361, -128.8292, 'Canada'
    65.28333, -126.8833, 'Canada'
    65.19306, -126.4689, 'Canada'
    64.91667, -125.5833, 'Canada'
    63.46639, -123.6928, 'Canada'
    62.69667, -123.065, 'Canada'
    61.88778, -121.6017, 'Canada'
    81.40072, -71.38333, 'Canada'
    56.63333, -76.1, 'Canada'
    81.40072, -76.70937, 'Canada'
    80.01667, -85.75, 'Canada'
    64.16667, -95.5, 'Canada'
    52.8, -118.1167, 'Canada'
    63.94054, -138.59157, 'Canada'
    60.44943, -133.52016, 'Canada'
    62.3377, -140.83835, 'Canada'
    65.87808, -89.36717, 'Canada'
    73.0054, -80.68725, 'Canada'
    66.87932, -64.69728, 'Canada'
    72.81072, -79.32928, 'Canada'
    
    % ===== RUSSIA =====
    % Russian European north
    67.58333, 64.18333, 'Russia'
    67.33333, 63.73333, 'Russia'
    68.3, 54.5, 'Russia'
    68.23333, 53.85, 'Russia'
    67.7723958, 34.182046, 'Russia'
    67.06556, 62.92508, 'Russia'
    
    % West Siberia
    65.314861, 72.864194, 'Russia'
    65.23375, 72.518417, 'Russia'
    69.71667, 66.75, 'Russia'
    70.11667, 75.58333, 'Russia'
    70.28333, 68.9, 'Russia'
    70.2755, 68.89164, 'Russia'
    70.2955, 68.88347, 'Russia'
    70.30139, 68.84131, 'Russia'
    70.27417, 68.89078, 'Russia'
    66.31325, 76.903478, 'Russia'
    67.47791, 76.69529, 'Russia'
    66.723483, 66.080488, 'Russia'
    70.8893, 78.4171, 'Russia'
    70.8906, 78.4308, 'Russia'
    70.8699, 78.5478, 'Russia'
    73.32694, 70.08528, 'Russia'
    73.32861, 70.08833, 'Russia'
    66.709167, 66.567778, 'Russia'
    66.696667, 66.358889, 'Russia'
    68.225278, 69.144444, 'Russia'
    66.54395, 66.73125, 'Russia'
    66.569882, 66.880857, 'Russia'
    66.05003, 76.64494, 'Russia'
    
    % Central Siberia
    72.38333, 99.5, 'Russia'
    74.53333, 98.6, 'Russia'
    71.58333, 128.7833, 'Russia'
    71.86167, 141.0102, 'Russia'
    71.7855, 129.4192, 'Russia'
    71.7855, 71.41917, 'Russia'
    69.43333, 88.46667, 'Russia'
    67.48135, 86.4347, 'Russia'
    62.0133, 129.65683, 'Russia'
    62.31618, 129.49952, 'Russia'
    56.76038, 118.18903, 'Russia'
    56.906264, 118.28067, 'Russia'
    72.369775, 126.48063, 'Russia'
    
    % North East Siberia
    70.91667, 156.6333, 'Russia'
    70.08333, 159.5833, 'Russia'
    70.08333, 159.9167, 'Russia'
    69.48333, 156.9833, 'Russia'
    69.38333, 158.4667, 'Russia'
    69.08333, 158.9, 'Russia'
    68.81667, 161, 'Russia'
    68.9255, 161.5046, 'Russia'
    68.7417, 161.5042, 'Russia'
    68.8, 160.95, 'Russia'
    68.51667, 161.4333, 'Russia'
    68.83333, 161.0333, 'Russia'
    69.31667, 154.9833, 'Russia'
    69.85, 159.5, 'Russia'
    68.41667, 161.2167, 'Russia'
    70.55, 147.4333, 'Russia'
    70.56667, 147.4167, 'Russia'
    68.73333, 158.9, 'Russia'
    69.16667, 154.4333, 'Russia'
    69.98333, 153.5833, 'Russia'
    68.7, 161.55, 'Russia'
    68.71667, 161.5333, 'Russia'
    68.74, 161.3928, 'Russia'
    
    % Chukotka
    64.78333, 176.9667, 'Russia'
    64.08333, 177.0667, 'Russia'
    64.56667, 177.2, 'Russia'
    65.6, -171.05, 'Russia'
    65.539667, -171.63023, 'Russia'
    64.63333, 176.9667, 'Russia'
    
    % Kamchatka
    55.7507, 160.2896, 'Russia'
    55.76524, 160.3206, 'Russia'
    55.89092, 160.5375, 'Russia'
    
    % ===== GREENLAND =====
    74.473, -20.5528, 'Greenland'
    74.466, -20.5643, 'Greenland'
    69.25, -53.5, 'Greenland'
    
    % ===== SVALBARD =====
    % Italy/Svalbard
    78.92, 11.86, 'Svalbard'
    
    % Norway/Svalbard
    78.1793, 16.467, 'Svalbard'
    78.200956, 15.836412, 'Svalbard'
    
    % Poland/Svalbard
    77.56667, 14.5, 'Svalbard'
    78.68333, 11.83333, 'Svalbard'
    % Calypsostranda sites (Poland/Svalbard - P1)
    77.52900, 14.49300, 'Svalbard'
    77.53100, 14.48300, 'Svalbard'
    77.53300, 14.49300, 'Svalbard'
    77.53900, 14.49300, 'Svalbard'
    77.54300, 14.51300, 'Svalbard'
    77.54300, 14.53300, 'Svalbard'
    77.54300, 14.55300, 'Svalbard'
    77.54500, 14.49300, 'Svalbard'
    77.54800, 14.47300, 'Svalbard'
    77.54800, 14.48800, 'Svalbard'
    77.54800, 14.50300, 'Svalbard'
    77.55800, 14.43300, 'Svalbard'
    77.55800, 14.46300, 'Svalbard'
    77.55800, 14.52300, 'Svalbard'
    77.55800, 14.55300, 'Svalbard'
    77.55800, 14.58300, 'Svalbard'
    77.57100, 14.47300, 'Svalbard'
    77.57300, 14.45300, 'Svalbard'
    77.57700, 14.47300, 'Svalbard'
    77.57800, 14.45300, 'Svalbard'
    77.58300, 14.45300, 'Svalbard'
    77.58300, 14.47300, 'Svalbard'
    % Kafføyra sites (Poland/Svalbard - P2)
    78.72833, 11.72167, 'Svalbard'
    78.73333, 11.71667, 'Svalbard'
    78.73833, 11.71167, 'Svalbard'
    
    % Sweden/Svalbard
    78.05, 13.61667, 'Svalbard'
    
    % ===== SCANDINAVIA (Assign to Russia per user request) =====
    68.33333, 18.83333, 'Scandinavia'  % Abisko, Sweden
    61.6775823, 8.3693057, 'Scandinavia'  % Juvvasshøe, Norway
    62.2958, 9.352, 'Scandinavia'  % Snøheim DB2, Norway
};

% Convert to matrix for easier processing
site_lats = cell2mat(sites(:,1));
site_lons = cell2mat(sites(:,2));
site_regions = sites(:,3);

% Initialize output
if isscalar(lat)
    region = 'Unknown';
else
    region = cell(size(lat));
    region(:) = {'Unknown'};
end

% Create lookup map for faster matching
lookup_map = containers.Map();
for i = 1:length(site_lats)
    key = sprintf('%.6f_%.6f', site_lats(i), site_lons(i));
    lookup_map(key) = site_regions{i};
end

% Match coordinates to regions with tolerance
tolerance = 0.01; % degrees (~1 km)

for i = 1:length(lat)
    % Calculate distance to all known sites
    lat_diff = abs(site_lats - lat(i));
    lon_diff = abs(site_lons - lon(i));
    
    % Find sites within tolerance
    matches = (lat_diff < tolerance) & (lon_diff < tolerance);
    
    if any(matches)
        % Take the first match (should only be one anyway)
        idx = find(matches, 1);
        % fprintf('found a match for the unique location!\n')
        if isscalar(lat)
            region = site_regions{idx};
        else
            region{i} = site_regions{idx};
        end
    else 
        fprintf('uh oh! could not find a match for the unique location! (%.5f,%.5f)\n', lat(i), lon(i))    
    end

end

end

% 
% function categoryTable = assignEnvironmentalCategories(latitudes, longitudes)
%     % Load the permafrost shapefile
%     shapefile = 'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\CALM\ggd318_map_circumarctic\permaice.shp'; 
%     permafrost = shaperead(shapefile);
%     shape_info = shapeinfo(shapefile);
%     source_proj = shape_info.CoordinateReferenceSystem;
% 
%     % Project the polygon coordinates
%     [X, Y] = deal({permafrost.X}, {permafrost.Y});
%     [polyLat, polyLon] = cellfun(@(x, y) projinv(source_proj, x, y), X, Y, 'UniformOutput', false);
%     combo_values = {permafrost.COMBO};
% 
%     % Initialize results
%     n = length(latitudes);
%     permafrostType = repmat("Unknown", n, 1);
%     groundIceType = repmat("Unknown", n, 1);
%     landformType = repmat("Unknown", n, 1);
% 
%     % Loop through each point
%     for i = 1:n
%         lat = latitudes(i);
%         lon = longitudes(i);
%         found = false;
% 
%         % Check which polygon it falls into
%         for j = 1:length(polyLat)
%             if inpolygon(lon, lat, polyLon{j}, polyLat{j})
%                 code = lower(combo_values{j});
%                 if numel(code) >= 3
%                     % Extract and label types
%                     switch code(1)
%                         case 'c', permafrostType(i) = "Continuous";
%                         case 'd', permafrostType(i) = "Discontinuous";
%                         case 's', permafrostType(i) = "Sporadic";
%                         case 'i', permafrostType(i) = "Isolated";
%                     end
%                     switch code(2)
%                         case 'h', groundIceType(i) = "High";
%                         case 'm', groundIceType(i) = "Medium";
%                         case 'l', groundIceType(i) = "Low";
%                     end
%                     switch code(3)
%                         case 'f', landformType(i) = "Lowlands";
%                         case 'r', landformType(i) = "Mountains";
%                     end
%                 end
%                 found = true;
%                 break;
%             end
%         end
% 
%         if ~found
%             % Optional: log unmatched coordinates
%             fprintf("No match for site at (%.2f, %.2f)\n", lat, lon);
%         end
%     end
% 
%     % Create table for output
%     categoryTable = table(latitudes, longitudes, permafrostType, groundIceType, landformType);
% end


function categoryTable = assignEnvironmentalCategories(latitudes, longitudes)

    % Load shapefile
    shapefile = 'C:\Users\aymane.ahajjam\Desktop\UND\Defense resiliency platform\Datasets\CALM\ggd318_map_circumarctic\permaice.shp';
    permafrost = shaperead(shapefile);
    shape_info = shapeinfo(shapefile);
    source_proj = shape_info.CoordinateReferenceSystem;

    % Convert polygons to lat/lon
    [X, Y] = deal({permafrost.X}, {permafrost.Y});
    [polyLat, polyLon] = cellfun(@(x,y) projinv(source_proj,x,y), ...
                                X, Y, 'UniformOutput', false);

    combo_values = {permafrost.COMBO};

    n = numel(latitudes);
    permafrostType = repmat("Unknown", n, 1);
    groundIceType  = repmat("Unknown", n, 1);
    landformType   = repmat("Unknown", n, 1);

    % =========================
    % STAGE 1: Polygon matching
    % =========================
    for i = 1:n
        lat = latitudes(i);
        lon = longitudes(i);

        for j = 1:numel(polyLat)
            if inpolygon(lon, lat, polyLon{j}, polyLat{j})
                [permafrostType(i), groundIceType(i), landformType(i)] = ...
                    decodeCombo(combo_values{j});
                break;
            end
        end
    end

    % ======================================
    % STAGE 2: Nearest-site label propagation
    % ======================================
    knownIdx   = permafrostType ~= "Unknown";
    unknownIdx = ~knownIdx;

    if any(unknownIdx) && any(knownIdx)

        knownLat = latitudes(knownIdx);
        knownLon = longitudes(knownIdx);

        for i = find(unknownIdx)'
            d = distance(latitudes(i), longitudes(i), knownLat, knownLon);
            [~, k] = min(deg2km(d));

            % Copy class from nearest known site
            permafrostType(i) = permafrostType(k);
            groundIceType(i)  = groundIceType(k);
            landformType(i)   = landformType(k);
        end
    end

    categoryTable = table(latitudes, longitudes, ...
                          permafrostType, groundIceType, landformType);
end


function [pf, gi, lf] = decodeCombo(combo)
    pf = "Unknown"; gi = "Unknown"; lf = "Unknown";
    code = lower(combo);

    if numel(code) >= 3
        switch code(1)
            case 'c', pf = "Continuous";
            case 'd', pf = "Discontinuous";
            case 's', pf = "Sporadic";
            case 'i', pf = "Isolated";
        end
        switch code(2)
            case 'h', gi = "High";
            case 'm', gi = "Medium";
            case 'l', gi = "Low";
        end
        switch code(3)
            case 'f', lf = "Lowlands/thickOverburdenCover";
            case 'r', lf = "Mountains/thinOverburdenCover";
        end
    end
end



%%================== Local plotting function ==================
function localPlotSite(site_id, panelLetter, data, raw_data)

    % ---- Grab site row ----
    site_row = data(strcmp(string(data.site_id), string(site_id)), :);
    if isempty(site_row)
        axis off;
        text(0.5,0.5,"Site not found","HorizontalAlignment","center",'Interpreter','latex');
        return;
    end

    % ---- Parse lat/lon from site_id formatted as "LAT_LON_..." ----
    parts = split(string(site_id), '_');
    if numel(parts) < 2
        axis off;
        text(0.5,0.5,"Bad site\_id format","HorizontalAlignment","center",'Interpreter','latex');
        return;
    end

    lat = str2double(parts(1));
    lon = str2double(parts(2));
    if isnan(lat) || isnan(lon)
        axis off;
        text(0.5,0.5,"Cannot parse lat/lon","HorizontalAlignment","center",'Interpreter','latex');
        return;
    end

    % ---- Get raw series near lat/lon ----
    site_ts = raw_data(raw_data.Lat==lat & raw_data.Long==lon, :);    
    if isempty(site_ts)
        axis off;
        text(0.5,0.5,sprintf("No raw points within %.2f^{\\circ}", tol), ...
            "HorizontalAlignment","center",'Interpreter','latex');
        return;
    end

    site_ts = sortrows(site_ts, 'Year');
    
    years = site_ts.Year;
    alt   = site_ts.Max;

    valid = ~isnan(years) & ~isnan(alt);
    years = years(valid);
    alt   = alt(valid);

    if numel(years) < 2
        axis off;
        text(0.5,0.5,"Not enough points","HorizontalAlignment","center",'Interpreter','latex');
        return;
    end

    % ---- Overall (long-term) Theil–Sen trend ----
    slope = site_row.trend_sen_slope;

    y0 = min(years);
    b0 = median(alt - slope .* (years - y0), 'omitnan');

    yr = min(years):max(years);
    trend_line = b0 + slope .* (yr - y0);

    % ---- Class + confidence ----
    cls  = string(site_row.full_class);
    conf = string(site_row.class_confidence);

    % ---- MK p-value formatting (plain text in LaTeX) ----
    mk_p = site_row.trend_mk_pvalue;
    if isnan(mk_p)
        p_str = "$p=\mathrm{NA}$";
    elseif mk_p < 0.001
        p_str = "$p<0.001$";
    elseif mk_p < 0.1
        p_str = "$p < 0.1$";
    else
        p_str = "p=" + string(sprintf('%.3g', mk_p));
    end

    % ---- Class probability support ----
    p_cls = NaN;
    switch cls
        case "Rapid Thickening"
            p_cls = site_row.class_p_rapid_thickening;
        case "Gradual Thickening"
            p_cls = site_row.class_p_gradual_thickening;
        case "No Trend"
            p_cls = site_row.class_p_no_trend;
        case "Gradual Thinning"
            p_cls = site_row.class_p_gradual_thinning;
        case "Rapid Thinning"
            p_cls = site_row.class_p_rapid_thinning;
        case "Transitional"
            p_cls = NaN; % no class_p_transitional
    end

    % ---- Flags (your table stores as strings "True"/"False" in some exports) ----
    flag_rev  = localAsBool(site_row.rev_reversal);
    flag_lp   = localAsBool(site_row.class_flag_low_power);

    % ---- Transitional / trajectory trends (from your trajectory component) ----
    early_b    = site_row.rev_early_sen_slope;
    recent_b   = site_row.rev_recent_sen_slope;

    % Reconstruct early/recent windows exactly like your Python
    [early_end, recent_start] = localTrajectoryWindows(years);

    % Early window data
    idxE = years <= early_end;
    yearsE = years(idxE); altE = alt(idxE);

    % Recent window data
    idxR = years >= recent_start;
    yearsR = years(idxR); altR = alt(idxR);

    % Build early/recent trend lines (anchored per-window)
    hEarly = gobjects(1);
    hRecent = gobjects(1);

    if numel(yearsE) >= 2 && isfinite(early_b)
        y0E = min(yearsE);
        b0E = median(altE - early_b .* (yearsE - y0E), 'omitnan');
        yrE = min(yearsE):max(yearsE);
        trE = b0E + early_b .* (yrE - y0E);
    else
        yrE = []; trE = [];
    end

    if numel(yearsR) >= 2 && isfinite(recent_b)
        y0R = min(yearsR);
        b0R = median(altR - recent_b .* (yearsR - y0R), 'omitnan');
        yrR = min(yearsR):max(yearsR);
        trR = b0R + recent_b .* (yrR - y0R);
    else
        yrR = []; trR = [];
    end

    % ---- Plot ----
    hold on;

    hObs = plot(years, alt, 'ko-', ...
        'MarkerFaceColor','w', 'MarkerEdgeColor','k', ...
        'LineWidth',2, 'MarkerSize',7);

    hTr = plot(yr, trend_line, 'r--', 'LineWidth', 1.8);

    % Plot early/recent trends only when Transitional (or reversal flag)
    showTraj = (cls == "Transitional") || (flag_rev == 1);
    if showTraj
        if ~isempty(yrE)
            hEarly = plot(yrE, trE, '-', 'LineWidth', 1.8); % color auto
        end
        if ~isempty(yrR)
            hRecent = plot(yrR, trR, '-', 'LineWidth', 1.8); % color auto
        end
        % Optional: show window split markers
        xline(early_end, ':', 'LineWidth', 1.2);
        xline(recent_start, ':', 'LineWidth', 1.2);
    end

    hold off;

    grid on; box on;
    set(gca, 'LineWidth', 1, 'FontSize', 10);
    ylabel('ALT (cm)', 'FontSize', 14, 'Interpreter', 'latex');

    % ---- Legend (latex-safe) ----
    legItems = [hObs hTr];
    legLabels = {
        sprintf('Observed ALT ($\\beta_{\\mathrm{min}}=%+.2f\\,\\mathrm{cm\\,yr^{-1}}$)\n', site_row.trend_min_meaningful_slope), ...
        sprintf('Theil--Sen trend: $\\beta_{\\mathrm{Sen}}=%+.2f\\,\\mathrm{cm\\,yr^{-1}}$ (%s) {  }  {  }  {  }  {  } ', slope, p_str)
    };

    if showTraj
        if isgraphics(hEarly)
            legItems(end+1) = hEarly; %#ok<AGROW>
            legLabels{end+1} = sprintf('Early trend: $\\beta_{\\mathrm{early}}=%+.2f\\,\\mathrm{cm\\,yr^{-1}}$', early_b);
        end
        if isgraphics(hRecent)
            legItems(end+1) = hRecent; %#ok<AGROW>
            legLabels{end+1} = sprintf('Recent trend: $\\beta_{\\mathrm{recent}}=%+.2f\\,\\mathrm{cm\\,yr^{-1}}$', recent_b);
        end
        legLabels{end+1} = 'Window boundaries';
        % Note: boundaries are xlines, not in legend; keep label minimal if you want.
    end

    legend(legItems, legLabels, 'Location','northoutside', 'Box','on', 'Interpreter','latex', 'FontSize',12);

    % ---- Title (latex) ----
    if lon < 0
        coord_str = sprintf('$%.2f^{\\circ}\\mathrm{N},\\;%.2f^{\\circ}\\mathrm{W}$', lat, abs(lon));
    else
        coord_str = sprintf('$%.2f^{\\circ}\\mathrm{N},\\;%.2f^{\\circ}\\mathrm{E}$', lat, lon);
    end

    reg = string(site_row.region);
    title_str = sprintf('(%s) Site: %s (%s)', panelLetter, coord_str, reg);
    title(title_str, 'FontSize', 14, 'Interpreter', 'latex');
    set(gca, 'TitleHorizontalAlignment', 'left');

    % ---- Annotation box (latex; keep lines short) ----
    conf_ch = char(conf);
    conf_first = upper(conf_ch(1));
    conf_rest  = conf_ch(2:end);

    line1 = sprintf('\\textbf{Confidence: } %s%s', conf_first, conf_rest);

    line2 = "";
    % if ~isnan(p_cls)
    %     line2 = sprintf('$P(\\mathrm{class})=%.2f$', p_cls);%\\textbf{Support: } 
    % end


    flags = strings(0,1);
    if flag_rev,  flags(end+1)="Reversal"; end
    if flag_lp,   flags(end+1)="Low power"; end

    line4 = "";
    if ~isempty(flags)
        line4 = sprintf("\\textbf{Flags }: %s", strjoin(flags, ", "));
    end


    ann = string.empty;
    for L = [string(line1) string(line2) string(line4)]
        if strlength(L) > 0
            ann(end+1) = L; %#ok<AGROW>
        end
    end
    if cls=="Rapid Thickening" || cls=="Gradual Thickening" || (cls=="No Trend" && panelLetter=='a') 
        text(0.02, 0.97, char(strjoin(ann, newline)), ...
            'Units','normalized', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','top', ...
            'FontSize', 10, ...
            'BackgroundColor','w', ...
            'EdgeColor',[0.3 0.3 0.3], ...
            'Margin', 3, ...
            'Interpreter','latex');
    else
        if cls=="Rapid Thinning" || (cls=="Gradual Thinning") || (cls=="No Trend" && panelLetter=='b') 
            text(0.97, 0.97, char(strjoin(ann, newline)), ...
                'Units','normalized', ...
                'HorizontalAlignment','right', ...
                'VerticalAlignment','top', ...
                'FontSize', 10, ...
                'BackgroundColor','w', ...
                'EdgeColor',[0.3 0.3 0.3], ...
                'Margin', 3, ...
                'Interpreter','latex');
        else
            text(0.98, 0.02, char(strjoin(ann, newline)), ...
                'Units','normalized', ...
                'HorizontalAlignment','right', ...
                'VerticalAlignment','bottom', ...
                'FontSize', 10, ...
                'BackgroundColor','w', ...
                'EdgeColor',[0.3 0.3 0.3], ...
                'Margin', 3, ...
                'Interpreter','latex');
        end
    end

end

%%================== Helpers ==================
function b = localAsBool(x)
% Robust conversion of table scalar to boolean 0/1 for:
%   - logical
%   - numeric
%   - string/char "True"/"False" (any case)
    if islogical(x)
        b = double(x);
        return;
    end
    if isnumeric(x)
        b = double(x ~= 0);
        return;
    end
    s = lower(string(x));
    b = double( (s == "true") | (s == "1") | (s == "t") );
end

function [early_end, recent_start] = localTrajectoryWindows(years)
% Reproduce your Python window logic:
%   span = y1 - y0 + 1
%   if span >= 14:
%       early_end = y0 + 6
%       recent_start = y1 - 6
%   else:
%       half = max(3, int(span/2) - 1)
%       half = min(half, int((span - 1)//2))
%       early_end = y0 + half
%       recent_start = y1 - half

    y0 = min(years);
    y1 = max(years);
    span = y1 - y0 + 1;

    if span >= 14
        early_end = y0 + 6;
        recent_start = y1 - 6;
    else
        half = max(3, floor(span/2) - 1);
        half = min(half, floor((span - 1) / 2));
        early_end = y0 + half;
        recent_start = y1 - half;
    end
end


function [result_table,region_counts,alt_by_region,cov_by_region] = preprocess_CALM(filePath, verbose)
%-----------------------------
% 1) Load raw table
% -----------------------------
opts = detectImportOptions(filePath);
raw_data = readtable(filePath, opts);

if verbose == 1
    fprintf('\n============================================================\n');
    fprintf('CALM ALT Max — EDA REPORT\n');
    fprintf('File: %s\n', filePath);
    fprintf('============================================================\n\n');

    fprintf('[RAW] Rows: %d | Cols: %d\n', height(raw_data), width(raw_data));
    fprintf('[RAW] Variables:\n');
    disp(raw_data.Properties.VariableNames);
end
% Remove fully empty rows (Excel artifacts)
raw_data = rmmissing(raw_data, 'MinNumMissing', width(raw_data));
if verbose == 1
    fprintf('[RAW] After removing fully empty rows: %d rows\n\n', height(raw_data));
end
%-----------------------------
% 2) Parse key columns robustly
% -----------------------------
% Lat/Long/Year
Lat  = raw_data.Lat;   if ~isnumeric(Lat),  Lat  = str2double(string(Lat));  end
Long = raw_data.Long;  if ~isnumeric(Long), Long = str2double(string(Long)); end
Year = raw_data.Year;  if ~isnumeric(Year), Year = str2double(string(Year)); end

% Max parsing + censored flag (>130 -> 135, NA -> NaN)
mx_raw = raw_data.Max;
s = strtrim(string(mx_raw));
is_censored_high = contains(s, ">130", 'IgnoreCase', true);

s = replace(s, ">130", "135");
s = replace(s, "NA", "");
s(s=="") = missing;

Max = str2double(s);

% Row-level missingness summary (before filtering)
n_missing_Lat  = sum(~isfinite(Lat));
n_missing_Long = sum(~isfinite(Long));
n_missing_Year = sum(~isfinite(Year));
n_missing_Max  = sum(~isfinite(Max));

if verbose == 1
    fprintf('[MISSING BEFORE FILTER]\n');
    fprintf('  Lat missing:  %d\n', n_missing_Lat);
    fprintf('  Long missing: %d\n', n_missing_Long);
    fprintf('  Year missing: %d\n', n_missing_Year);
    fprintf('  Max missing:  %d\n\n', n_missing_Max);
end
%-----------------------------
% 3) Filter invalid rows (Lat/Long/Year/Max must be finite)
% -----------------------------
ok = isfinite(Lat) & isfinite(Long) & isfinite(Year) & isfinite(Max);
Lat = Lat(ok); Long = Long(ok); Year = Year(ok); Max = Max(ok);
is_censored_high = is_censored_high(ok);

% Stable site_id from fixed precision (8 decimals)
site_id = string(num2str(Lat,'%.8f')) + "_" + string(num2str(Long,'%.8f'));

if verbose == 1
    fprintf('[CLEAN] After dropping rows with invalid Lat/Long/Year/Max: %d rows\n', numel(Max));
    fprintf('[CLEAN] Censored-high rows (">130" mapped to 135): %d (%.2f%%)\n\n', ...
        sum(is_censored_high), 100*mean(is_censored_high));
end
%-----------------------------
% 4) Deduplicate true duplicates by (site_id, Year)
%     - Keep first Lat/Long (should be identical within site_id)
%     - Mean Max if duplicates exist
% -----------------------------
G = findgroups(site_id, Year);

siteList = splitapply(@(x) x(1), site_id, G);
yearList = splitapply(@(x) x(1), Year,    G);
latFirst = splitapply(@(x) x(1), Lat,     G);
lonFirst = splitapply(@(x) x(1), Long,    G);
meanMax  = splitapply(@(x) mean(x,'omitnan'), Max, G);
censAny  = splitapply(@(x) any(x), is_censored_high, G);

df = table(siteList, yearList, meanMax, latFirst, lonFirst, censAny, ...
    'VariableNames', {'site_id','Year','Max','Lat','Long','is_censored_high'});

% Duplicate count diagnostics (raw vs dedup)
raw_pairs = string(site_id) + "_" + string(Year);
n_raw = numel(raw_pairs);
n_unique_pairs = numel(unique(raw_pairs));
n_dupes = n_raw - n_unique_pairs;

if verbose == 1
    fprintf('[DEDUP]\n');
    fprintf('  Raw rows (after basic clean): %d\n', n_raw);
    fprintf('  Unique (site,year) pairs:     %d\n', n_unique_pairs);
    fprintf('  Duplicate rows removed:       %d\n\n', n_dupes);
end
[CALMlocations,~,~] = unique([df.Lat df.Long],"rows");
if verbose == 1
    fprintf('[SITES] Unique CALM locations (after dedup): %d\n', size(CALMlocations,1));
end

%-----------------------------
% 5) Arctic filter (>=60N)
% -----------------------------
df = df(df.Lat >= 60.0, :);
[CALMlocations,~,~] = unique([df.Lat df.Long],"rows");
if verbose == 1
    fprintf('[FILTER] Unique locations after Arctic filter (Lat>=60): %d\n', size(CALMlocations,1));
end

%-----------------------------
% 6) Keep sites with >= 8 unique years
% -----------------------------
G2 = findgroups(df.site_id);
years_per_site = splitapply(@(x) numel(unique(x)), df.Year, G2);
sites = splitapply(@(x) x(1), df.site_id, G2);

keep_mask = years_per_site >= 8;
valid_sites = sites(keep_mask);

df = df(ismember(df.site_id, valid_sites), :);
[CALMlocations,~,ic] = unique([df.Lat df.Long],"rows");
if verbose == 1
    fprintf('[FILTER] Unique locations after years_per_site>=8: %d\n\n', size(CALMlocations,1));
end
%-----------------------------
% 7) Assign MainRegion (requires assign_CALM_regions)
% -----------------------------
df.MainRegion = strings(height(df),1);

for i = 1:size(CALMlocations,1)
    tmp = df(i==ic,:);
    % assign_CALM_regions must return a region string
    df.MainRegion(i==ic) = strings(height(tmp),1) + assign_CALM_regions(tmp.Lat(1), tmp.Long(1));
end

result_table = df;
result_table.MainRegion = categorical(strtrim(string(result_table.MainRegion)));

%-----------------------------
% 8) Per-site coverage metrics
%     - NumberYears: number of observed unique years per site
%     - YearSpan: maxYear-minYear+1 per site
%     - MissingYearsInSpan: YearSpan - NumberYears
%     - Completeness: NumberYears / YearSpan
% -----------------------------
G3 = findgroups(result_table.site_id);

site_lat  = splitapply(@(x) x(1), result_table.Lat,  G3);
site_lon  = splitapply(@(x) x(1), result_table.Long, G3);
site_reg  = splitapply(@(x) x(1), string(result_table.MainRegion), G3);

site_years = splitapply(@(x) numel(unique(x)), result_table.Year, G3);
site_ymin  = splitapply(@(x) min(x), result_table.Year, G3);
site_ymax  = splitapply(@(x) max(x), result_table.Year, G3);

site_span = site_ymax - site_ymin + 1;
site_missing_in_span = site_span - site_years;
site_completeness = site_years ./ site_span;

site_summary = table( ...
    splitapply(@(x) x(1), result_table.site_id, G3), ...
    site_lat, site_lon, categorical(site_reg), site_years, site_ymin, site_ymax, ...
    site_span, site_missing_in_span, site_completeness, ...
    'VariableNames', {'site_id','Lat','Long','MainRegion','NumberYears','MinYear','MaxYear', ...
                      'YearSpan','MissingYearsInSpan','Completeness'} );

% Overall temporal range
all_min_year = min(result_table.Year);
all_max_year = max(result_table.Year);

if verbose == 1
    fprintf('[TEMPORAL COVERAGE]\n');
    fprintf('  Overall year range: %d–%d\n', all_min_year, all_max_year);
    fprintf('  Sites (after filters): %d\n', height(site_summary));
    fprintf('  Median observed years/site: %.0f\n', median(site_summary.NumberYears));
    fprintf('  Median missing years in span/site: %.0f\n', median(site_summary.MissingYearsInSpan));
    fprintf('  Median completeness (years/span): %.2f\n\n', median(site_summary.Completeness));
end

%-----------------------------
% 9) Summary stats by region
% -----------------------------
% Basic by-region counts
[region, NumSites] = groupcounts(site_summary.MainRegion);
region_counts = table(region, NumSites, 'VariableNames', {'MainRegion','NumSites'});

% ALT Max stats by region (row-level)
alt_by_region = groupsummary(result_table, "MainRegion", ...
    ["mean","median","min","max","std"], "Max");

% Coverage stats by region (site-level)
cov_by_region = groupsummary(site_summary, "MainRegion", ...
    ["mean","median","min","max"], ["NumberYears","MissingYearsInSpan","Completeness"]);

if verbose == 1
    fprintf('[REGION SUMMARY]\n');
    disp(region_counts);

    fprintf('[ALT MAX STATS BY REGION] (row-level)\n');
    disp(alt_by_region);

    fprintf('[COVERAGE STATS BY REGION] (site-level)\n');
    disp(cov_by_region);
%-----------------------------
% 10) Global distribution stats
% -----------------------------
    fprintf('[ALT MAX — GLOBAL DISTRIBUTION] (row-level)\n');
    fprintf('  N rows: %d\n', height(result_table));
    fprintf('  Mean:   %.3f\n', mean(result_table.Max));
    fprintf('  Std:    %.3f\n', std(result_table.Max));
    fprintf('  Min:    %.3f\n', min(result_table.Max));
    fprintf('  P25:    %.3f\n', prctile(result_table.Max,25));
    fprintf('  Median: %.3f\n', median(result_table.Max));
    fprintf('  P75:    %.3f\n', prctile(result_table.Max,75));
    fprintf('  Max:    %.3f\n\n', max(result_table.Max));
    
    fprintf('\nDone.\n');
end
end