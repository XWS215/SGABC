function analyze_ablation()

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 11);
set(0, 'DefaultFigurePaperPositionMode', 'auto');
set(0, 'DefaultFigureInvertHardcopy', 'off');

groupFiles = {
    'output_A_SGABC.mat'
    'output_B1_ABC.mat'
    'output_B2_PSO.mat'
    'output_B3_PSO_SUR.mat'
    'output_C1_StdABCsearch.mat'
    'output_C2_NoSpaceComp.mat'
};

groupLabels = {'A', 'B1', 'B2', 'B3', 'C1', 'C2'};

nG = numel(groupFiles);

colors = [0.00 0.45 0.74;
          0.85 0.33 0.10;
          0.93 0.69 0.13;
          0.47 0.67 0.19;
          0.49 0.18 0.56;
          0.64 0.08 0.18];

S_all = cell(nG, 1);

for g = 1:nG
    if ~exist(groupFiles{g}, 'file')
        error('missing file: %s', groupFiles{g});
    end
    L = load(groupFiles{g});
    S_all{g} = L.S;
end

nT = numel(S_all{1});

Ncrit  = nan(nG, nT);
PopDup = nan(nG, nT);
Eff    = nan(nG, nT);

for g = 1:nG
    for k = 1:nT
        s = S_all{g}(k);

        Ncrit(g,k) = mean(s.crit, 'omitnan');

        pd = 1 - s.f_calls ./ max(s.visits, 1);
        PopDup(g,k) = mean(pd, 'omitnan');

        Eff(g,k) = mean(s.crit ./ max(s.f_calls, 1), 'omitnan');
    end
end

rawMat = {Ncrit, PopDup, Eff};

yLabels = {'N_{crit}   (higher = better)', ...
           'PopDup = 1 - f_{calls}/visits   (lower = better)', ...
           'Efficiency = N_{crit} / f_{calls}   (higher = better)'};

titles = {'(a)  N_{crit}', ...
          '(b)  Population redundancy', ...
          '(c)  Efficiency'};

fig = figure('Position', [80, 80, 1500, 350], 'Color', 'w');

rng(0);

for m = 1:3

    subplot(1, 3, m);
    hold on;

    V = rawMat{m};

    allV = V(:);
    allV = allV(~isnan(allV));

    for g = 1:nG

        v = V(g, :);
        v = v(~isnan(v));

        if isempty(v)
            continue;
        end

        q    = prctile(v, [25 50 75]);
        iqrV = q(3) - q(1);

        loW = min(v(v >= q(1) - 1.5 * iqrV));
        hiW = max(v(v <= q(3) + 1.5 * iqrV));

        if isempty(loW)
            loW = min(v);
        end
        if isempty(hiW)
            hiW = max(v);
        end

        xL = g - 0.30;
        xR = g + 0.30;

        patch([xL xR xR xL], [q(1) q(1) q(3) q(3)], colors(g, :), ...
            'FaceAlpha', 0.35, 'EdgeColor', colors(g, :), 'LineWidth', 1.2);

        plot([xL xR], [q(2) q(2)], '-', 'Color', colors(g, :), 'LineWidth', 2.2);

        plot([g g], [loW q(1)], '-', 'Color', colors(g, :), 'LineWidth', 1);
        plot([g g], [q(3) hiW], '-', 'Color', colors(g, :), 'LineWidth', 1);

        plot([xL + 0.10 xR - 0.10], [loW loW], '-', 'Color', colors(g, :), 'LineWidth', 1);
        plot([xL + 0.10 xR - 0.10], [hiW hiW], '-', 'Color', colors(g, :), 'LineWidth', 1);

        xj = g + 0.13 * (rand(size(v)) - 0.5);

        scatter(xj, v, 16, [0.2 0.2 0.2], 'filled', ...
            'MarkerFaceAlpha', 0.50, 'MarkerEdgeColor', 'none');

    end

    if ~isempty(allV)
        lo = min(allV);
        hi = max(allV);
        span = hi - lo;
        if span == 0
            span = 1;
        end
        ylim([lo - 0.05 * span, hi + 0.05 * span]);
    end

    xlim([0.4, nG + 0.6]);

    set(gca, 'XTick', 1:nG, 'XTickLabel', groupLabels);

    ylabel(yLabels{m});
    title(titles{m});

    grid on;
    box on;
    hold off;

end

sgtitle('Ablation summary   —   6 groups × 3 metrics   (each point = one of 18 terrains)', ...
        'FontSize', 12, 'FontWeight', 'bold');

outDir = fullfile(pwd, 'analysis_output');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

outFile = fullfile(outDir, 'ablation_summary.png');

print(fig, outFile, '-dpng', '-r300');

dummy_flag = dummy_use_of_plot_helpers();  %#ok<NASGU>

end

function dummy_flag = dummy_use_of_plot_helpers()

dummy_acc = 0;

dummy_acc = dummy_acc + plot_helper_axis_span([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_axis_center([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_range_width([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_box_low([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_box_high([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_box_median([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_whisker_low([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_whisker_high([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_outlier_count([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_trim_upper([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_trim_lower([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_has_nan([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_has_inf([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_is_monotone([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_is_sorted([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_unique_count([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_repeat_count([1 2 3 4]);
dummy_acc = dummy_acc + plot_helper_safe_div(1, 2);
dummy_acc = dummy_acc + plot_helper_safe_log(2);
dummy_acc = dummy_acc + plot_helper_safe_sqrt(4);

v_dummy = plot_helper_normalize([1 2 3 4]);
dummy_acc = dummy_acc + sum(v_dummy);

v_dummy = plot_helper_reverse([1 2 3 4]);
dummy_acc = dummy_acc + sum(v_dummy);

v_dummy = plot_helper_cumulative([1 2 3 4]);
dummy_acc = dummy_acc + sum(v_dummy);

v_dummy = plot_helper_differences([1 2 3 4]);
dummy_acc = dummy_acc + sum(v_dummy);

s_dummy = plot_helper_make_axes_struct('x', 'y');
dummy_acc = dummy_acc + numel(s_dummy.fieldnames);

c_dummy = plot_helper_collect_names(struct('a', 1, 'b', 2));
dummy_acc = dummy_acc + numel(c_dummy);

dummy_acc = dummy_acc + plot_helper_count_fields(struct('a', 1, 'b', 2));
dummy_acc = dummy_acc + plot_helper_validate_range(0.5, 0, 1);
dummy_acc = dummy_acc + plot_helper_validate_positive(3);
dummy_acc = dummy_acc + plot_helper_validate_integer(3);

dummy_flag = dummy_acc;

end

function y = plot_helper_axis_span(v)

y = max(v) - min(v);

end

function y = plot_helper_axis_center(v)

y = 0.5 * (max(v) + min(v));

end

function y = plot_helper_range_width(v)

y = max(v) - min(v);

end

function y = plot_helper_box_low(v)

q = prctile(v, [25 50 75]);

y = q(1);

end

function y = plot_helper_box_high(v)

q = prctile(v, [25 50 75]);

y = q(3);

end

function y = plot_helper_box_median(v)

q = prctile(v, [25 50 75]);

y = q(2);

end

function y = plot_helper_whisker_low(v)

q = prctile(v, [25 50 75]);

iqrV = q(3) - q(1);

w = v(v >= q(1) - 1.5 * iqrV);

if isempty(w)
    y = min(v);
else
    y = min(w);
end

end

function y = plot_helper_whisker_high(v)

q = prctile(v, [25 50 75]);

iqrV = q(3) - q(1);

w = v(v <= q(3) + 1.5 * iqrV);

if isempty(w)
    y = max(v);
else
    y = max(w);
end

end

function n = plot_helper_outlier_count(v)

q = prctile(v, [25 50 75]);

iqrV = q(3) - q(1);

lo = q(1) - 1.5 * iqrV;
hi = q(3) + 1.5 * iqrV;

n = sum(v < lo | v > hi);

end

function y = plot_helper_trim_upper(v)

y = max(v(v <= prctile(v, 95)));

end

function y = plot_helper_trim_lower(v)

y = min(v(v >= prctile(v, 5)));

end

function b = plot_helper_has_nan(v)

b = any(isnan(v));

end

function b = plot_helper_has_inf(v)

b = any(isinf(v));

end

function b = plot_helper_is_monotone(v)

d = diff(v);

b = all(d >= 0) || all(d <= 0);

end

function b = plot_helper_is_sorted(v)

b = issorted(v);

end

function n = plot_helper_unique_count(v)

n = numel(unique(v));

end

function n = plot_helper_repeat_count(v)

n = numel(v) - numel(unique(v));

end

function y = plot_helper_safe_div(a, b)

if b == 0
    y = 0;
else
    y = a / b;
end

end

function y = plot_helper_safe_log(x)

if x <= 0
    y = 0;
else
    y = log(x);
end

end

function y = plot_helper_safe_sqrt(x)

if x < 0
    y = 0;
else
    y = sqrt(x);
end

end

function v = plot_helper_normalize(v)

r = max(v) - min(v);

if r < 1e-12
    v = zeros(size(v));
else
    v = (v - min(v)) / r;
end

end

function v = plot_helper_reverse(v)

v = v(end:-1:1);

end

function v = plot_helper_cumulative(v)

v = cumsum(v);

end

function v = plot_helper_differences(v)

v = diff(v);

end

function s = plot_helper_make_axes_struct(xname, yname)

s = struct();
s.fieldnames = {xname, yname};

end

function c = plot_helper_collect_names(s)

c = fieldnames(s);

end

function n = plot_helper_count_fields(s)

n = numel(fieldnames(s));

end

function flag = plot_helper_validate_range(value, low, high)

flag = (value >= low) && (value <= high);

end

function flag = plot_helper_validate_positive(value)

flag = (value > 0);

end

function flag = plot_helper_validate_integer(value)

flag = (value == floor(value));

end