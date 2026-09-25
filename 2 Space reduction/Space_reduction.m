function sampling_dynamic_focus()

set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName',  'Times New Roman');
set(0, 'DefaultAxesFontSize', 10);
set(0, 'DefaultFigureColor', 'w');
set(0, 'DefaultAxesFontAngle', 'normal');
set(0, 'DefaultTextFontAngle', 'normal');
try
    set(0, 'DefaultAxesXTickLabelRotation', 0);
    set(0, 'DefaultAxesYTickLabelRotation', 0);
catch
end

FIG_POS   = [100, 100, 1500, 500];
FS_TITLE  = 11;
FS_LABEL  = 10;
FS_TICK   = 9;
FS_LEGEND = 8;

outdir = fullfile(pwd, 'output');
if ~exist(outdir, 'dir'), mkdir(outdir); end

LB = -5.12;  UB = 5.12;
Rthr = 0.5;
N = 400;

opts = struct();
opts.nRounds     = 8;
opts.GS          = 0.1;
opts.gMax        = 0.2;
opts.k0          = 2.0;
opts.kDecay      = 0.5;
opts.minPts      = 4;
opts.minVol      = 1e-6;
opts.exploreFrac = 0.30;

COL = struct();
COL.rand    = [0.45, 0.45, 0.45];
COL.lhs     = [0.20, 0.45, 0.70];
COL.dynamic = [0.80, 0.25, 0.20];
COL.noncrit = [0.82, 0.82, 0.82];

colors = { COL.rand, COL.lhs, COL.dynamic };
titles = {'Random', 'Static LHS', 'Dynamic Focus'};

rng(42);
res2 = run_experiment(2, N, LB, UB, Rthr, opts);
res3 = run_experiment(3, N, LB, UB, Rthr, opts);

% ================= Figure 1: 2D =================
figure('Position', FIG_POS, 'Color', 'w');
methods2 = {res2.X_rand, res2.X_lhs, res2.X_dyn};
for mi = 1:3
    subplot(2, 3, mi);
    plot_scatter_2d(res2.wfun, methods2{mi}, LB, UB, Rthr, colors{mi}, COL.noncrit, FS_LABEL, FS_TICK);
    hit = 100 * mean(res2.wfun(methods2{mi}) > Rthr);
    title(sprintf('%s  (hit %.1f%%)', titles{mi}, hit), ...
        'FontSize', FS_TITLE, 'FontWeight', 'bold', 'FontAngle', 'normal');
end
plot_stats_panel(res2, colors, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, N);
sgtitle(sprintf('D = 2  —  region = %s', res2.wfun_name), ...
    'FontSize', FS_TITLE + 2, 'FontWeight', 'bold', ...
    'FontAngle', 'normal', 'FontName', 'Times New Roman');
save_figure(outdir, 'fig1_2d_stats');

% ================= Figure 2: 3D =================
figure('Position', FIG_POS, 'Color', 'w');
methods3 = {res3.X_rand, res3.X_lhs, res3.X_dyn};
for mi = 1:3
    subplot(2, 3, mi);
    plot_scatter_3d(res3.wfun, methods3{mi}, LB, UB, Rthr, colors{mi}, COL.noncrit, FS_LABEL, FS_TICK);
    hit = 100 * mean(res3.wfun(methods3{mi}) > Rthr);
    title(sprintf('%s  (hit %.1f%%)', titles{mi}, hit), ...
        'FontSize', FS_TITLE, 'FontWeight', 'bold', 'FontAngle', 'normal');
end
plot_stats_panel(res3, colors, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, N);
sgtitle(sprintf('D = 3  —  region = %s', res3.wfun_name), ...
    'FontSize', FS_TITLE + 2, 'FontWeight', 'bold', ...
    'FontAngle', 'normal', 'FontName', 'Times New Roman');
save_figure(outdir, 'fig2_3d_stats');

fprintf('\n--- Summary ---\n');
fprintf('D = 2  |  Random %.1f%%  LHS %.1f%%  Dynamic %.1f%%\n', ...
    100*res2.hit_r(end), 100*res2.hit_l(end), 100*res2.hit_d(end));
fprintf('D = 3  |  Random %.1f%%  LHS %.1f%%  Dynamic %.1f%%\n', ...
    100*res3.hit_r(end), 100*res3.hit_l(end), 100*res3.hit_d(end));
end


function res = run_experiment(D, N, LB, UB, Rthr, opts)
rng(42);

[wfun, wfun_name] = make_importance(D);

X_rand = sample_random(N, D, LB, UB);
X_lhs  = sample_lhs(N, D, LB, UB);
[X_dyn, hist] = sample_dynamic_sgabc(N, D, LB, UB, wfun, Rthr, opts);

N_list = round(logspace(1.5, 3.2, 10));
n_trials = 3;
hit_r = zeros(size(N_list));
hit_l = zeros(size(N_list));
hit_d = zeros(size(N_list));
for i = 1:numel(N_list)
    Ni = N_list(i);
    ar = 0; al = 0; ad = 0;
    for t = 1:n_trials
        rng(100+t);
        ar = ar + mean(wfun(sample_random(Ni, D, LB, UB)) > Rthr);
        al = al + mean(wfun(sample_lhs(Ni, D, LB, UB)) > Rthr);
        [Xd, ~] = sample_dynamic_sgabc(Ni, D, LB, UB, wfun, Rthr, opts);
        ad = ad + mean(wfun(Xd) > Rthr);
    end
    hit_r(i) = ar / n_trials;
    hit_l(i) = al / n_trials;
    hit_d(i) = ad / n_trials;
end

res.X_rand    = X_rand;
res.X_lhs     = X_lhs;
res.X_dyn     = X_dyn;
res.hist      = hist;
res.wfun      = wfun;
res.wfun_name = wfun_name;
res.N_list    = N_list;
res.hit_r     = hit_r;
res.hit_l     = hit_l;
res.hit_d     = hit_d;
end


function [wfun, name_out] = make_importance(D)
c1 = [-2, zeros(1, D-1)];
c2 = [ 2, zeros(1, D-1)];
sig = 1.5;
wfun = @(X) max(exp(-sum((X - c1).^2, 2) / (2*sig^2)), ...
                exp(-sum((X - c2).^2, 2) / (2*sig^2)));
name_out = 'Two Gaussian Balls';
end


function X = sample_random(N, D, LB, UB)
X = LB + rand(N, D) .* (UB - LB);
end


function X = sample_lhs(N, D, LB, UB)
X = zeros(N, D);
for d = 1:D
    u = (rand(N, 1) + (0:N-1)') / N;
    u = u(randperm(N));
    X(:, d) = LB + u * (UB - LB);
end
end


function [X, hist] = sample_dynamic_sgabc(N, D, LB, UB, wfun, Rthr, opts)
nRounds      = opts.nRounds;
GS           = opts.GS;
gMax         = opts.gMax;
minPts       = opts.minPts;
minVol       = opts.minVol;
exploreFrac  = opts.exploreFrac;

batchSize = ceil(N / nRounds);
segs = cell(1, D);
for d = 1:D
    segs{d} = [LB, UB];
end
k = opts.k0;

X       = zeros(0, D);
allCrit = zeros(0, D);

hist.segs      = {};
hist.vol       = [];
hist.nsegs     = [];
hist.critCount = [];
hist.X         = {};

for r = 1:nRounds
    if size(allCrit, 1) < minPts
        Xr = sample_random(batchSize, D, LB, UB);
    else
        n_exp = round(exploreFrac * batchSize);
        n_box = batchSize - n_exp;
        Xr_box = sample_from_segs(segs, n_box, D, LB, UB);
        Xr_exp = sample_random(n_exp, D, LB, UB);
        Xr = [Xr_box; Xr_exp];
    end

    yr = wfun(Xr);
    X = [X; Xr];
    allCrit = [allCrit; Xr(yr >= Rthr, :)];

    if size(allCrit, 1) >= minPts
        [segs, k] = update_segs(segs, allCrit, LB, UB, k, opts.kDecay, ...
            GS, gMax, minVol, D);
    end

    hist.segs{end+1}      = segs;
    hist.vol(end+1)       = compute_seg_volume(segs, LB, UB);
    hist.nsegs(end+1)     = count_segs(segs);
    hist.critCount(end+1) = size(allCrit, 1);
    hist.X{end+1}         = Xr;
end

if size(X, 1) < N
    Xextra = sample_from_segs(segs, N - size(X,1), D, LB, UB);
    X = [X; Xextra];
end
X = X(1:N, :);
end


function X = sample_from_segs(segs, n, D, LB, UB)
X = zeros(n, D);
for d = 1:D
    s = segs{d};
    if isempty(s)
        X(:, d) = LB + rand(n, 1) * (UB - LB);
    else
        pk = randi(size(s, 1), n, 1);
        lo = s(pk, 1);
        hi = s(pk, 2);
        X(:, d) = lo + rand(n, 1) .* (hi - lo);
    end
end
X = max(LB, min(UB, X));
end


function [segs_new, k_new] = update_segs(segs, pts, LB, UB, k, kDecay, ...
    GS, gMax, minVol, D)
segs_new = cell(1, D);
for d = 1:D
    col = pts(:, d);
    mu = mean(col);
    sg = std(col);
    g_raw = k * sg;
    g = min(max(g_raw, GS), gMax);

    idx = unique(floor((col - mu) / g));
    lo = max(mu + idx * g, LB);
    hi = min(mu + (idx + 1) * g, UB);

    lo = LB + GS * floor((lo - LB) / GS);
    hi = LB + GS * ceil((hi - LB) / GS);
    hi = min(hi, UB);

    seg_raw = [lo(:), hi(:)];
    keep = seg_raw(1, :);
    for ii = 2:size(seg_raw, 1)
        if seg_raw(ii, 1) <= keep(end, 2) + 1e-9
            keep(end, 2) = max(keep(end, 2), seg_raw(ii, 2));
        else
            keep(end+1, :) = seg_raw(ii, :);
        end
    end

    keep = intersect_segs(segs{d}, keep);
    if isempty(keep)
        keep = segs{d};
    end

    segs_new{d} = keep;
end

vol = compute_seg_volume(segs_new, LB, UB);
if vol < minVol
    segs_new = segs;
end
k_new = k * kDecay;
end


function out = intersect_segs(A, B)
out = zeros(0, 2);
if isempty(A) || isempty(B)
    return;
end
for i = 1:size(A, 1)
    for j = 1:size(B, 1)
        lo = max(A(i, 1), B(j, 1));
        hi = min(A(i, 2), B(j, 2));
        if hi > lo + 1e-9
            out(end+1, :) = [lo, hi];
        end
    end
end
end


function v = compute_seg_volume(segs, LB, UB)
v = 1;
for d = 1:numel(segs)
    s = segs{d};
    len = 0;
    for i = 1:size(s, 1)
        len = len + (s(i, 2) - s(i, 1));
    end
    v = v * len / (UB - LB);
end
end


function n = count_segs(segs)
n = 0;
for d = 1:numel(segs)
    n = n + size(segs{d}, 1);
end
end


function plot_stats_panel(res, colors, FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND, N)

subplot(2, 3, 4);
hold on;
plot(res.N_list, 100*res.hit_r, 'o-', 'Color', colors{1}, 'LineWidth', 2, ...
    'MarkerFaceColor', colors{1}, 'MarkerSize', 5, 'DisplayName', 'Random');
plot(res.N_list, 100*res.hit_l, 's-', 'Color', colors{2}, 'LineWidth', 2, ...
    'MarkerFaceColor', colors{2}, 'MarkerSize', 5, 'DisplayName', 'LHS');
plot(res.N_list, 100*res.hit_d, '^-', 'Color', colors{3}, 'LineWidth', 2, ...
    'MarkerFaceColor', colors{3}, 'MarkerSize', 5, 'DisplayName', 'Dynamic');
set(gca, 'XScale', 'log', 'FontSize', FS_TICK, 'FontAngle', 'normal');
xlabel('Number of samples N', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
ylabel('Hit rate (%)', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
title('Hit rate vs N', 'FontSize', FS_TITLE, 'FontWeight', 'bold', 'FontAngle', 'normal');
legend('Location', 'east', 'FontSize', FS_LEGEND);
grid on; box on; xtickangle(0); hold off;

subplot(2, 3, 5);
hold on;
plot(1:numel(res.hist.vol), res.hist.vol, 'o-', 'Color', colors{3}, ...
    'LineWidth', 2, 'MarkerFaceColor', colors{3}, 'MarkerSize', 5);
set(gca, 'YScale', 'log', 'FontSize', FS_TICK, 'FontAngle', 'normal');
xlabel('Round', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
ylabel('Retained volume fraction', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
title('Retained volume', 'FontSize', FS_TITLE, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0); hold off;

subplot(2, 3, 6);
bar_data = 100 * [res.hit_r(end), res.hit_l(end), res.hit_d(end)];
b = bar(bar_data, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.6);
b.CData(1,:) = colors{1};
b.CData(2,:) = colors{2};
b.CData(3,:) = colors{3};
set(gca, 'XTickLabel', {'Random', 'LHS', 'Dynamic'}, ...
    'FontSize', FS_TICK, 'FontAngle', 'normal');
ylabel('Hit rate (%)', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
title(sprintf('Final hit rate (N = %d)', N), ...
    'FontSize', FS_TITLE, 'FontWeight', 'bold', 'FontAngle', 'normal');
grid on; box on; xtickangle(0);
for i = 1:3
    text(i, bar_data(i) + 1, sprintf('%.1f%%', bar_data(i)), ...
        'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
        'FontSize', FS_TICK, 'FontAngle', 'normal');
end
end


function plot_scatter_2d(wfun, X, LB, UB, Rthr, colorValue, nonColor, FS_LABEL, FS_TICK)
G = 120;
g = linspace(LB, UB, G);
[Xg, Yg] = meshgrid(g, g);
Zg = zeros(G, G);
for i = 1:G
    for j = 1:G
        pt = [Xg(i, j), Yg(i, j)];
        Zg(i, j) = wfun(pt);
    end
end
contour(Xg, Yg, Zg, [Rthr Rthr], 'r-', 'LineWidth', 2);
hold on;

isc = wfun(X) >= Rthr;
scatter(X(~isc,1), X(~isc,2), 22, nonColor, ...
    'filled', 'MarkerFaceAlpha', 0.55, ...
    'MarkerEdgeColor', [0.65 0.65 0.65], 'LineWidth', 0.2);
scatter(X(isc,1), X(isc,2), 45, colorValue, ...
    'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.45, ...
    'MarkerFaceAlpha', 0.9);

axis([LB UB LB UB]); axis square;
xlabel('x_1', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
ylabel('x_2', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
set(gca, 'FontSize', FS_TICK, 'FontAngle', 'normal', 'XTickLabelRotation', 0);
grid on; box on; xtickangle(0);
hold off;
end


function plot_scatter_3d(wfun, X, LB, UB, Rthr, colorValue, nonColor, FS_LABEL, FS_TICK)
G = 28;
g = linspace(LB, UB, G);
[Xg, Yg, Zg] = meshgrid(g, g, g);
pts = [Xg(:), Yg(:), Zg(:)];
wv = wfun(pts);
isCrit = wv >= Rthr;
critPts = pts(isCrit, :);

scatter3(critPts(:,1), critPts(:,2), critPts(:,3), ...
    14, [1.00 0.30 0.30], 'filled', ...
    'MarkerFaceAlpha', 0.06, 'MarkerEdgeColor', 'none');
hold on;

yr = wfun(X);
isHit = yr >= Rthr;
scatter3(X(~isHit,1), X(~isHit,2), X(~isHit,3), ...
    14, nonColor, 'filled', ...
    'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor', 'none');
scatter3(X(isHit,1), X(isHit,2), X(isHit,3), ...
    35, colorValue, 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 0.4, ...
    'MarkerFaceAlpha', 0.9);

axis([LB UB LB UB LB UB]);
axis square;
xlabel('x_1', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
ylabel('x_2', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
zlabel('x_3', 'FontSize', FS_LABEL, 'FontAngle', 'normal');
set(gca, 'FontSize', FS_TICK, 'FontAngle', 'normal', ...
    'XTickLabelRotation', 0, 'YTickLabelRotation', 0);
grid on; box on;
view(-37.5, 22);
hold off;
end


function save_figure(outdir, basename)
set(gcf, 'Color', 'w');
set(gcf, 'InvertHardcopy', 'off');
set(gcf, 'PaperPositionMode', 'auto');
print(gcf, fullfile(outdir, [basename '.png']), '-dpng', '-r300');
fprintf('Saved: %s.png\n', basename);
end