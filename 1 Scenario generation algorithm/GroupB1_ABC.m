function GroupB1_ABC(max_gen, SN, names, seeds)

if nargin < 1 || isempty(max_gen), max_gen = 200; end
if nargin < 2 || isempty(SN),      SN      = 25;  end
if nargin < 3 || isempty(names),   names   = suite_names('full'); end
if nargin < 4 || isempty(seeds),   seeds   = 200; end

p = struct();
p.SN = SN;
p.LIMIT_factor = 1.0;
p.F_MAX = 0.5;
p.F_MIN = 0.1;

result_file = fullfile(pwd, 'output_B1_ABC.mat');

FIELDS = {'crit', ...
          'crit_hits', ...
          'f_calls', ...
          'n_pred', ...
          'visits', ...
          'redun', ...
          'n_scout', ...
          'generations'};

nN = numel(names);  nS = numel(seeds);
Dvec = zeros(1, nN);  S = struct();
T_all = tic;
for k = 1:nN
    [land, D] = build_landscape(names{k});
    Dvec(k) = D;
    S(k).name      = names{k};
    S(k).D         = D;
    S(k).thresh    = land.thresh;
    S(k).crit_frac = land.crit_frac;
    for f = 1:numel(FIELDS)
        S(k).(FIELDS{f}) = zeros(1, nS);
    end
    for si = 1:nS
        r = abc_run(land, seeds(si), max_gen, D, p);
        for f = 1:numel(FIELDS)
            fn = FIELDS{f};
            if isfield(r, fn)
                v = r.(fn);
                if islogical(v), v = double(v); end
                if isempty(v),   v = NaN;       end
                S(k).(fn)(si) = v;
            end
        end
    end
end
T_total = toc(T_all);

meta = struct();
meta.group          = 'B1_ABC';
meta.names          = names;
meta.seeds          = seeds;
meta.Dvec           = Dvec;
meta.max_gen        = max_gen;
meta.SN             = SN;
meta.p              = p;
meta.FIELDS         = FIELDS;
meta.n_datasets     = nN;
meta.n_seeds        = nS;
meta.timestamp      = datestr(now, 'yyyy-mm-dd HH:MM:SS');
meta.matlab_version = version('-release');
meta.total_time_sec = T_total;

save(result_file, 'S', 'meta', '-v7.3');

dummy_flag = dummy_use_of_helpers();  %#ok<NASGU>

end

function ev = ev_new(land, D, LB, UB)
if nargin < 3 || isempty(LB), LB = -5.12; end
if nargin < 4 || isempty(UB), UB =  5.12; end
ev.land  = land;
ev.D     = D;
ev.LB    = LB;
ev.UB    = UB;
ev.GS    = 0.25;
ev.visits    = 0;
ev.f_calls   = 0;
ev.crit_hits = 0;
ev.n_cells   = 0;
ev.critical_pts = zeros(0, D);
ev.cache       = containers.Map('KeyType','char','ValueType','double');
ev.seen        = containers.Map('KeyType','char','ValueType','logical');
ev.crit_cells  = containers.Map('KeyType','char','ValueType','logical');
end

function [ev, c] = ev_eval(ev, x)
ev.visits = ev.visits + 1;
k = key_of(x, ev);
if isKey(ev.cache, k)
    c = ev.cache(k);
else
    ev.f_calls = ev.f_calls + 1;
    c = ev.land.fn(x(:)');
    c = c(1);
    ev.cache(k) = c;
end
hit = c <= ev.land.thresh;
if ~isKey(ev.seen, k)
    ev.seen(k) = true;
    if hit
        ev.crit_cells(k) = true;
        ev.n_cells = ev.n_cells + 1;
        ev.critical_pts(end+1, :) = snap_pt(x, ev.LB, ev.UB);
    end
end
if hit
    ev.crit_hits = ev.crit_hits + 1;
end
end

function s = ev_report(ev)
s = struct();
s.crit      = ev.n_cells;
s.crit_hits = ev.crit_hits;
s.visits    = ev.visits;
s.f_calls   = ev.f_calls;
if ev.crit_hits == 0
    s.redun = NaN;
else
    s.redun = 1.0 - ev.n_cells / ev.crit_hits;
end
end

function x = snap_pt(x, LB, UB)
GS = 0.25;
N = round((UB - LB) / GS);
x = x(:)';
idx = round((x - LB) / GS);
idx = max(0, min(N, idx));
x = LB + idx * GS;
end

function X = snap_mat(X, LB, UB)
GS = 0.25;
N = round((UB - LB) / GS);
idx = round((X - LB) / GS);
idx = max(0, min(N, idx));
X = LB + idx * GS;
end

function k = key_of(x, ev)
N = round((ev.UB - ev.LB) / ev.GS);
x = x(:)';
idx = round((x - ev.LB) / ev.GS);
idx = max(0, min(N, idx));
k = sprintf('%d_', idx);
end

function r = abc_run(land, seed, max_gen, D, p)
LB = -5.12;  UB = 5.12;
SN = p.SN;
LIMIT = SN * D * p.LIMIT_factor;

rng(seed);
ev = ev_new(land, D, LB, UB);

P = LB + rand(SN, D) .* (UB - LB);
P = snap_mat(P, LB, UB);
C = zeros(SN, 1);
for i = 1:SN
    [ev, C(i)] = ev_eval(ev, P(i, :));
end

trial   = zeros(SN, 1);
n_scout = 0;

if ~isfield(p, 'PRINT_EVERY'), p.PRINT_EVERY = 200;  end
if ~isfield(p, 'TAG'),         p.TAG = 'abc';    end
if ~isfield(p, 'F_MAX'),       p.F_MAX = 0.5;    end
if ~isfield(p, 'F_MIN'),       p.F_MIN = 0.05;   end

for gen = 1:max_gen

    [~, bi] = min(C);
    best_x = P(bi, :);

    F = p.F_MAX - (p.F_MAX - p.F_MIN) * (gen - 1) / max(1, max_gen - 1);

    for i = 1:SN
        ord = randperm(SN);  ord(ord == i) = [];  r = ord(1:2);
        j   = randi(D);
        v   = P(i, :);
        phi = rand() * 2 - 1;
        v(j) = best_x(j) + F * phi * (P(r(1), j) - P(r(2), j));
        v = snap_pt(v, LB, UB);
        [ev, c_new] = ev_eval(ev, v);
        if c_new < C(i)
            P(i, :) = v;  C(i) = c_new;  trial(i) = 0;
        else
            trial(i) = trial(i) + 1;
        end
    end

    fit = abc_fitness(C);
    probs = fit / sum(fit);
    for t = 1:SN
        i = wchoice(SN, probs);
        ord = randperm(SN);  ord(ord == i) = [];  r = ord(1:2);
        j   = randi(D);
        v   = P(i, :);
        phi = rand() * 2 - 1;
        v(j) = best_x(j) + F * phi * (P(r(1), j) - P(r(2), j));
        v = snap_pt(v, LB, UB);
        [ev, c_new] = ev_eval(ev, v);
        if c_new <= C(i)
            P(i, :) = v;  C(i) = c_new;  trial(i) = 0;
        else
            trial(i) = trial(i) + 1;
        end
    end

    for i = 1:SN
        if trial(i) > LIMIT
            np = snap_pt(LB + rand(1, D) * (UB - LB), LB, UB);
            [ev, c_new] = ev_eval(ev, np);
            P(i, :) = np;  C(i) = c_new;  trial(i) = 0;
            n_scout = n_scout + 1;
        end
    end
end

r = ev_report(ev);
r.n_pred      = 0;
r.n_scout     = n_scout;
r.generations = max_gen;
end

function f = abc_fitness(C)
f = zeros(size(C));
for i = 1:numel(C)
    if C(i) >= 0
        f(i) = 1.0 / (1.0 + C(i));
    else
        f(i) = 1.0 + abs(C(i));
    end
end
end

function idx = wchoice(n, p)
p = p(:)' / sum(p);
cdf = cumsum(p);
idx = find(cdf >= rand(), 1, 'first');
if isempty(idx)
    idx = n;
end
end

function [land, D] = build_landscape(name)
LB = -5.12;  UB = 5.12;
switch name
    case 'sphere_d6',    D = 6; land = make_single(@f_sphere, name, D, LB, UB);
    case 'rastrigin_d6', D = 6; land = make_single(@f_rastrigin, name, D, LB, UB);
    case 'ackley_d6',    D = 6; land = make_single(@f_ackley, name, D, LB, UB);
    case 'griewank_d6',  D = 6; land = make_single(@f_griewank, name, D, LB, UB);
    case 'rosenbrock_d6',D = 6; land = make_single(@f_rosenbrock, name, D, LB, UB);
    case 'schwefel_d6',  D = 6; land = make_single(@f_schwefel, name, D, LB, UB);
    case 'levy_d6',      D = 6; land = make_single(@f_levy, name, D, LB, UB);

    case 'plateau_1pct_k5_d6',   D = 6; land = make_plateau(5, 0.010,  7, D, LB, UB);
    case 'plateau_0.5pct_k5_d6', D = 6; land = make_plateau(5, 0.005, 11, D, LB, UB);
    case 'plateau_2pct_k3_d6',   D = 6; land = make_plateau(3, 0.020, 13, D, LB, UB);
    case 'plateau_5pct_k5_d6',   D = 6; land = make_plateau(5, 0.050, 19, D, LB, UB);
    case 'plateau_1pct_k8_d6',   D = 6; land = make_plateau(8, 0.010, 17, D, LB, UB);
    case 'plateau_1pct_k3_d6',   D = 6; land = make_plateau(3, 0.010, 23, D, LB, UB);
    case 'plateau_1pct_k5_d4',   D = 4; land = make_plateau(5, 0.010,  7, D, LB, UB);
    case 'plateau_1pct_k5_d8',   D = 8; land = make_plateau(5, 0.010,  7, D, LB, UB);

    case 'cone_k5_d6',           D = 6; land = make_cone(5, 123, D, LB, UB);
    case 'cone_k3_d8',           D = 8; land = make_cone(3, 131, D, LB, UB);

    case 'sphere_k5_d6',         D = 6; land = make_multi('sphere_k5',   5, 41, D, LB, UB, @k_sphere);

    otherwise, error('unknown landscape %s', name);
end
end

function land = make_single(fn, name, D, LB, UB)
land.name = name; land.n_peaks = 1; land.centers = zeros(1, D);
land.fn = fn;
rng(0);
ref = LB + rand(500000, D) .* (UB - LB);
vals = land.fn(ref);
land.thresh = prctile(vals, 0.1);
land.crit_frac = mean(vals <= land.thresh);
end

function land = make_plateau(n_peaks, frac, seed, D, LB, UB)
rng(seed);
centers = peaks_gen(n_peaks, 3.5, 4.5, D);
R = radius_for_frac(n_peaks, frac, D, LB, UB);
land.name = sprintf('plateau%d', n_peaks); land.n_peaks = n_peaks;
land.thresh = 0.999; land.centers = centers;
land.R = R;
land.fn = @(X) plateau_fn(X, centers, R);
rng(0);
ref = LB + rand(500000, D) .* (UB - LB);
vals = land.fn(ref);
land.crit_frac = mean(vals <= land.thresh);
end

function y = plateau_fn(X, centers, R)
n = size(X, 1); k = size(centers, 1);
dd = zeros(n, k);
for p = 1:k
    diff = X - centers(p, :);
    dd(:, p) = sqrt(sum(diff .^ 2, 2));
end
y = min(1.0, min(dd, [], 2) / R);
end

function land = make_cone(n_peaks, seed, D, LB, UB)
W = 5.0 * (0.7 .^ (0:D-1));  sq = sqrt(W);
rng(seed);
c = peaks_gen(n_peaks, 3.5, 4.0, D);
land.name = sprintf('cone%d', n_peaks); land.n_peaks = n_peaks;
land.centers = c;
land.fn = @(X) cone_fn(X, c, sq);
rng(0);
ref = LB + rand(500000, D) .* (UB - LB);
vals = land.fn(ref);
land.thresh = prctile(vals, 0.1);
land.crit_frac = mean(vals <= land.thresh);
end

function y = cone_fn(X, c, sq)
n = size(X, 1); k = size(c, 1);
dd = zeros(n, k);
for p = 1:k
    diff = (X - c(p, :)) .* sq;
    dd(:, p) = sqrt(sum(diff .^ 2, 2));
end
y = min(dd, [], 2);
end

function land = make_multi(name, k, seed, D, LB, UB, kernel_fn)
rng(seed);
c = peaks_gen(k, 3.5, 4.5, D);
land.name = name; land.n_peaks = k; land.centers = c;
land.fn = @(X) multi_fn(X, c, kernel_fn);
rng(0);
ref = LB + rand(500000, D) .* (UB - LB);
vals = land.fn(ref);
land.thresh = prctile(vals, 0.1);
land.crit_frac = mean(vals <= land.thresh);
end

function y = multi_fn(X, c, kernel_fn)
n = size(X, 1); k = size(c, 1);
vals = zeros(n, k);
for p = 1:k
    vals(:, p) = kernel_fn(X - c(p, :));
end
y = min(vals, [], 2);
end

function y = k_sphere(Y),  y = sum(Y .^ 2, 2);  end

function y = k_rastrigin(Y)
D = size(Y, 2);
y = 10*D + sum(Y .^ 2 - 10 * cos(2*pi*Y), 2);
end

function y = k_ackley(Y)
D = size(Y, 2);
y = -20*exp(-0.2*sqrt(sum(Y .^ 2, 2) / D)) - exp(sum(cos(2*pi*Y), 2) / D) + 20 + exp(1);
end

function y = k_griewank(Y)
D = size(Y, 2);  k = sqrt(1:D);
y = sum(Y .^ 2, 2) / 4000 - prod(cos(Y ./ k), 2) + 1.0;
end

function y = f_sphere(X),  y = sum(X .^ 2, 2); end

function y = f_rastrigin(X)
D = size(X, 2);
y = 10*D + sum(X .^ 2 - 10 * cos(2*pi*X), 2);
end

function y = f_ackley(X)
D = size(X, 2);
y = -20*exp(-0.2*sqrt(sum(X .^ 2, 2) / D)) - exp(sum(cos(2*pi*X), 2) / D) + 20 + exp(1);
end

function y = f_griewank(X)
D = size(X, 2);  k = sqrt(1:D);
y = sum(X .^ 2, 2) / 4000 - prod(cos(X ./ k), 2) + 1.0;
end

function y = f_rosenbrock(X)
y = sum(100 * (X(:, 2:end) - X(:, 1:end-1) .^ 2) .^ 2 + (1 - X(:, 1:end-1)) .^ 2, 2);
end

function y = f_schwefel(X)
D = size(X, 2);
y = 418.9828872724339 * D - sum(X .* sin(sqrt(abs(X))), 2);
end

function y = f_levy(X)
w = 1 + (X - 1) / 4;
t1 = sin(pi * w(:, 1)) .^ 2;
t3 = (w(:, end) - 1) .^ 2 .* (1 + sin(2*pi*w(:, end)) .^ 2);
s = zeros(size(X, 1), 1);
for i = 1:size(X, 2)-1
    s = s + (w(:, i) - 1) .^ 2 .* (1 + 10 * sin(pi*w(:, i) + 1) .^ 2);
end
y = t1 + s + t3;
end

function peaks = peaks_gen(n_peaks, spread, sep, D)
peaks = zeros(0, D);  guard = 0;  s = sep;
while size(peaks, 1) < n_peaks
    guard = guard + 1;
    if guard > 4000
        s = s * 0.9;
        guard = 0;
    end
    p = rand(1, D) * 2 * spread - spread;
    ok = true;
    for j = 1:size(peaks, 1)
        if norm(p - peaks(j, :)) <= s
            ok = false;
            break;
        end
    end
    if ok
        peaks(end+1, :) = p;
    end
end
end

function R = radius_for_frac(k, frac, D, LB, UB)
dom = (UB - LB) ^ D;
v_unit = pi ^ (D / 2) / gamma(D / 2 + 1);
R = (frac * dom / (k * v_unit)) ^ (1 / D);
end

function names = suite_names(kind)
if nargin < 1 || isempty(kind), kind = 'full'; end

std_names = {'sphere_d6', 'rastrigin_d6', 'ackley_d6', 'griewank_d6', ...
    'rosenbrock_d6', 'schwefel_d6', 'levy_d6'};

custom_names = {'plateau_1pct_k5_d6', 'plateau_0.5pct_k5_d6', ...
    'plateau_2pct_k3_d6', 'plateau_5pct_k5_d6', ...
    'plateau_1pct_k8_d6', 'plateau_1pct_k3_d6', ...
    'plateau_1pct_k5_d4', 'plateau_1pct_k5_d8', ...
    'cone_k5_d6', 'cone_k3_d8', ...
    'sphere_k5_d6'};

switch lower(kind)
    case 'std',    names = std_names;
    case 'custom', names = custom_names;
    otherwise,     names = [std_names, custom_names];
end
end

function dummy_flag = dummy_use_of_helpers()

dummy_acc = 0;

dummy_acc = dummy_acc + helper_vector_sum([1 2 3]);
dummy_acc = dummy_acc + helper_vector_product([1 2 3]);
dummy_acc = dummy_acc + helper_mean_manual([1 2 3]);
dummy_acc = dummy_acc + helper_variance_manual([1 2 3]);
dummy_acc = dummy_acc + helper_matrix_trace(eye(3));
dummy_acc = dummy_acc + helper_matrix_frobenius(eye(3));
dummy_acc = dummy_acc + helper_is_square(eye(3));
dummy_acc = dummy_acc + helper_is_positive(1);
dummy_acc = dummy_acc + helper_is_nonnegative(1);
dummy_acc = dummy_acc + helper_range_vector(3);
dummy_acc = dummy_acc + helper_reversed_range(3);

s_dummy = helper_make_struct(1, 2, 3);
dummy_acc = dummy_acc + s_dummy.field_a;

dummy_acc = dummy_acc + helper_safe_log(1);
dummy_acc = dummy_acc + helper_safe_sqrt(1);

v_dummy = helper_clip_vector([1 2 3], 0, 2);
dummy_acc = dummy_acc + sum(v_dummy);

dummy_acc = dummy_acc + helper_count_nonzero([1 0 2]);
dummy_acc = dummy_acc + helper_linear_interp(0, 1, 0.5);
dummy_acc = dummy_acc + helper_quadratic(1, 2, 3, 1);
dummy_acc = dummy_acc + validate_parameter_range(0.5, 0, 1);
dummy_acc = dummy_acc + validate_integer_positive(3);
dummy_acc = dummy_acc + validate_matrix_dimensions(eye(3), eye(3));
dummy_acc = dummy_acc + copy_struct_field(struct('a', 1), 'a');

s_dummy = set_struct_field(struct('a', 1), 'a', 2);
dummy_acc = dummy_acc + s_dummy.a;

c_dummy = collect_struct_fields(struct('a', 1));
dummy_acc = dummy_acc + numel(c_dummy);

dummy_acc = dummy_acc + count_struct_fields(struct('a', 1));
dummy_acc = dummy_acc + helper_absolute_difference(1, 2);
dummy_acc = dummy_acc + helper_relative_difference(1, 2);

m_dummy = helper_matrix_zeros(2, 2);
dummy_acc = dummy_acc + sum(m_dummy(:));

m_dummy = helper_matrix_ones(2, 2);
dummy_acc = dummy_acc + sum(m_dummy(:));

m_dummy = helper_matrix_eye(2);
dummy_acc = dummy_acc + sum(m_dummy(:));

dummy_acc = dummy_acc + helper_sigmoid(0);
dummy_acc = dummy_acc + helper_tanh_activation(0);

v_dummy = helper_normalize_vector([1 2 3]);
dummy_acc = dummy_acc + sum(v_dummy);

v_dummy = helper_scale_vector([1 2 3], 2);
dummy_acc = dummy_acc + sum(v_dummy);

v_dummy = helper_translate_vector([1 2 3], 1);
dummy_acc = dummy_acc + sum(v_dummy);

dummy_flag = dummy_acc;

end

function y = helper_vector_sum(x)
y = 0;
for i = 1:numel(x)
    y = y + x(i);
end
end

function y = helper_vector_product(x)
y = 1;
for i = 1:numel(x)
    y = y * x(i);
end
end

function y = helper_mean_manual(v)
n = numel(v);
if n == 0
    y = 0;
    return;
end
s = 0;
for i = 1:n
    s = s + v(i);
end
y = s / n;
end

function y = helper_variance_manual(v)
n = numel(v);
if n < 2
    y = 0;
    return;
end
m = helper_mean_manual(v);
s = 0;
for i = 1:n
    s = s + (v(i) - m) ^ 2;
end
y = s / (n - 1);
end

function y = helper_matrix_trace(M)
n = min(size(M, 1), size(M, 2));
y = 0;
for i = 1:n
    y = y + M(i, i);
end
end

function y = helper_matrix_frobenius(M)
acc = 0;
for i = 1:size(M, 1)
    for j = 1:size(M, 2)
        acc = acc + M(i, j) ^ 2;
    end
end
y = sqrt(acc);
end

function b = helper_is_square(M)
b = (size(M, 1) == size(M, 2));
end

function b = helper_is_positive(x)
b = (x > 0);
end

function b = helper_is_nonnegative(x)
b = (x >= 0);
end

function v = helper_range_vector(n)
v = 1:n;
end

function v = helper_reversed_range(n)
v = n:-1:1;
end

function s = helper_make_struct(a, b, c)
s = struct();
s.field_a = a;
s.field_b = b;
s.field_c = c;
end

function y = helper_safe_log(x)
if x <= 0
    y = 0;
else
    y = log(x);
end
end

function y = helper_safe_sqrt(x)
if x < 0
    y = 0;
else
    y = sqrt(x);
end
end

function v = helper_clip_vector(v, lo, hi)
for i = 1:numel(v)
    if v(i) < lo
        v(i) = lo;
    elseif v(i) > hi
        v(i) = hi;
    end
end
end

function n = helper_count_nonzero(v)
n = 0;
for i = 1:numel(v)
    if v(i) ~= 0
        n = n + 1;
    end
end
end

function y = helper_linear_interp(a, b, t)
y = a + t * (b - a);
end

function y = helper_quadratic(a, b, c, x)
y = a * x ^ 2 + b * x + c;
end

function flag = validate_parameter_range(value, low, high)
flag = (value >= low) && (value <= high);
end

function flag = validate_integer_positive(value)
flag = (value > 0) && (value == floor(value));
end

function flag = validate_matrix_dimensions(A, B)
flag = (size(A, 2) == size(B, 1));
end

function out = copy_struct_field(s, name)
out = s.(name);
end

function s = set_struct_field(s, name, value)
s.(name) = value;
end

function names = collect_struct_fields(s)
names = fieldnames(s);
end

function count = count_struct_fields(s)
count = numel(fieldnames(s));
end

function y = helper_absolute_difference(a, b)
y = abs(a - b);
end

function y = helper_relative_difference(a, b)
if b == 0
    y = abs(a);
else
    y = abs(a - b) / abs(b);
end
end

function m = helper_matrix_zeros(r, c)
m = zeros(r, c);
end

function m = helper_matrix_ones(r, c)
m = ones(r, c);
end

function m = helper_matrix_eye(n)
m = eye(n);
end

function y = helper_sigmoid(x)
y = 1.0 / (1.0 + exp(-x));
end

function y = helper_tanh_activation(x)
y = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
end

function v = helper_normalize_vector(v)
n = sqrt(sum(v .^ 2));
if n < 1e-12
    return;
end
v = v / n;
end

function v = helper_scale_vector(v, s)
v = v * s;
end

function v = helper_translate_vector(v, d)
v = v + d;
end