function mcx_ppath(varargin)
% MCX_PPATH  Baseline ("white") MCX run with the C3/C4 activation regions as their
% own labels; records detected-photon partial path lengths for later mua rescaling.
%
%   mcx_ppath()                          % all sources, 1e9 photons each
%   mcx_ppath('srclist', 46, 'nphoton', 2e8)
%
% Each activation region (voxels whose Cedalion dmua >= qmask x peak, mapped through
% voxel_to_vertex_brain) becomes one extra label carrying the BASE gray/white optics
% (label 6 = C3, 7 = C4), so the simulation is the unperturbed head. For every
% source-detector pair the detected photons give
%   <L_m> = sum_i w_i L_im / sum_i w_i     (w_i = exp(-sum_m mua_m L_im), baseline)
% which is exactly -dOD/dmua_m for a uniform change of label m. The first-order
% prediction dOD_lin = mean(dmua in region) * <L_region> is correlated against
% Cedalion's Adot peak dOD (recovered from the sample data by mcx_build_hrf).
%
% Outputs: cache/ppath_srcNN.mat (per-photon detid + ppath, for rescaling),
%          mcx_ppath.mat/.jdb (per-pair statistics), figures/ppath_vs_cedalion.png

opt = struct('nphoton', 1e9, ...       % photons per source (one MCX call)
             'qmask', 1e-2, ...        % region = dmua >= qmask x peak
             'detradius', 1.5, ...     % mm (3 mm diameter)
             'maxdetphoton', 1e7, ...
             'gpuid', '111', ...         % moli: 3090 + 2x 2080 SUPER (MCX order)
             'workload', [5248 3072 3072], ...  % photon share ~ CUDA cores
             'seed', 20260610, ...
             'srclist', [], ...
             'savephoton', true, ...
             'outfile', '');
for i = 1:2:numel(varargin)
    opt.(varargin{i}) = varargin{i + 1};
end
here = fileparts(mfilename('fullpath'));
if isempty(opt.outfile)
    opt.outfile = fullfile(here, 'mcx_ppath.mat');
end
cachedir = fullfile(here, 'cache');
if ~exist(cachedir, 'dir')
    mkdir(cachedir);
end

%% ---- head model + one label per activation region ----------------------
M = headmodel(here);
pairs = M.pairs;
trials = M.trials;
wl = M.wl;
npair = size(pairs, 1);
nwl = numel(wl);
ntrial = numel(trials);
ns = size(M.meas.srcpos, 1);
nlabel0 = size(M.prop0, 1) - 1;

vol = M.vol;
prop = M.prop0;
q = sqrt(sum(M.dmuavox.^2, 3));
q = q ./ max(q, [], 1);
dmua_mean = zeros(ntrial, nwl);                           % mean dmua inside each region
infrac = zeros(ntrial, nwl);                              % share of the integrated dmua
for t = 1:ntrial
    inmask = q(:, t) >= opt.qmask;
    if any(vol(M.vox(inmask)) > nlabel0)
        error('C3/C4 regions overlap at qmask=%g', opt.qmask);
    end
    base = unique(M.prop0(double(vol(M.vox(inmask))) + 1, :), 'rows');
    if size(base, 1) > 1
        error('region %s spans tissues with different optics', trials{t});
    end
    vol(M.vox(inmask)) = nlabel0 + t;
    prop(end + 1, :) = base;                              %#ok<AGROW> baseline optics
    for w = 1:nwl
        d = M.dmuavox(:, t, w);
        dmua_mean(t, w) = mean(d(inmask));
        infrac(t, w) = sum(d(inmask)) / sum(d);
    end
    fprintf('%s region: label %d, %d voxels (%.1f cm^3), %.1f%% of integrated dmua, mean dmua %s /mm\n', ...
            trials{t}, nlabel0 + t, nnz(inmask), nnz(inmask) * M.unitinmm^3 / 1000, ...
            100 * infrac(t, 1), mat2str(dmua_mean(t, :), 3));
end
lab = nlabel0 + (1:ntrial);
nmedia = size(prop, 1) - 1;
sds = sqrt(sum((double(M.meas.srcpos(pairs(:, 1), :)) - double(M.meas.detpos(pairs(:, 2), :))).^2, 2));
clear q;

%% ---- baseline MCX, one call per source ---------------------------------
clear cfg;
cfg.vol = vol;
cfg.prop = prop;
cfg.unitinmm = M.unitinmm;
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.issrcfrom0 = 1;
cfg.isreflect = 0;
cfg.issave2pt = 0;
cfg.savedetflag = 'dp';
cfg.maxdetphoton = opt.maxdetphoton;
cfg.nphoton = opt.nphoton;
cfg.gpuid = opt.gpuid;
if ~isempty(opt.workload)
    cfg.workload = opt.workload;
end
cfg.autopilot = 1;

srclist = opt.srclist;
if isempty(srclist)
    srclist = 1:ns;
end
ndet = zeros(npair, 1);
I0 = nan(npair, 1);                       % detected weight per launched photon
Lmean = nan(npair, nmedia);               % weighted mean partial path per label (mm)
Lse = nan(npair, ntrial);                 % standard error of <L_region>
hitfrac = nan(npair, ntrial);             % weighted fraction of photons visiting region
Ltot = nan(npair, 1);                     % weighted mean total path (mm)
for s = srclist(:)'
    dets = pairs(pairs(:, 1) == s, 2);
    if isempty(dets)
        continue
    end
    tic;
    cfg.srcpos = M.srcmcx(s, :);
    cfg.srcdir = M.srcdir(s, :);
    cfg.detpos = [M.detmcx(dets, :), repmat(opt.detradius / M.unitinmm, numel(dets), 1)];
    cfg.seed = opt.seed + s;
    [~, detp] = mcxlab(cfg);
    if ~isfield(detp, 'detid')                           % nothing detected
        detp.detid = zeros(0, 1);
        detp.ppath = zeros(0, nmedia);
    end
    if numel(detp.detid) >= opt.maxdetphoton
        warning('src %d: detected-photon buffer full, raise maxdetphoton', s);
    end
    detid = uint16(detp.detid);
    ppath = single(detp.ppath * M.unitinmm);             % mm
    clear detp;
    if opt.savephoton
        save(fullfile(cachedir, sprintf('ppath_src%02d.mat', s)), 'detid', 'ppath', 'dets', ...
             'prop', 'opt', '-v7.3');
    end
    w = exp(-double(ppath) * prop(2:end, 1));
    id = double(detid);
    nd = numel(dets);
    sw = accumarray(id, w, [nd 1]);
    [~, prow] = ismember([repmat(s, nd, 1), dets], pairs, 'rows');
    ndet(prow) = accumarray(id, 1, [nd 1]);
    I0(prow) = sw / opt.nphoton;
    for m = 1:nmedia
        Lmean(prow, m) = accumarray(id, w .* double(ppath(:, m)), [nd 1]) ./ sw;
    end
    Ltot(prow) = accumarray(id, w .* sum(double(ppath), 2), [nd 1]) ./ sw;
    for t = 1:ntrial
        L = double(ppath(:, lab(t)));
        dev2 = (w .* (L - Lmean(prow(id), lab(t)))).^2;
        Lse(prow, t) = sqrt(accumarray(id, dev2, [nd 1])) ./ sw;
        hitfrac(prow, t) = accumarray(id, w .* (L > 0), [nd 1]) ./ sw;
    end
    fprintf('src %2d/%d: %8d detected, per pair min %6d median %7d, %.1fs\n', s, ns, ...
            numel(detid), min(ndet(prow)), median(ndet(prow)), toc);
    clear ppath w id detid;
end

%% ---- partial-path statistics at the detector array ---------------------
ok = ndet > 0;
fprintf('\n== partial-path statistics (%d/%d pairs, %.0e photons/source, r=%.1f mm) ==\n', ...
        nnz(ok), npair, opt.nphoton, opt.detradius);
pc = @(x) pctl(x(ok), [0 5 50 95 100]);
fprintf('%-28s %10s %10s %10s %10s %10s\n', '', 'min', 'p5', 'median', 'p95', 'max');
prow_ = @(name, x, f) fprintf(['%-28s' repmat([' ' f], 1, 5) '\n'], name, pc(x));
prow_('detected photons', ndet, '%10.0f');
prow_('SDS (mm)', sds, '%10.1f');
prow_('<L_total> (mm)', Ltot, '%10.1f');
prow_('DPF = <L_total>/SDS', Ltot ./ sds, '%10.2f');
names = {'csf', 'gm', 'scalp', 'skull', 'wm'};
for m = 1:nlabel0
    prow_(sprintf('<L_%s> (mm)', names{m}), Lmean(:, m), '%10.2f');
end
for t = 1:ntrial
    prow_(sprintf('<L_%s> (mm)', trials{t}), Lmean(:, lab(t)), '%10.3g');
    prow_(sprintf('visit frac. %s', trials{t}), hitfrac(:, t), '%10.3g');
    big = ok & Lmean(:, lab(t)) > 0.1 * max(Lmean(:, lab(t)));
    fprintf('  %s: %d pairs with <L> > 10%% of max; their median rel. SE %.3g\n', trials{t}, ...
            nnz(big), median(Lse(big, t) ./ Lmean(big, lab(t))));
end

%% ---- linear correlation against Cedalion's sample data -----------------
[maxOD_ced, K, reltime] = mcx_build_hrf('fitonly', true);   % Adot peak dOD, pair x wl x trial
meas = M.meas;
dOD = double(meas.dOD);
dOD_lin = zeros(npair, nwl, ntrial);
for t = 1:ntrial
    for w = 1:nwl
        dOD_lin(:, w, t) = dmua_mean(t, w) * Lmean(:, lab(t));
    end
end
fprintf('\n== first-order MCX dOD_lin = mean(dmua) * <L_region> vs Cedalion ==\n');
fprintf('%-10s %6s %12s %12s %14s %14s\n', 'trial', 'wl', 'r(peakOD)', 'r(raw@peak)', ...
        'slope MCX/ced', 'median ratio');
for t = 1:ntrial
    [~, ip] = max(K(t, :));
    for w = 1:nwl
        x = maxOD_ced(ok, w, t);
        y = dOD_lin(ok, w, t);
        raw = dOD(ok, w, ip, t);
        big = abs(x) > 0.1 * max(abs(x));
        fprintf('%-10s %6g %12.4f %12.4f %14.3f %14.3f\n', trials{t}, wl(w), pearson(x, y), ...
                pearson(raw, y), (x' * y) / (x' * x), median(y(big) ./ x(big)));
    end
end
dl = dOD_lin(ok, :, 1) - dOD_lin(ok, :, 2);
dc = maxOD_ced(ok, :, 1) - maxOD_ced(ok, :, 2);
[~, ip] = max(K(1, :));
draw = dOD(ok, :, ip, 1) - dOD(ok, :, ip, 2);
fprintf('C3-C4 (rest cancels): r vs fitted peak %.4f, r vs raw dOD at t=%.2fs %.4f\n', ...
        pearson(dc(:), dl(:)), reltime(ip), pearson(draw(:), dl(:)));

out = struct('format', 'mcx-ppath-stats', 'pairs', int32(pairs), 'wavelengths', wl, ...
             'trialtype', {trials}, 'sds', sds, 'ndetected', ndet, 'I0', I0, ...
             'Lmean', Lmean, 'Lse', Lse, 'hitfrac', hitfrac, 'Ltotal', Ltot, ...
             'labels', {[names, trials]}, 'prop', prop, 'dmua_mean', dmua_mean, ...
             'region_dmua_fraction', infrac, 'dOD_lin', dOD_lin, ...
             'maxOD_cedalion', maxOD_ced, 'settings', opt);
savemeas(opt.outfile, out);                % .mat + .jdb
fprintf('saved %s\n', opt.outfile);
plotscatter(fullfile(here, 'figures'), maxOD_ced, dOD_lin, ok, wl, trials);
end

% -------------------------------------------------------------------------
function p = pctl(x, pc)
% nearest-rank percentiles (no Statistics Toolbox)
x = sort(x(:));
p = x(max(1, ceil(pc / 100 * numel(x))))';
end

function r = pearson(x, y)
c = corrcoef(x(:), y(:));
r = c(1, 2);
end

function plotscatter(figdir, ced, lin, ok, wl, trials)
if ~exist(figdir, 'dir')
    mkdir(figdir);
end
fig = figure('Visible', 'off', 'Position', [0 0 900 420]);
for t = 1:numel(trials)
    subplot(1, numel(trials), t);
    hold on;
    for w = 1:numel(wl)
        plot(ced(ok, w, t), lin(ok, w, t), '.', 'DisplayName', sprintf('%g nm', wl(w)));
    end
    v = [ced(ok, :, t); lin(ok, :, t)];
    plot([min(v(:)) max(v(:))], [min(v(:)) max(v(:))], 'k--', 'HandleVisibility', 'off');
    grid on;
    axis tight;
    xlabel('Cedalion Adot peak \DeltaOD');
    ylabel('MCX mean(\Delta\mu_a)\cdot<L_{region}>');
    title(trials{t});
    legend('Location', 'northwest');
end
print(fig, fullfile(figdir, 'ppath_vs_cedalion.png'), '-dpng', '-r120');
close(fig);
end
