function mcx_jacobian(varargin)
% MCX_JACOBIAN  Adjoint (reciprocity) Jacobians from one MCX fluence run per optode;
% the channel-space activation is then a weighted sum over the voxel pattern
%     dOD(p, wl, trial) = sum_v J_p(v) * dmua(v, wl, trial) / Phi_p,
%     J_p(v) = phi_src(v) * phi_det(v) * Vvox
% for the 497 source-detector pairs, with the FULL Cedalion activation (no labels).
%
%   mcx_jacobian()                       % 1e9 photons per optode, RTX 5090 on shui
%   mcx_jacobian('nphoton', 3e9)
%
% * one MCX run per optode: 56 pencil sources; 144 detectors as their adjoint, a
%   LAMBERTIAN (cos-theta, srcdir(4)=-Inf) inward disk of the detector radius launched
%   from the skin surface above the detector: the adjoint of a detector counting all
%   photons that exit through its disk, so that P_det(v) = (A/4) phi_det(v)
%   (opt.detlaunch='collimated' gives MCX's adjoint-mode collimated disk instead);
%   fluences are kept on disk (opt.fluencedir) and reused when the settings match
% * Phi_p from the fluences themselves, as a reciprocity average (as Cedalion's
%   compute_sensitivity) of the reading of each optode's fluence at the other one:
%   'kienle' (default): Phi_p = 4 R, R = mean diffuse reflectance of the source
%     fluence under the detector disk, from the interior fluence (Kienle & Patterson
%     1997 Eq. 7 with n=1: R = Phi(0)/4 + (D/2) dPhi/dz, fitted on the skin above the
%     detector); with the Lambertian adjoint, dOD/dmua(v) = phi_s phi_d V / (4 R)
%   'footprint': mean fluence of the tissue voxels within the detector radius
% * Cedalion's optics are wavelength independent -> one Jacobian set serves both
%   wavelengths, both conditions and every HRF phase (they differ only in dmua)
%
% Outputs: mcx_jacobian.jdb (peak dOD per pair; same fields as mcx_dod.jdb, so
% mcx_build_hrf('mcxfile', 'mcx_jacobian.jdb') turns it into an HRF measurement),
% <fluencedir>/jac_brain.mat: J_p/Phi_p (mm) on the brain voxels for reweighting.

opt = struct('nphoton', 1e9, ...       % photons per optode run (Cedalion: 1e8)
             'detradius', 1.5, ...     % mm, disk radius of the detector adjoint source
             'gpuid', '10', ...        % shui: RTX 5090 (MCX order)
             'workload', [], ...
             'seed', 20261027, ...
             'fluencedir', '', ...     % [] = mcx_ref/fluence (raw per-optode fluences)
             'norm', 'kienle', ...     % Phi_p: 'kienle' (4 x dref from fluence) | 'footprint'
             'detlaunch', 'lambertian', ...  % detector adjoint source: 'lambertian' | 'collimated'
             'savejac', true, ...
             'ppathfile', '', ...      % optional cross-check with mcx_ppath.jdb
             'outfile', '');
for i = 1:2:numel(varargin)
    opt.(varargin{i}) = varargin{i + 1};
end
here = fileparts(mfilename('fullpath'));
if isempty(opt.ppathfile)
    opt.ppathfile = fullfile(here, 'mcx_ppath.jdb');
end
if isempty(opt.outfile)
    opt.outfile = fullfile(here, 'mcx_jacobian.jdb');
end
if isempty(opt.fluencedir)
    opt.fluencedir = fullfile(here, 'fluence');
end
if ~exist(opt.fluencedir, 'dir')
    mkdir(opt.fluencedir);
end

%% ---- head model, cropped grid, optodes ----------------------------------
M = headmodel(here);
pairs = M.pairs;
trials = M.trials;
wl = M.wl;
npair = size(pairs, 1);
nwl = numel(wl);
ntrial = numel(trials);
ns = size(M.srcmcx, 1);
nd = size(M.detmcx, 1);
nopt = ns + nd;
dmua = reshape(M.dmuavox, numel(M.vox), []);              % [nvox x ntrial*nwl]

lo = zeros(1, 3);
hi = zeros(1, 3);
for d = 1:3
    k = find(any(any(M.vol > 0, mod(d, 3) + 1), mod(d + 1, 3) + 1));
    lo(d) = max(k(1) - 1, 1);
    hi(d) = min(k(end) + 1, size(M.vol, d));
end
vol = M.vol(lo(1):hi(1), lo(2):hi(2), lo(3):hi(3));
dims = size(vol);
[ii, jj, kk] = ind2sub(size(M.vol), M.vox);
voxc = sub2ind(dims, ii - lo(1) + 1, jj - lo(2) + 1, kk - lo(3) + 1);
optpos = [M.srcmcx; M.detmcx] - (lo - 1);                 % 0-based grid coords, cropped
optdir = [M.srcdir; M.detdir];
Vvox = M.unitinmm^3;

% footprint of each optode: tissue voxels whose centre is within the detector radius
foot = cell(nopt, 1);
rfoot = max(opt.detradius / M.unitinmm, 1);
for o = 1:nopt
    c = floor(optpos(o, :)) + 1;                          % MATLAB index of the voxel
    [gi, gj, gk] = ndgrid(max(c(1) - 3, 1):min(c(1) + 3, dims(1)), ...
                          max(c(2) - 3, 1):min(c(2) + 3, dims(2)), ...
                          max(c(3) - 3, 1):min(c(3) + 3, dims(3)));
    g = [gi(:), gj(:), gk(:)];
    idx = sub2ind(dims, g(:, 1), g(:, 2), g(:, 3));
    sel = sum((g - 0.5 - optpos(o, :)).^2, 2) <= rfoot^2 & vol(idx) > 0;
    foot{o} = idx(sel);
end

%% ---- one MCX fluence run per optode (cached) -----------------------------
clear cfg;
cfg.vol = vol;
cfg.prop = M.prop0;
cfg.unitinmm = M.unitinmm;
cfg.tstart = 0;                           % Cedalion: single 0-5 ns gate
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.issrcfrom0 = 1;
cfg.isreflect = 0;
cfg.isnormalized = 1;
cfg.outputtype = 'fluence';               % 1/mm^2 per launched photon
cfg.nphoton = opt.nphoton;
cfg.gpuid = opt.gpuid;
if ~isempty(opt.workload)
    cfg.workload = opt.workload;
end
cfg.autopilot = 1;

% skin-surface readout geometry (also the launch point of the detector adjoint)
Pk = drefprobe(vol, optpos, optdir, opt.detradius / M.unitinmm, M.prop0);
islamb = strcmp(opt.detlaunch, 'lambertian');
sig = [opt.nphoton, opt.detradius, dims, lo, opt.seed];
phi = zeros(prod(dims), nopt, 'single');
tall = tic;
nrun = 0;
for o = 1:nopt
    fname = fullfile(opt.fluencedir, sprintf('optode%03d.bin', o));
    sfile = [fname(1:end - 4) '.sig'];
    osig = sig;
    if o > ns && islamb
        osig = [sig, 1];                                  % Lambertian detector run
    end
    if exist(fname, 'file') && exist(sfile, 'file') && isequal(load(sfile, '-ascii'), osig)
        fid = fopen(fname, 'r');
        phi(:, o) = fread(fid, prod(dims), 'single=>single');
        fclose(fid);
        continue
    end
    cfg.srcpos = optpos(o, :);
    cfg.srcdir = [optdir(o, :), 0];
    if o <= ns
        cfg.srctype = 'pencil';
        cfg.srcparam1 = [0 0 0 0];
    else
        cfg.srctype = 'disk';
        cfg.srcparam1 = [opt.detradius / M.unitinmm 0 0 0];
        if islamb
            cfg.srcpos = Pk(o).b0;                        % on the skin surface
            cfg.srcdir(4) = -Inf;                         % Lambertian (analytical)
        end
    end
    cfg.seed = opt.seed + o;
    f = mcxlab(cfg);
    phi(:, o) = single(f.data(:));
    clear f;
    fid = fopen(fname, 'w');
    fwrite(fid, phi(:, o), 'single');
    fclose(fid);
    save(sfile, 'osig', '-ascii', '-double');
    nrun = nrun + 1;
    if mod(o, 20) == 0 || o == nopt
        fprintf('optode %3d/%d: %d MCX runs so far, %.1fs elapsed\n', o, nopt, nrun, toc(tall));
    end
end
fprintf('fluences: %d MCX runs, %d reused, %.1fs\n', nrun, nopt - nrun, toc(tall));

%% ---- pair Jacobians, normalization, weighted sums -----------------------
tj = tic;
q = sqrt(sum(M.dmuavox.^2, 3));
nlabel0 = size(M.prop0, 1) - 1;
qmask = 1e-2;
if exist(opt.ppathfile, 'file')
    pp = loadjd(opt.ppathfile);
    qmask = pp.settings.qmask;
end
inreg = (q ./ max(q, [], 1)) >= qmask;                    % C3/C4 regions (checks only)
clear q;
% pair intensities: Kienle dref of the source fluence at the detector; footprint mean
Phi_k = zeros(npair, 1);                  % 4 x Kienle dref (1/mm^2 per photon)
Phi_f = zeros(npair, 1);                  % footprint mean fluence
Rsd = zeros(npair, 1);                    % dref of the source fluence at the detector
for p = 1:npair
    s = pairs(p, 1);
    o = ns + pairs(p, 2);
    Rsd(p) = drefread(Pk(o), phi(:, s));
    Phi_k(p) = 4 * Rsd(p);
    Phi_f(p) = 0.5 * (mean(phi(foot{o}, s)) + mean(phi(foot{s}, o)));
end
switch opt.norm
    case 'kienle'
        Phi = Phi_k;
    case 'footprint'
        Phi = Phi_f;
    otherwise
        error('unknown norm %s', opt.norm);
end
Jsum = zeros(npair, 1);                   % sum_v J (unnormalized)
Jreg = zeros(npair, ntrial);
Jdmua = zeros(npair, ntrial * nwl);
if opt.savejac
    Lvox = zeros(numel(voxc), npair, 'single');
end
for s = 1:ns
    prow = find(pairs(:, 1) == s);
    if isempty(prow)
        continue
    end
    od = ns + pairs(prow, 2);
    J = phi(:, s) .* phi(:, od) * Vvox;                   % [nvoxel x ndet]
    Jsum(prow) = double(sum(J, 1, 'double'))';
    Jb = J(voxc, :);
    for t = 1:ntrial
        Jreg(prow, t) = sum(double(Jb(inreg(:, t), :)), 1)';
    end
    Jdmua(prow, :) = double(Jb)' * dmua;
    if opt.savejac
        Lvox(:, prow) = Jb ./ single(Phi(prow)');         % J/Phi on brain voxels (mm)
    end
end
clear J Jb;
Lsum = Jsum ./ Phi;                       % sum_v J/Phi = mean total path (mm)
Lreg = Jreg ./ Phi;
dOD_adj = permute(reshape(Jdmua ./ Phi, npair, ntrial, nwl), [1 3 2]);   % pair x wl x trial
ok = isfinite(Phi) & Phi > 0;
fprintf('Jacobians for %d pairs (%d valid): %.1fs\n', npair, nnz(ok), toc(tj));
if opt.savejac
    save(fullfile(opt.fluencedir, 'jac_brain.mat'), 'Lvox', 'voxc', 'lo', 'dims', 'pairs', ...
         'opt', '-v7.3', '-nocompression');
    clear Lvox;
end

%% ---- checks and comparison with Cedalion ---------------------------------
sds = sqrt(sum((double(M.meas.srcpos(pairs(:, 1), :)) - double(M.meas.detpos(pairs(:, 2), :))).^2, 2));
fprintf('\nmean total path sum_v J/Phi (mm): median %.1f [%.1f %.1f], DPF median %.2f\n', ...
        median(Lsum(ok)), min(Lsum(ok)), max(Lsum(ok)), median(Lsum(ok) ./ sds(ok)));
if exist('pp', 'var')
    fprintf('cross-check vs detected-photon run (mcx_ppath.jdb, same positions and radius):\n');
    Idet = double(pp.I0);                                 % detected weight per photon
    Ik = Rsd * pi * opt.detradius^2;                      % Kienle dref x disk area
    fprintf('  intensity: Kienle dref*area / detected  median %.3f [p5 %.3f p95 %.3f], r(log) %.4f\n', ...
            median(Ik(ok) ./ Idet(ok)), pctl(Ik(ok) ./ Idet(ok), 5), pctl(Ik(ok) ./ Idet(ok), 95), ...
            pearson(log(Ik(ok)), log(Idet(ok))));
    for nm = {'kienle', Phi_k; 'footprint', Phi_f}'
        F = nm{2};
        fprintf('  [%s] total path sum J/Phi / <L_total>: median %.3f', nm{1}, ...
                median(Jsum(ok) ./ F(ok) ./ double(pp.Ltotal(ok))));
        for t = 1:ntrial
            Lpp = double(pp.Lmean(:, nlabel0 + t));
            big = ok & Lpp > 0.1 * max(Lpp);
            fprintf('; %s region %.3f (r %.4f)', trials{t}, median(Jreg(big, t) ./ F(big) ./ Lpp(big)), ...
                    pearson(Jreg(ok, t) ./ F(ok), Lpp(ok)));
        end
        fprintf('\n');
    end
end

[maxOD_ced, K, reltime] = mcx_build_hrf('fitonly', true);
dOD = double(M.meas.dOD);
fprintf('\n== Jacobian dOD = sum_v (J/Phi) dmua(v) vs Cedalion (%d pairs) ==\n', nnz(ok));
fprintf('%-10s %6s %12s %12s %14s %14s\n', 'trial', 'wl', 'r(peakOD)', 'r(raw@peak)', ...
        'slope MCX/ced', 'median ratio');
for t = 1:ntrial
    [~, ip] = max(K(t, :));
    for w = 1:nwl
        x = maxOD_ced(ok, w, t);
        y = dOD_adj(ok, w, t);
        big = abs(x) > 0.1 * max(abs(x));
        fprintf('%-10s %6g %12.4f %12.4f %14.3f %14.3f\n', trials{t}, wl(w), pearson(x, y), ...
                pearson(dOD(ok, w, ip, t), y), (x' * y) / (x' * x), median(y(big) ./ x(big)));
    end
end
dl = dOD_adj(ok, :, 1) - dOD_adj(ok, :, 2);
dc = maxOD_ced(ok, :, 1) - maxOD_ced(ok, :, 2);
[~, ip] = max(K(1, :));
draw = dOD(ok, :, ip, 1) - dOD(ok, :, ip, 2);
fprintf('C3-C4 (rest cancels): r vs fitted peak %.4f, r vs raw dOD at t=%.2fs %.4f\n', ...
        pearson(dc(:), dl(:)), reltime(ip), pearson(draw(:), dl(:)));

% linear in amplitude: same fields as mcx_dod.jdb so mcx_build_hrf can use it
amp = linspace(0, 1, 21);
out = struct('format', 'mcx-jacobian-dod', 'complete', all(ok), ...
             'Description', ['Adjoint MCX Jacobian (one fluence run per optode) applied to ' ...
                             'the Cedalion synthetic activation on the voxelated ICBM-152 head'], ...
             'wavelengths', wl, 'pairs', int32(pairs), 'trialtype', {trials}, ...
             'amp', amp, 'dOD_amp', dOD_adj .* reshape(amp, 1, 1, 1, []), ...
             'maxOD', dOD_adj, 'maxOD_se', nan(size(dOD_adj)), ...
             'maxOD_cedalion', maxOD_ced, 'Phi', Phi, 'Phi_kienle', Phi_k, 'Phi_footprint', Phi_f, 'Ltotal', Lsum, 'Lregion', Lreg, ...
             'sds', sds, 'settings', opt);
savebj('', out, 'FileName', opt.outfile, 'Compression', 'zlib');
fprintf('saved %s\n', opt.outfile);
plotscatter(fullfile(here, 'figures'), maxOD_ced, dOD_adj, ok, wl, trials);
end

% -------------------------------------------------------------------------
function p = pctl(x, pc)
x = sort(x(:));
p = x(max(1, ceil(pc / 100 * numel(x))));
end

function r = pearson(x, y)
c = corrcoef(x(:), y(:));
r = c(1, 2);
end

function plotscatter(figdir, ced, adj, ok, wl, trials)
if ~exist(figdir, 'dir')
    mkdir(figdir);
end
fig = figure('Visible', 'off', 'Position', [0 0 900 420]);
for t = 1:numel(trials)
    subplot(1, numel(trials), t);
    hold on;
    for w = 1:numel(wl)
        plot(ced(ok, w, t), adj(ok, w, t), '.', 'DisplayName', sprintf('%g nm', wl(w)));
    end
    v = [ced(ok, :, t); adj(ok, :, t)];
    plot([min(v(:)) max(v(:))], [min(v(:)) max(v(:))], 'k--', 'HandleVisibility', 'off');
    grid on;
    axis tight;
    xlabel('Cedalion Adot peak \DeltaOD');
    ylabel('MCX Jacobian \Sigma_v (J/\Phi) \Delta\mu_a');
    title(trials{t});
    legend('Location', 'northwest');
end
print(fig, fullfile(figdir, 'jacobian_vs_cedalion.png'), '-dpng', '-r120');
close(fig);
end
