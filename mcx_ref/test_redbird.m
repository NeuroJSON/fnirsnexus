function test_redbird(alpha, beta, jac)
% TEST_REDBIRD  Run the redbird two-surface recon (s4_recon_surface) on the original
% Cedalion data and on the MCX reference data, side by side, without touching
% ../redbird_recon outputs.
%
%   test_redbird()              % alpha=0.003, beta=0.01 (s4 best) + a sweep
%   test_redbird([], [], 'cedalion')  % redbird's inverse with Cedalion's Adot instead
%   test_redbird([], [], 'matched')   % redbird's Cedalion-matched FEM operator (s4_recon_matched)
%   test_redbird([], [], 's4_op_matched_csf0.15_voxel.mat')  % any operator cache in ../redbird_recon
%
% The surface operator Anorm (FEM forward, data independent) is reused from
% ../redbird_recon/s4_op.mat; c_meas is recomputed from each dataset's pre-stimulus
% variance exactly as s4_recon_surface does. Then the same whitening, depth
% weighting, Tikhonov (rbreginv) and t=10 s snapshot. Writes recon_mcx.bgii and
% figures/redbird_cedalion_vs_mcx.png.

if nargin < 1 || isempty(alpha)
    alpha = 0.003;
end
if nargin < 2 || isempty(beta)
    beta = 0.01;
end
here = fileparts(mfilename('fullpath'));
rb = fullfile(here, '..', 'redbird_recon');

if nargin < 3 || isempty(jac)
    jac = 'redbird';
end
S = load(fullfile(rb, 's4_op.mat'), 'Anorm', 'ncortex', 'nscalp');
Anorm = S.Anorm;
if strcmp(jac, 'matched')
    jac = 's4_op_matched.mat';
end
if endsWith(jac, '.mat')                                  % any redbird operator cache
    S = load(fullfile(rb, jac), 'Anorm', 'ncortex', 'nscalp');
    Anorm = S.Anorm;
end
if strcmp(jac, 'cedalion')
    % Cedalion's Adot (mm) in redbird's [wl-major rows] x [HbO; HbR] x [cortex; scalp] layout
    C = load(fullfile(here, 'adot_cedalion.mat'));
    e = rbextinction([760 850], {'hbo', 'hbr'});           % 1/(mm uM)
    A = double(C.Adot);
    Anorm = -[A * e(1, 1), A * e(1, 2); A * e(2, 1), A * e(2, 2)];
    clear A C;
end
fprintf('Jacobian: %s\n', jac);
ncortex = S.ncortex;
nscalp = S.nscalp;
gt = loadbj(fullfile(rb, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);
truth = {double(gt.GIFTIData.MeshVertex3.Properties.HbO_C3), ...
         double(gt.GIFTIData.MeshVertex3.Properties.HbO_C4)};

data = {'Cedalion (Adot)', fullfile(rb, 'hrf_measurement.jdb'); ...
        'MCX reference', fullfile(here, 'hrf_measurement.mat')};
cond = {'C3', 'C4'};
res = cell(size(data, 1), 1);
fprintf('\n== redbird s4 two-surface recon, alpha=%.3g beta=%.3g, t=10 s ==\n', alpha, beta);
fprintf('%-16s %-4s %11s %7s %8s %8s %6s\n', 'data', 'cond', 'peak_err_mm', 'corr', ...
        'rec_pk', 'true_pk', 'leak');
for k = 1:size(data, 1)
    [rhs, cmeas] = loaddata(data{k, 2});
    res{k} = solve(Anorm, cmeas, rhs, alpha, beta, ncortex, nscalp);
    for t = 1:2
        [err, cc, pk, leak] = score(res{k}.hbo(:, t), res{k}.scalp(:, t), truth{t}, cortex);
        fprintf('%-16s %-4s %11.1f %7.3f %8.2f %8.2f %6.2f\n', data{k, 1}, cond{t}, ...
                err, cc, pk, max(truth{t}), leak);
    end
end

% noise-free signal of each dataset (same c_meas weighting): separates SNR from model error
[maxOD_ced, K, reltime] = mcx_build_hrf('fitonly', true);
[~, tk] = min(abs(reltime - 10.0));
mm = loadmeas(data{2, 2});
[npair, nwl] = size(maxOD_ced, [1 2]);
clean = {zeros(npair * nwl, 2), zeros(npair * nwl, 2)};
for w = 1:nwl
    rows = (w - 1) * npair + (1:npair);
    for t = 1:2
        clean{1}(rows, t) = -maxOD_ced(:, w, t) * K(t, tk);
        clean{2}(rows, t) = -double(mm.dOD_clean(:, w, tk, t));
    end
end
fprintf('\n== noise-free signal only (no resting-state background), same weighting ==\n');
for k = 1:2
    [~, cmeas] = loaddata(data{k, 2});
    r = solve(Anorm, cmeas, clean{k}, alpha, beta, ncortex, nscalp);
    for t = 1:2
        [err, cc, pk, leak] = score(r.hbo(:, t), r.scalp(:, t), truth{t}, cortex);
        fprintf('%-16s %-4s %11.1f %7.3f %8.2f %8.2f %6.2f\n', data{k, 1}, cond{t}, err, cc, pk, ...
                max(truth{t}), leak);
    end
end
fprintf('signal/noise at t=10 s (median over the 20 strongest channels, 850 nm): ');
for k = 1:2
    [rhs, cmeas] = loaddata(data{k, 2});
    for t = 1:2
        snr = abs(clean{k}(npair + (1:npair), t)) ./ sqrt(cmeas(npair + (1:npair)));
        snr = sort(snr, 'descend');
        fprintf('%s %s %.2f; ', data{k, 1}, cond{t}, median(snr(1:20)));
    end
end
fprintf('\n');

% alpha/beta sweep on both datasets (C3 / C4 peak error, corr)
fprintf('\n== sweep: peak_err_mm / corr ==\n%6s %6s | %-26s | %-26s\n', 'alpha', 'beta', ...
        data{1, 1}, data{2, 1});
dd = cell(2, 2);
for k = 1:2
    [dd{k, 1}, dd{k, 2}] = loaddata(data{k, 2});
end
for a = [0.001 0.003 0.01 0.03]
    for b = [0.03 0.01 0.003]
        fprintf('%6.3g %6.3g |', a, b);
        for k = 1:2
            r = solve(Anorm, dd{k, 2}, dd{k, 1}, a, b, ncortex, nscalp);
            for t = 1:2
                [err, cc] = score(r.hbo(:, t), r.scalp(:, t), truth{t}, cortex);
                fprintf(' %s %5.1f/%5.3f', cond{t}, err, cc);
            end
            fprintf(' |');
        end
        fprintf('\n');
    end
end

% save the MCX recon (cortex) and a side-by-side figure
props = struct('HbO_C3', single(res{2}.hbo(:, 1)), 'HbR_C3', single(res{2}.hbr(:, 1)), ...
               'HbO_C4', single(res{2}.hbo(:, 2)), 'HbR_C4', single(res{2}.hbr(:, 2)));
out = struct();
out.GIFTIHeader = struct('Version', '1.0', 'MetaData', ...
                         struct('Description', 'redbird two-surface recon of the MCX reference data (uM)', ...
                                'LengthUnit', 'mm', 'alpha', alpha, 'beta', beta));
out.GIFTIData.MeshVertex3 = struct('Data', single(cortex), 'Properties', props);
out.GIFTIData.MeshTri3 = struct('Data', gt.GIFTIData.MeshTri3.Data);
savebj('', out, 'FileName', fullfile(here, 'recon_mcx.bgii'), 'Compression', 'zlib');
plotcompare(fullfile(here, 'figures'), gt, cortex, truth, res, data(:, 1));
end

% -------------------------------------------------------------------------
function [rhs, cmeas] = loaddata(fname)
m = loadmeas(fname);
dOD = double(m.dOD);
[npair, nwl, ~, ntrial] = size(dOD);
taxis = double(m.time(:));
[~, tidx] = min(abs(taxis - 10.0));
pre = taxis < 0;
cmeas = zeros(npair * nwl, 1);
rhs = zeros(npair * nwl, ntrial);
for w = 1:nwl
    rows = (w - 1) * npair + (1:npair);
    cmeas(rows) = var(reshape(dOD(:, w, pre, :), npair, []), 0, 2);
    rhs(rows, :) = -squeeze(dOD(:, w, tidx, :));
end
end

function r = solve(Anorm, cmeas, rhs, alpha, beta, ncortex, nscalp)
Nsurf = ncortex + nscalp;
wrow = 1 ./ sqrt(cmeas + 1e-6 * max(cmeas));
Aw = wrow .* Anorm;
cn = sqrt(sum(Aw.^2, 1))';
Wc = 1 ./ (cn + beta * max(cn));
Aw = Aw .* Wc';
lambda = alpha * max(sum(Aw.^2, 2));
r = struct('hbo', zeros(ncortex, 2), 'hbr', zeros(ncortex, 2), 'scalp', zeros(nscalp, 2));
for t = 1:size(rhs, 2)
    z = rbreginv(Aw, wrow .* rhs(:, t), lambda);
    d = z .* Wc;
    r.hbo(:, t) = d(1:ncortex);
    r.scalp(:, t) = d(ncortex + (1:nscalp));
    r.hbr(:, t) = d(Nsurf + (1:ncortex));
end
end

function [err, cc, pk, leak] = score(cv, sv, tr, cortex)
[~, ir] = max(cv);
[~, it] = max(tr);
err = norm(cortex(ir, :) - cortex(it, :));
c = corrcoef(cv, tr);
cc = c(1, 2);
pk = max(cv);
leak = max(abs(sv)) / max(abs(cv));
end

function plotcompare(figdir, gt, cortex, truth, res, names)
if ~exist(figdir, 'dir')
    mkdir(figdir);
end
face = double(gt.GIFTIData.MeshTri3.Data);
n = 256;
tt = linspace(0, 1, n)';
cmap = [min(1, 2 * tt), 1 - abs(2 * tt - 1), min(1, 2 * (1 - tt))];
set(0, 'DefaultFigureVisible', 'off');
fig = figure('Visible', 'off', 'Position', [1 1 1100 1100], 'Color', 'w');
rows = {truth{1}, truth{2}, 'truth'; res{1}.hbo(:, 1), res{1}.hbo(:, 2), names{1}; ...
        res{2}.hbo(:, 1), res{2}.hbo(:, 2), names{2}};
views = {[-90 0], [90 0]};
for i = 1:3
    for t = 1:2
        val = rows{i, t};
        subplot(3, 2, (i - 1) * 2 + t);
        plotmesh([cortex, val(:)], face, 'edgecolor', 'none', 'facecolor', 'interp');
        view(views{t});
        axis equal off;
        camlight;
        lighting gouraud;
        mm = max(abs(val));
        if mm == 0
            mm = 1;
        end
        caxis([-mm mm]);
        colormap(cmap);
        colorbar;
        title(sprintf('%s C%d (pk %.2f uM)', rows{i, 3}, 2 + t, max(val)), 'Interpreter', 'none');
    end
end
print(fig, fullfile(figdir, 'redbird_cedalion_vs_mcx.png'), '-dpng', '-r100');
close(fig);
end
