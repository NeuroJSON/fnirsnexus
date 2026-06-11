function s3_recon_official(trial, maxiter)
% S3_RECON_OFFICIAL  Iterative Gauss-Newton fNIRS reconstruction through the OFFICIAL
% redbird path (rbrun/rbrunrecon) using the new recon.isratio flag + Rytov ('logphase').
%
%   s3_recon_official()        % Stim C4, 4 iterations
%   s3_recon_official(1, 3)    % Stim C3, 3 iterations
%
% Unlike s2_recon (a single fixed linear step), this updates the optical properties
% each iteration and re-runs the forward + Jacobian, so it reports a genuine
% multi-iteration residual curve. Homogeneous baseline (node-wise) + gray-matter
% reconstruction submesh.

if (nargin < 1 || isempty(trial));   trial = 2;   end     % 2 = Stim C4
if (nargin < 2 || isempty(maxiter)); maxiter = 4; end
here = fileparts(mfilename('fullpath'));
addtoolboxpaths();

%% inputs
mesh = loadjmesh(fullfile(here, 'head_mesh.bmsh'));
node = meshpart(mesh.MeshVertex3);
[etet, elabel] = meshpart(mesh.MeshTet4);
elem = [etet, elabel];                                    % 5th col = tissue label
Nn = size(node, 1);

meas = loadjd(fullfile(here, 'hrf_measurement.jdb'));
wl = double(meas.wavelengths(:))';
wlkey = arrayfun(@(x) num2str(x), wl, 'UniformOutput', false);
srcpos = double(meas.srcpos); detpos = double(meas.detpos);
pairs = double(meas.pairs); dOD = double(meas.dOD); taxis = double(meas.time(:));
trials = meas.trialtype; if (~iscell(trials)); trials = cellstr(trials); end
[~, tidx] = min(abs(taxis - 10.0));
fprintf('iterative recon: %s, t=%.2fs, maxiter=%d\n', trials{trial}, taxis(tidx), maxiter);

%% optodes onto scalp
face = volface(elem(:, 1:4)); nv = nodesurfnorm(node, face);
[srcpos, srcdir] = snap2surf(node, face, nv, srcpos);
[detpos, detdir] = snap2surf(node, face, nv, detpos);

%% homogeneous baseline (NODE-WISE param so multi-iter sync preserves out-of-mesh nodes)
bhbo = 28; bhbr = 14;
clear cfg;
cfg.node = node; cfg.elem = [elem(:, 1:4), ones(size(elem, 1), 1)];
cfg.seg = ones(size(elem, 1), 1);
cfg.srcpos = srcpos; cfg.srcdir = srcdir; cfg.detpos = detpos; cfg.detdir = detdir;
cfg.param = struct('hbo', bhbo * ones(Nn, 1), 'hbr', bhbr * ones(Nn, 1));   % node-wise
cfg.prop = containers.Map();
for w = 1:numel(wl); cfg.prop(wlkey{w}) = [0 0 1 1; 0.01 1.0 0 1.37]; end
cfg.omega = 0;
cfg = rbmeshprep(cfg);

%% source-detector map (full grid; activate only the measured pairs)
ns = size(srcpos, 1); nd = size(detpos, 1);
sd = rbsdmap(cfg);
rows = (pairs(:, 1) - 1) * nd + pairs(:, 2);
for w = 1:numel(wl)
    t = sd(wlkey{w}); t(:, 3) = 0; t(rows, 3) = 1; sd(wlkey{w}) = t;
end

%% ratiometric data I/I0 = exp(-dOD): per-wavelength [Ndet x Nsrc], 1 on unused pairs
ratio = containers.Map();
for w = 1:numel(wl)
    R = ones(nd, ns);
    for p = 1:size(pairs, 1)
        R(pairs(p, 2), pairs(p, 1)) = exp(-dOD(p, w, tidx, trial));
    end
    ratio(wlkey{w}) = R;
end

%% gray-matter reconstruction submesh (label 4)
gmel = elem(elem(:, 5) == 4, 1:4);
gmnodes = unique(gmel);
newidx = zeros(Nn, 1); newidx(gmnodes) = 1:numel(gmnodes);
clear recon;
recon.node = node(gmnodes, :);
recon.elem = newidx(gmel);
[recon.mapid, recon.mapweight] = tsearchn(recon.node, recon.elem, cfg.node);
recon.bulk = struct('hbo', bhbo, 'hbr', bhbr);
recon.param = struct('hbo', bhbo, 'hbr', bhbr);
recon.prop = containers.Map(wlkey, repmat({[]}, 1, numel(wl)));
recon.lambda = 0.1;
recon.isratio = 1;                                        % <-- ratiometric (fNIRS) data

%% iterative Gauss-Newton (Rytov)
[newrecon, resid, newcfg] = rbrun(cfg, recon, ratio, sd, 'mode', 'image', ...
                                  'maxiter', maxiter, 'reform', 'logphase', ...
                                  'lambda', recon.lambda);
resid = resid(:)';
fprintf('residual per iter      : %s\n', sprintf('%.4e ', resid));
fprintf('relative residual r/r1 : %s\n', sprintf('%.4f ', resid / resid(1)));

dhbo = newrecon.param.hbo(:) - bhbo;
fprintf('final dHbO range [%.3f %.3f] uM on %d GM nodes\n', min(dhbo), max(dhbo), numel(dhbo));

%% map onto cortex and render
gt = loadbj(fullfile(here, 'groundtruth.bgii'));
node_c = double(gt.GIFTIData.MeshVertex3.Data); face_c = double(gt.GIFTIData.MeshTri3.Data);
tag = strrep(strrep(trials{trial}, 'Stim ', ''), ' ', '');
cnode = knnsearch(recon.node, node_c);
cval = dhbo(cnode);
truth = double(gt.GIFTIData.MeshVertex3.Properties.(sprintf('HbO_%s', tag)));

n = 256; tt = linspace(0, 1, n)';
cmap = [min(1, 2 * tt), 1 - abs(2 * tt - 1), min(1, 2 * (1 - tt))];
vw = [-90 0]; if (strcmp(tag, 'C4')); vw = [90 0]; end
set(0, 'DefaultFigureVisible', 'off');
fig = figure('Visible', 'off', 'Position', [1 1 1300 520], 'Color', 'w');
subplot(1, 2, 1);
plotmesh([node_c, truth(:)], face_c, 'edgecolor', 'none', 'facecolor', 'interp');
view(vw); axis equal off; camlight; lighting gouraud; caxis([-5 5]); colormap(cmap); colorbar;
title(sprintf('truth %s (pk %.2f uM)', tag, max(truth)), 'Interpreter', 'none');
subplot(1, 2, 2);
plotmesh([node_c, cval], face_c, 'edgecolor', 'none', 'facecolor', 'interp');
view(vw); axis equal off; camlight; lighting gouraud;
mm = max(abs(cval)); if (mm == 0); mm = 1; end; caxis([-mm mm]); colormap(cmap); colorbar;
title(sprintf('redbird %s, %d iters (pk %.2f uM)', tag, maxiter, max(cval)), 'Interpreter', 'none');
print(fig, fullfile(here, sprintf('recon_iter_%s.png', tag)), '-dpng', '-r110');
fprintf('wrote recon_iter_%s.png\n', tag);
end

% ------------------------------------------------------------------
function [arr, val] = meshpart(x)
val = [];
if (isstruct(x))
    arr = double(x.Data);
    if (isfield(x, 'Properties') && isfield(x.Properties, 'Value')); val = double(x.Properties.Value); end
else
    arr = double(x);
end
end

function [p, dir] = snap2surf(node, face, nv, p)
surfidx = unique(face(:));
k = knnsearch(node(surfidx, :), p); ci = surfidx(k);
dir = -nv(ci, :); p = node(ci, :) + 1.0 * dir;
end

function addtoolboxpaths()
root = '/home/users/fangq/space/git/Project';
cand = {fullfile(root, 'github', 'redbird-m', 'matlab'), fullfile(root, 'github', 'iso2mesh'), ...
        fullfile(root, 'jsonlab'), fullfile(root, 'github', 'jnifty'), ...
        fullfile(root, 'github', 'jmesh', 'lib', 'matlab')};
for i = 1:numel(cand); if (exist(cand{i}, 'dir')); addpath(cand{i}); end; end
end
