function s4_recon_surface(alpha, beta)
% S4_RECON_SURFACE  Reconstruct on Cedalion's TWO-SURFACE unknown space (cortex +
% scalp), instead of the gray-matter volume band used by s2/s3. This makes the
% redbird LHS (parameter mesh) match Cedalion's `TwoSurfaceHeadModel` with
% brain_only=False.
%
%   s4_recon_surface()            % alpha=0.01, beta=1
%   s4_recon_surface(0.03, 0.5)
%
% Method:
%   * redbird FEM forward -> baseline detphi + fluence phi (homogeneous/5-tissue)
%   * NODAL Jacobian on the forward mesh (rbfemmatrix, no remap)
%   * conservatively aggregate (node->vertex SUM) the nodal Jacobian onto the
%     surface vertices Vsurf=[cortex; scalp] (each node -> nearest vertex, summed,
%     total sensitivity conserved) -> a 2-D surface operator
%   * Rytov normalize (J/ymodel), chromophore stack -> W
%   * measurement whitening with c_meas (pre-stim dOD variance) + depth weighting
%   * Tikhonov solve -> dHbO/dHbR on the two surfaces
% Reconstructs C3, C4 and the C3-C4 contrast; scores the cortex part against the
% ground truth (same 25k vertices) and reports scalp leakage. Caches the surface
% operator in s4_op.mat so alpha/beta/c_meas can be retuned without a new forward.

if nargin < 1 || isempty(alpha)
    alpha = 0.01;
end
if nargin < 2 || isempty(beta)
    beta  = 1;
end
here = fileparts(mfilename('fullpath'));
hm = fullfile(here, '..', 'hm_icbm152');

%% ---- data + surfaces ----------------------------------------------------
meas = loadjd(fullfile(here, 'hrf_measurement.jdb'));
wl = double(meas.wavelengths(:))';
wlkey = arrayfun(@(x) num2str(x), wl, 'UniformOutput', false);
srcpos = double(meas.srcpos);
detpos = double(meas.detpos);
pairs = double(meas.pairs);
dOD = double(meas.dOD);
taxis = double(meas.time(:));
trials = meas.trialtype;
if ~iscell(trials)
    trials = cellstr(trials);
end
[~, tidx] = min(abs(taxis - 10.0));

gt = loadbj(fullfile(here, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);            % RAS, 25000 (== truth grid)
ncortex = size(cortex, 1);

[scalp_v, ~] = readobjmesh(fullfile(hm, 'mask_scalp.obj')); % voxel coords
A = getaffine(loadnifti(fullfile(hm, 'mask_skin.nii')));
scalp = (A(1:3, 1:3) * scalp_v' + A(1:3, 4))';            % -> RAS
nscalp = size(scalp, 1);
Vsurf = [cortex; scalp];                                  % unknown vertices [cortex; scalp]
fprintf('two-surface unknowns: %d cortex + %d scalp = %d\n', ncortex, nscalp, size(Vsurf, 1));

npair = size(pairs, 1);
nwl = numel(wl);

%% ---- build (or load) the surface operator -------------------------------
cachef = fullfile(here, 's4_op.mat');
if exist(cachef, 'file')
    fprintf('loading cached surface operator %s\n', cachef);
    S = load(cachef);
    Anorm = S.Anorm;
    cmeas = S.cmeas;
    ncortex = S.ncortex;
    nscalp = S.nscalp;
else
    mesh = loadjmesh(fullfile(here, 'head_mesh.bmsh'));
    node = meshpart(mesh.MeshVertex3);
    [etet, elabel] = meshpart(mesh.MeshTet4);
    elem = [etet, elabel];
    nseg = max(elem(:, 5));
    face = volface(elem(:, 1:4));
    nv = nodesurfnorm(node, face);
    [srcpos, srcdir] = snap2surf(node, face, nv, srcpos);
    [detpos, detdir] = snap2surf(node, face, nv, detpos);

    clear cfg;
    cfg.node = node;
    cfg.elem = elem;
    cfg.seg = elem(:, 5);
    cfg.srcpos = srcpos;
    cfg.srcdir = srcdir;
    cfg.detpos = detpos;
    cfg.detdir = detdir;
    %            scalp bone CSF  gray white
    tis.hbo = [25 15 2 60 40];
    tis.hbr = [12 8 1 25 18];
    tis.scatamp = [1.2 1.35 0.4 1.65 1.5];
    tis.scatpow = [1 1 1 1 1];
    sel = min(1:nseg, numel(tis.hbo));
    cfg.param = struct('hbo', tis.hbo(sel), 'hbr', tis.hbr(sel), ...
                       'scatamp500', tis.scatamp(sel), 'scatpow500', tis.scatpow(sel));
    cfg.prop = containers.Map();
    for w = 1:nwl
        cfg.prop(wlkey{w}) = [0 0 1 1; repmat([0.015 1 0 1.37], nseg, 1)];
    end
    cfg.omega = 0;
    cfg = rbmeshprep(cfg);

    ns = size(srcpos, 1);
    nd = size(detpos, 1);
    sdfull = rbsdmap(cfg);
    kf = sdfull.keys;
    for i = 1:numel(kf)
        t = sdfull(kf{i});
        t(:, 3) = 1;
        sdfull(kf{i}) = t;
    end
    [detphi_base, phi] = rbrunforward(cfg, 'sd', sdfull);
    if isa(phi, 'containers.Map')
        phimap = phi;
    else
        phimap = phi(1).phi;
    end

    % conservative node->vertex aggregation (matches Cedalion's binary voxel->vertex
    % SUM): assign each forward node to its NEAREST surface vertex and SUM the nodal
    % Jacobian columns. rbfemmatrix's nodal Jacobian is volume-integrated, so the sum
    % conserves the total sensitivity (each node counted exactly once) and is robust
    % to non-uniform FEM node density -- unlike the previous nearest-node gather,
    % which sampled one node per vertex (aliasing + dropped nodes, not conservative).
    vnode = nearestnodes(Vsurf, node);                       % nearest surf vertex per fwd node
    Sagg = sparse(1:size(node, 1), vnode, 1, size(node, 1), size(Vsurf, 1)); % [Nfwd x Nsurf]
    sdmeas = [pairs(:, 1), pairs(:, 2) + ns, ones(npair, 1), ones(npair, 1)];
    Js = containers.Map();
    for w = 1:nwl
        cfgw = cfg;
        cfgw.prop = cfg.prop(wlkey{w});
        Jw = rbfemmatrix(cfgw, sdmeas, phimap(wlkey{w}), cfg.deldotdel, 1);  % [npair x Nfwd]
        Js(wlkey{w}) = Jw * Sagg;                                           % [npair x Nsurf] (summed)
        clear Jw;
    end
    Jchrome = rbjacchrome(Js, {'hbo', 'hbr'});
    Amat = rbmatflat(Jchrome);                              % [npair*nwl x 2*Nsurf]

    ymodel = zeros(npair * nwl, 1);
    for w = 1:nwl
        bw = detphi_base(detkey(detphi_base, wlkey{w}));
        ymodel((w - 1) * npair + (1:npair)) = bw(sub2ind(size(bw), pairs(:, 2), pairs(:, 1)));
    end
    Anorm = Amat ./ ymodel;                                 % Rytov (OD) operator

    % c_meas: per-(pair,wl) noise variance from the pre-stimulus baseline (reltime<0)
    pre = taxis < 0;
    cmeas = zeros(npair * nwl, 1);
    for w = 1:nwl
        v = var(reshape(dOD(:, w, pre, :), npair, []), 0, 2);   % across pre-stim & trials
        cmeas((w - 1) * npair + (1:npair)) = v;
    end
    save(cachef, 'Anorm', 'cmeas', 'ncortex', 'nscalp', '-v7.3');
    fprintf('cached surface operator %s\n', cachef);
end
Nsurf = ncortex + nscalp;

%% ---- weighting + Tikhonov inverse operator ------------------------------
wrow = 1 ./ sqrt(cmeas + 1e-6 * max(cmeas));               % measurement (c_meas) whitening
Aw = wrow .* Anorm;                                        % row-whiten
cn = sqrt(sum(Aw.^2, 1))';
Wc = 1 ./ (cn + beta * max(cn));                           % depth (column) weighting
Aw = Aw .* Wc';
lambda = alpha * max(sum(Aw.^2, 2));
fprintf('alpha=%.3g beta=%.3g lambda=%.3e\n', alpha, beta, lambda);

%% ---- reconstruct C3, C4, contrast ---------------------------------------
truthC3 = double(gt.GIFTIData.MeshVertex3.Properties.HbO_C3);
truthC4 = double(gt.GIFTIData.MeshVertex3.Properties.HbO_C4);
targets = {'C3', dOD(:, :, tidx, 1); 'C4', dOD(:, :, tidx, 2); ...
           'C3-C4', dOD(:, :, tidx, 1) - dOD(:, :, tidx, 2)};
hbo = containers.Map();
hbr = containers.Map();                                    % d = [Hbo(1:Nsurf); Hbr(Nsurf+...)]
for j = 1:size(targets, 1)
    rhs = zeros(npair * nwl, 1);
    for w = 1:nwl
        rhs((w - 1) * npair + (1:npair)) = -targets{j, 2}(:, w);
    end
    z = rbreginv(Aw, wrow .* rhs, lambda);
    d = z .* Wc;
    h = d(1:Nsurf);
    hbo(targets{j, 1}) = h;
    hbr(targets{j, 1}) = d(Nsurf + (1:Nsurf));
    cortex_h = h(1:ncortex);
    scalp_h = h(ncortex + (1:nscalp));
    fprintf('%-6s cortex dHbO[%.2f %.2f] scalp dHbO[%.2f %.2f]  leak=%.2f\n', ...
            targets{j, 1}, min(cortex_h), max(cortex_h), min(scalp_h), max(scalp_h), ...
            max(abs(scalp_h)) / max(abs(cortex_h)));
end

%% ---- save the CORTEX result to recon.bgii (JGIfTI, same grid as ground truth) ----
cidx = 1:ncortex;
hboC3 = hbo('C3');
hbrC3 = hbr('C3');
hboC4 = hbo('C4');
hbrC4 = hbr('C4');
props = struct('HbO_C3', single(hboC3(cidx)), 'HbR_C3', single(hbrC3(cidx)), ...
               'HbO_C4', single(hboC4(cidx)), 'HbR_C4', single(hbrC4(cidx)));
out = struct();
out.GIFTIHeader = struct('Version', '1.0', 'MetaData', ...
                         struct('Description', 'redbird two-surface DOT recon (cortex dHbO/dHbR, uM)', ...
                                'LengthUnit', 'mm', 'alpha', alpha, 'beta', beta));
out.GIFTIData.MeshVertex3 = struct('Data', single(cortex), 'Properties', props);
out.GIFTIData.MeshTri3 = struct('Data', gt.GIFTIData.MeshTri3.Data);
savebj('', out, 'FileName', fullfile(here, 'recon.bgii'), 'Compression', 'zlib');
fprintf('saved recon.bgii (two-surface cortex result)\n');

%% ---- score cortex vs truth ----------------------------------------------
for tag = {'C3', 'C4'}
    h = hbo(tag{1});
    cv = h(1:ncortex);
    tr = double(gt.GIFTIData.MeshVertex3.Properties.(sprintf('HbO_%s', tag{1})));
    [~, ir] = max(cv);
    [~, it] = max(tr);
    cc = corrcoef(cv, tr);
    fprintf('score %s: peak-err %.1f mm, corr %.3f, pk %.2f (truth %.2f)\n', ...
            tag{1}, norm(cortex(ir, :) - cortex(it, :)), cc(1, 2), max(cv), max(tr));
end

%% ---- render cortex: truth vs redbird-surface ----------------------------
face_c = double(gt.GIFTIData.MeshTri3.Data);
n = 256;
tt = linspace(0, 1, n)';
cmap = [min(1, 2 * tt), 1 - abs(2 * tt - 1), min(1, 2 * (1 - tt))];
set(0, 'DefaultFigureVisible', 'off');
fig = figure('Visible', 'off', 'Position', [1 1 1500 900], 'Color', 'w');
panels = {truthC3, 'truth C3', [-90 0]; truthC4, 'truth C4', [90 0]; truthC3 - truthC4, 'truth C3-C4', [0 90]
          hbo('C3'), 'redbird C3', [-90 0]; hbo('C4'), 'redbird C4', [90 0]; hbo('C3-C4'), 'redbird C3-C4', [0 90]};
for i = 1:6
    val = panels{i, 1};
    val = val(1:ncortex);
    subplot(2, 3, i);
    plotmesh([cortex, val(:)], face_c, 'edgecolor', 'none', 'facecolor', 'interp');
    view(panels{i, 3});
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
    title(sprintf('%s (pk %.2f)', panels{i, 2}, max(val)), 'Interpreter', 'none');
end
print(fig, fullfile(here, 'recon_surface.png'), '-dpng', '-r110');
fprintf('wrote recon_surface.png\n');
end

% ===================================================================
function [arr, val] = meshpart(x)
val = [];
if isstruct(x)
    arr = double(x.Data);
    if isfield(x, 'Properties') && isfield(x.Properties, 'Value')
        val = double(x.Properties.Value);
    end
else
    arr = double(x);
end
end

function k = detkey(m, ck)
if isKey(m, ck)
    k = ck;
else
    k = str2double(ck);
end
end

function [p, dir] = snap2surf(node, face, nv, p)
si = unique(face(:));
k = nearestnodes(node(si, :), p);
ci = si(k);
dir = -nv(ci, :);
p = node(ci, :) + 1.0 * dir;
end

function idx = nearestnodes(node, q)
if exist('knnsearch', 'file')
    idx = knnsearch(node, q);
    return
end
idx = zeros(size(q, 1), 1);
for i = 1:size(q, 1)
    d = node - q(i, :);
    [~, idx(i)] = min(sum(d .* d, 2));
end
end

function A = getaffine(nii)
h = nii.NIFTIHeader;
if isfield(h, 'Affine') && ~isempty(h.Affine)
    A = h.Affine;
    if size(A, 1) == 3
        A = [A; 0 0 0 1];
    end
else
    A = eye(4);
end
end
