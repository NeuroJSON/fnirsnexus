function s4_recon_matched(alpha, beta, measfile, csfmus, agg)
% S4_RECON_MATCHED  s4_recon_surface with the forward problem and the Jacobian
% aggregation matched to Cedalion's ForwardModel, to test whether redbird and Cedalion
% give the same reconstruction from the same data.
%
%   s4_recon_matched()                                   % alpha=0.003, beta=0.01, this folder's data
%   s4_recon_matched(0.003, 0.01, '../mcx_ref/hrf_measurement.jdb')
%   s4_recon_matched([], [], [], 0.15)                   % diffusion-equivalent CSF (CSF paper)
%   s4_recon_matched([], [], [], [], 'node')             % node->vertex aggregation instead
%
% Differences from s4_recon_surface (everything else is identical):
%   1. optics: Cedalion's tissue table (cedalion/dot/tissue_properties.py) in cfg.prop,
%      wavelength independent, n = 1, g = 0.001; no cfg.param (HbO/HbR-based mua)
%   2. aggregation, as Cedalion's compute_sensitivity (agg='voxel', default): the FEM
%      fluences are interpolated (barycentric, P = voxel-centre x node) to the centres of
%      the gray/white and scalp voxels, J(v) = -phi_s(v) phi_d(v) Vvox, and summed into
%      vertices with Cedalion's own voxel_to_vertex_brain/scalp maps (skull and CSF voxels
%      are in neither map and are dropped).
%      agg='node': rbfemmatrix nodal Jacobian, gray/white nodes -> nearest cortex vertex,
%      scalp nodes -> nearest scalp vertex (majority tissue label), skull/CSF dropped.
% The operator (data independent) is cached in s4_op_matched[_csf<mus>][_voxel].mat;
% c_meas is recomputed from measfile. Writes recon_matched_<tag>.bgii.

if nargin < 1 || isempty(alpha)
    alpha = 0.003;
end
if nargin < 2 || isempty(beta)
    beta = 0.01;
end
here = fileparts(mfilename('fullpath'));
if nargin < 3 || isempty(measfile)
    measfile = fullfile(here, 'hrf_measurement.jdb');
end
if nargin < 4 || isempty(csfmus)
    csfmus = 0.01;                                         % Cedalion's (nearly clear) CSF
end
if nargin < 5 || isempty(agg)
    agg = 'voxel';
end
hm = fullfile(here, '..', 'hm_icbm152');

%% ---- data + surfaces ----------------------------------------------------
meas = loadjd(measfile);
wl = double(meas.wavelengths(:))';
wlkey = arrayfun(@(x) num2str(x), wl, 'UniformOutput', false);
srcpos = double(meas.srcpos);
detpos = double(meas.detpos);
pairs = double(meas.pairs);
dOD = double(meas.dOD);
taxis = double(meas.time(:));
[~, tidx] = min(abs(taxis - 10.0));

gt = loadbj(fullfile(here, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);
ncortex = size(cortex, 1);
[scalp_v, ~] = readobjmesh(fullfile(hm, 'mask_scalp.obj'));
A = getaffine(loadnifti(fullfile(hm, 'mask_skin.nii')));
scalp = (A(1:3, 1:3) * scalp_v' + A(1:3, 4))';
nscalp = size(scalp, 1);
Nsurf = ncortex + nscalp;
npair = size(pairs, 1);
nwl = numel(wl);

%% ---- build (or load) the matched surface operator -----------------------
cachef = 's4_op_matched';
if csfmus ~= 0.01
    cachef = sprintf('%s_csf%g', cachef, csfmus);
end
if strcmp(agg, 'voxel')
    cachef = [cachef '_voxel'];
end
cachef = fullfile(here, [cachef '.mat']);
if exist(cachef, 'file')
    fprintf('loading cached matched operator %s\n', cachef);
    S = load(cachef);
    Anorm = S.Anorm;
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
    % Cedalion tissue_properties.py, [mua mus g n] (1/mm); mesh labels 1..5 =
    % scalp bone CSF gray white
    ced = [0.0191 0.66 0.001 1      % scalp
           0.0136 0.86 0.001 1      % skull
           0.0026 csfmus 0.001 1    % CSF
           0.0186 1.10 0.001 1      % gray
           0.0186 1.10 0.001 1];    % white
    if nseg > size(ced, 1)
        error('mesh has %d tissue labels, expected 5', nseg);
    end
    cfg.prop = containers.Map();
    for w = 1:nwl
        cfg.prop(wlkey{w}) = [0 0 1 1; ced(1:nseg, :)];
    end
    cfg.omega = 0;
    cfg = rbmeshprep(cfg);

    ns = size(srcpos, 1);
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

    nn = size(node, 1);
    sdmeas = [pairs(:, 1), pairs(:, 2) + ns, ones(npair, 1), ones(npair, 1)];
    Js = containers.Map();
    if strcmp(agg, 'voxel')
        % Cedalion's compute_sensitivity: voxel-centre fluence products, summed into
        % vertices with its voxel_to_vertex_brain/scalp maps
        nii = loadnifti(fullfile(hm, 'mask_skin.nii'));
        dims = size(nii.NIFTIData);
        clear nii;
        [bvox, bvert] = readvox2vert(fullfile(hm, 'voxel_to_vertex_brain.mtx.gz'), dims);
        [svox, svert] = readvox2vert(fullfile(hm, 'voxel_to_vertex_scalp.mtx.gz'), dims);
        vox = [bvox; svox];
        nvox = numel(vox);
        [ii, jj, kk] = ind2sub(dims, vox);
        pts = double((A(1:3, 1:3) * ([ii, jj, kk] - 1)' + A(1:3, 4))');  % voxel centres, RAS
        [ti, bc] = pointLocation(triangulation(double(elem(:, 1:4)), double(node)), pts);
        outside = isnan(ti);
        cols = ones(nvox, 4);
        vals = zeros(nvox, 4);
        cols(~outside, :) = elem(ti(~outside), 1:4);
        vals(~outside, :) = bc(~outside, :);
        cols(outside, 1) = nearestnodes(node, pts(outside, :));
        vals(outside, 1) = 1;
        P = sparse(repmat((1:nvox)', 4, 1), cols(:), vals(:), nvox, nn);
        G = sparse(1:nvox, [bvert; ncortex + svert], 1, nvox, Nsurf);
        Vvox = abs(det(A(1:3, 1:3)));
        fprintf('voxel aggregation: %d brain + %d scalp voxels (%d outside the mesh -> nearest node)\n', ...
                numel(bvox), numel(svox), nnz(outside));
        clear pts ti bc cols vals ii jj kk;
        for w = 1:nwl
            phw = phimap(wlkey{w});
            Pphi = zeros(nvox, size(phw, 2), 'single');
            for c = 1:20:size(phw, 2)
                cc = c:min(c + 19, size(phw, 2));
                Pphi(:, cc) = single(P * phw(:, cc));
            end
            Jw = zeros(npair, Nsurf);
            for s = 1:ns
                prow = find(pairs(:, 1) == s);
                if isempty(prow)
                    continue
                end
                Jv = double(Pphi(:, s) .* Pphi(:, ns + pairs(prow, 2)));
                Jw(prow, :) = -Vvox * (G' * Jv)';
            end
            Js(wlkey{w}) = Jw;
            clear Pphi Jw;
        end
        clear P G;
    else
    % tissue-aware conservative aggregation (Cedalion's voxel_to_vertex_brain/scalp):
    % gray/white nodes -> nearest cortex vertex, scalp nodes -> nearest scalp vertex
    cnt = sparse(reshape(elem(:, 1:4), [], 1), repmat(elem(:, 5), 4, 1), 1, nn, nseg);
    [~, nlab] = max(cnt, [], 2);
    nlab = full(nlab);
    inb = find(nlab == 4 | nlab == 5);
    ins = find(nlab == 1);
    vb = nearestnodes(cortex, node(inb, :));
    vs = nearestnodes(scalp, node(ins, :));
    Sagg = sparse([inb; ins], [vb; ncortex + vs], 1, nn, Nsurf);
    fprintf('aggregation: %d gray/white nodes -> cortex, %d scalp nodes -> scalp, %d skull/CSF nodes dropped\n', ...
            numel(inb), numel(ins), nn - numel(inb) - numel(ins));

    for w = 1:nwl
        cfgw = cfg;
        cfgw.prop = cfg.prop(wlkey{w});
        Jw = rbfemmatrix(cfgw, sdmeas, phimap(wlkey{w}), cfg.deldotdel, 1);
        Js(wlkey{w}) = Jw * Sagg;
        clear Jw;
    end
    end
    Jchrome = rbjacchrome(Js, {'hbo', 'hbr'});
    Amat = rbmatflat(Jchrome);

    ymodel = zeros(npair * nwl, 1);
    for w = 1:nwl
        bw = detphi_base(detkey(detphi_base, wlkey{w}));
        ymodel((w - 1) * npair + (1:npair)) = bw(sub2ind(size(bw), pairs(:, 2), pairs(:, 1)));
    end
    Anorm = Amat ./ ymodel;
    save(cachef, 'Anorm', 'ncortex', 'nscalp', '-v7.3');
    fprintf('cached matched operator %s\n', cachef);
end

% c_meas from this dataset's pre-stimulus baseline (as in s4_recon_surface)
pre = taxis < 0;
cmeas = zeros(npair * nwl, 1);
for w = 1:nwl
    cmeas((w - 1) * npair + (1:npair)) = var(reshape(dOD(:, w, pre, :), npair, []), 0, 2);
end

%% ---- weighting + Tikhonov (identical to s4_recon_surface) ---------------
wrow = 1 ./ sqrt(cmeas + 1e-6 * max(cmeas));
Aw = wrow .* Anorm;
cn = sqrt(sum(Aw.^2, 1))';
Wc = 1 ./ (cn + beta * max(cn));
Aw = Aw .* Wc';
lambda = alpha * max(sum(Aw.^2, 2));
fprintf('alpha=%.3g beta=%.3g lambda=%.3e  data=%s\n', alpha, beta, lambda, measfile);

hbo = zeros(ncortex, 2);
hbr = zeros(ncortex, 2);
tags = {'C3', 'C4'};
for t = 1:2
    rhs = zeros(npair * nwl, 1);
    for w = 1:nwl
        rhs((w - 1) * npair + (1:npair)) = -dOD(:, w, tidx, t);
    end
    d = rbreginv(Aw, wrow .* rhs, lambda) .* Wc;
    hbo(:, t) = d(1:ncortex);
    hbr(:, t) = d(Nsurf + (1:ncortex));
    tr = double(gt.GIFTIData.MeshVertex3.Properties.(['HbO_' tags{t}]));
    [~, ir] = max(hbo(:, t));
    [~, it] = max(tr);
    cc = corrcoef(hbo(:, t), tr);
    fprintf('score %s: peak-err %.1f mm, corr %.3f, pk %.2f (truth %.2f), leak %.2f\n', tags{t}, ...
            norm(cortex(ir, :) - cortex(it, :)), cc(1, 2), max(hbo(:, t)), max(tr), ...
            max(abs(d(ncortex + (1:nscalp)))) / max(abs(hbo(:, t))));
end

[~, tag] = fileparts(fileparts(measfile));
props = struct('HbO_C3', single(hbo(:, 1)), 'HbR_C3', single(hbr(:, 1)), ...
               'HbO_C4', single(hbo(:, 2)), 'HbR_C4', single(hbr(:, 2)));
out = struct();
out.GIFTIHeader = struct('Version', '1.0', 'MetaData', ...
                         struct('Description', 'redbird two-surface recon, Cedalion-matched forward (uM)', ...
                                'LengthUnit', 'mm', 'alpha', alpha, 'beta', beta));
out.GIFTIData.MeshVertex3 = struct('Data', single(cortex), 'Properties', props);
out.GIFTIData.MeshTri3 = struct('Data', gt.GIFTIData.MeshTri3.Data);
savebj('', out, 'FileName', fullfile(here, sprintf('recon_matched_%s.bgii', tag)), 'Compression', 'zlib');
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

function [vox, vert] = readvox2vert(fname, dims)
% Cedalion's Matrix-Market voxel->vertex map (rows = C-order flat voxel index)
tmp = tempname;
gunzip(fname, tmp);
[~, base] = fileparts(fname);
fid = fopen(fullfile(tmp, base), 'rt');
line = fgetl(fid);
while startsWith(line, '%')
    line = fgetl(fid);
end
d = textscan(fid, '%f %f %f');
fclose(fid);
rmdir(tmp, 's');
r = d{1} - 1;
k = mod(r, dims(3));
j = mod(floor(r / dims(3)), dims(2));
i = floor(r / (dims(2) * dims(3)));
vox = sub2ind(dims, i + 1, j + 1, k + 1);
vert = d{2};
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
