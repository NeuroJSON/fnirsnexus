function mcx_cedalion_check(varargin)
% MCX_CEDALION_CHECK  Recompute Cedalion's sensitivity with mcxlab using EXACTLY the
% conventions of cedalion.dot.ForwardModel, to test whether it reproduces the
% downloaded precomputed Adot that generated the sample data.
%
%   mcx_cedalion_check()                    % 1e8 photons per optode (Cedalion default)
%
% Cedalion conventions reproduced (forward_model.py):
%   * optode position: nearest scalp-mesh vertex -> nearest scalp voxel (integer IJK),
%     passed to MCX with issrcfrom0=1 (i.e. the voxel CORNER), no move to the skin
%   * every optode (sources AND detectors) a collimated pencil beam along
%     -(vertex normal); fluence, single 0-5 ns gate, isnormalized
%   * _fluence_at_optodes: emitter fluence in the voxel floor(pos) of the receiver,
%     stepping along the receiver's direction (<= 50 mm) while it is zero
%   * compute_sensitivity: Adot = (phi_s .* phi_d) @ voxel_to_vertex_brain / normfactor
%     x Vvox, normfactor = (f[s->d] + f[d->s]) / 2
% Output: mcx_cedalion_check.jdb and a comparison table vs Cedalion's peak dOD; with
% saveadot, adot_cedalion.mat: the full Adot [pair x vertex] (25k cortex + 10k scalp,
% Cedalion's voxel_to_vertex_brain/scalp maps, mm), the original Adot reproduced (the
% copy on Cedalion's server has since been replaced).

opt = struct('nphoton', 1e8, ...
             'gpuid', '10', ...        % shui: RTX 5090
             'saveadot', true, ...     % export the full Adot (brain + scalp vertices)
             'seed', 20261028, ...
             'fluencedir', '', ...
             'outfile', '');
for i = 1:2:numel(varargin)
    opt.(varargin{i}) = varargin{i + 1};
end
here = fileparts(mfilename('fullpath'));
if isempty(opt.fluencedir)
    opt.fluencedir = fullfile(here, 'fluence_cedalion');
end
if isempty(opt.outfile)
    opt.outfile = fullfile(here, 'mcx_cedalion_check.jdb');
end
if ~exist(opt.fluencedir, 'dir')
    mkdir(opt.fluencedir);
end

%% ---- model, Cedalion optode positions (no skin move) ---------------------
M = headmodel(here);
hm = fullfile(here, '..', 'hm_icbm152');
A = M.A;
ras2ijk = @(p) (A(1:3, 1:3) \ (p' - A(1:3, 4)))';
[sv, sf] = readobjmesh(fullfile(hm, 'mask_scalp.obj'));
sn = nodesurfnorm(sv, sf);
[cs, cdirs] = snapoptode(ras2ijk(double(M.meas.srcpos)), sv, sn, M.vol == 3);
[cdet, cdird] = snapoptode(ras2ijk(double(M.meas.detpos)), sv, sn, M.vol == 3);
pairs = M.pairs;
trials = M.trials;
wl = M.wl;
npair = size(pairs, 1);
nwl = numel(wl);
ntrial = numel(trials);
ns = size(cs, 1);
nopt = ns + size(cdet, 1);
optijk = [cs; cdet];                                       % integer, 0-based (Cedalion)
optdir = [cdirs; cdird];

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
optpos = optijk - (lo - 1);                               % cropped grid, still the corner
dmua = reshape(M.dmuavox, numel(M.vox), []);

%% ---- one pencil-beam fluence per optode (cached) -------------------------
clear cfg;
cfg.vol = vol;
cfg.prop = M.prop0;
cfg.unitinmm = M.unitinmm;
cfg.tstart = 0;
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.issrcfrom0 = 1;
cfg.isnormalized = 1;
cfg.outputtype = 'fluence';
cfg.srctype = 'pencil';
cfg.nphoton = opt.nphoton;
cfg.gpuid = opt.gpuid;
cfg.autopilot = 1;
sig = [opt.nphoton, dims, lo, opt.seed];
phi = zeros(prod(dims), nopt, 'single');
tall = tic;
nrun = 0;
for o = 1:nopt
    fname = fullfile(opt.fluencedir, sprintf('optode%03d.bin', o));
    sfile = [fname(1:end - 4) '.sig'];
    if exist(fname, 'file') && exist(sfile, 'file') && isequal(load(sfile, '-ascii'), sig)
        fid = fopen(fname, 'r');
        phi(:, o) = fread(fid, prod(dims), 'single=>single');
        fclose(fid);
        continue
    end
    cfg.srcpos = optpos(o, :);
    cfg.srcdir = optdir(o, :);
    cfg.seed = opt.seed + o;
    f = mcxlab(cfg);
    phi(:, o) = single(f.data(:));
    clear f;
    fid = fopen(fname, 'w');
    fwrite(fid, phi(:, o), 'single');
    fclose(fid);
    save(sfile, 'sig', '-ascii', '-double');
    nrun = nrun + 1;
    if mod(o, 20) == 0 || o == nopt
        fprintf('optode %3d/%d: %d MCX runs so far, %.1fs elapsed\n', o, nopt, nrun, toc(tall));
    end
end
fprintf('fluences: %d MCX runs, %d reused, %.1fs\n', nrun, nopt - nrun, toc(tall));

%% ---- Cedalion normalization and sensitivity ------------------------------
% fluence_at_optodes[e, r]: emitter e's fluence in the voxel floor(pos_r), stepping
% along r's direction (1 voxel per step, <= 50 mm) while zero
maxstep = ceil(50 / M.unitinmm);
recvox = cell(nopt, 1);
for r = 1:nopt
    lst = zeros(maxstep, 1);
    for st = 0:maxstep - 1
        p = floor(optpos(r, :) + st * optdir(r, :)) + 1;
        lst(st + 1) = sub2ind(dims, p(1), p(2), p(3));
    end
    recvox{r} = lst;
end
fat = @(e, r) firstpos(phi(recvox{r}, e));
nf = zeros(npair, 1);
Jdmua = zeros(npair, ntrial * nwl);
Jsum = zeros(npair, 1);
if opt.saveadot
    [svox, svert] = readvox2vert(fullfile(hm, 'voxel_to_vertex_scalp.mtx.gz'), size(M.vol));
    [si, sj, sk] = ind2sub(size(M.vol), svox);
    svoxc = sub2ind(dims, si - lo(1) + 1, sj - lo(2) + 1, sk - lo(3) + 1);
    nbrain = max(M.vert);
    nscalp = max(svert);
    Agg = [sparse(voxc, M.vert, 1, prod(dims), nbrain), ...
           sparse(svoxc, svert, 1, prod(dims), nscalp)];   % [nvoxel x (brain+scalp)]
    Adot = zeros(npair, nbrain + nscalp, 'single');
end
for s = 1:ns
    prow = find(pairs(:, 1) == s);
    od = ns + pairs(prow, 2);
    J = phi(:, s) .* phi(:, od) * M.unitinmm^3;
    for k = 1:numel(prow)
        nf(prow(k)) = 0.5 * (fat(s, od(k)) + fat(od(k), s));
    end
    Jsum(prow) = double(sum(J, 1, 'double'))';
    Jdmua(prow, :) = double(J(voxc, :))' * dmua;
    if opt.saveadot
        for k = 1:numel(prow)
            Adot(prow(k), :) = single((double(J(:, k))' * Agg) / nf(prow(k)));
        end
    end
end
clear J;
if opt.saveadot
    is_brain = [true(1, nbrain), false(1, nscalp)];
    save(fullfile(here, 'adot_cedalion.mat'), 'Adot', 'is_brain', 'pairs', '-v7');
    fprintf('saved adot_cedalion.mat: %d pairs x (%d brain + %d scalp) vertices\n', npair, ...
            nbrain, nscalp);
end
dOD = permute(reshape(Jdmua ./ nf, npair, ntrial, nwl), [1 3 2]);    % pair x wl x trial
ok = nf > 0;

%% ---- compare --------------------------------------------------------------
[maxOD_ced, K, ~] = mcx_build_hrf('fitonly', true);
mj = loadjd(fullfile(here, 'mcx_jacobian.jdb'));          % physical model (for reference)
fprintf('\n== MCX with Cedalion''s exact conventions vs Cedalion''s precomputed Adot ==\n');
fprintf('%-10s %10s %14s %14s %22s\n', 'trial', 'r', 'slope', 'median ratio', ...
        'physical model slope');
for t = 1:ntrial
    x = maxOD_ced(ok, 2, t);
    y = dOD(ok, 2, t);
    z = double(mj.maxOD(ok, 2, t));
    big = abs(x) > 0.1 * max(abs(x));
    fprintf('%-10s %10.4f %14.3f %14.3f %22.3f\n', trials{t}, pearson(x, y), (x' * y) / (x' * x), ...
            median(y(big) ./ x(big)), (x' * z) / (x' * x));
end
fprintf('total path sum J/norm: median %.1f mm (physical model %.1f mm)\n', ...
        median(Jsum(ok) ./ nf(ok)), median(double(mj.Ltotal(ok))));

out = struct('format', 'mcx-cedalion-check', 'pairs', int32(pairs), 'wavelengths', wl, ...
             'trialtype', {trials}, 'maxOD', dOD, 'maxOD_cedalion', maxOD_ced, ...
             'normfactor', nf, 'Ltotal', Jsum ./ nf, 'optijk', optijk, 'optdir', optdir, ...
             'settings', opt);
savebj('', out, 'FileName', opt.outfile, 'Compression', 'zlib');
fprintf('saved %s\n', opt.outfile);
end

% -------------------------------------------------------------------------
function v = firstpos(x)
k = find(x > 0, 1);
if isempty(k)
    v = 0;
else
    v = double(x(k));
end
end

function r = pearson(x, y)
c = corrcoef(x(:), y(:));
r = c(1, 2);
end
