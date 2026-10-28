function mcx_forward(varargin)
% MCX_FORWARD  Non-inverse-crime channel-space forward data for the ICBM-152 /
% NinjaNIRS-22 image-recon challenge, computed by voxel Monte Carlo (mcxlab) on the
% raw segmented ICBM-152 head instead of Cedalion's precomputed sensitivity Adot.
%
%   mcx_forward()                              % all sources, default settings
%   mcx_forward('nphoton', 1e8, 'nbatch', 20)  % more photons
%   mcx_forward('srclist', 1:4)                % only some sources (resumable)
%
% Cedalion generated the reference (cedalion_ref/hmstandardization_challenge_prep)
% as   dOD_peak = Adot * E * [dHbO; dHbR]   (Rytov-linear, on the 25k cortex vertices)
% and reconstructs with the same Adot -> inverse crime. Here the identical activation
% (cedalion_ref/spatial_act.csv) is instead placed into the voxel head through
% Cedalion's own voxel->vertex map (voxel_to_vertex_brain.mtx.gz) and propagated
% with MCX, using the SAME tissue labels/optical properties, optode snapping,
% inward source direction, time gate and Prahl extinction as Cedalion's
% ForwardModel.compute_fluence_mcx. What differs: a fully nonlinear perturbation-MC
% intensity ratio at real detectors replaces the linear adjoint Adot.
%
% Method (perturbation MC, exact for the simulated photons):
%   * the activation dmua(voxel) is quantized into extra labels (linear bins + a few
%     log-spaced tail bins; each label takes the MEAN dmua of its voxels, so the
%     integrated perturbation is conserved per label) -> one label volume
%   * mua does not change MCX photon paths (absorption = weight decay), and Cedalion's
%     scattering is wavelength independent -> ONE MCX run per source serves both
%     wavelengths, both conditions and any activation amplitude
%   * for each detected photon (partial paths L_m per label):
%        w_base = exp(-sum_m mua0_m L_m),  w_act(a) = w_base * exp(-a * sum_m dmua_m L_m)
%     dOD(a) = -ln( sum w_act(a) / sum w_base ) per source-detector pair, for a grid
%     of amplitudes a in [0,1] (the HRF regressor is scaled into [0,1] by Cedalion)
%
% Output: mcx_dod.mat/.jdb with dOD_amp [pair x wl x trial x amp], the
% peak maxOD = dOD(a=1), its MC standard error, baseline intensities and settings.
% Per-source results are cached in cache/ so an interrupted run resumes.
% Next step: mcx_build_hrf.m (temporal augmentation -> hrf_measurement.mat).
%
% Requires mcxlab (+ utils), iso2mesh (readobjmesh, nodesurfnorm), jsonlab/jnifti.

opt = struct('nphoton', 1e8, ...       % photons per MCX batch
             'nbatch', 20, ...         % batches per source (independent seeds)
             'nlin', 32, ...           % linear dmua bins on [1/nlin, 1] x peak
             'nlog', 8, ...            % log-spaced tail bins on [qmin, 1/nlin] x peak
             'qmin', 1e-4, ...         % drop voxels below qmin x peak (Gaussian tail)
             'detradius', 3, ...       % detector radius (mm)
             'maxdetphoton', 5e6, ...  % detected-photon buffer per batch
             'namp', 21, ...           % activation amplitudes a = linspace(0,1,namp)
             'gpuid', '11', ...        % both GPUs
             'seed', 20260610, ...
             'srclist', [], ...        % [] = all sources
             'outfile', '');
for i = 1:2:numel(varargin)
    opt.(varargin{i}) = varargin{i + 1};
end
here = fileparts(mfilename('fullpath'));
if isempty(opt.outfile)
    opt.outfile = fullfile(here, 'mcx_dod.mat');
end
cachedir = fullfile(here, 'cache');
if ~exist(cachedir, 'dir')
    mkdir(cachedir);
end

%% ---- 1-3. Cedalion-matched head model, optodes, activation -> voxels --------
M = headmodel(here);
vol = M.vol;
unitinmm = M.unitinmm;
prop0 = M.prop0;
wl = M.wl;
pairs = M.pairs;
trials = M.trials;
srcmcx = M.srcmcx;
srcdir = M.srcdir;
detmcx = M.detmcx;
vox = M.vox;
dmuavox = M.dmuavox;
npair = size(pairs, 1);
nwl = numel(wl);
ntrial = numel(trials);
ns = size(M.meas.srcpos, 1);

% bin each trial on q = |dmua|/peak (norm over wavelengths): nlog log tail + nlin linear
q = sqrt(sum(dmuavox.^2, 3));
q = q ./ max(q, [], 1);
edges = unique([0, logspace(log10(opt.qmin), log10(1 / opt.nlin), opt.nlog + 1), ...
                linspace(1 / opt.nlin, 1, opt.nlin)]);
bin = zeros(size(q));
for t = 1:ntrial
    bin(:, t) = discretize(q(:, t), edges) - 1;           % 0 = below qmin (dropped)
    bin(isnan(bin(:, t)), t) = 0;
    dropped = sum(q(bin(:, t) == 0, t)) / sum(q(:, t));
    fprintf('  %s: %d active voxels, dropped tail fraction %.2e\n', trials{t}, ...
            nnz(bin(:, t)), dropped);
end

% base tissues with identical properties share labels (gm == wm in Cedalion)
[~, ~, baseclass] = unique(prop0(double(vol(vox)) + 1, :), 'rows');
key = [baseclass, bin];
isact = any(bin > 0, 2);
[ukey, ~, lid] = unique(key(isact, :), 'rows');
nlabel0 = size(prop0, 1) - 1;
nnew = size(ukey, 1);
if nlabel0 + nnew > 255
    error('%d perturbation labels exceed uint8; reduce nlin/nlog', nnew);
end
actvox = vox(isact);
volp = vol;
volp(actvox) = uint8(nlabel0 + lid);

% each new label: base-tissue optics + MEAN dmua of its voxels (conserves the integral)
prop = [prop0; prop0(double(vol(actvox(accumarray(lid, (1:numel(lid))', [], @min)))) + 1, :)];
dmua = zeros(size(prop, 1) - 1, ntrial, nwl);             % per label (row m = label m)
dact = dmuavox(isact, :, :);
qerr = zeros(ntrial, nwl);
for t = 1:ntrial
    for w = 1:nwl
        mlab = accumarray(lid, dact(:, t, w), [nnew 1], @mean);
        dmua(nlabel0 + (1:nnew), t, w) = mlab;
        qerr(t, w) = norm(mlab(lid) - dact(:, t, w)) / norm(dact(:, t, w));
    end
end
fprintf('%d perturbation labels; voxel-wise quantization rel. L2 error %s\n', nnew, ...
        mat2str(qerr(:)', 3));
clear M dmuavox dact q bin key isact lid;

%% ---- 4. MCX per source, perturbation-MC reweighting ---------------------
amp = linspace(0, 1, opt.namp);
clear cfg;
cfg.vol = volp;
cfg.prop = prop;
cfg.unitinmm = unitinmm;
cfg.tstart = 0;                     % Cedalion: single 0-5 ns gate
cfg.tend = 5e-9;
cfg.tstep = 5e-9;
cfg.issrcfrom0 = 1;
cfg.isreflect = 0;                  % n=1 everywhere (Cedalion), no mismatch
cfg.issave2pt = 0;                  % detected photons only
cfg.savedetflag = 'dp';
cfg.maxdetphoton = opt.maxdetphoton;
cfg.nphoton = opt.nphoton;
cfg.gpuid = opt.gpuid;
cfg.autopilot = 1;

srclist = opt.srclist;
if isempty(srclist)
    srclist = 1:ns;
end
sig = [opt.nphoton, opt.nlin, opt.nlog, opt.qmin, opt.detradius, opt.namp, nnew];
for s = srclist(:)'
    cachef = fullfile(cachedir, sprintf('src%02d.mat', s));
    dets = pairs(pairs(:, 1) == s, 2);
    if isempty(dets)
        continue
    end
    S = struct('dets', dets, 'sig', sig, 'Sb', zeros(numel(dets), nwl, ntrial, opt.namp, 0), ...
               'nb', zeros(numel(dets), 0));
    if exist(cachef, 'file')
        C = load(cachef);
        if isequal(C.S.sig, sig) && isequal(C.S.dets, dets)
            S = C.S;
        end
    end
    cfg.srcpos = srcmcx(s, :);
    cfg.srcdir = srcdir(s, :);
    cfg.detpos = [detmcx(dets, :), repmat(opt.detradius / unitinmm, numel(dets), 1)];
    for b = size(S.Sb, 5) + 1:opt.nbatch
        tic;
        cfg.seed = opt.seed + 1000 * s + b;
        [~, detp] = mcxlab(cfg);
        nd = numel(detp.detid);
        if nd >= opt.maxdetphoton
            warning('src %d batch %d: detected-photon buffer full, raise maxdetphoton', s, b);
        end
        L = single(detp.ppath) * unitinmm;               % [nd x nmedia] mm
        id = double(detp.detid);
        clear detp;
        wbase = exp(-double(L * single(prop(2:end, 1))));
        Sbb = zeros(numel(dets), nwl, ntrial, opt.namp);  % det x wl x trial x amp
        for t = 1:ntrial
            for w = 1:nwl
                X = double(L * single(dmua(:, t, w)));    % sum_m dmua_m L_m per photon
                for k = 1:opt.namp
                    Sbb(:, w, t, k) = accumarray(id, wbase .* exp(-amp(k) * X), [numel(dets) 1]);
                end
            end
        end
        S.Sb(:, :, :, :, b) = Sbb;
        S.nb(:, b) = accumarray(id, 1, [numel(dets) 1]);
        clear L X wbase id;
        save(cachef, 'S');
        fprintf('src %2d/%d batch %2d/%d: %8d photons detected, min/pair %6d, %.1fs\n', ...
                s, ns, b, opt.nbatch, nd, min(S.nb(:, b)), toc);
    end
end

%% ---- 5. assemble per-pair dOD(amplitude) --------------------------------
dOD_amp = nan(npair, nwl, ntrial, opt.namp);
maxOD_se = nan(npair, nwl, ntrial);
I0 = nan(npair, nwl);                                     % detected weight / launched photon
ndet = zeros(npair, 1);
for s = 1:ns
    cachef = fullfile(cachedir, sprintf('src%02d.mat', s));
    if ~exist(cachef, 'file')
        continue
    end
    C = load(cachef);
    if ~isequal(C.S.sig, sig) || size(C.S.Sb, 5) < opt.nbatch
        warning('src %d incomplete or stale cache, skipped', s);
        continue
    end
    Sb = C.S.Sb(:, :, :, :, 1:opt.nbatch);
    Ssum = sum(Sb, 5);
    [~, prow] = ismember([repmat(s, numel(C.S.dets), 1), C.S.dets], pairs, 'rows');
    dOD_amp(prow, :, :, :) = -log(Ssum ./ Ssum(:, :, :, 1));
    odb = -log(Sb(:, :, :, end, :) ./ Sb(:, :, :, 1, :));   % per-batch peak dOD
    maxOD_se(prow, :, :) = std(odb, 0, 5) / sqrt(opt.nbatch);
    I0(prow, :) = reshape(Ssum(:, :, 1, 1), [], nwl) / (opt.nphoton * opt.nbatch);
    ndet(prow) = sum(C.S.nb(:, 1:opt.nbatch), 2);
end
maxOD = dOD_amp(:, :, :, end);
done = all(isfinite(maxOD(:, 1, 1)));
fprintf('pairs done %d/%d; peak dOD range [%.3g %.3g], median rel. SE %.3g\n', ...
        nnz(isfinite(maxOD(:, 1, 1))), npair, min(maxOD(:)), max(maxOD(:)), ...
        median(abs(maxOD_se(:) ./ maxOD(:)), 'omitnan'));

out = struct('format', 'mcx-pmc-dod', ...
             'Description', ['MCX perturbation-MC dOD of the Cedalion synthetic activation ' ...
                             'on the voxelated ICBM-152 head'], ...
             'complete', done, ...
             'wavelengths', wl, 'pairs', int32(pairs), 'trialtype', {trials}, ...
             'amp', amp, ...
             'dOD_amp', dOD_amp, ...       % pair x wl x trial x amp
             'maxOD', maxOD, ...           % pair x wl x trial, a = 1
             'maxOD_se', maxOD_se, ...
             'I0', I0, 'ndetected', ndet, ...
             'srcpos_ijk', srcmcx, 'srcdir', srcdir, 'detpos_ijk', detmcx, ...
             'prop', prop, 'dmua', dmua, 'quant_relerr', qerr, ...
             'settings', opt, 'unitinmm', unitinmm);
savemeas(opt.outfile, out);                % .mat + .jdb
fprintf('saved %s\n', opt.outfile);
end
