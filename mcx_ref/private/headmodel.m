function M = headmodel(here)
% HEADMODEL  Cedalion-matched voxel head model shared by the mcx_ref scripts:
% label volume + optics, snapped optodes, and the activation mapped to voxels.
%
% Returns M with fields vol, A, unitinmm, prop0, meas, wl, pairs, trials, srcmcx,
% srcdir, detmcx, detdir (MCX grid coords, issrcfrom0=1; inward unit directions), vox (brain voxel
% linear index), vert (its 1-based cortex vertex),
% dmuavox [nvox x ntrial x nwl] (1/mm).

hm = fullfile(here, '..', 'hm_icbm152');

%% ---- 1. label volume + optical properties (Cedalion's convention) -------
% read_segmentation_masks: label = position in the mask_files dict (1-based)
masknames = {'mask_csf', 'mask_gray', 'mask_skin', 'mask_bone', 'mask_white'};
tissue = {'csf', 'gm', 'scalp', 'skull', 'wm'};
vol = [];
for i = 1:numel(masknames)
    nii = loadnifti(fullfile(hm, [masknames{i} '.nii']));
    if isempty(vol)
        vol = zeros(size(nii.NIFTIData), 'uint8');
        A = getaffine(nii);
    end
    if any(vol(nii.NIFTIData > 0))
        error('masks overlap at %s; Cedalion would sum the labels', masknames{i});
    end
    vol(nii.NIFTIData > 0) = i;
end
unitinmm = abs(det(A(1:3, 1:3)))^(1 / 3);
fprintf('label volume %s, unitinmm=%g\n', mat2str(size(vol)), unitinmm);

% cedalion/dot/tissue_properties.py  [mua(1/mm) mus(1/mm) g n], wavelength independent
%                  mua     mus    g      n
tisprop = struct('csf',   [0.0026 0.0100 0.001 1], ...
                 'gm',    [0.0186 1.1000 0.001 1], ...
                 'scalp', [0.0191 0.6600 0.001 1], ...
                 'skull', [0.0136 0.8600 0.001 1], ...
                 'wm',    [0.0186 1.1000 0.001 1]);
prop0 = [0 0 1 1; cell2mat(cellfun(@(t) tisprop.(t), tissue', 'UniformOutput', false))];

%% ---- 2. optodes: RAS -> IJK, snap to scalp voxel, inward normal --------
meas = loadjd(fullfile(here, '..', 'redbird_recon', 'hrf_measurement.jdb'));
wl = double(meas.wavelengths(:))';
pairs = double(meas.pairs);
trials = meas.trialtype;
if ~iscell(trials)
    trials = cellstr(trials);
end
npair = size(pairs, 1);
nwl = numel(wl);
ntrial = numel(trials);
ns = size(meas.srcpos, 1);

ras2ijk = @(p) (A(1:3, 1:3) \ (p' - A(1:3, 4)))';         % 0-based voxel-centre coords
[sv, sf] = readobjmesh(fullfile(hm, 'mask_scalp.obj'));   % already 0-based IJK
sn = nodesurfnorm(sv, sf);
% Cedalion's nearest-scalp-voxel snap can sit up to ~5 voxels below the skin (the
% decimated scalp surface cuts inside the voxel skin); fine for its adjoint fluence
% lookup but MCX detectors must touch the exit surface. So snap to the scalp voxels
% that face the EXTERIOR air (interior cavities excluded).
outside = ~imfill(vol > 0, 'holes');
skin = (vol == 3) & imdilate(outside, cat(3, [0 0 0; 0 1 0; 0 0 0], [0 1 0; 1 1 1; 0 1 0], ...
                                          [0 0 0; 0 1 0; 0 0 0]));
[srcijk, srcdir, srcshift] = snapoptode(ras2ijk(double(meas.srcpos)), sv, sn, vol == 3, skin);
[detijk, detdir, detshift] = snapoptode(ras2ijk(double(meas.detpos)), sv, sn, vol == 3, skin);
fprintf('optodes moved from Cedalion''s scalp voxel to the skin surface: src median %.1f max %.1f, det median %.1f max %.1f voxels\n', ...
        median(srcshift), max(srcshift), median(detshift), max(detshift));
srcmcx = srcijk + 0.5;                                    % voxel centre, issrcfrom0=1
detmcx = detijk + 0.5;

%% ---- 3. activation -> voxels (Cedalion's voxel_to_vertex_brain) ---------
[vox, vert] = readvox2vert(fullfile(hm, 'voxel_to_vertex_brain.mtx.gz'), size(vol));
fprintf('brain voxel map: %d voxels, labels %s\n', numel(vox), mat2str(unique(vol(vox))'));

act = readspatialact(fullfile(here, '..', 'cedalion_ref', 'spatial_act.csv'), trials);
E = prahl(wl);                                            % [nwl x 2] 1/(mm M), HbO HbR
dmuavox = zeros(numel(vox), ntrial, nwl);                 % dmua per brain voxel (1/mm)
for t = 1:ntrial
    dmuavox(:, t, :) = reshape(act(vert, :, t) * 1e-6 * E', [], 1, nwl);  % uM -> M
end

M = struct('vol', vol, 'A', A, 'unitinmm', unitinmm, 'prop0', prop0, 'meas', meas, ...
           'wl', wl, 'pairs', pairs, 'trials', {trials}, 'srcmcx', srcmcx, ...
           'srcdir', srcdir, 'detmcx', detmcx, 'detdir', detdir, 'vox', vox, 'vert', vert, ...
           'dmuavox', dmuavox);
end
