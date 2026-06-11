% RENDER_SCALP  Render the SCALP-surface portion of the two-surface recon (the
% superficial leakage) for C3, C4, and the C3-C4 contrast. Uses the cached operator.
here = fileparts(mfilename('fullpath'));
hm = fullfile(here, '..', 'hm_icbm152');

S = load(fullfile(here, 's4_op.mat'));
Anorm = S.Anorm;
cmeas = S.cmeas;
ncortex = S.ncortex;
nscalp = S.nscalp;
Nsurf = ncortex + nscalp;
m = loadbj(fullfile(here, 'hrf_measurement.jdb'));
wl = double(m.wavelengths(:))';
nwl = numel(wl);
dOD = double(m.dOD);
pairs = double(m.pairs);
npair = size(pairs, 1);
taxis = double(m.time(:));
[~, tidx] = min(abs(taxis - 10.0));

% scalp surface (voxel -> RAS), with faces for rendering
[sv, sf] = readobjmesh(fullfile(hm, 'mask_scalp.obj'));
nii = loadnifti(fullfile(hm, 'mask_skin.nii'));
h = nii.NIFTIHeader;
A = h.Affine;
if size(A, 1) == 3
    A = [A; 0 0 0 1];
end
scalp = (A(1:3, 1:3) * sv' + A(1:3, 4))';

% weighting (alpha=0.003, beta=0.01) and inversion
alpha = 0.003;
beta = 0.01;
wrow = 1 ./ sqrt(cmeas + 1e-6 * max(cmeas));
Aw = wrow .* Anorm;
cn = sqrt(sum(Aw.^2, 1))';
Wc = 1 ./ (cn + beta * max(cn));
Aw = Aw .* Wc';
lam = alpha * max(sum(Aw.^2, 2));

targets = {'C3', dOD(:, :, tidx, 1); 'C4', dOD(:, :, tidx, 2); 'C3-C4', dOD(:, :, tidx, 1) - dOD(:, :, tidx, 2)};
n = 256;
tt = linspace(0, 1, n)';
cmap = [min(1, 2 * tt), 1 - abs(2 * tt - 1), min(1, 2 * (1 - tt))];
set(0, 'DefaultFigureVisible', 'off');
fig = figure('Visible', 'off', 'Position', [1 1 1500 520], 'Color', 'w');
for j = 1:3
    rhs = zeros(npair * nwl, 1);
    for w = 1:nwl
        rhs((w - 1) * npair + (1:npair)) = -targets{j, 2}(:, w);
    end
    z = rbreginv(Aw, wrow .* rhs, lam);
    d = z .* Wc;
    scalp_h = d(ncortex + (1:nscalp));            % scalp-surface portion
    subplot(1, 3, j);
    plotmesh([scalp, scalp_h(:)], sf(:, 1:3), 'edgecolor', 'none', 'facecolor', 'interp');
    view([0 90]);
    axis equal off;
    camlight;
    lighting gouraud;
    mm = max(abs(scalp_h));
    if mm == 0
        mm = 1;
    end
    caxis([-mm mm]);
    colormap(cmap);
    colorbar;
    title(sprintf('scalp leakage %s (pk %.2f uM)', targets{j, 1}, max(scalp_h)), 'Interpreter', 'none');
end
print(fig, fullfile(here, 'recon_scalp.png'), '-dpng', '-r110');
fprintf('wrote recon_scalp.png\n');
exit;
