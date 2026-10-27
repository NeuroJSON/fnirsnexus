function plot_recon_compare()
% PLOT_RECON_COMPARE  Cortex HbO (t_rel = 10 s) of truth, Cedalion ImageRecon and the
% redbird variants, on the original (Cedalion Adot) and the MCX reference data.
% Writes figures/recon_compare.png. Titles give the peak and the peak error vs truth.

here = fileparts(mfilename('fullpath'));
rb = fullfile(here, '..', 'redbird_recon');
gt = loadbj(fullfile(rb, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);
face = double(gt.GIFTIData.MeshTri3.Data);

% {label, file} per row, per dataset column group
cols = {'original', 'MCX'};
rows = {{'truth', fullfile(rb, 'groundtruth.bgii'); 'truth', fullfile(rb, 'groundtruth.bgii')}, ...
        {'Cedalion ImageRecon', fullfile(here, 'recon_cedalion_orig.bgii'); ...
         'Cedalion ImageRecon', fullfile(here, 'recon_cedalion_mcx.bgii')}, ...
        {'redbird s4 (voxel agg)', fullfile(rb, 'recon.bgii'); ...
         'redbird matched', fullfile(rb, 'recon_matched_mcx_ref.bgii')}, ...
        {'redbird matched', fullfile(rb, 'recon_matched_redbird_recon.bgii'); ...
         'redbird matched + Ced. scale', fullfile(here, 'recon_mcx.bgii')}};
conds = {'C3', [-90 0]; 'C4', [90 0]};
truthpk = zeros(2, 1);
for c = 1:2
    tr = double(gt.GIFTIData.MeshVertex3.Properties.(['HbO_' conds{c, 1}]));
    [~, truthpk(c)] = max(tr);
end

n = 256;
tt = linspace(0, 1, n)';
cmap = [min(1, 2 * tt), 1 - abs(2 * tt - 1), min(1, 2 * (1 - tt))];
set(0, 'DefaultFigureVisible', 'off');
fig = figure('Visible', 'off', 'Position', [1 1 1700 1500], 'Color', 'w');
for r = 1:numel(rows)
    for d = 1:2
        g = loadbj(rows{r}{d, 2});
        P = g.GIFTIData.MeshVertex3.Properties;
        for c = 1:2
            val = double(P.(['HbO_' conds{c, 1}]));
            subplot(numel(rows), 4, (r - 1) * 4 + (d - 1) * 2 + c);
            plotmesh([cortex, val(:)], face, 'edgecolor', 'none', 'facecolor', 'interp');
            view(conds{c, 2});
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
            [pk, ip] = max(val);
            err = norm(cortex(ip, :) - cortex(truthpk(c), :));
            title(sprintf('%s | %s data %s\npk %.2f uM, err %.1f mm', rows{r}{d, 1}, cols{d}, ...
                          conds{c, 1}, pk, err), 'Interpreter', 'none', 'FontSize', 8);
        end
    end
end
if ~exist(fullfile(here, 'figures'), 'dir')
    mkdir(fullfile(here, 'figures'));
end
print(fig, fullfile(here, 'figures', 'recon_compare.png'), '-dpng', '-r90');
close(fig);
fprintf('wrote figures/recon_compare.png\n');
end
