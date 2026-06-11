% RENDER_LATERAL  Render the redbird cortical ΔHbO recon in the SAME lateral views
% and color scale (RdBu, +/-1 uM) Cedalion uses, for side-by-side comparison with
% the Cedalion figures (notebook cells 52, PDF p.16).
here = fileparts(mfilename('fullpath'));

gt = loadbj(fullfile(here, 'groundtruth.bgii'));
rc = loadbj(fullfile(here, 'recon.bgii'));
node = double(gt.GIFTIData.MeshVertex3.Data);
face = double(gt.GIFTIData.MeshTri3.Data);
G = gt.GIFTIData.MeshVertex3.Properties;
R = rc.GIFTIData.MeshVertex3.Properties;

% RdBu_r-like diverging colormap (blue -> white -> red)
n = 256;
t = linspace(0, 1, n)';
cmap = [min(1, 2 * t), 1 - abs(2 * t - 1), min(1, 2 * (1 - t))];

% panels: {value, title, view([az el]), caxis}
% caxis: 5 uM for truth panels; 0 => autoscale to each recon panel's own max
P = {double(G.HbO_C3(:)), 'truth C3',   [-90 0], 5
     double(R.HbO_C3(:)), 'redbird C3', [-90 0], 0
     double(G.HbO_C4(:)), 'truth C4',   [90 0], 5
     double(R.HbO_C4(:)), 'redbird C4', [90 0], 0};

set(0, 'DefaultFigureVisible', 'off');
fig = figure('Visible', 'off', 'Position', [1 1 1200 950], 'Color', 'w');
for i = 1:4
    subplot(2, 2, i);
    plotmesh([node, P{i, 1}], face, 'edgecolor', 'none', 'facecolor', 'interp');
    view(P{i, 3});
    axis equal off;
    camlight;
    lighting gouraud;
    cm = P{i, 4};
    if cm == 0
        cm = max(abs(P{i, 1}));
        if cm == 0
            cm = 1;
        end
    end
    caxis([-cm cm]);
    colormap(cmap);
    colorbar;
    title(sprintf('%s  (pk %.2f uM)', P{i, 2}, max(P{i, 1})), 'Interpreter', 'none');
end
print(fig, fullfile(here, 'recon_lateral.png'), '-dpng', '-r110');
fprintf('wrote recon_lateral.png\n');
exit;
