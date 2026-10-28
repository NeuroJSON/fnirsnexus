function inverse_match()
% INVERSE_MATCH  Convert redbird's s4 inverse into Cedalion's ImageRecon one change at a
% time, for several Jacobians, and compare every map with Cedalion's own output.
% Both inverses are  x = P A' (A P A' + lambda C)^-1 y :
%   S0 redbird s4 : joint HbO/HbR, C = pre-stim dOD variance, P = Wc^2
%                   (Wc = 1/(|col of C^-1/2 A| + beta max), beta=0.01),
%                   lambda = 0.003 * max row energy of C^-1/2 A Wc
%   S1  + C = Cedalion c_meas (resting-state OD variance)
%   S2  + P = I (no depth weighting)
%   S3  + lambda = 0.5 * max eig(A A')            (Cedalion alpha_meas = 0.5)
%   S4  + per-wavelength mua, conc = pinv(E) mua  (= Cedalion ImageRecon exactly)
% Writes figures/inverse_match_<operator>.png (truth, Cedalion, S0..S4; orig & MCX data).

here = fileparts(mfilename('fullpath'));
rb = fullfile(here, '..', 'redbird_recon');
gt = loadbj(fullfile(rb, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);
face = double(gt.GIFTIData.MeshTri3.Data);
nc = size(cortex, 1);
truth = [double(gt.GIFTIData.MeshVertex3.Properties.HbO_C3), double(gt.GIFTIData.MeshVertex3.Properties.HbO_C4)];
E = log(10) / 10 * [586 1548.52; 1058 691.32];             % 1/(mm M), Prahl
Ec = E * 1e-6;                                            % 1/(mm uM)
Cc = load(fullfile(here, 'cmeas_cedalion.mat'));
cced = Cc.cmeas(:);                                       % [wl1 pairs; wl2 pairs]

A0 = load(fullfile(here, 'adot_cedalion.mat'));
npair = size(A0.pairs, 1);
ops = struct('name', {}, 'tag', {}, 'A', {});             % A{w}: mua Jacobian (mm) per wl
ops(1) = struct('name', 'Cedalion Adot', 'tag', 'cedalion', 'A', {{double(A0.Adot), double(A0.Adot)}});
clear A0;
erb = rbextinction([760 850], {'hbo'});
opf = {'s4_op_matched_csf0.15_voxel.mat', 'redbird matched (Cedalion optics, CSF 0.15, voxel)', 'matched'; ...
       's4_op.mat', 'redbird s4 (own optics, voxel)', 's4'; ...
       's4_op_matched_csf0.15_voxel_rownorm.mat', 'redbird matched + Cedalion per-channel scale', 'matched_rownorm'};
for k = 1:size(opf, 1)
    S = load(fullfile(rb, opf{k, 1}), 'Anorm');
    h = size(S.Anorm, 2) / 2;
    ops(end + 1) = struct('name', opf{k, 2}, 'tag', opf{k, 3}, ...
                          'A', {{-S.Anorm(1:npair, 1:h) / erb(1), -S.Anorm(npair + (1:npair), 1:h) / erb(2)}}); %#ok<AGROW>
    clear S;
end

data = {'orig', fullfile(rb, 'hrf_measurement.jdb'); 'mcx', fullfile(here, 'hrf_measurement.mat')};
steps = {'S0 redbird s4', 'S1 +Cedalion c_meas', 'S2 +no depth weight', 'S3 +Cedalion lambda', ...
         'S4 +per-wavelength (=Cedalion)'};
Y = cell(2, 1);
cpre = cell(2, 1);
ced = cell(2, 1);
for d = 1:2
    m = loadmeas(data{d, 2});
    dOD = double(m.dOD);
    taxis = double(m.time(:));
    [~, tidx] = min(abs(taxis - 10.0));
    Y{d} = [squeeze(dOD(:, 1, tidx, :)); squeeze(dOD(:, 2, tidx, :))];   % [wl-stacked pairs x trial]
    pre = taxis < 0;
    cpre{d} = [var(reshape(dOD(:, 1, pre, :), npair, []), 0, 2); var(reshape(dOD(:, 2, pre, :), npair, []), 0, 2)];
    g = loadbj(fullfile(here, sprintf('recon_cedalion_%s.bgii', data{d, 1})));
    ced{d} = [double(g.GIFTIData.MeshVertex3.Properties.HbO_C3), double(g.GIFTIData.MeshVertex3.Properties.HbO_C4)];
end

fprintf('\n%-50s %-31s %-4s %-12s %-12s\n', 'operator', 'inverse step', 'data', 'C3 r/err', 'C4 r/err');
for k = 1:numel(ops)
    nv = size(ops(k).A{1}, 2);
    Ac = [ops(k).A{1} * Ec(1, 1), ops(k).A{1} * Ec(1, 2); ops(k).A{2} * Ec(2, 1), ops(k).A{2} * Ec(2, 2)];  % OD per uM
    maps = zeros(nc, 2, numel(steps), 2);                % vertex x cond x step x data
    for d = 1:2
        for s = 1:numel(steps)
            if s == 1
                C = cpre{d};
            else
                C = cced;
            end
            if s <= 4
                if s <= 2
                    Aw = Ac ./ sqrt(C + 1e-6 * max(C));
                    cn = sqrt(sum(Aw.^2, 1));
                    P = (1 ./ (cn + 0.01 * max(cn))).^2;
                else
                    P = ones(1, 2 * nv);
                end
                AP = Ac .* P;
                F = AP * Ac';
                if s <= 3
                    Wr = 1 ./ sqrt(C + 1e-6 * max(C));
                    lam = 0.003 * max(sum((Wr .* Ac).^2 .* P, 2));
                    Cs = C + 1e-6 * max(C);
                else
                    lam = 0.5 * max(eig((F + F') / 2));
                    Cs = C;
                end
                x = AP' * ((F + lam * diag(Cs)) \ Y{d});
                hbo = x(1:nc, :);
            else
                mua = zeros(nv, 2, 2);                    % vertex x wl x trial
                for w = 1:2
                    Aw = ops(k).A{w};
                    F = Aw * Aw';
                    lam = 0.5 * max(eig((F + F') / 2));
                    rows = (w - 1) * npair + (1:npair);
                    mua(:, w, :) = reshape(Aw' * ((F + lam * diag(C(rows))) \ Y{d}(rows, :)), nv, 1, 2);
                end
                pe = pinv(E);
                hbo = 1e6 * squeeze(pe(1, 1) * mua(1:nc, 1, :) + pe(1, 2) * mua(1:nc, 2, :));
            end
            maps(:, :, s, d) = hbo;
            out = cell(1, 2);
            for t = 1:2
                cc = corrcoef(hbo(:, t), ced{d}(:, t));
                [~, ip] = max(hbo(:, t));
                [~, it] = max(truth(:, t));
                out{t} = sprintf('%.3f/%5.1f', cc(1, 2), norm(cortex(ip, :) - cortex(it, :)));
            end
            fprintf('%-50s %-31s %-4s %-12s %-12s\n', ops(k).name, steps{s}, data{d, 1}, out{:});
        end
    end
    if k > 1
        render(fullfile(here, 'figures', sprintf('inverse_match_%s.png', ops(k).tag)), ops(k).name, ...
               cortex, face, truth, ced, maps, steps);
    end
end
end

% -------------------------------------------------------------------------
function render(fname, opname, cortex, face, truth, ced, maps, steps)
n = 256;
tt = linspace(0, 1, n)';
cmap = [min(1, 2 * tt), 1 - abs(2 * tt - 1), min(1, 2 * (1 - tt))];
rowname = [{'truth', 'Cedalion ImageRecon'}, steps];
views = {[-90 0], [90 0]};
dn = {'original', 'MCX'};
[~, it3] = max(truth(:, 1));
[~, it4] = max(truth(:, 2));
itr = [it3 it4];
set(0, 'DefaultFigureVisible', 'off');
fig = figure('Visible', 'off', 'Position', [1 1 1600 2600], 'Color', 'w');
nr = numel(rowname);
for r = 1:nr
    for d = 1:2
        for t = 1:2
            if r == 1
                val = truth(:, t);
            elseif r == 2
                val = ced{d}(:, t);
            else
                val = maps(:, t, r - 2, d);
            end
            ax = axes('Position', [0.01 + ((d - 1) * 2 + t - 1) * 0.245, 1 - r / nr + 0.004, 0.235, 1 / nr - 0.026]);
            plotmesh([cortex, val(:)], face, 'edgecolor', 'none', 'facecolor', 'interp');
            view(views{t});
            axis(ax, 'equal', 'off');
            camlight;
            lighting gouraud;
            mm = max(abs(val));
            if mm == 0
                mm = 1;
            end
            caxis([-mm mm]);
            colormap(cmap);
            [pk, ip] = max(val);
            if r == 2
                rc = 1;
            elseif r > 2
                c = corrcoef(val, ced{d}(:, t));
                rc = c(1, 2);
            else
                rc = NaN;
            end
            title(sprintf('%s | %s C%d\npk %.2f uM, err %.1f mm, r vs Ced %.3f', rowname{r}, dn{d}, 2 + t, ...
                          pk, norm(cortex(ip, :) - cortex(itr(t), :)), rc), 'Interpreter', 'none', 'FontSize', 9);
        end
    end
end
annotation(fig, 'textbox', [0 0.985 1 0.015], 'String', ['Jacobian: ' opname], 'EdgeColor', 'none', ...
           'HorizontalAlignment', 'center', 'FontSize', 13, 'FontWeight', 'bold', 'Interpreter', 'none');
print(fig, fname, '-dpng', '-r80');
close(fig);
fprintf('wrote %s\n', fname);
end
