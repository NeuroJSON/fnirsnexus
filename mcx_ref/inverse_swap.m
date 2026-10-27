function inverse_swap()
% INVERSE_SWAP  Separate the Jacobian from the inverse: every operator (Cedalion's Adot,
% redbird matched, redbird s4) x every inverse (Cedalion ImageRecon, redbird s4), on the
% original and the MCX data. Each HbO map (t=10 s) is compared with Cedalion's own
% ImageRecon output (recon_cedalion_{orig,mcx}.bgii): vertex-wise r, peak distance.
%
% Cedalion's inverse (image_recon.py, alpha_spatial=None, no basis functions, mua2conc):
%   per wavelength  mua = A' (A A' + lambda C)^-1 y,  lambda = alpha_meas * max eig(A A'),
%   alpha_meas = 0.5, C = diag(c_meas) (resting-state OD variance, cmeas_cedalion.mat);
%   conc = pinv(E) [mua_760; mua_850], E = Prahl (1/(mm M)).
% redbird's inverse (s4_recon_surface): joint HbO/HbR, c_meas (pre-stim) row whitening,
%   depth weighting 1/(|col| + beta max), Tikhonov alpha (alpha=0.003, beta=0.01).

here = fileparts(mfilename('fullpath'));
rb = fullfile(here, '..', 'redbird_recon');
gt = loadbj(fullfile(rb, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);
nc = size(cortex, 1);
truth = {double(gt.GIFTIData.MeshVertex3.Properties.HbO_C3), double(gt.GIFTIData.MeshVertex3.Properties.HbO_C4)};
E = log(10) / 10 * [586 1548.52; 1058 691.32];             % 1/(mm M), Prahl (Cedalion)
erb = rbextinction([760 850], {'hbo', 'hbr'});              % 1/(mm uM), redbird
Cc = load(fullfile(here, 'cmeas_cedalion.mat'));

C = load(fullfile(here, 'adot_cedalion.mat'));
npair = size(C.pairs, 1);
ops = {'Cedalion Adot', mua2anorm(double(C.Adot), erb), double(C.Adot), double(C.Adot)};
for f = {'s4_op_matched_csf0.15_voxel.mat', 's4_op.mat'}
    S = load(fullfile(rb, f{1}), 'Anorm');
    ops(end + 1, :) = {f{1}, S.Anorm, -S.Anorm(1:npair, 1:end / 2) / erb(1, 1), ...
                       -S.Anorm(npair + (1:npair), 1:end / 2) / erb(2, 1)}; %#ok<AGROW>
end
data = {'orig', fullfile(rb, 'hrf_measurement.jdb'); 'mcx', fullfile(here, 'hrf_measurement.jdb')};

fprintf('\n%-36s %-9s %-4s %-4s %8s %10s %10s %8s\n', 'operator', 'inverse', 'data', 'cond', ...
        'r_vs_ced', 'pk_dist_mm', 'err_truth', 'pk_uM');
for d = 1:2
    m = loadjd(data{d, 2});
    dOD = double(m.dOD);
    taxis = double(m.time(:));
    [~, tidx] = min(abs(taxis - 10.0));
    ced = loadbj(fullfile(here, sprintf('recon_cedalion_%s.bgii', data{d, 1})));
    pre = taxis < 0;
    cmeas_rb = [var(reshape(dOD(:, 1, pre, :), npair, []), 0, 2); var(reshape(dOD(:, 2, pre, :), npair, []), 0, 2)];
    for k = 1:size(ops, 1)
        for inv = {'cedalion', 'redbird'}
            for t = 1:2
                y = squeeze(dOD(:, :, tidx, t));                % pair x wl
                if strcmp(inv{1}, 'cedalion')
                    mua = zeros(size(ops{k, 3}, 2), 2);
                    for w = 1:2
                        Aw = ops{k, 2 + w};
                        F = Aw * Aw';
                        lam = 0.5 * max(eig((F + F') / 2));
                        mua(:, w) = Aw' * ((F + lam * diag(Cc.cmeas(:, w))) \ y(:, w));
                    end
                    conc = (pinv(E) * mua')' * 1e6;             % uM, [vertex x HbO/HbR]
                    hbo = conc(1:nc, 1);
                else
                    hbo = redbirdinv(ops{k, 2}, cmeas_rb, -y(:), 0.003, 0.01, nc);
                end
                cv = double(ced.GIFTIData.MeshVertex3.Properties.(sprintf('HbO_C%d', 2 + t)));
                [pk, ip] = max(hbo);
                [~, ic] = max(cv);
                [~, it] = max(truth{t});
                cc = corrcoef(hbo, cv);
                fprintf('%-36s %-9s %-4s C%-3d %8.3f %10.1f %10.1f %8.2f\n', ops{k, 1}, inv{1}, ...
                        data{d, 1}, 2 + t, cc(1, 2), norm(cortex(ip, :) - cortex(ic, :)), ...
                        norm(cortex(ip, :) - cortex(it, :)), pk);
            end
        end
    end
end
end

% -------------------------------------------------------------------------
function Anorm = mua2anorm(A, e)
% Cedalion mua Jacobian (mm) -> redbird Rytov layout [wl rows] x [HbO; HbR] (per uM)
Anorm = -[A * e(1, 1), A * e(1, 2); A * e(2, 1), A * e(2, 2)];
end

function hbo = redbirdinv(Anorm, cmeas, rhs, alpha, beta, nc)
% s4_recon_surface's inverse: c_meas whitening, depth weighting, Tikhonov
wrow = 1 ./ sqrt(cmeas + 1e-6 * max(cmeas));
Aw = wrow .* Anorm;
cn = sqrt(sum(Aw.^2, 1))';
Wc = 1 ./ (cn + beta * max(cn));
Aw = Aw .* Wc';
lambda = alpha * max(sum(Aw.^2, 2));
d = rbreginv(Aw, wrow .* rhs, lambda) .* Wc;
hbo = d(1:nc);
end
