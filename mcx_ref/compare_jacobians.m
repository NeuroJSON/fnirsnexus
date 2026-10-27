function compare_jacobians(opfile)
% COMPARE_JACOBIANS  Compare the per-vertex mua Jacobians (dOD/dmua, mm) of
%   (1) redbird (FEM diffusion; -Anorm/eps; opfile, default ../redbird_recon/s4_op.mat;
%       e.g. compare_jacobians('s4_op_matched.mat') for the Cedalion-matched forward)
%   (2) Cedalion (its ForwardModel conventions, adot_cedalion.mat; = the data's Adot)
%   (3) the physically validated MCX model (fluence/jac_brain.mat, cortex only)
% on the shared two-surface unknowns (25k cortex + 10k scalp vertices), per channel.

here = fileparts(mfilename('fullpath'));
rb = fullfile(here, '..', 'redbird_recon');
if nargin < 1 || isempty(opfile)
    opfile = 's4_op.mat';
end
S = load(fullfile(rb, opfile), 'Anorm', 'ncortex', 'nscalp');
fprintf('redbird operator: %s\n', opfile);
C = load(fullfile(here, 'adot_cedalion.mat'));
npair = size(C.pairs, 1);
nc = S.ncortex;
ns = S.nscalp;
w = 2;                                                    % 850 nm
e = rbextinction(850, {'hbo'});                           % 1/(mm uM), redbird's own
Jr = -double(S.Anorm((w - 1) * npair + (1:npair), 1:nc + ns)) / e;  % [pair x (cortex+scalp)]
Jc = double(C.Adot);                                      % [pair x (cortex+scalp)]

M = headmodel(here);
P = load(fullfile(here, 'fluence', 'jac_brain.mat'), 'Lvox');
Agg = sparse(1:numel(M.vert), M.vert, 1, numel(M.vert), nc);
Jp = (double(P.Lvox)' * Agg);                             % [pair x cortex]
clear P;

% channels that see the cortex (top 50% of Cedalion cortex sensitivity)
sc = sum(Jc(:, 1:nc), 2);
sel = sc >= median(sc);
fprintf('\n== per-vertex mua Jacobian, 850 nm, %d channels (top half by cortex sensitivity) ==\n', nnz(sel));
names = {'redbird FEM', 'MCX physical'};
J = {Jr, [Jp, nan(npair, ns)]};
for k = 1:2
    rc = rowcorr(J{k}(sel, 1:nc), Jc(sel, 1:nc));
    ratio = sum(J{k}(sel, 1:nc), 2) ./ sc(sel);
    fprintf('%-13s vs Cedalion: cortex pattern r median %.3f [p10 %.3f], cortex total ratio median %.3f [p10 %.3f p90 %.3f]\n', ...
            names{k}, median(rc), pctl(rc, 10), median(ratio), pctl(ratio, 10), pctl(ratio, 90));
end
cs = @(X) sum(X(sel, nc + (1:ns)), 2) ./ sum(X(sel, 1:nc), 2);
fprintf('scalp/cortex total sensitivity per channel: Cedalion median %.2f, redbird %.2f\n', ...
        median(cs(Jc)), median(cs(Jr)));
rs = rowcorr(Jr(sel, nc + (1:ns)), Jc(sel, nc + (1:ns)));
fprintf('scalp pattern r (redbird vs Cedalion) median %.3f\n', median(rs));
% the same comparison after pooling cortex vertices into coarse spatial bins: separates
% mesh-sampling noise (few FEM nodes per vertex) from real differences of the field
gt = loadbj(fullfile(rb, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);
for h = [5 10 20]
    [~, ~, bid] = unique(floor(cortex / h), 'rows');
    B = sparse(1:nc, bid, 1, nc, max(bid));
    rb_ = rowcorr(Jr(sel, 1:nc) * B, Jc(sel, 1:nc) * B);
    rp_ = rowcorr(Jp(sel, :) * B, Jc(sel, 1:nc) * B);
    fprintf('pooled into %2d mm bins (%5d bins): cortex pattern r median redbird %.3f [p10 %.3f], MCX physical %.3f\n', ...
            h, max(bid), median(rb_), pctl(rb_, 10), median(rp_));
end
fprintf('channel-to-channel cortex totals: r(redbird, Cedalion) %.3f, r(MCX physical, Cedalion) %.3f, r(redbird, MCX physical) %.3f\n', ...
        pear(sum(Jr(:, 1:nc), 2), sc), pear(sum(Jp, 2), sc), pear(sum(Jr(:, 1:nc), 2), sum(Jp, 2)));
end

function r = rowcorr(A, B)
A = A - mean(A, 2);
B = B - mean(B, 2);
r = sum(A .* B, 2) ./ sqrt(sum(A.^2, 2) .* sum(B.^2, 2));
end

function r = pear(x, y)
c = corrcoef(x, y);
r = c(1, 2);
end

function p = pctl(x, pc)
x = sort(x(~isnan(x)));
p = x(max(1, ceil(pc / 100 * numel(x))));
end
