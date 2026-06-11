% S4_SWEEP  Using the cached two-surface operator, sweep depth weighting (beta) and
% Tikhonov (alpha) to suppress scalp leakage and localize the cortex. No forward.
here = fileparts(mfilename('fullpath'));
addpath('/home/users/fangq/space/git/Project/github/redbird-m/matlab');
addpath('/home/users/fangq/space/git/Project/jsonlab');

S = load(fullfile(here, 's4_op.mat'));               % Anorm, cmeas, ncortex, nscalp
Anorm = S.Anorm; cmeas = S.cmeas; ncortex = S.ncortex; nscalp = S.nscalp; Nsurf = ncortex + nscalp;
m = loadbj(fullfile(here, 'hrf_measurement.jdb'));
wl = double(m.wavelengths(:))'; nwl = numel(wl);
dOD = double(m.dOD); pairs = double(m.pairs); npair = size(pairs, 1);
taxis = double(m.time(:)); [~, tidx] = min(abs(taxis - 10.0));
gt = loadbj(fullfile(here, 'groundtruth.bgii'));
cortex = double(gt.GIFTIData.MeshVertex3.Data);
trC3 = double(gt.GIFTIData.MeshVertex3.Properties.HbO_C3);
trC4 = double(gt.GIFTIData.MeshVertex3.Properties.HbO_C4);
itC3 = find(trC3 == max(trC3), 1); itC4 = find(trC4 == max(trC4), 1);

wrow = 1 ./ sqrt(cmeas + 1e-6 * max(cmeas));
rhsC3 = zeros(npair * nwl, 1); rhsC4 = rhsC3;
for w = 1:nwl
    rhsC3((w-1)*npair + (1:npair)) = -dOD(:, w, tidx, 1);
    rhsC4((w-1)*npair + (1:npair)) = -dOD(:, w, tidx, 2);
end

fprintf('%6s %6s | %-7s %-7s %-6s | %-7s %-7s %-6s\n', 'alpha','beta','errC3','corrC3','leakC3','errC4','corrC4','leakC4');
for alpha = [0.003 0.01 0.05]
  for beta = [1 0.1 0.03 0.01 0.003]
    Aw = (wrow .* Anorm);
    cn = sqrt(sum(Aw.^2, 1))'; Wc = 1 ./ (cn + beta * max(cn));
    Aw = Aw .* Wc';
    lam = alpha * max(sum(Aw.^2, 2));
    out = zeros(1, 6);
    rhss = {rhsC3, rhsC4}; its = [itC3 itC4]; trs = {trC3, trC4};
    for t = 1:2
        z = rbreginv(Aw, wrow .* rhss{t}, lam); d = z .* Wc; h = d(1:Nsurf);
        cv = h(1:ncortex); sv = h(ncortex+(1:nscalp));
        [~, ir] = max(cv); cc = corrcoef(cv, trs{t});
        out((t-1)*3 + (1:3)) = [norm(cortex(ir,:) - cortex(its(t),:)), cc(1,2), max(abs(sv))/max(abs(cv))];
    end
    fprintf('%6.3g %6.3g | %7.1f %7.3f %6.2f | %7.1f %7.3f %6.2f\n', alpha, beta, out(1),out(2),out(3),out(4),out(5),out(6));
  end
end
exit
