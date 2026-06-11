function s2_recon(peaktime, alpha, beta)
% S2_RECON  Linear (single Gauss-Newton step) DOT image reconstruction in
%           redbird-m, porting the Cedalion ImageRecon step of
%           hmstandardization_challenge_prep.
%
%   s2_recon()        % reconstruct at the HRF peak (t ~ 10 s)
%   s2_recon(10.0)    % reconstruct at a chosen reltime (s)
%
% A single Gauss-Newton iteration == a linear reconstruction. To match Cedalion's
% OD-domain linear operator on a SPARSE fNIRS montage we build the operator by hand
% rather than through rbrun's high-level recon path (whose detector extraction
% assumes a dense source x detector grid):
%
%   1. load FEM head mesh (s1_build_mesh.m) and HRF dOD (export_inputs.py)
%   2. project optodes onto the scalp; homogeneous baseline (single bulk HbO/HbR,
%      fixed scattering)
%   3. baseline forward solve -> detphi_base, fluence phi
%   4. rbjac -> per-wavelength mua Jacobian for the measured pairs; remap onto a
%      coarse reconstruction mesh; rbjacchrome -> HbO/HbR Jacobian
%   5. Rytov normalize each (wavelength,pair) row by detphi_base => the OD
%      sensitivity (== Cedalion's Adot). Solve  Anorm*[dHbO;dHbR] = -dOD  with
%      Tikhonov regularization (rbreginv). Anorm is built once and applied to
%      every condition.
%   6. map node-wise dHbO/dHbR onto the 25k ICBM cortex surface, save binary JGIfTI
%
% Requires redbird-m, iso2mesh, jsonlab/jmesh on the path.

if nargin < 1 || isempty(peaktime)
    peaktime = 10.0;
end
if nargin < 2 || isempty(alpha)
    alpha = 0.01;
end
if nargin < 3 || isempty(beta)
    beta  = 0.01;
end
here = fileparts(mfilename('fullpath'));

%% ---- load inputs --------------------------------------------------------
mesh = loadjmesh(fullfile(here, 'head_mesh.bmsh'));
node = meshpart(mesh.MeshVertex3);                    % Nn x3
[etet, elabel] = meshpart(mesh.MeshTet4);             % tets + tissue label (1=EC, 2=brain)
if isempty(elabel)
    elabel = ones(size(etet, 1), 1);
end
elem = [etet, elabel];
nseg = max(elem(:, 5));
fprintf('mesh: %d nodes, %d tets, %d regions\n', size(node, 1), size(elem, 1), nseg);

meas   = loadjd(fullfile(here, 'hrf_measurement.jdb'));
wl     = double(meas.wavelengths(:))';                % [760 850]
wlkey  = arrayfun(@(x) num2str(x), wl, 'UniformOutput', false);
srcpos = double(meas.srcpos);                         % Nsrc x3 RAS
detpos = double(meas.detpos);                         % Ndet x3 RAS
pairs  = double(meas.pairs);                          % Npair x2 [srcIdx detIdxLocal]
dOD    = double(meas.dOD);                            % Npair x Nwl x Ntime x Ntrial
taxis  = double(meas.time(:));
trials = meas.trialtype;
if ~iscell(trials)
    trials = cellstr(trials);
end
[~, tidx] = min(abs(taxis - peaktime));
fprintf('reconstructing at reltime=%.2f s (index %d)\n', taxis(tidx), tidx);

%% ---- build (or load cached) the linear OD->dHbO/dHbR operator Anorm ------
% the forward solve + Jacobian is the only expensive step and is independent of
% the regularization and reconstruction time point; cache it so lambda/time can be
% tuned instantly.
cachef = fullfile(here, 'recon_op.mat');
if exist(cachef, 'file')
    fprintf('loading cached recon operator: %s\n', cachef);
    S = load(cachef);
    Anorm = S.Anorm;
    rnode = S.rnode;
    nrec = S.nrec;
    npair = S.npair;
    nwl = S.nwl;
    widx = S.widx;
else
    %% ---- project optodes onto the scalp surface -----------------------------
    face = volface(elem(:, 1:4));
    nv   = nodesurfnorm(node, face);
    [srcpos, srcdir] = snap2surf(node, face, nv, srcpos);
    [detpos, detdir] = snap2surf(node, face, nv, detpos);

    %% ---- homogeneous-baseline forward model ---------------------------------
    clear cfg;
    cfg.node = node;
    cfg.elem = elem;
    cfg.seg = elem(:, 5);
    cfg.srcpos = srcpos;
    cfg.srcdir = srcdir;
    cfg.detpos = detpos;
    cfg.detdir = detdir;
    % per-tissue baseline (columns = scalp, bone, CSF, gray, white). param drives
    % rbmeshprep to compute mua (from HbO/HbR) and musp (scatter power law) at each
    % wavelength; the recon estimates CHANGES about this baseline. Sliced to nseg so a
    % 2-region or 5-region mesh both work.
    %            scalp  bone  CSF  gray  white
    tis.hbo      = [25    15    2    60    40];   % uM
    tis.hbr      = [12     8    1    25    18];   % uM
    tis.scatamp  = [1.2  1.35  0.4  1.65  1.5];   % scatamp500
    tis.scatpow  = [1.0   1.0  1.0   1.0  1.0];
    sel = min(1:nseg, numel(tis.hbo));
    cfg.param = struct('hbo', tis.hbo(sel), 'hbr', tis.hbr(sel), ...
                       'scatamp500', tis.scatamp(sel), 'scatpow500', tis.scatpow(sel));
    cfg.prop = containers.Map();
    for w = 1:numel(wl)
        cfg.prop(wlkey{w}) = [0 0 1 1; repmat([0.015 1.0 0 1.37], nseg, 1)];  % overwritten
    end
    cfg.omega = 0;                                        % CW
    cfg = rbmeshprep(cfg);

    %% ---- baseline forward on the FULL grid (fluence + baseline detphi) ------
    % run with all source-detector pairs so detphi_base is full [Ndet x Nsrc] and phi
    % holds every detector adjoint field; the sparse montage is selected only in rbjac.
    ns = size(srcpos, 1);
    sdfull = rbsdmap(cfg);
    k = sdfull.keys;
    for i = 1:numel(k)
        t = sdfull(k{i});
        t(:, 3) = 1;
        sdfull(k{i}) = t;
    end
    [detphi_base, phi] = rbrunforward(cfg, 'sd', sdfull);
    b1 = detphi_base(wlkey{1});
    fprintf('baseline forward %s size %s nan=%d range[%.2e %.2e]\n', wlkey{1}, ...
            mat2str(size(b1)), sum(isnan(b1(:))), min(abs(b1(:))), max(abs(b1(:))));

    %% ---- Jacobian restricted to the gray-matter (cortex) layer --------------
    % NOTE: this GM-only restriction is a deliberate redbird SIMPLIFICATION to suppress
    % the superficial bias -- it does NOT match Cedalion. Cedalion's notebook runs
    % ImageRecon(..., brain_only=False), so it reconstructs on BOTH the cortex AND the
    % scalp surface (is_brain only selects the cortex for display; it leaks onto the
    % scalp, see PDF p.18). The faithful cortex+scalp two-surface recon is in
    % s4_recon_surface.m. Here we instead confine the recon space to the GM layer
    % (label 4): build the NODAL Jacobian on the forward mesh, then keep only the GM-node
    % columns (zero/drop all others), forcing the update onto the cortex.
    sd = build_sd(pairs, ns, wlkey);                          % measured pairs (global idx)
    if isa(phi, 'containers.Map')
        phimap = phi;
    else
        phimap = phi(1).phi;
    end
    gmnodes = unique(elem(elem(:, 5) == 4, 1:4));             % gray-matter nodes
    rnode   = node(gmnodes, :);
    nrec    = numel(gmnodes);
    Jr = containers.Map();
    for w = 1:numel(wlkey)
        cfgw = cfg;
        cfgw.prop = cfg.prop(wlkey{w});           % single-wl 2-row prop
        Jw = rbfemmatrix(cfgw, sd(wlkey{w}), phimap(wlkey{w}), cfg.deldotdel, 1); % [Npair x Nfwd]
        Jr(wlkey{w}) = Jw(:, gmnodes);                        % keep GM columns only
        clear Jw;
    end
    fprintf('Jacobian (GM only): keys %s, Jr size %s (of %d fwd nodes)\n', ...
            strjoin(keys(Jr), ','), mat2str(size(Jr(wlkey{1}))), size(node, 1));
    Jchrome = rbjacchrome(Jr, {'hbo', 'hbr'});               % struct .hbo/.hbr (wl-stacked)
    A = rbmatflat(Jchrome);                                  % [Npair*Nwl x 2*nrec]

    % canonical wavelength block order = keys(Jr) (the order rbmatflat stacks).
    % map each block to its char key (for detphi_base) and dOD wavelength index.
    canon = keys(Jr);
    nwl   = numel(canon);
    npair = size(pairs, 1);
    ck = cell(1, nwl);
    widx = zeros(1, nwl);
    for w = 1:nwl
        if ischar(canon{w})
            ck{w} = canon{w};
        else
            ck{w} = num2str(canon{w});
        end
        widx(w) = find(wl == str2double(ck{w}), 1);
    end

    % baseline measurements stacked the same way (wavelength blocks)
    ymodel = zeros(npair * nwl, 1);
    for w = 1:nwl
        bw = detphi_base(detkey(detphi_base, ck{w}));
        ymodel((w - 1) * npair + (1:npair)) = bw(sub2ind(size(bw), pairs(:, 2), pairs(:, 1)));
    end
    Anorm = A ./ ymodel;                                     % Rytov: d log(detphi)/d conc
    save(cachef, 'Anorm', 'rnode', 'nrec', 'npair', 'nwl', 'widx', '-v7.3');
    fprintf('cached recon operator: %s\n', cachef);
end  % cache if/else

%% ---- reconstruct each condition (one linear solve = one GN step) --------
% rbreginvunder adds lambda ABSOLUTELY to diag(A*A'); scale it relative to the
% operator (alpha plays the role of Cedalion's alpha_meas).
% depth (spatial) compensation: rescale columns by their sensitivity norm so deep
% (low-sensitivity) nodes are not suppressed -- the analog of Cedalion alpha_spatial.
% Solve in the weighted space  (A W) z = rhs, then dconc = W z.
cn   = sqrt(sum(Anorm.^2, 1))';                         % [2*nrec x 1] column norms
W    = 1 ./ (cn + beta * max(cn));
Aw   = Anorm .* W';
lambda = alpha * max(sum(Aw.^2, 2));                    % relative to weighted op
fprintf('alpha=%.3g beta=%.3g -> lambda=%.3e\n', alpha, beta, lambda);
ntrial = numel(trials);
recon_hbo = zeros(nrec, ntrial);
recon_hbr = zeros(nrec, ntrial);
for t = 1:ntrial
    rhs = zeros(npair * nwl, 1);                         % -dOD stacked by wavelength
    for w = 1:nwl
        rhs((w - 1) * npair + (1:npair)) = -dOD(:, widx(w), tidx, t);
    end
    z = rbreginv(Aw, rhs, lambda);                       % weighted-space solution
    dconc = z .* W;                                      % [2*nrec x 1] = [dHbO; dHbR]
    recon_hbo(:, t) = dconc(1:nrec);
    recon_hbr(:, t) = dconc(nrec + (1:nrec));
    fprintf('%-8s ||dOD||=%.3e  dHbO range [%.3f %.3f] uM\n', ...
            trials{t}, norm(rhs), min(recon_hbo(:, t)), max(recon_hbo(:, t)));
end

%% ---- map recon-mesh result onto the 25k cortex and save as JGIfTI -------
gt = loadbj(fullfile(here, 'groundtruth.bgii'));             % loadjd lacks .bgii ext
cortex = double(meshpart(gt.GIFTIData.MeshVertex3.Data));    % 25000 x3
cnode  = nearestnodes(rnode, cortex);                        % nearest recon node
props  = struct();
for t = 1:ntrial
    tag = strrep(strrep(trials{t}, 'Stim ', ''), ' ', '');   % C3 / C4
    props.(sprintf('HbO_%s', tag)) = single(recon_hbo(cnode, t));
    props.(sprintf('HbR_%s', tag)) = single(recon_hbr(cnode, t));
end
out = struct();
out.GIFTIHeader = struct('Version', '1.0', ...
                         'MetaData', struct('Description', 'redbird linear DOT reconstruction (dHbO/dHbR uM)', ...
                                            'LengthUnit', 'mm', 'ReconTime_s', taxis(tidx)));
out.GIFTIData.MeshVertex3 = struct('Data', single(cortex), 'Properties', props);
out.GIFTIData.MeshTri3    = struct('Data', gt.GIFTIData.MeshTri3.Data);
savebj('', out, 'FileName', fullfile(here, 'recon.bgii'), 'Compression', 'zlib');
fprintf('saved recon.bgii\n');
end

% =========================================================================
function [arr, val] = meshpart(x)
val = [];
if isstruct(x)
    arr = double(x.Data);
    if isfield(x, 'Properties') && isfield(x.Properties, 'Value')
        val = double(x.Properties.Value);
    end
else
    arr = double(x);
end
end

% -------------------------------------------------------------------------
function k = detkey(m, charkey)
% return the key of Map m matching wavelength charkey, honoring char/numeric keyType
if isKey(m, charkey)
    k = charkey;
    return
end
nk = str2double(charkey);
if isKey(m, nk)
    k = nk;
    return
end
error('wavelength key %s not found in detphi map', charkey);
end

% -------------------------------------------------------------------------
function [p, dir] = snap2surf(node, face, nv, p)
surfidx = unique(face(:));
k   = nearestnodes(node(surfidx, :), p);
ci  = surfidx(k);
dir = -nv(ci, :);
p   = node(ci, :) + 1.0 * dir;
end

% -------------------------------------------------------------------------
function idx = nearestnodes(node, q)
if exist('knnsearch', 'file')
    idx = knnsearch(node, q);
    return
end
idx = zeros(size(q, 1), 1);
for i = 1:size(q, 1)
    d = node - q(i, :);
    [~, idx(i)] = min(sum(d .* d, 2));
end
end

% -------------------------------------------------------------------------
function sd = build_sd(pairs, ns, wlkey)
% measured pairs as a per-wavelength redbird sd Map:
%   [src(1..Ns)  det(Ns+local)  active=1  cw=1]
tab = [pairs(:, 1), pairs(:, 2) + ns, ones(size(pairs, 1), 1), ones(size(pairs, 1), 1)];
sd = containers.Map();
for w = 1:numel(wlkey)
    sd(wlkey{w}) = tab;
end
end

% -------------------------------------------------------------------------
function [rnode, relem] = coarse_recon_mesh(node)
% coarse recon mesh; meshabox 3rd arg is MAX TET VOLUME (mm^3), not edge length.
% 300 mm^3 ~ 12 mm edge -> a few thousand nodes, far coarser than the forward mesh.
bb0 = min(node, [], 1);
bb1 = max(node, [], 1);
[rnode, ~, relem] = meshabox(bb0 - 1, bb1 + 1, 300);
relem = relem(:, 1:4);
end
