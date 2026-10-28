function [maxOD_ced, K, reltime] = mcx_build_hrf(varargin)
% MCX_BUILD_HRF  Turn the MCX peak dOD (mcx_jacobian.m) into a block-averaged HRF
% measurement with exactly Cedalion's temporal workflow, and write it in the
% redbird-hrf-dod schema (drop-in for ../redbird_recon/hrf_measurement.jdb).
%
%   mcx_build_hrf()                  % -> hrf_measurement.mat/.jdb (+ figures/)
%   maxOD_ced = mcx_build_hrf('fitonly', true)  % only fit/validate vs Cedalion
%
% Cedalion's notebook: syn(t) = maxOD * reg_trial(t) (Gamma HRF tau=0 sigma=3 T=3
% convolved with 10 s boxcars, scaled to max 1) is added to the resting OD, band-pass
% filtered (Butterworth 4, 0.02-0.5 Hz, sosfiltfilt), epoched [-1, 20] s by linear
% interpolation, baseline-corrected and averaged. Because od_augm carries a
% trial_type dimension, BOTH conditions average all 14 epochs, so
%     hrf_C(t) = R(t) + K_C(t) * maxOD_C,     C in {C3, C4}
% with a common resting-state block average R and kernels K_C = pipeline(reg_C).
% We port the (noise-free) pipeline, fit maxOD_C^cedalion from hrf_C3-hrf_C4 (exactly
% rank 2), recover R, and rebuild
%     hrf_C^mcx(t) = R(t) + pipeline( dOD_C^mcx(a = reg_C(t)) )
% using the nonlinear MCX dOD(amplitude) table. The fit residual validates the port.
% The resting background (real physiology/noise and c_meas) is thus kept identical.

opt = struct('fitonly', false, ...
             'mcxfile', '', ...
             'outfile', '', ...
             'restfile', '', ...          % only its time axis is used
             'fband', [0.02 0.5], ...
             'butter_order', 4, ...
             'epoch', [1 20]);            % seconds before/after onset
for i = 1:2:numel(varargin)
    opt.(varargin{i}) = varargin{i + 1};
end
here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
if isempty(opt.mcxfile)
    opt.mcxfile = fullfile(here, 'mcx_jacobian.mat');
end
if isempty(opt.outfile)
    opt.outfile = fullfile(here, 'hrf_measurement.mat');
end
if isempty(opt.restfile)
    % the 3309-sample run (0..368 s) matches Cedalion's nn22 resting-state recording
    opt.restfile = fullfile(root, 'nn22_resting_state', 'sub-08', 'nirs', ...
                            'sub-08_task-RS_run-02_nirs.snirf');
end

%% ---- inputs ----------------------------------------------------------------
meas = loadjd(fullfile(root, 'redbird_recon', 'hrf_measurement.jdb'));
trials = meas.trialtype;
if ~iscell(trials)
    trials = cellstr(trials);
end
dOD = double(meas.dOD);                                   % pair x wl x reltime x trial
[npair, nwl, nrel, ntrial] = size(dOD);
time = double(h5read(opt.restfile, '/nirs/data1/time'));
time = time(:)';
fs = 1 / mean(diff(time));

stim = readstim(fullfile(root, 'cedalion_ref', 'synthetic_hrf.snirf'), trials);
onset_all = sort([stim.onset]);
fprintf('%d samples, fs=%.4f Hz; %d events (%s)\n', numel(time), fs, numel(onset_all), ...
        strjoin(cellfun(@(c, s) sprintf('%s:%d', c, numel(s.onset)), trials, ...
                        num2cell(stim), 'UniformOutput', false), ', '));

%% ---- temporal kernels K_C = pipeline(reg_C) --------------------------------
hrf = gammabasis(fs, 0, 3, 3);
reg = zeros(ntrial, numel(time));
for t = 1:ntrial
    reg(t, :) = hrfregressor(time, stim(t).onset, stim(t).duration, stim(t).value, hrf);
end
[sos, g] = butterband(opt.fband / (fs / 2), opt.butter_order);
[K, reltime] = blockavg(reg, time, fs, onset_all, opt.epoch, sos, g);   % ntrial x nrel
if max(abs(reltime - double(meas.time(:))')) > 1e-6
    error('reltime axis mismatch with hrf_measurement.jdb');
end

%% ---- fit Cedalion's peak dOD and the common resting term R -----------------
Y1 = reshape(permute(dOD(:, :, :, 1), [3 1 2]), nrel, []);  % nrel x (pair*wl)
Y2 = reshape(permute(dOD(:, :, :, 2), [3 1 2]), nrel, []);
M = [K(1, :)', -K(2, :)'];
coef = M \ (Y1 - Y2);                                      % 2 x (pair*wl)
res = norm(Y1 - Y2 - M * coef, 'fro') / norm(Y1 - Y2, 'fro');
fprintf('temporal-port check: rank-2 fit residual of hrf_C3-hrf_C4 = %.2e\n', res);
if res > 1e-3
    warning('temporal port does not reproduce Cedalion (residual %.2e)', res);
end
maxOD_ced = reshape(coef', npair, nwl, ntrial);            % pair x wl x trial
R = 0.5 * ((Y1 - K(1, :)' * coef(1, :)) + (Y2 - K(2, :)' * coef(2, :)));
R = permute(reshape(R, nrel, npair, nwl), [2 3 1]);        % pair x wl x reltime
if opt.fitonly
    return
end

%% ---- MCX synthetic signal through the same pipeline -------------------------
mc = loadmeas(opt.mcxfile);
if ~isequal(double(mc.pairs), double(meas.pairs)) || ~mc.complete
    error('%s is incomplete or does not match the measurement pairs', opt.mcxfile);
end
amp = double(mc.amp(:))';
tab = double(mc.dOD_amp);                                  % pair x wl x trial x amp
maxOD = double(mc.maxOD);
dOD_clean = zeros(npair, nwl, nrel, ntrial);
for t = 1:ntrial
    % nonlinear dOD(a(t)): linear interpolation on the uniform amplitude grid
    T = reshape(tab(:, :, t, :), npair * nwl, []);
    x = reg(t, :) * (numel(amp) - 1);
    i0 = min(floor(x), numel(amp) - 2);
    f = x - i0;
    syn = T(:, i0 + 1) .* (1 - f) + T(:, i0 + 2) .* f;    % (pair*wl) x ntime
    ba = blockavg(syn, time, fs, onset_all, opt.epoch, sos, g);
    dOD_clean(:, :, :, t) = reshape(ba, npair, nwl, nrel);
end
dOD_new = R + dOD_clean;

r = corrcoef(maxOD(:), maxOD_ced(:));
r = r(1, 2);
pos = abs(maxOD_ced(:)) > 0.1 * max(abs(maxOD_ced(:)));
fprintf('MCX vs Cedalion peak dOD: corr %.4f, median ratio %.3f (strong channels)\n', ...
        r, median(maxOD(pos) ./ maxOD_ced(pos)));

out = meas;
out.dOD = single(dOD_new);                                 % R + MCX signal (Cedalion-like)
out.dOD_clean = single(dOD_clean);                         % MCX signal only, no rest noise
out.maxOD = maxOD;                                         % MCX peak dOD (a = 1)
out.maxOD_se = double(mc.maxOD_se);
out.maxOD_cedalion = maxOD_ced;                            % Adot peak dOD (fitted)
out.Description = ['Cedalion synthetic-HRF workflow with the channel-space activation ' ...
                   'recomputed by MCX perturbation MC on the voxelated ICBM-152 head'];
[~, mcname, mcext] = fileparts(opt.mcxfile);
out.Generator = sprintf('fnirsnexus/mcx_ref: %s (%s%s) -> mcx_build_hrf.m', mc.format, mcname, mcext);
if isfield(mc, 'settings')
    out.ForwardSettings = mc.settings;                     % MCX forward settings
end
out.TemporalSettings = opt;
savemeas(opt.outfile, out);                % .mat + .jdb
fprintf('saved %s\n', opt.outfile);

plotcompare(fullfile(here, 'figures'), maxOD, maxOD_ced, double(meas.wavelengths), ...
            trials, reltime, dOD, dOD_new, dOD_clean);
end

% -------------------------------------------------------------------------
function stim = readstim(fname, trials)
info = h5info(fname, '/nirs');
stim = repmat(struct('onset', [], 'duration', [], 'value', []), 1, numel(trials));
for i = 1:numel(info.Groups)
    gname = info.Groups(i).Name;
    if isempty(regexp(gname, '/stim\d+$', 'once'))
        continue
    end
    name = h5read(fname, [gname '/name']);
    if iscell(name)
        name = name{1};
    end
    t = find(strcmp(strtrim(name), trials));
    d = h5read(fname, [gname '/data']);                   % 3 x nevent (onset;dur;value)
    if size(d, 1) ~= 3
        d = d';
    end
    stim(t) = struct('onset', d(1, :), 'duration', d(2, :), 'value', d(3, :));
end
end

function r = gammabasis(fs, tau, sigma, T)
% cedalion.models.glm.Gamma: x^2 exp(-x^2), x=(t-tau)/sigma, boxcar T, max-normalized
n = ceil((4.1 * sigma + tau + T) * fs);
th = (0:n - 1) / fs;
x2 = ((th - tau) / sigma).^2;
r = x2 .* exp(-x2);
r(th < tau) = 0;
if T > 0
    w = fix(T * fs);
    r = conv(r, ones(1, w) / w);
    r = r(1:n);
end
r = r / max(r);
end

function reg = hrfregressor(time, onset, duration, value, hrf)
% cedalion design_matrix.hrf_regressors + build_stim_array (basis starts at t=0)
s = zeros(size(time));
for i = 1:numel(onset)
    i0 = find(time >= onset(i), 1);                       % numpy searchsorted (left)
    i1 = find(time >= onset(i) + duration(i), 1);
    if isempty(i1)
        i1 = numel(time) + 1;
    end
    i1 = max(i1, i0 + 1);
    s(i0:i1 - 1) = value(i);
end
reg = conv(s, hrf);
reg = reg(1:numel(time));
reg = reg / max(reg) * max(value);                        % hrf_regs *= stim.value.max()
end

function [sos, g] = butterband(wn, order)
[z, p, k] = butter(order, wn, 'bandpass');
[sos, g] = zp2sos(z, p, k);
end

function [ba, reltime] = blockavg(x, time, fs, onsets, win, sos, g)
% filter (rows of x = channels) -> epochs (cedalion to_epochs) -> baseline -> mean
y = filtfilt(sos, g, x')';
edges = [time(1) - 0.5 * (time(2) - time(1)), 0.5 * (time(1:end - 1) + time(2:end)), ...
         time(end) + 0.5 * (time(end) - time(end - 1))];
nb = ceil(win(1) * fs);
na = ceil(win(2) * fs);
reltime = (-nb:na) * round(1 / fs, 3);
ba = zeros(size(x, 1), numel(reltime));
ne = 0;
for e = 1:numel(onsets)
    oi = find(edges <= onsets(e), 1, 'last') - 1;         % 0-based sample of onset
    if oi - nb < 0 || oi + na >= numel(time)
        continue
    end
    ep = interp1(time, y', onsets(e) + reltime, 'linear', 'extrap')';
    ba = ba + ep - mean(ep(:, reltime < 0), 2);
    ne = ne + 1;
end
ba = ba / ne;
end

function plotcompare(figdir, maxOD, maxOD_ced, wl, trials, reltime, dOD_old, dOD_new, dOD_clean)
wl = wl(:)';
if ~exist(figdir, 'dir')
    mkdir(figdir);
end
fig = figure('Visible', 'off', 'Position', [0 0 1400 420]);
subplot(1, 3, 1);
hold on;
mk = {'o', 's'};
for t = 1:numel(trials)
    for w = 1:numel(wl)
        plot(maxOD_ced(:, w, t), maxOD(:, w, t), mk{t}, 'MarkerSize', 3, ...
             'DisplayName', sprintf('%s %g nm', trials{t}, wl(w)));
    end
end
lim = [min([maxOD(:); maxOD_ced(:)]) max([maxOD(:); maxOD_ced(:)])];
plot(lim, lim, 'k--', 'HandleVisibility', 'off');
axis tight;
grid on;
xlabel('Cedalion Adot peak \DeltaOD');
ylabel('MCX peak \DeltaOD');
legend('Location', 'northwest');
for t = 1:numel(trials)
    [~, ip] = max(abs(maxOD(:, 1, t)));
    subplot(1, 3, 1 + t);
    plot(reltime, squeeze(dOD_old(ip, :, :, t))', '--', reltime, ...
         squeeze(dOD_new(ip, :, :, t))', '-', reltime, squeeze(dOD_clean(ip, :, :, t))', ':');
    grid on;
    xlabel('t_{rel} (s)');
    ylabel('\DeltaOD');
    title(sprintf('%s, pair %d', trials{t}, ip));
    legend([arrayfun(@(x) sprintf('Cedalion %g', x), wl, 'UniformOutput', false), ...
            arrayfun(@(x) sprintf('MCX %g', x), wl, 'UniformOutput', false), ...
            arrayfun(@(x) sprintf('MCX clean %g', x), wl, 'UniformOutput', false)], ...
           'Location', 'northwest', 'FontSize', 7);
end
print(fig, fullfile(figdir, 'mcx_vs_cedalion.png'), '-dpng', '-r120');
close(fig);
end
