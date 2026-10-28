function m = loadmeas(fname)
% LOADMEAS  Load a measurement/summary bundle: .mat or .jdb (JData; mcx_ref writes both,
% Cedalion's original ../redbird_recon/hrf_measurement.jdb is .jdb only); same fields.
[~, ~, ext] = fileparts(fname);
if strcmpi(ext, '.mat')
    m = load(fname);
else
    m = loadjd(fname);
end
end
