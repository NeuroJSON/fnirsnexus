function export_snirf(varargin)
% EXPORT_SNIRF  Write the MCX reference measurement as a SNIRF file with exactly the
% layout of Cedalion's cedalion_ref/synthetic_hrf.snirf, so SNIRF readers (Cedalion,
% Homer3, MNE-NIRS) can load it in place of the original.
%
%   export_snirf()                                  % dOD -> hrf_measurement.snirf
%   export_snirf('field', 'dOD_clean')              % activation only -> hrf_measurement_clean.snirf
%
% The probe (source/detector/landmark positions and labels), stim onsets and the
% measurement list are copied from the Cedalion file; only dataTimeSeries is replaced.
% Layout: one /nirs/data1 block, dataTimeSeries [190 time x 1988 channels], channel
% k = (trial-1)*994 + (pair-1)*2 + wavelength; measurementList(k) holds sourceIndex,
% detectorIndex, wavelengthIndex, dataType 99999, dataTypeLabel "HRF dOD" and
% dataTypeIndex = condition (1 "Stim C3", 2 "Stim C4", the names of stim1/stim2).
% Requires jsnirfy (https://github.com/NeuroJSON/jsnirfy) with its bundled EasyH5.

opt = struct('infile', '', ...           % [] = mcx_ref/hrf_measurement.mat
             'field', 'dOD', ...         % 'dOD' (R + activation) | 'dOD_clean' (activation only)
             'reffile', '', ...          % [] = ../cedalion_ref/synthetic_hrf.snirf
             'outfile', '', ...
             'jsnirfy', '');             % [] = ../../jsnirfy (sibling checkout)
for i = 1:2:numel(varargin)
    opt.(varargin{i}) = varargin{i + 1};
end
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
if isempty(opt.infile)
    opt.infile = fullfile(here, 'hrf_measurement.mat');
end
if isempty(opt.reffile)
    opt.reffile = fullfile(root, 'cedalion_ref', 'synthetic_hrf.snirf');
end
if isempty(opt.outfile)
    sfx = '';
    if strcmp(opt.field, 'dOD_clean')
        sfx = '_clean';
    end
    opt.outfile = fullfile(here, ['hrf_measurement' sfx '.snirf']);
end
if isempty(opt.jsnirfy)
    opt.jsnirfy = fullfile(root, '..', 'jsnirfy');
end
if ~exist('savesnirf', 'file')
    addpath(opt.jsnirfy, fullfile(opt.jsnirfy, 'external', 'easyh5'));
end

m = loadmeas(opt.infile);
dod = double(m.(opt.field));                               % [pair x wl x time x trial]
[npair, nwl, nt, ntrial] = size(dod);

s = loadsnirf(opt.reffile);
ml = s.nirs.data.measurementList;                         % channel order check vs the reference
src = [ml.sourceIndex];
det = [ml.detectorIndex];
wv = [ml.wavelengthIndex];
cond = [ml.dataTypeIndex];
[w, p, t] = ndgrid(1:nwl, 1:npair, 1:ntrial);
if numel(ml) ~= npair * nwl * ntrial || ~isequal(src(:), double(m.pairs(p(:), 1))) || ...
        ~isequal(det(:), double(m.pairs(p(:), 2))) || ~isequal(wv(:), w(:)) || ~isequal(cond(:), t(:))
    error('export_snirf: measurement list of %s does not match the pair order of %s', opt.reffile, opt.infile);
end
if ~isequal(cellstr([s.nirs.stim.name]), cellstr(m.trialtype)) || ...
        max(abs(double(s.nirs.data.time(:)) - double(m.time(:)))) > 1e-9
    error('export_snirf: stim names or time axis differ from the reference');
end

s.nirs.data.dataTimeSeries = reshape(permute(dod, [3 2 1 4]), nt, []);  % time x (wl, pair, trial)
s.nirs.metaDataTags.SubjectID = 'ICBM152';
s.nirs.metaDataTags.MeasurementDate = 'unknown';
s.nirs.metaDataTags.MeasurementTime = 'unknown';
s.nirs.metaDataTags.TimeUnit = 's';
s.nirs.metaDataTags.FrequencyUnit = 'Hz';
s.nirs.metaDataTags.Description = sprintf('%s [%s]', m.Description, opt.field);
s.nirs.metaDataTags.Generator = 'fnirsnexus/mcx_ref: hrf_measurement -> export_snirf.m (jsnirfy)';
s.formatVersion = '1.1';
s.nirs.aux = [];                                           % none; savesnirf expects the field
if exist(opt.outfile, 'file')
    delete(opt.outfile);                                   % EasyH5 appends to an existing file
end
savesnirf(s, opt.outfile);
fid = H5F.open(opt.outfile, 'H5F_ACC_RDWR', 'H5P_DEFAULT');  % drop the placeholder aux node
if H5L.exists(fid, '/nirs/aux', 'H5P_DEFAULT')
    H5L.delete(fid, '/nirs/aux', 'H5P_DEFAULT');
end
H5F.close(fid);
fprintf('saved %s (%s: %d time x %d channels)\n', opt.outfile, opt.field, size(s.nirs.data.dataTimeSeries));
end
