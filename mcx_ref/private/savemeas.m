function savemeas(fname, out)
% SAVEMEAS  Write a measurement/summary bundle as both MATLAB v7 .mat (one variable per
% field) and binary JData .jdb (zlib-compressed), with identical content.
[p, n] = fileparts(fname);
save(fullfile(p, [n '.mat']), '-struct', 'out', '-v7');
savebj('', out, 'FileName', fullfile(p, [n '.jdb']), 'Compression', 'zlib');
end
