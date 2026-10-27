function [vox, vert] = readvox2vert(fname, dims)
% Matrix-Market (nvoxel x nvertex) 0/1 map; rows = C-order flat voxel index
tmp = tempname;
gunzip(fname, tmp);
[~, base] = fileparts(fname);
fid = fopen(fullfile(tmp, base), 'rt');
line = fgetl(fid);
while startsWith(line, '%')
    line = fgetl(fid);
end
sz = sscanf(line, '%d');
if sz(1) ~= prod(dims)
    error('voxel map has %d rows, volume has %d voxels', sz(1), prod(dims));
end
d = textscan(fid, '%f %f %f');
fclose(fid);
rmdir(tmp, 's');
r = d{1} - 1;                                             % 0-based C-order index
k = mod(r, dims(3));
j = mod(floor(r / dims(3)), dims(2));
i = floor(r / (dims(2) * dims(3)));
vox = sub2ind(dims, i + 1, j + 1, k + 1);
vert = d{2};
end
