function A = getaffine(nii)
h = nii.NIFTIHeader;
if isfield(h, 'Affine') && ~isempty(h.Affine)
    A = h.Affine;
    if size(A, 1) == 3
        A = [A; 0 0 0 1];
    end
else
    A = eye(4);
    if isfield(h, 'QuaternOffset')
        A(1:3, 4) = [h.QuaternOffset.x; h.QuaternOffset.y; h.QuaternOffset.z];
    end
end
end
