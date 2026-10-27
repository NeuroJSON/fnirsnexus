function P = drefprobe(vol, pos, dir, rdet, prop)
% P = drefprobe(vol, pos, dir, rdet, prop)
% Geometry for reading the diffuse reflectance under a detector disk from an
% interior fluence (Kienle & Patterson 1997, Eq. 7). With a matched boundary
% (n=1, R_fres=0, as in Cedalion):
%     R = Phi(0)/4 + (D/2) dPhi/dz        (z along the inward normal)
% Phi(0) and dPhi/dz come from a fit  ln Phi = a + b z + c u + d w  over the tissue
% voxels under the disk (|lateral| <= rdet, 0 < z <= zmax), u/w the lateral axes, so
% the intercept is the value at the disk centre on the skin surface.
%
% pos: [n x 3] optode positions (0-based grid, issrcfrom0=1); dir: [n x 3] inward unit
% normals; rdet: disk radius (voxels); prop: [label+1 x 4] optics (D of the skin voxels)
% P(i): idx (voxel linear indices), X (design matrix), D (diffusion coefficient), b0
% (skin surface point). Use drefread(P(i), phi) to get R for any fluence volume phi.

zmax = 3.5;
outside = ~imfill(vol > 0, 'holes');
dims = size(vol);
n = size(pos, 1);
P = repmat(struct('idx', [], 'X', [], 'D', 0, 'b0', [0 0 0]), n, 1);
for i = 1:n
    d = dir(i, :) / norm(dir(i, :));
    % skin surface along the normal: march outward until reaching exterior air
    p = pos(i, :);
    for k = 1:400
        q = p - 0.025 * d;
        v = floor(q) + 1;
        if any(v < 1) || any(v > dims) || outside(v(1), v(2), v(3))
            break
        end
        p = q;
    end
    b0 = p;                                               % last point inside tissue
    % lateral axes
    [~, m] = min(abs(d));
    u = zeros(1, 3);
    u(m) = 1;
    u = u - dot(u, d) * d;
    u = u / norm(u);
    w = cross(d, u);
    c = floor(b0) + 1;
    rr = ceil(max(rdet, 1) + zmax + 1);
    [gi, gj, gk] = ndgrid(max(c(1) - rr, 1):min(c(1) + rr, dims(1)), ...
                          max(c(2) - rr, 1):min(c(2) + rr, dims(2)), ...
                          max(c(3) - rr, 1):min(c(3) + rr, dims(3)));
    g = [gi(:), gj(:), gk(:)];
    idx = sub2ind(dims, g(:, 1), g(:, 2), g(:, 3));
    rel = (g - 0.5) - b0;                                 % voxel centres, 0-based grid
    z = rel * d';
    lu = rel * u';
    lw = rel * w';
    for rlat = [max(rdet, 1), 1.5 * max(rdet, 1), 2 * max(rdet, 1)]
        sel = vol(idx) > 0 & z > 0 & z <= zmax & (lu.^2 + lw.^2) <= rlat^2;
        if nnz(sel) >= 8
            break
        end
    end
    lab = mode(double(vol(idx(sel & z <= 1.5))));
    if isnan(lab)
        lab = 3;
    end
    P(i).idx = idx(sel);
    P(i).X = [ones(nnz(sel), 1), z(sel), lu(sel), lw(sel)];
    P(i).D = 1 / (3 * (prop(lab + 1, 1) + prop(lab + 1, 2) * (1 - prop(lab + 1, 3))));
    P(i).b0 = b0;
end
end
