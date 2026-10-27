function [R, phi0, dphidz] = drefread(P, phi)
% [R, phi0, dphidz] = drefread(P, phi)
% Diffuse reflectance (1/mm^2 per launched photon) under a detector disk, from the
% fluence column/volume phi, using the probe P of drefprobe (Kienle1997 Eq. 7, n=1):
%     R = phi0/4 + (D/2) dphi/dz
v = double(phi(P.idx));
ok = v > 0;
if nnz(ok) < 4
    R = NaN;
    phi0 = NaN;
    dphidz = NaN;
    return
end
c = P.X(ok, :) \ log(v(ok));
phi0 = exp(c(1));
dphidz = c(2) * phi0;
R = 0.25 * phi0 + 0.5 * P.D * dphidz;
end
