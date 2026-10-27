function [pos, dir, shift] = snapoptode(p, sv, sn, scalpmask, skinmask)
% Cedalion ForwardModel: direction = -(normal of the nearest scalp vertex); position =
% nearest scalp vertex, then the nearest scalp voxel to it (0-based voxel coords).
% With skinmask, the position is then moved to the nearest skin-surface voxel and
% shift reports that move (voxels).
c = mean(sv, 1);
pos = nearestvoxel(sv(nearestvertex(p, sv), :), scalpmask);
dir = zeros(size(p));
for i = 1:size(p, 1)
    iv = nearestvertex(p(i, :), sv);
    d = -sn(iv, :) / norm(sn(iv, :));
    if dot(d, c - sv(iv, :)) < 0                         % make sure it points inward
        d = -d;
    end
    dir(i, :) = d;
end
shift = zeros(size(p, 1), 1);
if nargin > 4
    ced = pos;
    pos = nearestvoxel(ced, skinmask);
    shift = sqrt(sum((pos - ced).^2, 2));
end
end

function iv = nearestvertex(p, sv)
iv = zeros(size(p, 1), 1);
for i = 1:size(p, 1)
    [~, iv(i)] = min(sum((sv - p(i, :)).^2, 2));
end
end

function pos = nearestvoxel(p, mask)
[ii, jj, kk] = ind2sub(size(mask), find(mask));
xyz = [ii, jj, kk] - 1;
pos = zeros(size(p));
for i = 1:size(p, 1)
    [~, ix] = min(sum((xyz - p(i, :)).^2, 2));
    pos(i, :) = xyz(ix, :);
end
end
