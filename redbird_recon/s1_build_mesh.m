function s1_build_mesh(outfile)
% S1_BUILD_MESH  Build a tetrahedral ICBM-152 head mesh for redbird FEM recon.
%
%   s1_build_mesh()             % writes head_mesh.bmsh in this folder
%   s1_build_mesh('mesh.jmsh')  % custom output (.jmsh text or .bmsh binary)
%
% Cedalion's ICBM-152 head model is surface + voxel based (it runs a Monte-Carlo
% forward), so it contains NO tetrahedral mesh; redbird is FEM and needs one. We
% build it from the 5 ICBM-152 tissue masks shipped with the example using iso2mesh
% vol2mesh (CGAL). This reuses Cedalion's segmentation and yields a full 5-tissue
% model (1=scalp 2=bone 3=CSF 4=gray 5=white), which is more faithful than the
% 2-tissue mesh obtainable from the brain+scalp surfaces (and avoids the tetgen
% "invalid PLC" self-intersections those decimated surfaces trigger when merged).
%
% Node coordinates are mapped voxel(IJK)->RAS(mm) via the NIfTI affine so the mesh
% shares the SNIRF optode frame. Isolated nodes are removed (they create zero rows
% -> singular FEM matrix).
%
% Requires iso2mesh (vol2mesh, removeisolatednode, savejmesh) and jsonlab/jnifti.

if nargin < 1
    outfile = fullfile(fileparts(mfilename('fullpath')), 'head_mesh.bmsh');
end
here = fileparts(mfilename('fullpath'));
hm   = fullfile(here, '..', 'hm_icbm152');

%% 1. combine the 5 tissue masks into one labelled volume (inner overwrites outer)
masknames = {'mask_skin', 'mask_bone', 'mask_csf', 'mask_gray', 'mask_white'};
seg = [];
nii = [];
for i = 1:numel(masknames)
    nii = loadnifti(fullfile(hm, [masknames{i} '.nii']));
    vol = nii.NIFTIData;
    if isempty(seg)
        seg = zeros(size(vol), 'uint8');
    end
    seg(vol > 0) = i;                                    % 1..5
end
fprintf('segmentation %s, labels %s\n', mat2str(size(seg)), mat2str(unique(seg(:))'));

%% 2. tetrahedral mesh (CGAL); raise maxvol for a coarser/faster mesh
opt = struct('radbound', 2, 'distbound', 1);
maxvol = 30;
[node, elem, face] = vol2mesh(seg, 1:size(seg, 1), 1:size(seg, 2), 1:size(seg, 3), ...
                              opt, maxvol, 1, 'cgalmesh');

%% 3. drop isolated nodes (reindex only the index columns, keep label/bid columns)
elabel = elem(:, 5);
fbid = face(:, 4);
[node, e4, f3] = removeisolatednode(node, elem(:, 1:4), face(:, 1:3));
elem = [e4, elabel];
face = [f3, fbid];
fprintf('mesh: %d nodes, %d tets, regions %s\n', ...
        size(node, 1), size(elem, 1), mat2str(unique(elem(:, 5))'));

%% 4. map node coordinates voxel(IJK)->RAS(mm)
A = getaffine(nii);
node(:, 1:3) = (A(1:3, 1:3) * (node(:, 1:3) - 1)' + A(1:3, 4))';
fprintf('node RAS bbox: x[%.1f %.1f] y[%.1f %.1f] z[%.1f %.1f]\n', ...
        min(node(:, 1)), max(node(:, 1)), min(node(:, 2)), max(node(:, 2)), ...
        min(node(:, 3)), max(node(:, 3)));

%% 5. save as JMesh (tissue label in elem 5th column)
savejmesh(node, face, elem, outfile, 'Dimension', 3, ...
          'MeshTitle', 'ICBM-152 head FEM mesh (RAS mm)', ...
          'Comment', 'scalp=1 bone=2 csf=3 gray=4 white=5');
fprintf('saved %s\n', outfile);
end

% -------------------------------------------------------------------------
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
